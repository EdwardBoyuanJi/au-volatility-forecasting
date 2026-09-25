from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.io import utc_now, write_parquet


def _datetime_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=days), end)
        yield cursor, chunk_end
        cursor = chunk_end


def _normalise_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "trading"}
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(f"TqSdk trading calendar missing fields: {sorted(missing)}")
    result = calendar.copy()
    result["cal_date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["is_open"] = result["trading"].astype(bool).astype(int)
    open_date = result["cal_date"].where(result["is_open"].eq(1))
    result["pretrade_date"] = open_date.ffill().shift(1)
    result["timestamp_original"] = result["cal_date"].dt.strftime("%Y-%m-%d 00:00:00")
    result["timezone_original"] = "Asia/Shanghai"
    local = result["cal_date"].dt.tz_localize("Asia/Shanghai")
    result["timestamp_utc"] = local.dt.tz_convert("UTC")
    result["timestamp_shanghai"] = local
    # Exchange holiday schedules are generally published before the year begins.
    # This explicit conservative rule is used only for point-in-time auditing.
    known = pd.to_datetime(result["cal_date"].dt.year.astype(str) + "-01-01 23:59:59")
    result["available_at"] = known.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    result["retrieved_at_utc"] = utc_now()
    return result.sort_values("cal_date").reset_index(drop=True)


def _normalise_contracts(symbol_info: pd.DataFrame) -> pd.DataFrame:
    result = symbol_info.copy()
    if "instrument_id" not in result.columns:
        result = result.reset_index()
        if "instrument_id" not in result.columns:
            index_column = result.columns[0]
            result = result.rename(columns={index_column: "instrument_id"})
    result["ts_code"] = result["instrument_id"].astype(str)
    expiry_seconds = pd.to_numeric(result.get("expire_datetime"), errors="coerce")
    expiry_utc = pd.to_datetime(expiry_seconds, unit="s", utc=True, errors="coerce")
    result["delist_date"] = expiry_utc.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    result["last_ddate"] = result["delist_date"]
    result["list_date"] = pd.NaT
    result["timestamp_original"] = result["instrument_id"].astype(str)
    result["timezone_original"] = "Asia/Shanghai"
    result["timestamp_utc"] = pd.NaT
    result["timestamp_shanghai"] = pd.NaT
    result["available_at"] = utc_now()
    result["retrieved_at_utc"] = utc_now()
    return result.sort_values("instrument_id").reset_index(drop=True)


def _eligible_contracts(
    contracts: pd.DataFrame,
    start_date,
    end_date,
    *,
    listing_lead_buffer_days: int,
) -> list[str]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    delivery_year = pd.to_numeric(contracts.get("delivery_year"), errors="coerce")
    delivery_month = pd.to_numeric(contracts.get("delivery_month"), errors="coerce")
    delivery_month_start = pd.to_datetime(
        {
            "year": delivery_year,
            "month": delivery_month,
            "day": pd.Series(1, index=contracts.index),
        },
        errors="coerce",
    )
    expiry = pd.to_datetime(contracts.get("delist_date"), errors="coerce")
    not_expired_before_sample = expiry.isna() | expiry.ge(start - pd.Timedelta(days=7))
    not_too_far_after_sample = delivery_month_start.isna() | delivery_month_start.le(
        end + pd.Timedelta(days=int(listing_lead_buffer_days))
    )
    product_match = contracts["instrument_id"].astype(str).str.match(
        r"^SHFE\.au\d{4}$", na=False
    )
    selected = contracts.loc[
        product_match & not_expired_before_sample & not_too_far_after_sample,
        "instrument_id",
    ]
    return sorted(selected.astype(str).unique().tolist())


def _normalise_bars(
    bars: pd.DataFrame,
    symbol: str,
    *,
    duration_seconds: int,
) -> pd.DataFrame:
    required = {"datetime", "open", "high", "low", "close", "volume", "close_oi"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"TqSdk K-line response missing fields: {sorted(missing)}")
    result = bars.copy()
    numeric_timestamp = pd.to_numeric(result["datetime"], errors="coerce")
    result["timestamp_utc"] = pd.to_datetime(
        numeric_timestamp, unit="ns", utc=True, errors="coerce"
    )
    result["timestamp_shanghai"] = result["timestamp_utc"].dt.tz_convert(
        "Asia/Shanghai"
    )
    result["timestamp_original"] = result["timestamp_shanghai"].astype(str)
    result["timezone_original"] = "Asia/Shanghai"
    result["available_at"] = result["timestamp_utc"] + pd.Timedelta(
        seconds=int(duration_seconds)
    )
    result["retrieved_at_utc"] = utc_now()
    result["ts_code"] = symbol
    result["trade_time"] = result["timestamp_shanghai"].dt.tz_localize(None)
    result["vol"] = pd.to_numeric(result["volume"], errors="coerce")
    result["oi"] = pd.to_numeric(result["close_oi"], errors="coerce")
    result["amount"] = np.nan
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    keep = [
        "ts_code",
        "trade_time",
        "timestamp_original",
        "timezone_original",
        "timestamp_utc",
        "timestamp_shanghai",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "oi",
        "retrieved_at_utc",
    ]
    result = result[keep].dropna(
        subset=["timestamp_utc", "open", "high", "low", "close"]
    )
    return result.sort_values("timestamp_utc").reset_index(drop=True)


def _create_api():
    user = os.getenv("TQ_USER")
    password = os.getenv("TQ_PASSWORD")
    if not user or not password:
        raise RuntimeError("TQ_USER and TQ_PASSWORD are required for TqSdk.")
    try:
        from tqsdk import TqApi, TqAuth
    except ImportError as exc:  # pragma: no cover - exercised in a real download
        raise RuntimeError("Install requirements.txt before using TqSdk.") from exc
    return TqApi(auth=TqAuth(user, password))


def download_tqsdk_data(
    config: dict,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Path]:
    """Download SHFE AU contracts, calendar, and 5-minute bars from TqSdk Pro.

    All true contracts that can plausibly overlap the requested sample are
    requested. The representative contract is selected later from prior-day
    observed open interest and volume, never from a continuous back-adjusted
    series or future information.
    """

    settings = config["data_sources"]["tqsdk"]
    root = Path(config["_project_root"])
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    calendar_start = start - pd.Timedelta(days=7)
    calendar_end = end + pd.Timedelta(
        days=int(config["dates"]["future_calendar_buffer_days"])
    )
    # Include the evening preceding the first requested trade date. End at the
    # model cutoff so the next trade day's night session is not pulled in.
    download_start = calendar_start + pd.Timedelta(hours=20)
    cutoff_hour, cutoff_minute = map(
        int, config["project"]["model_cutoff_time"].split(":")[:2]
    )
    download_end = end + pd.Timedelta(hours=cutoff_hour, minutes=cutoff_minute)
    duration_seconds = int(settings["duration_seconds"])

    api = _create_api()
    try:
        symbols = api.query_quotes(
            ins_class="FUTURE",
            exchange_id=settings["exchange"],
            product_id=settings["product"],
        )
        if not symbols:
            raise RuntimeError("TqSdk returned no SHFE AU futures contracts.")
        contracts = _normalise_contracts(api.query_symbol_info(list(symbols)))
        contract_path = write_parquet(contracts, raw / "tqsdk_fut_basic_au.parquet")

        calendar = _normalise_calendar(
            api.get_trading_calendar(
                start_dt=calendar_start.date(), end_dt=calendar_end.date()
            )
        )
        calendar_path = write_parquet(
            calendar, raw / "tqsdk_shfe_trade_calendar.parquet"
        )

        eligible = _eligible_contracts(
            contracts,
            start,
            end,
            listing_lead_buffer_days=int(settings["listing_lead_buffer_days"]),
        )
        if not eligible:
            raise RuntimeError("No TqSdk AU contracts overlap the requested sample.")
        pieces: list[pd.DataFrame] = []
        for symbol in eligible:
            expiry = contracts.loc[
                contracts["instrument_id"].astype(str).eq(symbol), "delist_date"
            ]
            symbol_end = download_end
            if not expiry.empty and pd.notna(expiry.iloc[0]):
                symbol_end = min(
                    symbol_end,
                    pd.Timestamp(expiry.iloc[0]) + pd.Timedelta(days=1),
                )
            if symbol_end <= download_start:
                continue
            for chunk_start, chunk_end in _datetime_chunks(
                download_start, symbol_end, int(settings["chunk_days"])
            ):
                try:
                    frame = api.get_kline_data_series(
                        symbol=symbol,
                        duration_seconds=duration_seconds,
                        start_dt=chunk_start.to_pydatetime(),
                        end_dt=chunk_end.to_pydatetime(),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"TqSdk failed for {symbol}, {chunk_start} to {chunk_end}: {exc}"
                    ) from exc
                if frame is not None and not frame.empty:
                    pieces.append(
                        _normalise_bars(
                            frame,
                            symbol,
                            duration_seconds=duration_seconds,
                        )
                    )
                pause = float(settings.get("request_pause_seconds", 0.0))
                if pause > 0:
                    time.sleep(pause)
        if not pieces:
            raise RuntimeError("TqSdk returned no AU 5-minute rows for the requested range.")
        bars = pd.concat(pieces, ignore_index=True)
        bars = bars.drop_duplicates(["ts_code", "timestamp_utc"], keep="last")
        bars = bars.sort_values(["ts_code", "timestamp_utc"]).reset_index(drop=True)
        bars_path = write_parquet(bars, raw / "tqsdk_au_5m.parquet")
    finally:
        api.close()
    return {"contracts": contract_path, "calendar": calendar_path, "bars": bars_path}
