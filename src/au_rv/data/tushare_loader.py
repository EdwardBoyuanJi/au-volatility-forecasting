from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from au_rv.config import resolve_path
from au_rv.io import utc_now, write_parquet
from au_rv.time_utils import add_timestamp_columns

MINUTE_FIELDS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount", "oi"]


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=days - 1, hours=23, minutes=59), end)
        yield cursor, chunk_end
        cursor = chunk_end.normalize() + pd.Timedelta(days=1)


def _normalize_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    result = calendar.copy()
    result["cal_date"] = pd.to_datetime(result["cal_date"], format="%Y%m%d", errors="coerce")
    result["pretrade_date"] = pd.to_datetime(
        result.get("pretrade_date"), format="%Y%m%d", errors="coerce"
    )
    result["timestamp_original"] = result["cal_date"].dt.strftime("%Y-%m-%d 00:00:00")
    result["timezone_original"] = "Asia/Shanghai"
    local = result["cal_date"].dt.tz_localize("Asia/Shanghai")
    result["timestamp_utc"] = local.dt.tz_convert("UTC")
    result["timestamp_shanghai"] = local
    # Exchange holiday calendars are normally published before the year starts.
    # Jan 1 of the calendar year is a conservative, explicit availability rule.
    known = pd.to_datetime(result["cal_date"].dt.year.astype(str) + "-01-01 23:59:59")
    result["available_at"] = known.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    return result


def _normalize_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    result = contracts.copy()
    for column in ("list_date", "delist_date", "last_ddate"):
        if column in result:
            result[column] = pd.to_datetime(result[column], format="%Y%m%d", errors="coerce")
    result["timestamp_original"] = result["list_date"].dt.strftime("%Y-%m-%d 00:00:00")
    result["timezone_original"] = "Asia/Shanghai"
    local = result["list_date"].dt.tz_localize("Asia/Shanghai")
    result["timestamp_utc"] = local.dt.tz_convert("UTC")
    result["timestamp_shanghai"] = local
    result["available_at"] = result["timestamp_utc"]
    return result


def download_tushare_data(
    config: dict,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Path]:
    """Download real AU contracts, SHFE calendar, and 5-minute bars.

    Every true AU contract overlapping the requested range is downloaded.
    Contract selection is deliberately deferred to feature construction, where
    only t-1 open interest and volume are allowed.
    """

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing.")
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements.txt before downloading Tushare data.") from exc

    settings = config["data_sources"]["tushare"]
    root = Path(config["_project_root"])
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    future_end = end.normalize() + pd.Timedelta(
        days=int(config["dates"]["future_calendar_buffer_days"])
    )

    pro = ts.pro_api(token)
    contracts = pro.fut_basic(
        exchange=settings["exchange"],
        fut_type="1",
        fields=(
            "ts_code,symbol,exchange,name,fut_code,list_date,delist_date,"
            "last_ddate,trade_time_desc"
        ),
    )
    product = settings["product"].upper()
    contracts = contracts[
        contracts["fut_code"].astype(str).str.upper().eq(product)
        | contracts["symbol"].astype(str).str.upper().str.startswith(product)
    ].copy()
    contracts = _normalize_contracts(contracts)
    contract_path = write_parquet(contracts, raw / "tushare_fut_basic_au.parquet")

    calendar = pro.fut_trade_cal(
        exchange=settings["exchange"],
        start_date=start.strftime("%Y%m%d"),
        end_date=future_end.strftime("%Y%m%d"),
    )
    calendar = _normalize_calendar(calendar)
    calendar_path = write_parquet(calendar, raw / "tushare_shfe_trade_calendar.parquet")

    overlapping = contracts[
        contracts["list_date"].le(end.normalize())
        & contracts["delist_date"].ge(start.normalize())
    ].copy()
    pieces: list[pd.DataFrame] = []
    for row in overlapping.itertuples(index=False):
        contract_start = max(start, pd.Timestamp(row.list_date))
        contract_end = min(end, pd.Timestamp(row.delist_date) + pd.Timedelta(hours=23, minutes=59))
        for chunk_start, chunk_end in _date_chunks(
            contract_start, contract_end, int(settings["chunk_days"])
        ):
            frame = pro.ft_mins(
                ts_code=row.ts_code,
                freq=settings["minute_frequency"],
                start_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                fields=",".join(MINUTE_FIELDS),
            )
            if frame is not None and not frame.empty:
                missing = set(MINUTE_FIELDS).difference(frame.columns)
                if missing:
                    raise ValueError(f"Tushare ft_mins missing fields: {sorted(missing)}")
                pieces.append(frame[MINUTE_FIELDS].copy())
            time.sleep(float(settings["request_pause_seconds"]))

    if not pieces:
        raise RuntimeError("Tushare returned no AU 5-minute rows for the requested range.")
    bars = pd.concat(pieces, ignore_index=True)
    bars = bars.drop_duplicates(["ts_code", "trade_time"], keep="last")
    bars = add_timestamp_columns(bars, "trade_time", "Asia/Shanghai")
    bars["available_at"] = bars["timestamp_utc"]
    bars["retrieved_at_utc"] = utc_now()
    for column in ("open", "high", "low", "close", "vol", "amount", "oi"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.sort_values(["ts_code", "timestamp_utc"]).reset_index(drop=True)
    bars_path = write_parquet(bars, raw / "tushare_au_5m.parquet")
    return {"contracts": contract_path, "calendar": calendar_path, "bars": bars_path}
