from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.data.tqsdk_loader import _create_api
from au_rv.features.realized_variance import prepare_au_bars
from au_rv.io import read_parquet, utc_now, write_parquet


def normalise_au_option_catalog(symbol_info: pd.DataFrame) -> pd.DataFrame:
    result = symbol_info.copy()
    if "instrument_id" not in result.columns:
        result = result.reset_index()
        if "instrument_id" not in result.columns:
            result = result.rename(columns={result.columns[0]: "instrument_id"})
    result = result[
        result["instrument_id"].astype(str).str.match(r"^SHFE\.au\d{4}[CP]\d+$")
    ].copy()
    result["ts_code"] = result["instrument_id"].astype(str)
    result["underlying_symbol"] = result["underlying_symbol"].astype(str)
    result["option_class"] = result["option_class"].astype(str).str.upper()
    result["strike_price"] = pd.to_numeric(result["strike_price"], errors="coerce")
    expiry_seconds = pd.to_numeric(result["expire_datetime"], errors="coerce")
    expiry_utc = pd.to_datetime(expiry_seconds, unit="s", utc=True, errors="coerce")
    result["expiry_timestamp_shanghai"] = expiry_utc.dt.tz_convert("Asia/Shanghai")
    result["expiry_date"] = (
        result["expiry_timestamp_shanghai"].dt.tz_localize(None).dt.normalize()
    )
    for column in ("price_tick", "volume_multiple"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["retrieved_at_utc"] = utc_now()
    keep = [
        "ts_code",
        "instrument_id",
        "instrument_name",
        "underlying_symbol",
        "option_class",
        "strike_price",
        "expiry_timestamp_shanghai",
        "expiry_date",
        "price_tick",
        "volume_multiple",
        "delivery_year",
        "delivery_month",
        "last_exercise_datetime",
        "expired",
        "retrieved_at_utc",
    ]
    keep = [column for column in keep if column in result.columns]
    return (
        result[keep]
        .dropna(subset=["ts_code", "underlying_symbol", "strike_price", "expiry_date"])
        .drop_duplicates("ts_code", keep="last")
        .sort_values(["expiry_date", "underlying_symbol", "strike_price", "option_class"])
        .reset_index(drop=True)
    )


def build_au_futures_daily(
    bars: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    prepared = prepare_au_bars(bars, calendar).sort_values(
        ["trade_date", "ts_code", "timestamp_utc"]
    )
    return (
        prepared.groupby(["trade_date", "ts_code"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("vol", lambda values: values.sum(min_count=1)),
            close_oi=("oi", "last"),
            first_timestamp_utc=("timestamp_utc", "first"),
            last_timestamp_utc=("timestamp_utc", "last"),
        )
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )


def select_atm_option_pairs(
    catalog: pd.DataFrame,
    futures_daily: pd.DataFrame,
    open_trade_dates: pd.Series,
    *,
    horizons: list[int],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        pd.to_datetime(open_trade_dates, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    futures = futures_daily.copy()
    futures["trade_date"] = pd.to_datetime(futures["trade_date"]).dt.normalize()
    futures_lookup = futures.set_index(["trade_date", "ts_code"])
    pairs = catalog[
        catalog["option_class"].isin(["CALL", "PUT"])
    ].copy()
    pair_counts = pairs.groupby(
        ["underlying_symbol", "expiry_date", "strike_price"]
    )["option_class"].nunique()
    complete_index = pair_counts[pair_counts.eq(2)].index
    pairs = pairs.set_index(
        ["underlying_symbol", "expiry_date", "strike_price"]
    ).loc[complete_index].reset_index()
    start = pd.Timestamp(start_date).normalize() if start_date is not None else dates.min()
    end = pd.Timestamp(end_date).normalize() if end_date is not None else dates.max()
    rows: list[dict] = []
    for (underlying, expiry), group in pairs.groupby(
        ["underlying_symbol", "expiry_date"], sort=True
    ):
        expiry = pd.Timestamp(expiry).normalize()
        expiry_position = int(dates.searchsorted(expiry, side="left"))
        if expiry_position >= len(dates) or dates[expiry_position] != expiry:
            continue
        for horizon in sorted(set(int(value) for value in horizons)):
            signal_position = expiry_position - horizon
            if signal_position < 0:
                continue
            signal_date = pd.Timestamp(dates[signal_position]).normalize()
            entry_position = signal_position + 1
            if signal_date < start or signal_date > end or entry_position >= len(dates):
                continue
            key = (signal_date, str(underlying))
            if key not in futures_lookup.index:
                continue
            future = futures_lookup.loc[key]
            if isinstance(future, pd.DataFrame):
                future = future.iloc[-1]
            future_close = float(future["close"])
            strike_distances = (
                group[["strike_price"]]
                .drop_duplicates()
                .assign(distance=lambda frame: (frame["strike_price"] - future_close).abs())
                .sort_values(["distance", "strike_price"])
            )
            if strike_distances.empty:
                continue
            strike = float(strike_distances.iloc[0]["strike_price"])
            selected = group[group["strike_price"].eq(strike)]
            calls = selected[selected["option_class"].eq("CALL")]
            puts = selected[selected["option_class"].eq("PUT")]
            if calls.empty or puts.empty:
                continue
            rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": pd.Timestamp(dates[entry_position]).normalize(),
                    "expiry_date": expiry,
                    "horizon": horizon,
                    "underlying_symbol": str(underlying),
                    "signal_future_close": future_close,
                    "strike_price": strike,
                    "moneyness_abs": abs(strike / future_close - 1.0),
                    "call_symbol": str(calls.iloc[0]["ts_code"]),
                    "put_symbol": str(puts.iloc[0]["ts_code"]),
                    "price_tick": float(selected["price_tick"].dropna().iloc[0]),
                    "volume_multiple": float(
                        selected["volume_multiple"].dropna().iloc[0]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["signal_date", "horizon", "underlying_symbol"]
    ).reset_index(drop=True)


def normalise_option_daily_bars(
    bars: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    result = bars.copy()
    numeric_timestamp = pd.to_numeric(result["datetime"], errors="coerce")
    result["timestamp_utc"] = pd.to_datetime(
        numeric_timestamp, unit="ns", utc=True, errors="coerce"
    )
    result["timestamp_shanghai"] = result["timestamp_utc"].dt.tz_convert(
        "Asia/Shanghai"
    )
    result["trade_date"] = (
        result["timestamp_shanghai"].dt.tz_localize(None).dt.normalize()
    )
    result["ts_code"] = symbol
    result["retrieved_at_utc"] = utc_now()
    for column in ("open", "high", "low", "close", "volume", "open_oi", "close_oi"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    keep = [
        "trade_date",
        "ts_code",
        "timestamp_utc",
        "timestamp_shanghai",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_oi",
        "close_oi",
        "retrieved_at_utc",
    ]
    return (
        result[keep]
        .dropna(subset=["trade_date", "open", "close"])
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )


def download_au_option_strategy_data(config: dict) -> dict[str, Path]:
    settings = config["data_sources"]["au_options"]
    root = Path(config["_project_root"])
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    futures_bars = read_parquet(raw / "tqsdk_au_5m.parquet")
    calendar = read_parquet(raw / "tqsdk_shfe_trade_calendar.parquet")
    futures_daily = build_au_futures_daily(futures_bars, calendar)

    api = _create_api()
    try:
        symbols = [
            str(value)
            for value in api.query_quotes(ins_class="OPTION", exchange_id="SHFE")
            if str(value).startswith("SHFE.au")
        ]
        if not symbols:
            raise RuntimeError("TqSdk returned no SHFE AU option contracts.")
        catalog_pieces = []
        batch_size = int(settings.get("metadata_batch_size", 200))
        for offset in range(0, len(symbols), batch_size):
            catalog_pieces.append(
                api.query_symbol_info(symbols[offset : offset + batch_size])
            )
        catalog = normalise_au_option_catalog(
            pd.concat(catalog_pieces, ignore_index=True)
        )
        catalog_path = write_parquet(
            catalog, raw / "tqsdk_au_option_catalog.parquet"
        )
        open_dates = calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"]
        selections = select_atm_option_pairs(
            catalog,
            futures_daily,
            open_dates,
            horizons=[int(value) for value in config["features"]["horizons"]],
            start_date=settings.get("selection_start_date"),
            end_date=settings.get("selection_end_date"),
        )
        if selections.empty:
            raise RuntimeError("No ATM AU option pairs could be selected.")
        selection_path = write_parquet(
            selections, raw / "tqsdk_au_option_strategy_pairs.parquet"
        )

        symbol_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for row in selections.itertuples(index=False):
            for symbol in (row.call_symbol, row.put_symbol):
                start = pd.Timestamp(row.signal_date).normalize()
                end = pd.Timestamp(row.expiry_date).normalize() + pd.Timedelta(days=1)
                if symbol in symbol_windows:
                    prior_start, prior_end = symbol_windows[symbol]
                    symbol_windows[symbol] = (min(start, prior_start), max(end, prior_end))
                else:
                    symbol_windows[symbol] = (start, end)
        pieces: list[pd.DataFrame] = []
        for number, (symbol, (start, end)) in enumerate(
            sorted(symbol_windows.items()), start=1
        ):
            frame = api.get_kline_data_series(
                symbol=symbol,
                duration_seconds=86400,
                start_dt=start.to_pydatetime(),
                end_dt=end.to_pydatetime(),
            )
            if frame is not None and not frame.empty:
                pieces.append(normalise_option_daily_bars(frame, symbol))
            pause = float(settings.get("request_pause_seconds", 0.0))
            if pause > 0 and number < len(symbol_windows):
                time.sleep(pause)
        if not pieces:
            raise RuntimeError("TqSdk returned no selected AU option daily bars.")
        daily = (
            pd.concat(pieces, ignore_index=True)
            .drop_duplicates(["trade_date", "ts_code"], keep="last")
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )
        option_path = write_parquet(
            daily, raw / "tqsdk_au_option_daily_strategy.parquet"
        )
        futures_path = write_parquet(
            futures_daily, raw / "tqsdk_au_futures_daily_all_contracts.parquet"
        )
    finally:
        api.close()
    return {
        "catalog": catalog_path,
        "pairs": selection_path,
        "option_daily": option_path,
        "futures_daily": futures_path,
    }
