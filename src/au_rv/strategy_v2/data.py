from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.data.au_option_loader import normalise_option_daily_bars
from au_rv.data.tqsdk_loader import _create_api
from au_rv.io import read_parquet, utc_now, write_parquet


def opportunity_id(row: pd.Series) -> str:
    signal = pd.Timestamp(row["signal_date"]).strftime("%Y%m%d")
    return f"{signal}_h{int(row['horizon'])}_{row['underlying_symbol'].split('.')[-1]}_k{int(row['strike_price'])}"


def select_protective_wings(
    opportunities: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    wing_width_fraction: float,
) -> pd.DataFrame:
    result = opportunities.copy()
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.normalize()
    catalog = catalog.copy()
    catalog["expiry_date"] = pd.to_datetime(catalog["expiry_date"]).dt.normalize()
    rows: list[dict] = []
    for _, row in result[result["signal_distribution_short"].fillna(False)].iterrows():
        contracts = catalog[
            catalog["underlying_symbol"].astype(str).eq(str(row["underlying_symbol"]))
            & catalog["expiry_date"].eq(row["expiry_date"])
        ]
        target_width = float(row["signal_future_close"]) * float(wing_width_fraction)
        put_candidates = contracts[
            contracts["option_class"].eq("PUT")
            & contracts["strike_price"].le(float(row["strike_price"]) - target_width)
        ].sort_values("strike_price", ascending=False)
        call_candidates = contracts[
            contracts["option_class"].eq("CALL")
            & contracts["strike_price"].ge(float(row["strike_price"]) + target_width)
        ].sort_values("strike_price")
        if put_candidates.empty or call_candidates.empty:
            continue
        put = put_candidates.iloc[0]
        call = call_candidates.iloc[0]
        record = row.to_dict()
        record.update(
            {
                "opportunity_id": opportunity_id(row),
                "put_wing_symbol": str(put["ts_code"]),
                "put_wing_strike": float(put["strike_price"]),
                "call_wing_symbol": str(call["ts_code"]),
                "call_wing_strike": float(call["strike_price"]),
                "put_wing_width": float(row["strike_price"] - put["strike_price"]),
                "call_wing_width": float(call["strike_price"] - row["strike_price"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def normalise_tick_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    opportunity: str,
    role: str,
    window: str,
) -> pd.DataFrame:
    result = frame.copy()
    numeric = pd.to_numeric(result["datetime"], errors="coerce")
    result["timestamp_utc"] = pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
    result["timestamp_shanghai"] = result["timestamp_utc"].dt.tz_convert("Asia/Shanghai")
    result["ts_code"] = symbol
    result["opportunity_id"] = opportunity
    result["leg_role"] = role
    result["window_type"] = window
    result["retrieved_at_utc"] = utc_now()
    numeric_columns = [
        "last_price",
        "bid_price1",
        "bid_volume1",
        "ask_price1",
        "ask_volume1",
        "volume",
        "open_interest",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    keep = [
        "opportunity_id",
        "leg_role",
        "window_type",
        "ts_code",
        "timestamp_utc",
        "timestamp_shanghai",
        *numeric_columns,
        "retrieved_at_utc",
    ]
    return result[keep].dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")


def _combine_date(day: pd.Timestamp, clock: time) -> datetime:
    return datetime.combine(pd.Timestamp(day).date(), clock)


def download_strategy_v2_market_data(
    candidates: pd.DataFrame,
    *,
    project_root: Path,
) -> dict[str, Path]:
    raw_dir = Path(project_root) / "data/raw/strategy_v2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = _create_api()
    tick_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    try:
        wing_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for row in candidates.itertuples(index=False):
            for symbol in (row.put_wing_symbol, row.call_wing_symbol):
                start = pd.Timestamp(row.signal_date).normalize()
                end = pd.Timestamp(row.expiry_date).normalize() + pd.Timedelta(days=1)
                if symbol in wing_windows:
                    prior_start, prior_end = wing_windows[symbol]
                    wing_windows[symbol] = (min(prior_start, start), max(prior_end, end))
                else:
                    wing_windows[symbol] = (start, end)
        for symbol, (start, end) in sorted(wing_windows.items()):
            daily = api.get_kline_data_series(
                symbol=symbol,
                duration_seconds=86400,
                start_dt=start.to_pydatetime(),
                end_dt=end.to_pydatetime(),
            )
            if daily is not None and not daily.empty:
                daily_pieces.append(normalise_option_daily_bars(daily, symbol))

        for row in candidates.itertuples(index=False):
            legs = {
                "short_call": row.call_symbol,
                "short_put": row.put_symbol,
                "long_call_wing": row.call_wing_symbol,
                "long_put_wing": row.put_wing_symbol,
                "underlying_future": row.underlying_symbol,
            }
            signal_date = pd.Timestamp(row.signal_date).normalize()
            entry_date = pd.Timestamp(row.entry_date).normalize()
            expiry_date = pd.Timestamp(row.expiry_date).normalize()
            windows = [
                (
                    "signal_close",
                    _combine_date(signal_date, time(14, 54)),
                    _combine_date(signal_date, time(15, 1)),
                    {"short_call", "short_put"},
                ),
                (
                    "entry_night",
                    _combine_date(entry_date - pd.Timedelta(days=1), time(20, 55)),
                    _combine_date(entry_date - pd.Timedelta(days=1), time(21, 6)),
                    set(legs),
                ),
                (
                    "entry_day",
                    _combine_date(entry_date, time(8, 55)),
                    _combine_date(entry_date, time(9, 6)),
                    set(legs),
                ),
                (
                    "expiry_close",
                    _combine_date(expiry_date, time(14, 54)),
                    _combine_date(expiry_date, time(15, 1)),
                    {"underlying_future"},
                ),
            ]
            for window_name, start, end, included_roles in windows:
                for role in sorted(included_roles):
                    symbol = str(legs[role])
                    frame = api.get_tick_data_series(
                        symbol=symbol,
                        start_dt=start,
                        end_dt=end,
                    )
                    if frame is None or frame.empty:
                        continue
                    tick_pieces.append(
                        normalise_tick_frame(
                            frame,
                            symbol=symbol,
                            opportunity=str(row.opportunity_id),
                            role=role,
                            window=window_name,
                        )
                    )
    finally:
        api.close()
    if not daily_pieces:
        raise RuntimeError("No protective-wing daily bars were downloaded.")
    if not tick_pieces:
        raise RuntimeError("No strategy-v2 execution ticks were downloaded.")
    wing_daily = (
        pd.concat(daily_pieces, ignore_index=True)
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
    )
    ticks = (
        pd.concat(tick_pieces, ignore_index=True)
        .drop_duplicates(
            ["opportunity_id", "leg_role", "window_type", "timestamp_utc"],
            keep="last",
        )
        .sort_values(["opportunity_id", "window_type", "timestamp_utc", "leg_role"])
    )
    candidates_path = write_parquet(candidates, raw_dir / "strategy_v2_candidates.parquet")
    daily_path = write_parquet(wing_daily, raw_dir / "strategy_v2_wing_daily.parquet")
    tick_path = write_parquet(ticks, raw_dir / "strategy_v2_execution_ticks.parquet")
    return {"candidates": candidates_path, "wing_daily": daily_path, "ticks": tick_path}


def load_strategy_v2_market_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(project_root) / "data/raw/strategy_v2"
    return (
        read_parquet(raw_dir / "strategy_v2_candidates.parquet"),
        read_parquet(raw_dir / "strategy_v2_wing_daily.parquet"),
        read_parquet(raw_dir / "strategy_v2_execution_ticks.parquet"),
    )
