from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pandas as pd

from au_rv.data.tqsdk_loader import _create_api
from au_rv.io import read_parquet, utc_now, write_csv, write_parquet
from au_rv.strategy_v2.data import normalise_tick_frame, opportunity_id


def prepare_real_quote_opportunities(opportunities: pd.DataFrame) -> pd.DataFrame:
    result = opportunities.copy()
    for column in ("signal_date", "entry_date", "expiry_date"):
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    result["opportunity_id"] = [opportunity_id(row) for _, row in result.iterrows()]
    return result


def _combine_date(day: pd.Timestamp, clock: time) -> datetime:
    return datetime.combine(pd.Timestamp(day).date(), clock)


def _previous_trade_date(calendar_dates: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    prior = calendar_dates[calendar_dates < pd.Timestamp(date).normalize()]
    if len(prior) == 0:
        return pd.Timestamp(date).normalize() - pd.Timedelta(days=1)
    return pd.Timestamp(prior[-1]).normalize()


def _save_progress(
    raw_dir: Path,
    existing_ticks: pd.DataFrame,
    new_tick_pieces: list[pd.DataFrame],
    coverage_rows: list[dict],
) -> pd.DataFrame:
    pieces = [existing_ticks] if not existing_ticks.empty else []
    pieces.extend(new_tick_pieces)
    ticks = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not ticks.empty:
        ticks = (
            ticks.drop_duplicates(
                ["opportunity_id", "leg_role", "window_type", "timestamp_utc"],
                keep="last",
            )
            .sort_values(["opportunity_id", "window_type", "timestamp_utc", "leg_role"])
            .reset_index(drop=True)
        )
        write_parquet(ticks, raw_dir / "real_quote_ticks.parquet")
    coverage_path = raw_dir / "download_coverage.csv"
    old_coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    coverage = pd.concat([old_coverage, pd.DataFrame(coverage_rows)], ignore_index=True)
    if not coverage.empty:
        coverage = coverage.drop_duplicates("opportunity_id", keep="last")
        write_csv(coverage, coverage_path)
    return ticks


def download_real_quote_ticks(
    opportunities: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    project_root: Path,
    raw_dir_relative: str,
    flush_every: int,
) -> dict[str, Path]:
    raw_dir = Path(project_root) / raw_dir_relative
    raw_dir.mkdir(parents=True, exist_ok=True)
    opportunities = prepare_real_quote_opportunities(opportunities)
    candidates_path = write_parquet(opportunities, raw_dir / "opportunities_snapshot.parquet")
    tick_path = raw_dir / "real_quote_ticks.parquet"
    existing_ticks = read_parquet(tick_path) if tick_path.exists() else pd.DataFrame()
    coverage_path = raw_dir / "download_coverage.csv"
    attempted = set()
    if coverage_path.exists():
        attempted = set(pd.read_csv(coverage_path)["opportunity_id"].astype(str))
    calendar_date_column = "trade_date" if "trade_date" in calendar else "date"
    if "trading" in calendar:
        calendar = calendar[calendar["trading"].astype(bool)]
    elif "is_open" in calendar:
        calendar = calendar[calendar["is_open"].eq(1)]
    calendar_dates = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(calendar[calendar_date_column])
            .dt.normalize()
            .unique()
        )
    )
    api = _create_api()
    buffer: list[pd.DataFrame] = []
    coverage_buffer: list[dict] = []
    completed = 0
    try:
        for row in opportunities.itertuples(index=False):
            oid = str(row.opportunity_id)
            if oid in attempted:
                continue
            entry_date = pd.Timestamp(row.entry_date).normalize()
            prior_trade = _previous_trade_date(calendar_dates, entry_date)
            expiry_date = pd.Timestamp(row.expiry_date).normalize()
            legs = {
                "short_call": str(row.call_symbol),
                "short_put": str(row.put_symbol),
                "underlying_future": str(row.underlying_symbol),
            }
            windows = [
                (
                    "entry_night",
                    _combine_date(prior_trade, time(20, 55)),
                    _combine_date(prior_trade, time(21, 6)),
                    tuple(legs),
                ),
                (
                    "entry_day",
                    _combine_date(entry_date, time(8, 55)),
                    _combine_date(entry_date, time(9, 6)),
                    tuple(legs),
                ),
                (
                    "expiry_close",
                    _combine_date(expiry_date, time(14, 54)),
                    _combine_date(expiry_date, time(15, 1)),
                    ("underlying_future",),
                ),
            ]
            counts: dict[str, int] = {}
            for window_name, start, end, roles in windows:
                for role in roles:
                    symbol = legs[role]
                    frame = api.get_tick_data_series(
                        symbol=symbol,
                        start_dt=start,
                        end_dt=end,
                    )
                    count = 0 if frame is None else len(frame)
                    counts[f"{window_name}_{role}"] = count
                    if frame is None or frame.empty:
                        continue
                    buffer.append(
                        normalise_tick_frame(
                            frame,
                            symbol=symbol,
                            opportunity=oid,
                            role=role,
                            window=window_name,
                        )
                    )
            coverage_buffer.append(
                {
                    "opportunity_id": oid,
                    "signal_date": row.signal_date,
                    "entry_date": row.entry_date,
                    "expiry_date": row.expiry_date,
                    **counts,
                    "downloaded_at_utc": utc_now(),
                }
            )
            completed += 1
            if completed % int(flush_every) == 0:
                existing_ticks = _save_progress(
                    raw_dir, existing_ticks, buffer, coverage_buffer
                )
                buffer = []
                coverage_buffer = []
                print(f"saved {completed} newly attempted opportunities")
    finally:
        if buffer or coverage_buffer:
            existing_ticks = _save_progress(
                raw_dir, existing_ticks, buffer, coverage_buffer
            )
        api.close()
    if existing_ticks.empty:
        raise RuntimeError("No historical real-quote ticks were downloaded.")
    return {
        "opportunities": candidates_path,
        "ticks": tick_path,
        "coverage": coverage_path,
    }


def load_real_quote_ticks(project_root: Path, raw_dir_relative: str) -> pd.DataFrame:
    return read_parquet(Path(project_root) / raw_dir_relative / "real_quote_ticks.parquet")
