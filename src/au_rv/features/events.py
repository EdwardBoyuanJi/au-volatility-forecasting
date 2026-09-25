from __future__ import annotations

import pandas as pd

from au_rv.time_utils import model_cutoff


def build_event_counts(
    events: pd.DataFrame,
    all_open_trade_dates: pd.Series,
    feature_trade_dates: pd.Series,
    *,
    horizons: list[int],
    cutoff_time: str,
) -> pd.DataFrame:
    open_dates = pd.Index(
        sorted(
            pd.to_datetime(pd.Series(all_open_trade_dates))
            .dropna()
            .dt.normalize()
            .unique()
        )
    )
    event_frame = events.copy()
    event_frame["timestamp_utc"] = pd.to_datetime(event_frame["timestamp_utc"], utc=True)
    event_frame["available_at"] = pd.to_datetime(event_frame["available_at"], utc=True)
    event_frame["event_year"] = event_frame["timestamp_utc"].dt.tz_convert(
        "America/New_York"
    ).dt.year
    coverage = {
        event_type: set(
            event_frame.loc[event_frame["event_type"].eq(event_type), "event_year"].unique()
        )
        for event_type in ("cpi", "nfp", "fomc")
    }
    rows: list[dict] = []
    normalized_feature_dates = (
        pd.to_datetime(pd.Series(feature_trade_dates)).dropna().dt.normalize().unique()
    )
    for trade_date in sorted(normalized_feature_dates):
        cutoff = model_cutoff(trade_date, cutoff_time)
        cutoff_utc = cutoff.tz_convert("UTC")
        position = int(open_dates.searchsorted(pd.Timestamp(trade_date), side="left"))
        row: dict = {
            "trade_date": pd.Timestamp(trade_date),
            "calendar_available_at": cutoff_utc,
            "calendar_missing_flag": False,
        }
        for horizon in horizons:
            if position >= len(open_dates) or position + horizon >= len(open_dates):
                for event_type in ("cpi", "nfp", "fomc"):
                    row[f"{event_type}_count_{horizon}d"] = pd.NA
                row["calendar_missing_flag"] = True
                continue
            end_date = pd.Timestamp(open_dates[position + horizon])
            # "End of the h-th AU trade day" is the same post-close model
            # cutoff, not calendar-day midnight. A US release later that
            # Shanghai evening belongs beyond the realized target window.
            end_local = model_cutoff(end_date, cutoff_time)
            end_utc = end_local.tz_convert("UTC")
            years = set(range(cutoff.year, end_local.year + 1))
            if any(not years.issubset(coverage[event_type]) for event_type in coverage):
                row["calendar_missing_flag"] = True
            known = event_frame[event_frame["available_at"].le(cutoff_utc)]
            in_window = known[
                known["timestamp_utc"].gt(cutoff_utc)
                & known["timestamp_utc"].le(end_utc)
            ]
            for event_type in ("cpi", "nfp", "fomc"):
                row[f"{event_type}_count_{horizon}d"] = int(
                    in_window["event_type"].eq(event_type).sum()
                )
        rows.append(row)
    return pd.DataFrame(rows)
