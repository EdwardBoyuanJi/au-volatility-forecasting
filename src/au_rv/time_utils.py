from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")


def model_cutoff(trade_date: str | date | pd.Timestamp, cutoff_time: str = "15:05") -> pd.Timestamp:
    day = pd.Timestamp(trade_date).date()
    hour, minute = map(int, cutoff_time.split(":")[:2])
    return pd.Timestamp.combine(day, time(hour, minute)).tz_localize(SHANGHAI)


def add_timestamp_columns(
    frame: pd.DataFrame,
    source_column: str,
    original_timezone: str,
    *,
    original_column: str = "timestamp_original",
) -> pd.DataFrame:
    result = frame.copy()
    original = pd.to_datetime(result[source_column], errors="coerce")
    zone = ZoneInfo(original_timezone)
    if getattr(original.dt, "tz", None) is None:
        localized = original.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    else:
        localized = original.dt.tz_convert(zone)
    result[original_column] = result[source_column].astype(str)
    result["timezone_original"] = original_timezone
    result["timestamp_utc"] = localized.dt.tz_convert("UTC")
    result["timestamp_shanghai"] = localized.dt.tz_convert("Asia/Shanghai")
    return result


def _next_open_date(day: date, open_dates: np.ndarray) -> pd.Timestamp | pd.NaT:
    value = np.datetime64(day)
    position = int(np.searchsorted(open_dates, value, side="left"))
    if position >= len(open_dates):
        return pd.NaT
    return pd.Timestamp(open_dates[position])


def assign_shfe_trade_date(
    timestamps_shanghai: pd.Series,
    open_trade_dates: pd.Series | list,
) -> pd.Series:
    """Map SHFE night bars to the next open trade date.

    Bars at or after 21:00 map to the first open date strictly after their
    calendar date. Bars after midnight and day-session bars map to the first
    open date on or after their calendar date. This also handles Friday-night
    bars and holiday gaps without using natural-date grouping.
    """

    stamps = pd.to_datetime(timestamps_shanghai)
    if getattr(stamps.dt, "tz", None) is None:
        stamps = stamps.dt.tz_localize("Asia/Shanghai")
    dates = pd.to_datetime(pd.Series(open_trade_dates)).dt.normalize().drop_duplicates().sort_values()
    open_values = dates.to_numpy(dtype="datetime64[D]")
    mapped: list[pd.Timestamp | pd.NaT] = []
    for stamp in stamps:
        if pd.isna(stamp):
            mapped.append(pd.NaT)
            continue
        calendar_day = stamp.date()
        if stamp.time() >= time(21, 0):
            start = calendar_day + pd.Timedelta(days=1)
            mapped.append(_next_open_date(start, open_values))
        else:
            mapped.append(_next_open_date(calendar_day, open_values))
    return pd.to_datetime(pd.Series(mapped, index=timestamps_shanghai.index)).dt.normalize()


def shfe_session_label(timestamp_shanghai: pd.Timestamp, trade_date: pd.Timestamp) -> str | None:
    stamp = pd.Timestamp(timestamp_shanghai)
    clock = stamp.time()
    day = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    if clock >= time(21, 0) or clock <= time(2, 30):
        return f"{day}:night"
    if time(9, 0) <= clock <= time(10, 15):
        return f"{day}:day_am1"
    if time(10, 30) <= clock <= time(11, 30):
        return f"{day}:day_am2"
    if time(13, 30) <= clock <= time(15, 0):
        return f"{day}:day_pm"
    return None


def availability_mask(frame: pd.DataFrame, available_column: str, cutoff_column: str) -> pd.Series:
    available = pd.to_datetime(frame[available_column], utc=True, errors="coerce")
    cutoff = pd.to_datetime(frame[cutoff_column], utc=True, errors="coerce")
    return available.notna() & cutoff.notna() & available.le(cutoff)
