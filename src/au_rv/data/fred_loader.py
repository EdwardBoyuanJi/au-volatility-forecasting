from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from au_rv.io import utc_now, write_parquet

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _date_chunks(start_date, end_date, years: int):
    cursor = pd.Timestamp(start_date).normalize()
    final = pd.Timestamp(end_date).normalize()
    while cursor <= final:
        chunk_end = min(cursor + pd.DateOffset(years=int(years)) - pd.Timedelta(days=1), final)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _download_initial_release_series(
    series_id: str,
    api_key: str,
    start_date,
    end_date,
    conservative_time_et: str,
) -> pd.DataFrame:
    parameters = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "realtime_end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        "observation_start": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "observation_end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        "output_type": 4,
        # FRED documents a lower format-specific limit for JSON output. Page
        # explicitly so a multi-year daily history cannot be silently cut off.
        "limit": 2000,
    }
    observations: list[dict] = []
    offset = 0
    while True:
        page_parameters = {**parameters, "offset": offset}
        response = requests.get(
            FRED_OBSERVATIONS_URL, params=page_parameters, timeout=60
        )
        if not response.ok:
            try:
                detail = response.json().get("error_message", "")
            except (ValueError, AttributeError):
                detail = ""
            suffix = f": {detail}" if detail else ""
            # Do not call raise_for_status(): requests includes the complete URL,
            # including the FRED API key, in its exception string.
            raise RuntimeError(
                f"FRED {series_id} request failed with HTTP {response.status_code}{suffix}"
            )
        payload = response.json()
        page = payload.get("observations", [])
        observations.extend(page)
        offset += len(page)
        total = int(payload.get("count", len(observations)))
        if not page or offset >= total:
            break
    if not observations:
        raise RuntimeError(f"FRED returned no observations for {series_id}.")
    frame = pd.DataFrame(observations)
    frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    frame["observation_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "realtime_start", "value"]).copy()
    frame["series_id"] = series_id
    frame["timestamp_original"] = (
        frame["realtime_start"].dt.strftime("%Y-%m-%d") + f" {conservative_time_et}"
    )
    frame["timezone_original"] = "America/New_York"
    local = pd.to_datetime(frame["timestamp_original"]).dt.tz_localize(
        ZoneInfo("America/New_York"), ambiguous="infer", nonexistent="shift_forward"
    )
    frame["timestamp_utc"] = local.dt.tz_convert("UTC")
    frame["timestamp_shanghai"] = local.dt.tz_convert("Asia/Shanghai")
    frame["available_at"] = frame["timestamp_utc"]
    frame["retrieved_at_utc"] = utc_now()
    return frame[
        [
            "series_id",
            "observation_date",
            "value",
            "realtime_start",
            "timestamp_original",
            "timezone_original",
            "timestamp_utc",
            "timestamp_shanghai",
            "available_at",
            "retrieved_at_utc",
        ]
    ]


def download_fred_data(config: dict, start_date, end_date) -> Path:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is missing.")
    settings = config["data_sources"]["fred"]
    if not settings["use_initial_release_vintage"]:
        raise ValueError("Formal mode requires use_initial_release_vintage=true.")
    frames = []
    for series_id in settings["series"].values():
        for chunk_start, chunk_end in _date_chunks(
            start_date,
            end_date,
            years=int(settings.get("vintage_chunk_years", 4)),
        ):
            frames.append(
                _download_initial_release_series(
                    series_id,
                    key,
                    chunk_start,
                    chunk_end,
                    settings["conservative_available_time_et"],
                )
            )
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["series_id", "observation_date", "available_at"]
    )
    combined = combined.drop_duplicates(["series_id", "observation_date"], keep="first")
    root = Path(config["_project_root"])
    return write_parquet(combined, root / "data/raw/fred_initial_release_daily.parquet")
