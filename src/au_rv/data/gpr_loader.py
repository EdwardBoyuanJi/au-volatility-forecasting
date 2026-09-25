from __future__ import annotations

import io
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from au_rv.config import resolve_path
from au_rv.io import utc_now, write_parquet


def normalise_gpr_frame(
    frame: pd.DataFrame,
    *,
    release_lag_days: int,
    conservative_time_et: str,
) -> pd.DataFrame:
    """Normalise the author's current daily GPR workbook.

    The public workbook is republished weekly and recent observations can be
    revised.  The author exposes vintages, but downloading hundreds of complete
    3 MB workbooks would be disproportionate for a daily indicator.  We use the
    current workbook with a conservative seven-calendar-day observable lag and
    preserve an explicit vintage warning in every row.
    """

    required = {"date", "GPRD"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GPR workbook missing columns: {sorted(missing)}")
    result = frame.copy()
    result["observation_date"] = pd.to_datetime(result["date"], errors="coerce")
    result["value"] = pd.to_numeric(result["GPRD"], errors="coerce")
    result = result.dropna(subset=["observation_date", "value"]).copy()
    result["series_id"] = "GPRD"
    result["release_assumption_date"] = (
        result["observation_date"].dt.normalize()
        + pd.to_timedelta(int(release_lag_days), unit="D")
    )
    result["timestamp_original"] = (
        result["release_assumption_date"].dt.strftime("%Y-%m-%d")
        + f" {conservative_time_et}"
    )
    result["timezone_original"] = "America/New_York"
    local = pd.to_datetime(result["timestamp_original"]).dt.tz_localize(
        ZoneInfo("America/New_York"), ambiguous="infer", nonexistent="shift_forward"
    )
    result["timestamp_utc"] = local.dt.tz_convert("UTC")
    result["timestamp_shanghai"] = local.dt.tz_convert("Asia/Shanghai")
    result["available_at"] = result["timestamp_utc"]
    result["retrieved_at_utc"] = utc_now()
    result["vintage_mode"] = "current_workbook_conservative_weekly_lag"
    result["vintage_warning"] = (
        "Current author workbook may contain historical revisions; available_at "
        "uses a conservative weekly lag and is not a true archived vintage."
    )
    columns = [
        "series_id",
        "observation_date",
        "value",
        "release_assumption_date",
        "timestamp_original",
        "timezone_original",
        "timestamp_utc",
        "timestamp_shanghai",
        "available_at",
        "retrieved_at_utc",
        "vintage_mode",
        "vintage_warning",
    ]
    return result[columns].sort_values("observation_date").reset_index(drop=True)


def download_gpr_data(config: dict) -> Path:
    settings = config["data_sources"]["gpr"]
    response = requests.get(
        settings["url"],
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AU-RV-research/1.0)"},
    )
    if not response.ok:
        raise RuntimeError(f"GPR download failed with HTTP {response.status_code}")
    raw = pd.read_excel(io.BytesIO(response.content), engine="xlrd")
    result = normalise_gpr_frame(
        raw,
        release_lag_days=int(settings["conservative_release_lag_calendar_days"]),
        conservative_time_et=settings["conservative_available_time_et"],
    )
    return write_parquet(result, resolve_path(config, settings["output_path"]))
