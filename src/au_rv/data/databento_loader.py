from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from au_rv.io import utc_now, write_json, write_parquet


def _request_parameters(
    config: dict,
    start_date,
    end_date,
    *,
    available_end=None,
) -> dict:
    settings = config["data_sources"]["databento"]
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    requested_end = end + pd.Timedelta(days=1)
    if available_end is not None:
        available_end = pd.Timestamp(available_end)
        available_end = (
            available_end.tz_localize("UTC")
            if available_end.tzinfo is None
            else available_end.tz_convert("UTC")
        )
        requested_end = min(requested_end, available_end)
    if requested_end <= start:
        raise ValueError(
            "Databento has no data available inside the requested date range."
        )
    return {
        "dataset": settings["dataset"],
        "symbols": list(settings["symbols"]),
        "stype_in": settings["stype_in"],
        "stype_out": settings["stype_out"],
        "schema": settings["schema"],
        "start": start.isoformat(),
        "end": requested_end.isoformat(),
    }


def _client():
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is missing.")
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements.txt before using Databento.") from exc
    return db.Historical(key)


def estimate_databento_cost(config: dict, start_date, end_date) -> dict:
    client = _client()
    settings = config["data_sources"]["databento"]
    dataset_range = client.metadata.get_dataset_range(settings["dataset"])
    download_parameters = _request_parameters(
        config,
        start_date,
        end_date,
        available_end=dataset_range["end"],
    )
    # Metadata.get_cost prices the requested input symbology and schema but
    # does not accept the timeseries-only stype_out argument.
    cost_parameters = {
        key: value
        for key, value in download_parameters.items()
        if key != "stype_out"
    }
    cost = float(client.metadata.get_cost(**cost_parameters))
    result = {
        "estimated_cost_usd": cost,
        "estimated_at_utc": utc_now().isoformat(),
        "dataset_range": dataset_range,
        "cost_request": cost_parameters,
        "download_request": download_parameters,
        "warning": (
            "Databento bills actual bytes. Its estimate can over-report ranges "
            "that are not discrete multiples of ten minutes."
        ),
    }
    root = Path(config["_project_root"])
    write_json(result, root / "data/raw/databento_cost_estimate.json")
    return result


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index()
    timestamp_candidates = ["ts_event", "timestamp", "index"]
    timestamp_column = next(
        (column for column in timestamp_candidates if column in result.columns), None
    )
    if timestamp_column is None:
        raise ValueError("Databento OHLCV response has no ts_event timestamp.")
    result["ts_event"] = pd.to_datetime(result[timestamp_column], utc=True, errors="coerce")
    if "symbol" in result.columns and "instrument_id" in result.columns:
        continuous = result["symbol"].astype(str).str.match(
            r"^[A-Z0-9]+\.[cnv]\.\d+$", na=False
        )
        if continuous.any():
            # Databento's DataFrame keeps the requested smart symbol in
            # ``symbol`` even with stype_out=instrument_id. Preserve the actual
            # instrument ID so roll days never create a return across contracts.
            result.loc[continuous, "symbol"] = (
                "instrument_id:"
                + result.loc[continuous, "instrument_id"].astype(str)
            )
    if "symbol" not in result.columns:
        if "raw_symbol" in result.columns:
            result["symbol"] = result["raw_symbol"]
        elif "instrument_id" in result.columns:
            result["symbol"] = "instrument_id:" + result["instrument_id"].astype(str)
        else:
            raise ValueError("Databento response has no symbol or instrument_id.")
    # Parent-symbol responses can include exchange-listed futures spreads. When
    # raw GC symbols are present, retain only outrights such as GCM5 or GCZ25.
    # Continuous-symbol downloads use instrument IDs and do not need this filter.
    outright_pattern = re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$")
    labels = result["symbol"].astype(str)
    raw_gc = labels.str.startswith("GC")
    result = result[~raw_gc | labels.str.match(outright_pattern)].copy()
    if result.empty:
        raise ValueError(
            "Databento response contained no outright GC contracts after excluding spreads."
        )
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Databento OHLCV response missing fields: {sorted(missing)}")
    result["timestamp_original"] = result["ts_event"].astype(str)
    result["timezone_original"] = "UTC"
    result["timestamp_utc"] = result["ts_event"]
    result["timestamp_shanghai"] = result["ts_event"].dt.tz_convert("Asia/Shanghai")
    # ohlcv-1m timestamps denote the interval start; the bar is public at its end.
    result["available_at"] = result["ts_event"] + pd.Timedelta(minutes=1)
    result["retrieved_at_utc"] = utc_now()
    keep = [
        "symbol",
        "timestamp_original",
        "timezone_original",
        "timestamp_utc",
        "timestamp_shanghai",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "retrieved_at_utc",
    ]
    return result[keep].sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def download_databento_data(
    config: dict,
    start_date,
    end_date,
    *,
    force_execute: bool = False,
) -> Path:
    """Estimate first, then download only if the configuration gate allows it."""

    estimate = estimate_databento_cost(config, start_date, end_date)
    settings = config["data_sources"]["databento"]
    execute = bool(settings["execute_download"]) or force_execute
    if not execute:
        raise RuntimeError(
            "Databento cost was estimated but download is disabled. "
            "Set data_sources.databento.execute_download=true or pass --download-databento."
        )
    if estimate["estimated_cost_usd"] > float(settings["maximum_cost_usd"]):
        raise RuntimeError(
            f"Estimated Databento cost ${estimate['estimated_cost_usd']:.4f} exceeds "
            f"configured cap ${float(settings['maximum_cost_usd']):.4f}."
        )
    client = _client()
    parameters = estimate["download_request"]
    store = client.timeseries.get_range(**parameters)
    frame = _normalise_ohlcv(store.to_df())
    root = Path(config["_project_root"])
    return write_parquet(frame, root / "data/raw/databento_gc_1m.parquet")
