#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from au_rv.config import load_config
from au_rv.data.slv_iv_loader import download_slv_iv_data
from au_rv.data.fred_loader import _download_initial_release_series
from au_rv.data.tqsdk_loader import _create_api, _datetime_chunks, _normalise_bars
from au_rv.io import write_parquet


def _load_v6_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    result["_project_root"] = str(ROOT)
    result["_config_path"] = str(path)
    return result


def _download_au_main(settings: dict, *, force: bool) -> Path:
    raw_dir = ROOT / settings["backfill"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / "tqsdk_au_main_5m_2018_2019.parquet"
    if output.is_file() and not force:
        return output
    start = pd.Timestamp(settings["backfill"]["au_start"]) + pd.Timedelta(hours=20)
    end = pd.Timestamp(settings["backfill"]["au_end"]) + pd.Timedelta(hours=15, minutes=5)
    api = _create_api()
    pieces: list[pd.DataFrame] = []
    try:
        for chunk_start, chunk_end in _datetime_chunks(start, end, 180):
            frame = api.get_kline_data_series(
                symbol="KQ.m@SHFE.au",
                duration_seconds=300,
                start_dt=chunk_start.to_pydatetime(),
                end_dt=chunk_end.to_pydatetime(),
            )
            if frame is not None and not frame.empty:
                pieces.append(
                    _normalise_bars(
                        frame,
                        "KQ.m@SHFE.au",
                        duration_seconds=300,
                    )
                )
    finally:
        api.close()
    if not pieces:
        raise RuntimeError("TqSdk returned no 2018-2019 AU main-contract bars.")
    bars = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["ts_code", "timestamp_utc"], keep="last")
        .sort_values(["ts_code", "timestamp_utc"])
        .reset_index(drop=True)
    )
    return write_parquet(bars, output)


def _download_early_slv(settings: dict, *, force: bool) -> Path:
    raw_dir = ROOT / settings["backfill"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    iv_path = raw_dir / "databento_slv_iv_30d_2018_2019.parquet"
    if iv_path.is_file() and not force:
        return iv_path
    config = copy.deepcopy(load_config(ROOT / "config.yaml"))
    slv = config["data_sources"]["slv_options"]
    slv.update(
        {
            "execute_download": True,
            "maximum_cost_usd": float(
                settings["backfill"]["databento_maximum_cost_usd"]
            ),
            "definitions_path": str(
                raw_dir / "databento_slv_option_definitions_2018_2019.parquet"
            ),
            "underlying_path": str(
                raw_dir / "databento_slv_underlying_2018_2019.parquet"
            ),
            "option_prices_path": str(
                raw_dir / "databento_slv_option_ohlcv_2018_2019.parquet"
            ),
            "iv_path": str(iv_path),
            "definition_cache_dir": str(raw_dir / "slv_definition_snapshots"),
            "option_cache_dir": str(raw_dir / "slv_option_batches"),
            "cost_estimate_path": str(raw_dir / "databento_slv_cost_estimate.json"),
        }
    )
    outputs = download_slv_iv_data(
        config,
        settings["backfill"]["slv_start"],
        settings["backfill"]["slv_end"],
        force_execute=True,
    )
    return outputs["iv"]


def _download_legacy_dollar(settings: dict, *, force: bool) -> Path:
    raw_dir = ROOT / settings["backfill"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / "fred_legacy_broad_dollar.parquet"
    if output.is_file() and not force:
        return output
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the legacy dollar series.")
    base_config = load_config(ROOT / "config.yaml")
    frame = _download_initial_release_series(
        str(settings["backfill"]["fred_legacy_series"]),
        api_key,
        settings["backfill"]["fred_legacy_start"],
        settings["backfill"]["fred_legacy_end"],
        base_config["data_sources"]["fred"]["conservative_available_time_et"],
    )
    return write_parquet(frame, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download isolated 2018-2019 warm-up data for the five-year strategy study."
    )
    parser.add_argument("--config", default="strategy_v6_5y_config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("tqsdk", "slv", "fred"),
        default=("tqsdk", "slv", "fred"),
    )
    args = parser.parse_args()
    settings = _load_v6_config(ROOT / args.config)
    if "tqsdk" in args.sources:
        print("au_backfill:", _download_au_main(settings, force=args.force))
    if "slv" in args.sources:
        print("slv_backfill:", _download_early_slv(settings, force=args.force))
    if "fred" in args.sources:
        print(
            "fred_legacy_backfill:",
            _download_legacy_dollar(settings, force=args.force),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
