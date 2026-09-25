#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import ensure_project_directories, load_config
from au_rv.data.databento_loader import (
    download_databento_data,
    estimate_databento_cost,
)
from au_rv.data.event_calendar_loader import download_event_calendars
from au_rv.data.fred_loader import download_fred_data
from au_rv.data.gpr_loader import download_gpr_data
from au_rv.data.slv_iv_loader import download_slv_iv_data
from au_rv.data.tqsdk_loader import download_tqsdk_data

load_dotenv(ROOT / ".env")


def arguments():
    parser = argparse.ArgumentParser(description="Download only the fixed project data sources.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--download-databento", action="store_true")
    parser.add_argument("--download-slv-options", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["all", "tqsdk", "fred", "events", "databento", "gpr", "slv"],
        default=["all"],
        help="Download only selected sources; default is all sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    config = load_config(args.config)
    ensure_project_directories(config)
    end = pd.Timestamp(
        args.end
        or config["dates"]["end_date"]
        or pd.Timestamp.now(tz="Asia/Shanghai").date()
    )
    start = pd.Timestamp(args.start or config["dates"]["start_date"])
    if args.small:
        start = max(
            start,
            end - pd.Timedelta(days=int(config["dates"]["small_download_days"]) * 2),
        )
    print(f"Requested data range: {start.date()} to {end.date()}")
    failures: list[str] = []
    selected = set(args.sources)
    if "all" in selected:
        selected = {"tqsdk", "fred", "events", "databento", "gpr", "slv"}

    if "databento" in selected and os.getenv("DATABENTO_API_KEY"):
        try:
            estimate = estimate_databento_cost(config, start, end)
            print(f"Databento estimated cost: ${estimate['estimated_cost_usd']:.6f}")
        except Exception as exc:
            failures.append(f"Databento estimate: {exc}")
    elif "databento" in selected:
        failures.append("Databento estimate: DATABENTO_API_KEY is missing")
    if args.estimate_only:
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        return 0

    if "tqsdk" in selected and os.getenv("TQ_USER") and os.getenv("TQ_PASSWORD"):
        try:
            outputs = download_tqsdk_data(config, start, end)
            print("TqSdk files:", ", ".join(str(path) for path in outputs.values()))
        except Exception as exc:
            failures.append(f"TqSdk: {exc}")
    elif "tqsdk" in selected:
        failures.append("TqSdk: TQ_USER or TQ_PASSWORD is missing")

    if "fred" in selected and os.getenv("FRED_API_KEY"):
        try:
            print("FRED file:", download_fred_data(config, start, end))
        except Exception as exc:
            failures.append(f"FRED: {exc}")
    elif "fred" in selected:
        failures.append("FRED: FRED_API_KEY is missing")

    if "events" in selected:
        try:
            event_end = end + pd.Timedelta(
                days=int(config["dates"]["future_calendar_buffer_days"])
            )
            print("Event calendar file:", download_event_calendars(config, start, event_end))
        except Exception as exc:
            failures.append(f"Official event calendars: {exc}")

    if "gpr" in selected:
        try:
            print("GPR file:", download_gpr_data(config))
        except Exception as exc:
            failures.append(f"GPR: {exc}")

    if "databento" in selected and (
        args.download_databento
        or config["data_sources"]["databento"]["execute_download"]
    ):
        try:
            print(
                "Databento file:",
                download_databento_data(
                    config, start, end, force_execute=args.download_databento
                ),
            )
        except Exception as exc:
            failures.append(f"Databento download: {exc}")
    elif "databento" in selected:
        print("Databento download skipped after estimate (configuration gate is false).")

    if "slv" in selected and (
        args.download_slv_options
        or config["data_sources"]["slv_options"]["execute_download"]
    ):
        try:
            outputs = download_slv_iv_data(
                config, start, end, force_execute=args.download_slv_options
            )
            print("SLV OPRA files:", ", ".join(str(path) for path in outputs.values()))
        except Exception as exc:
            failures.append(f"SLV OPRA: {exc}")
    elif "slv" in selected:
        print("SLV OPRA download skipped (configuration gate is false).")

    if failures:
        print("\nIncomplete sources:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
