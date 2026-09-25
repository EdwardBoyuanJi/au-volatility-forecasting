#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from au_rv.io import read_parquet
from au_rv.strategy_v3.data import download_real_quote_ticks
from au_rv.strategy_v3.settings import load_strategy_v3_v4_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download historical top-book quotes for v3/v4 strategy research."
    )
    parser.add_argument(
        "--scope",
        choices=("old-signals", "all"),
        default="old-signals",
        help="Download the 22 original short signals first, or every option opportunity.",
    )
    args = parser.parse_args()
    settings = load_strategy_v3_v4_settings(ROOT)
    opportunities = read_parquet(
        ROOT / "data/outputs/volatility_strategy_opportunities.parquet"
    )
    if args.scope == "old-signals":
        opportunities = opportunities[opportunities["signal_model_timed_short"].eq(-1)]
    calendar = read_parquet(ROOT / "data/raw/tqsdk_shfe_trade_calendar.parquet")
    outputs = download_real_quote_ticks(
        opportunities,
        calendar,
        project_root=ROOT,
        raw_dir_relative=str(settings["data"]["raw_dir"]),
        flush_every=int(settings["data"]["flush_every_opportunities"]),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
