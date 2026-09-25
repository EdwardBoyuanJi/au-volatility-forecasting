#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.strategy_v6.backtest import run_strategy_v6_five_year_backtest
from au_rv.strategy_v6.settings import load_strategy_v6_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-pipeline",
        action="store_true",
        help="Reuse already built extended features, forecasts, and opportunities.",
    )
    args = parser.parse_args()
    settings, model_config = load_strategy_v6_settings(ROOT)
    outputs = run_strategy_v6_five_year_backtest(
        settings,
        model_config,
        rebuild_pipeline=not args.reuse_pipeline,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
