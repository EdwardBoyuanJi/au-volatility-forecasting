#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.models.strategy_forecasts import generate_strategy_forecasts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate point-in-time forecasts for the trading strategy."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print("strategy_forecasts:", generate_strategy_forecasts(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
