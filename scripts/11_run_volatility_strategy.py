#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.strategy.volatility_straddle import run_volatility_strategy_backtest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the volatility-forecast ATM straddle strategy."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    outputs = run_volatility_strategy_backtest(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
