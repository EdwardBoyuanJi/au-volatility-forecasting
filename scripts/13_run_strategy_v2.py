#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.strategy_v2.backtest import run_strategy_v2_backtest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the independently versioned, protected volatility strategy v2."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    outputs = run_strategy_v2_backtest(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
