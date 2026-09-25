#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.strategy_v3.settings import load_strategy_v3_v4_settings
from au_rv.strategy_v4.backtest import run_strategy_v4_backtest


def main() -> int:
    outputs = run_strategy_v4_backtest(load_strategy_v3_v4_settings(ROOT))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
