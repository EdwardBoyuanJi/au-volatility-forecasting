#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.strategy_v5.backtest import run_strategy_v5_backtest
from au_rv.strategy_v5.settings import load_strategy_v5_settings


def main() -> int:
    outputs = run_strategy_v5_backtest(load_strategy_v5_settings(ROOT))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
