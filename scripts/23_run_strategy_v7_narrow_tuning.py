#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.strategy_v6.real_backtest import run_real_topbook_backtest


def main() -> int:
    config_path = ROOT / "strategy_v7_narrow_tuning_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["_project_root"] = str(ROOT)
    settings["_config_path"] = str(config_path)
    outputs = run_real_topbook_backtest(settings)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
