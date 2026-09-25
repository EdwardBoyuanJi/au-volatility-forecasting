#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.strategy_v6.real_backtest import run_real_topbook_backtest


def _risk_name(value: float) -> str:
    percent = value * 100.0
    if math.isclose(percent, round(percent)):
        return f"{int(round(percent)):02d}"
    return f"{percent:04.1f}".replace(".", "p")


def _expand_grid(settings: dict) -> list[dict]:
    template = dict(settings["strategy_template"])
    risks = [float(value) for value in settings["grid"]["risk_budget_fractions"]]
    intervals = [
        int(value) for value in settings["grid"]["hedge_interval_trading_days"]
    ]
    if len(risks) != 6 or len(intervals) != 3:
        raise RuntimeError("The frozen v8 experiment must remain a 6 x 3 grid.")
    if len(set(risks)) != len(risks) or len(set(intervals)) != len(intervals):
        raise RuntimeError("Grid values must be unique.")
    strategies = []
    for risk in risks:
        for interval in intervals:
            strategies.append(
                {
                    **template,
                    "name": f"narrow_rb{_risk_name(risk)}_hedge{interval}d",
                    "risk_budget_fraction": risk,
                    "hedge_interval_trading_days": interval,
                }
            )
    if len(strategies) != 18 or len({item["name"] for item in strategies}) != 18:
        raise RuntimeError("Grid expansion did not produce 18 unique strategies.")
    return strategies


def main() -> int:
    config_path = ROOT / "strategy_v8_narrow_risk_hedge_grid_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["strategies"] = _expand_grid(settings)
    settings["_project_root"] = str(ROOT)
    settings["_config_path"] = str(config_path)
    settings["_manifest_extra_paths"] = [
        "scripts/25_run_strategy_v8_narrow_risk_hedge_grid.py",
        "scripts/26_summarize_strategy_v8_narrow_risk_hedge_grid.py",
    ]
    outputs = run_real_topbook_backtest(settings)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
