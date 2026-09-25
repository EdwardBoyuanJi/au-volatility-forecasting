#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_strategy import (
    EXPECTED_REFERENCE_METRICS,
    FINAL_STRATEGY_NAME,
    validate_final_strategy_settings,
)
from au_rv.strategy_v6.real_backtest import run_real_topbook_backtest


def _verify_reference(metrics_path: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 1 or metrics.iloc[0]["strategy"] != FINAL_STRATEGY_NAME:
        raise RuntimeError("Reference verification requires exactly the frozen strategy.")
    row = metrics.iloc[0]
    for field, expected in EXPECTED_REFERENCE_METRICS.items():
        actual = float(row[field])
        if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(
                f"Reference mismatch for {field}: expected {expected}, found {actual}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen AU narrow-spread 10% risk / 2-day hedge strategy."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "final_strategy_10pct_2d_config.yaml"),
        help="Path to the frozen strategy YAML.",
    )
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="Require exact equality to the bundled 2021-07-01/2026-07-27 snapshot.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    validate_final_strategy_settings(settings)
    settings["_project_root"] = str(ROOT)
    settings["_config_path"] = str(config_path)
    settings["_manifest_extra_paths"] = [
        "scripts/27_run_final_strategy_10pct_2d.py",
        "scripts/28_verify_final_strategy_package.py",
        "src/au_rv/final_strategy/spec.py",
    ]
    outputs = run_real_topbook_backtest(settings)
    if args.verify_reference:
        _verify_reference(outputs["metrics"])
        print("reference_metrics_verified: true")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
