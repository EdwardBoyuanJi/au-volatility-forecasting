#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config, resolve_path
from au_rv.io import write_csv
from au_rv.models.control_experiment import (
    add_fdr_adjustment,
    family_distribution_summary,
    generate_detailed_control_report,
    incremental_control_effects,
    selection_test_champions,
    unified_model_ranking,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild detailed control-experiment summaries from saved metrics."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics_path = resolve_path(config, config["outputs"]["control_metrics_path"])
    if not metrics_path.is_file():
        raise RuntimeError(f"Control metrics do not exist: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics = add_fdr_adjustment(metrics)
    write_csv(metrics, metrics_path)
    incremental = incremental_control_effects(metrics)
    unified = unified_model_ranking(metrics)
    family_summary = family_distribution_summary(metrics)
    selection_test = selection_test_champions(metrics)
    outputs = {
        "incremental": write_csv(
            incremental,
            resolve_path(config, config["outputs"]["control_incremental_path"]),
        ),
        "unified": write_csv(
            unified, resolve_path(config, config["outputs"]["control_unified_path"])
        ),
        "family_summary": write_csv(
            family_summary,
            resolve_path(config, config["outputs"]["control_family_summary_path"]),
        ),
        "selection_test": write_csv(
            selection_test,
            resolve_path(config, config["outputs"]["control_selection_test_path"]),
        ),
    }
    outputs["report"] = generate_detailed_control_report(
        config,
        metrics,
        incremental,
        unified,
        family_summary,
        selection_test,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
