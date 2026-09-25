#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config, resolve_path
from au_rv.io import read_parquet, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether untouched post-freeze observations are available for strategy v2."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    latest = pd.to_datetime(features["trade_date"]).max().normalize()
    effective = pd.Timestamp(config["strategy_v2"]["forward_effective_trade_date"]).normalize()
    post_freeze = features[pd.to_datetime(features["trade_date"]).dt.normalize().ge(effective)]
    status = {
        "strategy_version": config["strategy_v2"]["strategy_version"],
        "forward_effective_trade_date": effective,
        "latest_feature_trade_date": latest,
        "post_freeze_feature_rows": int(len(post_freeze)),
        "mature_forward_trades": 0,
        "status": (
            "awaiting_post_freeze_data"
            if post_freeze.empty
            else "post_freeze_data_available_run_signal_pipeline"
        ),
        "interpretation": (
            "Historical backtest results are research evidence, not an untouched forward test."
        ),
    }
    destination = resolve_path(
        config,
        Path(config["strategy_v2"]["output_dir"]) / "forward_status.json",
    )
    write_json(status, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
