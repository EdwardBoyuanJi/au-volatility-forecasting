#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.io import read_parquet
from au_rv.strategy_v2.backtest import build_enhanced_opportunities
from au_rv.strategy_v2.data import (
    download_strategy_v2_market_data,
    select_protective_wings,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download protective-wing bars and historical top-book ticks for strategy v2."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    enhanced, _ = build_enhanced_opportunities(config)
    catalog = read_parquet(ROOT / "data/raw/tqsdk_au_option_catalog.parquet")
    candidates = select_protective_wings(
        enhanced,
        catalog,
        wing_width_fraction=float(config["strategy_v2"]["wing_width_fraction"]),
    )
    if candidates.empty:
        raise RuntimeError("No strategy-v2 distribution-filter candidates have protective wings.")
    print(
        "distribution candidates:",
        int(enhanced["signal_distribution_short"].sum()),
        "download candidates:",
        len(candidates),
    )
    outputs = download_strategy_v2_market_data(candidates, project_root=ROOT)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
