#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.features.build import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    feature_path, audit_path = build_feature_table(config)
    print("Feature table:", feature_path)
    print("Leakage audit:", audit_path)


if __name__ == "__main__":
    main()
