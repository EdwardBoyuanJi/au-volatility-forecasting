#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_model.spec import validate_final_model_settings
from au_rv.final_model.training import train_final_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and freeze the final AU 5/20/40-day dual-model forecast engine.")
    parser.add_argument("--config", default=str(ROOT / "final_prediction_model_config.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_final_model_settings(settings)
    for name, path in train_final_models(settings, ROOT).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
