#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.models.latest_forecast import generate_latest_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    csv_path, json_path = generate_latest_forecast(load_config(args.config))
    print("Latest forecast CSV:", csv_path)
    print("Latest forecast JSON:", json_path)


if __name__ == "__main__":
    main()
