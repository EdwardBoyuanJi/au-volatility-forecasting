#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.models.walk_forward import run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print("Walk-forward predictions:", run_walk_forward(load_config(args.config)))


if __name__ == "__main__":
    main()
