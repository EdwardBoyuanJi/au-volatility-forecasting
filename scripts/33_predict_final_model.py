#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_model.inference import load_model_bundles, predict_feature_rows, predictions_to_json


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Invoke the frozen AU realized-volatility models on feature rows.")
    parser.add_argument("--input", default=str(ROOT / "data/model_inputs/latest_feature_snapshot.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "models/final_prediction_model"))
    parser.add_argument("--output", default=str(ROOT / "data/outputs/final_prediction_model/predictions.csv"))
    parser.add_argument("--json", action="store_true", help="Also write a JSON file next to the CSV output.")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    result = predict_feature_rows(_read(source), load_model_bundles(args.model_dir))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    if args.json:
        destination.with_suffix(".json").write_text(predictions_to_json(result), encoding="utf-8")
    print(result[["forecast_date", "horizon", "role", "forecast_vol_pct", "model_agreement"]].to_string(index=False))
    print(f"output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
