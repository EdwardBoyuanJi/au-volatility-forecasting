from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model_bundles(model_dir: str | Path) -> dict[tuple[str, int], dict]:
    directory = Path(model_dir)
    bundles: dict[tuple[str, int], dict] = {}
    for path in sorted(directory.glob("*_h*.joblib")):
        bundle = joblib.load(path)
        key = (str(bundle["role"]), int(bundle["horizon"]))
        if key in bundles:
            raise RuntimeError(f"Duplicate model bundle for {key}.")
        bundles[key] = bundle
    expected = {(role, horizon) for role in ("main", "robust") for horizon in (5, 20, 40)}
    missing = expected.difference(bundles)
    if missing:
        raise RuntimeError(f"Missing frozen model bundles: {sorted(missing)}")
    return bundles


def _predict_bundle(bundle: dict, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_order = list(bundle["feature_order"])
    missing = [column for column in feature_order if column not in rows.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")
    numeric = rows[feature_order].apply(pd.to_numeric, errors="coerce")
    bad = numeric.columns[numeric.isna().any()].tolist()
    if bad:
        raise ValueError(f"Input contains missing/non-numeric values in: {bad}")
    prediction = np.asarray(bundle["pipeline"].predict(numeric), dtype=float)
    epsilon = float(bundle["epsilon"])
    if bundle["objective"] == "mse_log":
        log_rv = prediction
        rv = np.maximum(np.exp(log_rv) - epsilon, epsilon)
    elif bundle["objective"] == "qlike":
        rv = np.maximum(prediction, epsilon)
        log_rv = np.log(rv + epsilon)
    else:
        raise RuntimeError(f"Unknown model objective: {bundle['objective']}")
    return log_rv, rv


def predict_feature_rows(
    rows: pd.DataFrame,
    bundles: dict[tuple[str, int], dict],
) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("Input feature table is empty.")
    annualization = int(next(iter(bundles.values()))["annualization_days"])
    records: list[dict] = []
    for (role, horizon), bundle in sorted(bundles.items(), key=lambda item: (item[0][1], item[0][0])):
        log_rv, rv = _predict_bundle(bundle, rows)
        residual_quantiles = {
            float(key): float(value)
            for key, value in bundle["residual_log_quantiles"].items()
        }
        for position, (_, source) in enumerate(rows.iterrows()):
            record = {
                "forecast_date": source.get("trade_date", source.get("forecast_date", None)),
                "selected_au_contract": source.get("selected_au_contract", None),
                "horizon": int(horizon),
                "role": role,
                "model_name": bundle["model_name"],
                "objective": bundle["objective"],
                "alpha": float(bundle["alpha"]),
                "forecast_log_rv": float(log_rv[position]),
                "forecast_rv": float(rv[position]),
                "forecast_vol": float(np.sqrt(annualization * rv[position])),
                "forecast_vol_pct": float(100.0 * np.sqrt(annualization * rv[position])),
                "training_end": bundle["training_end"],
                "model_version": bundle["model_version"],
            }
            for probability, residual in residual_quantiles.items():
                label = f"q{int(round(probability * 100)):02d}"
                projected_rv = max(float(np.exp(log_rv[position] + residual)), bundle["epsilon"])
                record[f"forecast_vol_{label}_pct"] = float(
                    100.0 * np.sqrt(annualization * projected_rv)
                )
            current_rv = source.get("rv", np.nan)
            if pd.notna(current_rv):
                record["current_vol_pct"] = float(100.0 * np.sqrt(annualization * float(current_rv)))
                record["direction_vs_current"] = (
                    "UP" if float(rv[position]) > float(current_rv) else "DOWN"
                )
            records.append(record)
    output = pd.DataFrame(records)
    for (forecast_date, horizon), group in output.groupby(["forecast_date", "horizon"], dropna=False):
        directions = set(group["direction_vs_current"].dropna()) if "direction_vs_current" in group else set()
        agreement = "AGREE" if len(directions) == 1 else "DISAGREE"
        output.loc[group.index, "model_agreement"] = agreement
        values = group["forecast_vol_pct"].to_numpy(dtype=float)
        output.loc[group.index, "model_spread_vol_points"] = float(values.max() - values.min())
    return output.sort_values(["forecast_date", "horizon", "role"]).reset_index(drop=True)


def predictions_to_json(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].astype(str)
    clean = clean.where(pd.notna(clean), None)
    return json.dumps(clean.to_dict("records"), ensure_ascii=False, indent=2, default=str)
