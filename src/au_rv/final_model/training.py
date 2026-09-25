from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from au_rv.models.control_experiment import ExperimentSpec, _mature_training, base_features, control_blocks
from au_rv.models.ridge_har_x import make_model_pipeline, prediction_to_log_and_rv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec(role: str, settings: dict, horizon: int) -> ExperimentSpec:
    configured = settings["models"][role]
    blocks = tuple(configured["blocks"])
    columns = list(base_features(configured["family"], horizon))
    available = control_blocks(horizon)
    for block in blocks:
        columns.extend(available[block])
    return ExperimentSpec(
        model_name=configured["model_name"],
        family=configured["family"],
        objective=configured["objective"],
        blocks=blocks,
        features=tuple(dict.fromkeys(columns)),
    )


def _evaluation(history: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    records: list[dict] = []
    mature = history[history["actual_rv"].notna()].copy()
    for horizon in (5, 20, 40):
        scoped = mature[mature["horizon"].astype(int).eq(horizon)]
        persistence = scoped[scoped["model_name"].eq("persistence__raw__none")].set_index("trade_date")
        for model_name, group in scoped.groupby("model_name"):
            group = group.set_index("trade_date").loc[persistence.index]
            actual = np.maximum(group["actual_rv"].to_numpy(dtype=float), epsilon)
            forecast = np.maximum(group["forecast_rv"].to_numpy(dtype=float), epsilon)
            current = np.maximum(group["rv_at_signal"].to_numpy(dtype=float), epsilon)
            persist = np.maximum(persistence["forecast_rv"].to_numpy(dtype=float), epsilon)
            log_error = np.log(actual) - np.log(forecast)
            persist_error = np.log(actual) - np.log(persist)
            ratio = actual / forecast
            denominator = float(np.sum(persist_error**2))
            records.append(
                {
                    "horizon": horizon,
                    "model_name": model_name,
                    "sample_count": len(group),
                    "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
                    "rmse_log_rv": float(np.sqrt(np.mean(log_error**2))),
                    "mae_log_rv": float(np.mean(np.abs(log_error))),
                    "oos_r2_vs_persistence": 1.0 - float(np.sum(log_error**2)) / denominator if denominator else np.nan,
                    "direction_accuracy": float(np.mean(np.sign(forecast - current) == np.sign(actual - current))),
                }
            )
    return pd.DataFrame(records).sort_values(["horizon", "model_name"]).reset_index(drop=True)


def train_final_models(settings: dict, root: Path) -> dict[str, Path]:
    feature_path = root / settings["inputs"]["feature_history_path"]
    history_path = root / settings["inputs"]["walk_forward_history_path"]
    features = pd.read_parquet(feature_path).copy()
    history = pd.read_parquet(history_path).copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    requested_cutoff = pd.Timestamp(settings["project"]["model_cutoff"]).normalize()
    eligible_dates = features.loc[features["trade_date"].le(requested_cutoff), "trade_date"]
    if eligible_dates.empty:
        raise RuntimeError("No feature row exists on or before the frozen cutoff.")
    cutoff = eligible_dates.max()
    current = features[features["trade_date"].eq(cutoff)].iloc[-1]
    model_dir = root / settings["outputs"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    epsilon = float(settings["project"]["epsilon"])
    annualization = int(settings["project"]["annualization_days"])
    quantiles = [float(value) for value in settings["training"]["interval_quantiles"]]
    manifest_models: list[dict] = []
    coefficient_rows: list[dict] = []
    reference_rows: list[pd.DataFrame] = []
    for horizon in [int(value) for value in settings["training"]["horizons"]]:
        specs = {role: _spec(role, settings, horizon) for role in ("main", "robust")}
        common_features = tuple(dict.fromkeys(feature for spec in specs.values() for feature in spec.features))
        features["experiment_common_valid"] = (
            features["feature_valid_flag"].astype(bool)
            & features[list(common_features)].notna().all(axis=1)
        )
        for role, spec in specs.items():
            training = _mature_training(
                features,
                anchor=cutoff,
                horizon=horizon,
                required_features=spec.features,
            )
            if len(training) < int(settings["training"]["minimum_training_samples"]):
                raise RuntimeError(f"{role} {horizon}d has only {len(training)} mature rows.")
            alpha = float(settings["training"]["alpha_by_model"][role])
            pipeline = make_model_pipeline(alpha, objective=spec.objective)
            target_column = f"target_rv_{horizon}d" if spec.objective == "qlike" else f"target_log_rv_{horizon}d"
            target = training[target_column].clip(lower=epsilon) if spec.objective == "qlike" else training[target_column]
            pipeline.fit(training[list(spec.features)], target)
            model_history = history[
                history["horizon"].astype(int).eq(horizon)
                & history["model_name"].eq(spec.model_name)
                & history["actual_rv"].notna()
                & pd.to_datetime(history["maturity_date"]).le(cutoff)
            ].copy()
            residual = np.log(np.maximum(model_history["actual_rv"].to_numpy(dtype=float), epsilon)) - model_history["forecast_log_rv"].to_numpy(dtype=float)
            residual_quantiles = {str(value): float(np.quantile(residual, value)) for value in quantiles}
            version_payload = f"{settings['project']['version']}|{role}|{horizon}|{training.trade_date.iloc[-1]}|{alpha}|{len(training)}"
            model_version = hashlib.sha256(version_payload.encode("utf-8")).hexdigest()[:16]
            bundle = {
                "pipeline": pipeline,
                "role": role,
                "horizon": horizon,
                "model_name": spec.model_name,
                "family": spec.family,
                "objective": spec.objective,
                "blocks": list(spec.blocks),
                "feature_order": list(spec.features),
                "alpha": alpha,
                "training_start": str(pd.Timestamp(training["trade_date"].iloc[0]).date()),
                "training_end": str(pd.Timestamp(training["trade_date"].iloc[-1]).date()),
                "training_samples": len(training),
                "forecast_cutoff": str(cutoff.date()),
                "annualization_days": annualization,
                "epsilon": epsilon,
                "residual_log_quantiles": residual_quantiles,
                "residual_calibration_samples": len(residual),
                "model_version": model_version,
            }
            destination = model_dir / f"{role}_h{horizon}.joblib"
            joblib.dump(bundle, destination)
            estimator = pipeline.steps[-1][1]
            scaler = pipeline.steps[0][1]
            for feature, scaled_coefficient, scale in zip(spec.features, estimator.coef_, scaler.scale_):
                coefficient_rows.append(
                    {
                        "role": role,
                        "horizon": horizon,
                        "model_name": spec.model_name,
                        "feature": feature,
                        "scaled_coefficient": float(scaled_coefficient),
                        "raw_feature_effect": float(scaled_coefficient / scale),
                    }
                )
            manifest_models.append(
                {
                    "role": role,
                    "horizon": horizon,
                    "path": str(destination.relative_to(root)),
                    "sha256": _sha256(destination),
                    "model_version": model_version,
                    "model_name": spec.model_name,
                    "objective": spec.objective,
                    "alpha": alpha,
                    "feature_count": len(spec.features),
                    "training_start": bundle["training_start"],
                    "training_end": bundle["training_end"],
                    "training_samples": len(training),
                }
            )
    latest_input_path = root / settings["outputs"]["latest_input_path"]
    latest_input_path.parent.mkdir(parents=True, exist_ok=True)
    required = sorted(set(feature for entry in manifest_models for feature in joblib.load(root / entry["path"])["feature_order"]))
    identity = [column for column in ("trade_date", "model_cutoff", "selected_au_contract", "rv") if column in features.columns]
    pd.DataFrame([current[identity + required].to_dict()]).to_csv(latest_input_path, index=False)
    from au_rv.final_model.inference import load_model_bundles, predict_feature_rows, predictions_to_json

    reference = predict_feature_rows(pd.read_csv(latest_input_path), load_model_bundles(model_dir))
    reference_csv = root / settings["outputs"]["reference_csv_path"]
    reference_json = root / settings["outputs"]["reference_json_path"]
    evaluation_path = root / settings["outputs"]["evaluation_path"]
    coefficient_path = root / settings["outputs"]["coefficient_path"]
    for path in (reference_csv, reference_json, evaluation_path, coefficient_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(reference_csv, index=False)
    reference_json.write_text(predictions_to_json(reference), encoding="utf-8")
    _evaluation(history, epsilon).to_csv(evaluation_path, index=False)
    pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
    manifest = {
        "model_package_version": settings["project"]["version"],
        "forecast_cutoff": str(cutoff.date()),
        "annualization_days": annualization,
        "feature_history_sha256": _sha256(feature_path),
        "walk_forward_history_sha256": _sha256(history_path),
        "models": manifest_models,
        "latest_input": str(latest_input_path.relative_to(root)),
        "reference_predictions": str(reference_csv.relative_to(root)),
    }
    manifest_path = root / settings["outputs"]["manifest_path"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model_dir": model_dir,
        "manifest": manifest_path,
        "latest_input": latest_input_path,
        "reference_csv": reference_csv,
        "reference_json": reference_json,
        "evaluation": evaluation_path,
        "coefficients": coefficient_path,
    }
