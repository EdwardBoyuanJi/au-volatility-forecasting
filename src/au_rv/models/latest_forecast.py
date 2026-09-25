from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from au_rv.config import config_fingerprint, git_commit, resolve_path
from au_rv.io import read_parquet, write_csv, write_json
from au_rv.models.conditional_error import conditional_error_estimate
from au_rv.models.prediction_intervals import (
    mature_residuals,
    project_quantiles,
    residual_quantiles,
)
from au_rv.models.ridge_har_x import (
    feature_sets,
    make_model_pipeline,
    model_objective,
    model_version,
    prediction_to_log_and_rv,
    select_alpha_purged,
)
from au_rv.models.walk_forward import _enough_history, mature_training_rows


def _nan_to_none(records: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for record in records:
        cleaned.append(
            {
                key: (
                    None
                    if value is None
                    or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
                    or pd.isna(value)
                    else value
                )
                for key, value in record.items()
            }
        )
    return cleaned


def select_champion_models(
    metrics: pd.DataFrame,
    horizons: list[int],
    *,
    selection_metric: str = "qlike",
) -> dict[int, str]:
    """Choose one deployable candidate per horizon from formal OOS metrics."""

    required = {
        "horizon",
        "model_name",
        "scope_type",
        "scope_value",
        "sample_count",
        selection_metric,
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Model metrics missing champion fields: {sorted(missing)}")
    deployable = set(feature_sets(int(horizons[0])))
    selected: dict[int, str] = {}
    for horizon in horizons:
        candidates = metrics[
            metrics["horizon"].astype(int).eq(int(horizon))
            & metrics["scope_type"].eq("all")
            & metrics["scope_value"].eq("all")
            & metrics["model_name"].isin(deployable)
        ].copy()
        candidates[selection_metric] = pd.to_numeric(
            candidates[selection_metric], errors="coerce"
        )
        candidates = candidates.dropna(subset=[selection_metric]).sort_values(
            [selection_metric, "model_name"]
        )
        if candidates.empty:
            raise RuntimeError(f"Horizon {horizon}: no deployable OOS champion candidate.")
        selected[int(horizon)] = str(candidates.iloc[0]["model_name"])
    return selected


def generate_latest_forecast_frame(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
    *,
    save_models: bool = True,
    model_names_by_horizon: dict[int, str] | None = None,
    selection_metric: str = "qlike",
) -> pd.DataFrame:
    ordered = features.sort_values("trade_date")
    if ordered.empty:
        raise RuntimeError("The feature table is empty.")
    row = ordered.iloc[-1]
    if not bool(row["feature_valid_flag"]):
        raise RuntimeError(
            f"Latest feature date {pd.Timestamp(row['trade_date']).date()} is invalid; "
            f"no stale fallback forecast was created. "
            f"Reason: {row.get('data_quality_notes', 'unknown')}"
        )
    forecast_date = pd.Timestamp(row["trade_date"]).normalize()
    settings = config["model"]
    epsilon = float(config["project"]["epsilon"])
    annualization = int(config["project"]["annualization_days"])
    probabilities = [float(value) for value in settings["interval_quantiles"]]
    fingerprint = config_fingerprint(config)
    rows: list[dict] = []
    for horizon in [int(value) for value in config["features"]["horizons"]]:
        model_name = (model_names_by_horizon or {}).get(horizon, "full")
        sets = feature_sets(horizon)
        if model_name not in sets:
            raise ValueError(
                f"Horizon {horizon}: {model_name!r} is not a deployable model."
            )
        columns = sets[model_name]
        objective = model_objective(model_name)
        training = mature_training_rows(
            features,
            forecast_date=forecast_date,
            horizon=horizon,
            feature_order=columns,
            training_window=settings["training_window"],
            rolling_training_years=int(settings["rolling_training_years"]),
        )
        if not _enough_history(training, forecast_date, settings):
            raise RuntimeError(
                f"Horizon {horizon}: mature history does not meet both "
                f"{settings['minimum_training_samples']} samples and "
                f"{settings['minimum_training_years']} years "
                f"(available rows: {len(training)})."
            )
        target = (
            f"target_rv_{horizon}d"
            if objective == "qlike"
            else f"target_log_rv_{horizon}d"
        )
        alpha, cv_scores = select_alpha_purged(
            training[columns],
            training[target],
            horizon=horizon,
            alpha_grid=[float(value) for value in settings["alpha_grid"]],
            n_splits=int(settings["inner_cv_splits"]),
            minimum_test_samples=int(settings["minimum_cv_test_samples"]),
            objective=objective,
            epsilon=epsilon,
            qlike_max_iter=int(settings.get("qlike_max_iter", 2000)),
            qlike_tol=float(settings.get("qlike_tol", 1e-9)),
        )
        pipeline = make_model_pipeline(
            alpha,
            objective=objective,
            qlike_max_iter=int(settings.get("qlike_max_iter", 2000)),
            qlike_tol=float(settings.get("qlike_tol", 1e-9)),
        )
        fit_target = training[target]
        if objective == "qlike":
            fit_target = fit_target.clip(lower=epsilon)
        pipeline.fit(training[columns], fit_target)
        forecast_log_values, forecast_rv_values = prediction_to_log_and_rv(
            pipeline,
            pd.DataFrame([row[columns].to_dict()], columns=columns),
            objective=objective,
            epsilon=epsilon,
        )
        forecast_log = float(forecast_log_values[0])
        forecast_rv = float(forecast_rv_values[0])
        forecast_vol = float(np.sqrt(annualization * forecast_rv))
        mature = mature_residuals(
            predictions,
            forecast_date=forecast_date,
            horizon=horizon,
            model_name=model_name,
        )
        quantile_map = residual_quantiles(
            mature["residual"] if not mature.empty else pd.Series(dtype=float),
            probabilities,
            int(settings["minimum_interval_samples"]),
        )
        event_flag = any(
            float(row[f"{kind}_count_{horizon}d"]) > 0
            for kind in ("cpi", "nfp", "fomc")
        )
        version = model_version(
            fingerprint, horizon, model_name, training["trade_date"].iloc[-1], alpha
        )
        output = {
            "forecast_date": forecast_date,
            "model_cutoff": row["model_cutoff"],
            "horizon": horizon,
            "model_name": model_name,
            "training_objective": objective,
            "selection_metric": selection_metric,
            "forecast_log_rv": forecast_log,
            "forecast_rv": forecast_rv,
            "forecast_vol": forecast_vol,
            "forecast_vol_pct": 100.0 * forecast_vol,
        }
        projected = project_quantiles(
            forecast_log,
            quantile_map,
            epsilon=epsilon,
            annualization_days=annualization,
        )
        for probability in probabilities:
            label = f"q{int(round(probability * 100)):02d}"
            output[f"forecast_rv_{label}"] = projected[f"forecast_rv_{label}"]
            output[f"forecast_vol_{label}"] = projected[f"forecast_vol_{label}"]
        output.update(
            conditional_error_estimate(
                mature,
                current_forecast_vol=forecast_vol,
                current_event_flag=event_flag,
                minimum_samples=int(settings["minimum_conditional_samples"]),
                rolling_window=int(settings["rolling_error_window"]),
                volatility_quantiles=[
                    float(value)
                    for value in settings["conditional_volatility_quantiles"]
                ],
            )
        )
        output.update(
            {
                "selected_au_contract": row["selected_au_contract"],
                "data_quality_flag": row["data_quality_flag"],
                "model_version": version,
            }
        )
        rows.append(output)
        if save_models:
            model_dir = resolve_path(config, config["outputs"]["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            bundle = {
                "pipeline": pipeline,
                "feature_order": columns,
                "alpha": alpha,
                "alpha_cv_scores": cv_scores,
                "training_start": training["trade_date"].iloc[0],
                "training_end": training["trade_date"].iloc[-1],
                "horizon": horizon,
                "model_name": model_name,
                "training_objective": objective,
                "epsilon": epsilon,
                "config_version": config["project"]["config_version"],
                "config_fingerprint": fingerprint,
                "data_cutoff": row["model_cutoff"],
                "code_version": git_commit(config["_project_root"]),
                "model_version": version,
            }
            joblib.dump(bundle, model_dir / f"ridge_h{horizon}.joblib")
    return pd.DataFrame(rows)


def generate_latest_forecast(config: dict) -> tuple[Path, Path]:
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    predictions = read_parquet(resolve_path(config, config["outputs"]["predictions_path"]))
    metrics = pd.read_csv(resolve_path(config, config["outputs"]["metrics_path"]))
    horizons = [int(value) for value in config["features"]["horizons"]]
    selection_metric = str(config["model"].get("champion_selection_metric", "qlike"))
    champions = select_champion_models(
        metrics, horizons, selection_metric=selection_metric
    )
    latest = generate_latest_forecast_frame(
        features,
        predictions,
        config,
        save_models=True,
        model_names_by_horizon=champions,
        selection_metric=selection_metric,
    )
    csv_path = write_csv(latest, resolve_path(config, config["outputs"]["latest_csv_path"]))
    json_path = write_json(
        _nan_to_none(latest.to_dict("records")),
        resolve_path(config, config["outputs"]["latest_json_path"]),
    )
    return csv_path, json_path
