from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.config import config_fingerprint, resolve_path
from au_rv.io import read_parquet, write_parquet
from au_rv.models.conditional_error import conditional_error_estimate
from au_rv.models.prediction_intervals import (
    mature_residuals,
    project_quantiles,
    residual_quantiles,
)
from au_rv.models.ridge_har_x import (
    MODEL_NAMES,
    feature_sets,
    make_model_pipeline,
    model_objective,
    model_version,
    prediction_to_log_and_rv,
    select_alpha_purged,
)


@dataclass
class FittedState:
    pipeline: object
    alpha: float
    training_objective: str
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    feature_order: list[str]
    version: str


def cadence_key(value, frequency: str) -> tuple:
    """Return a causal calendar bucket for prediction or model refitting."""

    date = pd.Timestamp(value).normalize()
    normalized = str(frequency).strip().lower()
    if normalized == "daily":
        return ("daily", date.strftime("%Y-%m-%d"))
    if normalized == "weekly":
        iso = date.isocalendar()
        return ("weekly", int(iso.year), int(iso.week))
    if normalized == "monthly":
        return ("monthly", date.year, date.month)
    if normalized == "quarterly":
        return ("quarterly", date.year, (date.month - 1) // 3 + 1)
    if normalized == "yearly":
        return ("yearly", date.year)
    raise ValueError(
        "Unsupported cadence. Use daily, weekly, monthly, quarterly, or yearly."
    )


def mature_training_rows(
    features: pd.DataFrame,
    *,
    forecast_date,
    horizon: int,
    feature_order: list[str],
    training_window: str,
    rolling_training_years: int,
) -> pd.DataFrame:
    date = pd.Timestamp(forecast_date).normalize()
    target = f"target_log_rv_{horizon}d"
    maturity = f"target_maturity_date_{horizon}d"
    data = features.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data[maturity] = pd.to_datetime(data[maturity], errors="coerce").dt.normalize()
    mask = (
        data["trade_date"].lt(date)
        & data[maturity].le(date)
        & data[target].notna()
        & data[feature_order].notna().all(axis=1)
        & data["feature_valid_flag"].astype(bool)
    )
    result = data.loc[mask].sort_values("trade_date")
    if training_window == "rolling":
        lower = date - pd.DateOffset(years=int(rolling_training_years))
        result = result[result["trade_date"].ge(lower)]
    return result


def _enough_history(data: pd.DataFrame, forecast_date, settings: dict) -> bool:
    if len(data) < int(settings["minimum_training_samples"]):
        return False
    earliest_allowed = data["trade_date"].min() + pd.DateOffset(
        years=int(settings["minimum_training_years"])
    )
    return pd.Timestamp(forecast_date) >= earliest_allowed


def _fit_state(
    training: pd.DataFrame,
    *,
    horizon: int,
    model_name: str,
    settings: dict,
    fingerprint: str,
    epsilon: float,
) -> FittedState:
    columns = feature_sets(horizon)[model_name]
    objective = model_objective(model_name)
    target = (
        f"target_rv_{horizon}d"
        if objective == "qlike"
        else f"target_log_rv_{horizon}d"
    )
    alpha, _ = select_alpha_purged(
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
    start, end = training["trade_date"].iloc[[0, -1]]
    return FittedState(
        pipeline=pipeline,
        alpha=alpha,
        training_objective=objective,
        training_start=pd.Timestamp(start),
        training_end=pd.Timestamp(end),
        feature_order=columns,
        version=model_version(fingerprint, horizon, model_name, end, alpha),
    )


def _event_flag(row: pd.Series, horizon: int) -> bool:
    return any(
        float(row.get(f"{event_type}_count_{horizon}d", 0) or 0) > 0
        for event_type in ("cpi", "nfp", "fomc")
    )


def run_walk_forward_frame(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = features.sort_values("trade_date").reset_index(drop=True).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    horizons = [int(value) for value in config["features"]["horizons"]]
    settings = config["model"]
    epsilon = float(config["project"]["epsilon"])
    annualization = int(config["project"]["annualization_days"])
    probabilities = [float(value) for value in settings["interval_quantiles"]]
    fingerprint = config_fingerprint(config)
    cache: dict[tuple, FittedState] = {}
    emitted_prediction_periods: set[tuple] = set()
    records: list[dict] = []

    for row in data.itertuples(index=False):
        forecast_date = pd.Timestamp(row.trade_date).normalize()
        row_series = pd.Series(row._asdict())
        if not bool(row_series.get("feature_valid_flag", False)):
            continue
        prediction_period = cadence_key(
            forecast_date, settings["prediction_frequency"]
        )
        if prediction_period in emitted_prediction_periods:
            continue
        # Weekly/monthly/etc. forecasts use the first feature-valid row in the
        # new period, which is knowable at that moment and needs no look-ahead.
        emitted_prediction_periods.add(prediction_period)
        for horizon in horizons:
            sets = feature_sets(horizon)
            reference_training = mature_training_rows(
                data,
                forecast_date=forecast_date,
                horizon=horizon,
                feature_order=sets["har"],
                training_window=settings["training_window"],
                rolling_training_years=int(settings["rolling_training_years"]),
            )
            if not _enough_history(reference_training, forecast_date, settings):
                continue
            target_log = row_series.get(f"target_log_rv_{horizon}d", np.nan)
            target_rv = row_series.get(f"target_rv_{horizon}d", np.nan)
            maturity_date = row_series.get(f"target_maturity_date_{horizon}d", pd.NaT)
            event_flag = _event_flag(row_series, horizon)
            for model_name in MODEL_NAMES:
                if model_name == "persistence":
                    forecast_log = float(row_series["log_rv_1d"])
                    forecast_rv = max(float(np.exp(forecast_log) - epsilon), 0.0)
                    alpha = np.nan
                    training_objective = "persistence"
                    train_start = reference_training["trade_date"].iloc[0]
                    train_end = reference_training["trade_date"].iloc[-1]
                    version = model_version(
                        fingerprint, horizon, model_name, train_end, None
                    )
                else:
                    columns = sets[model_name]
                    training = mature_training_rows(
                        data,
                        forecast_date=forecast_date,
                        horizon=horizon,
                        feature_order=columns,
                        training_window=settings["training_window"],
                        rolling_training_years=int(settings["rolling_training_years"]),
                    )
                    if not _enough_history(training, forecast_date, settings):
                        continue
                    cache_key = (
                        horizon,
                        model_name,
                        *cadence_key(
                            forecast_date, settings["retrain_frequency"]
                        ),
                    )
                    state = cache.get(cache_key)
                    if state is None:
                        state = _fit_state(
                            training,
                            horizon=horizon,
                            model_name=model_name,
                            settings=settings,
                            fingerprint=fingerprint,
                            epsilon=epsilon,
                        )
                        cache[cache_key] = state
                    current_features = pd.DataFrame(
                        [[row_series[column] for column in state.feature_order]],
                        columns=state.feature_order,
                    )
                    forecast_log_values, forecast_rv_values = prediction_to_log_and_rv(
                        state.pipeline,
                        current_features,
                        objective=state.training_objective,
                        epsilon=epsilon,
                    )
                    forecast_log = float(forecast_log_values[0])
                    forecast_rv = float(forecast_rv_values[0])
                    alpha = state.alpha
                    training_objective = state.training_objective
                    train_start = state.training_start
                    train_end = state.training_end
                    version = state.version
                forecast_vol = float(np.sqrt(annualization * forecast_rv))
                record = {
                    "forecast_date": forecast_date,
                    "model_cutoff": row_series["model_cutoff"],
                    "horizon": horizon,
                    "model_name": model_name,
                    "training_start": train_start,
                    "training_end": train_end,
                    "alpha": alpha,
                    "training_objective": training_objective,
                    "forecast_log_rv": forecast_log,
                    "forecast_rv": forecast_rv,
                    "forecast_vol": forecast_vol,
                    "forecast_vol_pct": 100.0 * forecast_vol,
                    "actual_log_rv": target_log,
                    "actual_rv": target_rv,
                    "residual": (
                        float(target_log - forecast_log)
                        if pd.notna(target_log)
                        else np.nan
                    ),
                    "target_maturity_date": maturity_date,
                    "rv_t": row_series.get("rv", np.nan),
                    "event_window_flag": event_flag,
                    "selected_au_contract": row_series.get(
                        "selected_au_contract", None
                    ),
                    "data_quality_flag": row_series.get("data_quality_flag", "unknown"),
                    "model_version": version,
                }
                if model_name == "full":
                    prior = pd.DataFrame(records)
                    mature = mature_residuals(
                        prior,
                        forecast_date=forecast_date,
                        horizon=horizon,
                        model_name="full",
                    )
                    quantile_map = residual_quantiles(
                        mature.get("residual", pd.Series(dtype=float)),
                        probabilities,
                        int(settings["minimum_interval_samples"]),
                    )
                    record.update(
                        project_quantiles(
                            forecast_log,
                            quantile_map,
                            epsilon=epsilon,
                            annualization_days=annualization,
                        )
                    )
                    record["residual_sample_count"] = int(len(mature))
                    record.update(
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
                records.append(record)
    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise RuntimeError(
            "Walk-forward produced no forecasts. Check data quality and minimum training history."
        )
    return predictions.sort_values(
        ["forecast_date", "horizon", "model_name"]
    ).reset_index(drop=True)


def run_walk_forward(config: dict) -> Path:
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    predictions = run_walk_forward_frame(features, config)
    path = resolve_path(config, config["outputs"]["predictions_path"])
    return write_parquet(predictions, path)
