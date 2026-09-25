from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = [
    "persistence",
    "har",
    "har_jump",
    "har_jump_macro",
    "full",
    "harq",
    "qlike_har",
    "qlike_harq",
    "qlike_har_jump_macro",
    "qlike_harq_jump_macro",
]

QLIKE_MODEL_NAMES = {
    "qlike_har",
    "qlike_harq",
    "qlike_har_jump_macro",
    "qlike_harq_jump_macro",
}


def feature_sets(horizon: int) -> dict[str, list[str]]:
    har = ["log_rv_1d", "log_rv_5d", "log_rv_22d"]
    harq = har + ["harq_log_rv_rq_interaction_1d"]
    jump = har + ["jump_var_1d", "jump_var_5d"]
    macro = jump + [
        "log_comex_nonoverlap_rv_1d",
        "abs_broad_dollar_ret_1d",
        "abs_us10y_real_chg_1d",
        "abs_usdcny_ret_1d",
    ]
    full = macro + [
        f"cpi_count_{horizon}d",
        f"nfp_count_{horizon}d",
        f"fomc_count_{horizon}d",
    ]
    harq_macro = harq + [
        "jump_var_1d",
        "jump_var_5d",
        "log_comex_nonoverlap_rv_1d",
        "abs_broad_dollar_ret_1d",
        "abs_us10y_real_chg_1d",
        "abs_usdcny_ret_1d",
    ]
    return {
        "har": har,
        "har_jump": jump,
        "har_jump_macro": macro,
        "full": full,
        "harq": harq,
        "qlike_har": har,
        "qlike_harq": harq,
        "qlike_har_jump_macro": macro,
        "qlike_harq_jump_macro": harq_macro,
    }


def model_objective(model_name: str) -> str:
    return "qlike" if model_name in QLIKE_MODEL_NAMES else "mse_log"


def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def make_qlike_pipeline(
    alpha: float,
    *,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> Pipeline:
    """Penalized QLIKE estimator with a positive log-link forecast.

    ``GammaRegressor`` minimizes half the Gamma deviance.  Apart from the
    constant factor two, that deviance is exactly the observation-level QLIKE
    loss.  Its log link guarantees strictly positive variance forecasts.
    """

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "qlike",
                GammaRegressor(
                    alpha=float(alpha),
                    max_iter=int(max_iter),
                    tol=float(tol),
                ),
            ),
        ]
    )


def make_model_pipeline(
    alpha: float,
    *,
    objective: str,
    qlike_max_iter: int = 2000,
    qlike_tol: float = 1e-9,
) -> Pipeline:
    if objective == "mse_log":
        return make_ridge_pipeline(alpha)
    if objective == "qlike":
        return make_qlike_pipeline(
            alpha,
            max_iter=qlike_max_iter,
            tol=qlike_tol,
        )
    raise ValueError(f"Unsupported training objective: {objective!r}")


def prediction_to_log_and_rv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    objective: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(pipeline.predict(X), dtype=float)
    if objective == "mse_log":
        forecast_log = prediction
        forecast_rv = np.maximum(np.exp(forecast_log) - epsilon, 0.0)
    elif objective == "qlike":
        forecast_rv = np.maximum(prediction, epsilon)
        forecast_log = np.log(forecast_rv + epsilon)
    else:
        raise ValueError(f"Unsupported training objective: {objective!r}")
    return forecast_log, forecast_rv


def _mean_qlike(actual_rv, forecast_rv, epsilon: float) -> float:
    actual = np.maximum(np.asarray(actual_rv, dtype=float), epsilon)
    forecast = np.maximum(np.asarray(forecast_rv, dtype=float), epsilon)
    ratio = actual / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def select_alpha_purged(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int,
    alpha_grid: list[float],
    n_splits: int,
    minimum_test_samples: int,
    objective: str = "mse_log",
    epsilon: float = 1e-12,
    qlike_max_iter: int = 2000,
    qlike_tol: float = 1e-9,
) -> tuple[float, pd.DataFrame]:
    """Select alpha using only past folds and a horizon-sized purge gap."""

    n = len(X)
    possible_splits = min(int(n_splits), max(2, n // max(minimum_test_samples, 1) - 1))
    if possible_splits < 2 or n <= horizon + 2 * minimum_test_samples:
        fallback = float(alpha_grid[len(alpha_grid) // 2])
        return fallback, pd.DataFrame(
            [
                {
                    "alpha": fallback,
                    "training_objective": objective,
                    "mean_validation_loss": np.nan,
                    "mean_validation_mse": np.nan,
                    "mean_validation_qlike": np.nan,
                    "fold_count": 0,
                }
            ]
        )
    splitter = TimeSeriesSplit(n_splits=possible_splits, gap=int(horizon))
    rows: list[dict] = []
    for alpha in alpha_grid:
        losses: list[float] = []
        for train_index, validation_index in splitter.split(X):
            if len(train_index) < 2 or len(validation_index) < 1:
                continue
            pipeline = make_model_pipeline(
                float(alpha),
                objective=objective,
                qlike_max_iter=qlike_max_iter,
                qlike_tol=qlike_tol,
            )
            train_target = y.iloc[train_index]
            if objective == "qlike":
                train_target = train_target.clip(lower=epsilon)
            pipeline.fit(X.iloc[train_index], train_target)
            forecast = pipeline.predict(X.iloc[validation_index])
            if objective == "mse_log":
                loss = mean_squared_error(y.iloc[validation_index], forecast)
            elif objective == "qlike":
                loss = _mean_qlike(
                    y.iloc[validation_index],
                    forecast,
                    epsilon,
                )
            else:
                raise ValueError(f"Unsupported training objective: {objective!r}")
            losses.append(float(loss))
        mean_loss = float(np.mean(losses)) if losses else np.nan
        rows.append(
            {
                "alpha": float(alpha),
                "training_objective": objective,
                "mean_validation_loss": mean_loss,
                "mean_validation_mse": mean_loss if objective == "mse_log" else np.nan,
                "mean_validation_qlike": mean_loss if objective == "qlike" else np.nan,
                "fold_count": len(losses),
            }
        )
    scores = pd.DataFrame(rows)
    valid = scores.dropna(subset=["mean_validation_loss"]).sort_values(
        ["mean_validation_loss", "alpha"]
    )
    if valid.empty:
        selected = float(alpha_grid[len(alpha_grid) // 2])
    else:
        selected = float(valid.iloc[0]["alpha"])
    return selected, scores


def model_version(
    config_fingerprint: str,
    horizon: int,
    model_name: str,
    training_end,
    alpha: float | None,
) -> str:
    payload = (
        f"{config_fingerprint}|{horizon}|{model_name}|"
        f"{pd.Timestamp(training_end).isoformat()}|{alpha}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
