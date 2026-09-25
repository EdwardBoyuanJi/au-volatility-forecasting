from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def qlike(actual_rv, forecast_rv, epsilon: float = 1e-12) -> float:
    return float(np.mean(qlike_losses(actual_rv, forecast_rv, epsilon)))


def qlike_losses(actual_rv, forecast_rv, epsilon: float = 1e-12) -> np.ndarray:
    actual = np.maximum(np.asarray(actual_rv, dtype=float), epsilon)
    forecast = np.maximum(np.asarray(forecast_rv, dtype=float), epsilon)
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def diebold_mariano_losses(
    losses_model,
    losses_benchmark,
    *,
    horizon: int,
) -> tuple[float, float, int]:
    first = np.asarray(losses_model, dtype=float)
    second = np.asarray(losses_benchmark, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    differential = first[valid] - second[valid]
    return _diebold_mariano_differential(differential, horizon=horizon)


def diebold_mariano(
    errors_model,
    errors_benchmark,
    *,
    horizon: int,
) -> tuple[float, float, int]:
    first = np.asarray(errors_model, dtype=float)
    second = np.asarray(errors_benchmark, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    differential = first[valid] ** 2 - second[valid] ** 2
    return _diebold_mariano_differential(differential, horizon=horizon)


def _diebold_mariano_differential(
    differential: np.ndarray,
    *,
    horizon: int,
) -> tuple[float, float, int]:
    n = len(differential)
    if n < max(10, horizon + 2):
        return np.nan, np.nan, n
    centered = differential - differential.mean()
    lag = min(max(int(horizon) - 1, 0), n - 2)
    long_run_variance = float(np.dot(centered, centered) / n)
    for order in range(1, lag + 1):
        covariance = float(np.dot(centered[order:], centered[:-order]) / n)
        weight = 1.0 - order / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return np.nan, np.nan, n
    statistic = differential.mean() / math.sqrt(long_run_variance / n)
    # Harvey-Leybourne-Newbold finite-sample adjustment.
    adjustment_term = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if adjustment_term > 0:
        statistic *= math.sqrt(adjustment_term)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    return float(statistic), p_value, n


def direction_accuracy(frame: pd.DataFrame) -> float:
    valid = frame[["forecast_rv", "actual_rv", "rv_t"]].dropna()
    if valid.empty:
        return np.nan
    forecast_direction = np.sign(valid["forecast_rv"] - valid["rv_t"])
    actual_direction = np.sign(valid["actual_rv"] - valid["rv_t"])
    return float(np.mean(forecast_direction.eq(actual_direction)))


def interval_coverage(frame: pd.DataFrame, lower: str, upper: str) -> float:
    valid = frame[["actual_rv", lower, upper]].dropna()
    if valid.empty:
        return np.nan
    return float(
        np.mean(
            valid["actual_rv"].ge(valid[lower])
            & valid["actual_rv"].le(valid[upper])
        )
    )
