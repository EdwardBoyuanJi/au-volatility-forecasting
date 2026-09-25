from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from au_rv.models.walk_forward import cadence_key


@dataclass(frozen=True)
class Garch11Parameters:
    mean: float
    omega: float
    alpha: float
    beta: float


def _decode_parameters(theta: np.ndarray) -> tuple[float, float, float]:
    long_run_variance = float(np.exp(np.clip(theta[0], -20.0, 20.0)))
    logits = np.array([theta[1], theta[2], 0.0], dtype=float)
    logits -= logits.max()
    weights = np.exp(logits) / np.exp(logits).sum()
    alpha = float(0.999 * weights[0])
    beta = float(0.999 * weights[1])
    omega = max(long_run_variance * (1.0 - alpha - beta), 1e-10)
    return omega, alpha, beta


def _filter_variance(
    residuals: np.ndarray,
    *,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float,
) -> np.ndarray:
    variance = np.empty_like(residuals, dtype=float)
    previous = max(float(initial_variance), 1e-8)
    for index, residual in enumerate(residuals):
        if index > 0:
            previous = omega + alpha * residuals[index - 1] ** 2 + beta * previous
        variance[index] = max(previous, 1e-10)
    return variance


def fit_garch11(returns, *, maximum_iterations: int = 500) -> Garch11Parameters:
    """Gaussian QMLE GARCH(1,1) on decimal daily returns.

    The optimization is performed in percentage-return units for numerical
    stability.  Parameters returned by this function are converted back to
    decimal-return variance units.
    """

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 30:
        raise ValueError("GARCH(1,1) requires at least 30 finite returns.")
    mean_decimal = float(np.mean(values))
    residuals = (values - mean_decimal) * 100.0
    sample_variance = max(float(np.var(residuals, ddof=1)), 1e-6)
    alpha0, beta0 = 0.05, 0.90
    remainder0 = 0.999 - alpha0 - beta0
    initial = np.array(
        [
            np.log(sample_variance),
            np.log(alpha0 / remainder0),
            np.log(beta0 / remainder0),
        ]
    )

    def objective(theta):
        omega, alpha, beta = _decode_parameters(np.asarray(theta, dtype=float))
        variance = _filter_variance(
            residuals,
            omega=omega,
            alpha=alpha,
            beta=beta,
            initial_variance=sample_variance,
        )
        return 0.5 * float(np.sum(np.log(variance) + residuals**2 / variance))

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={"maxiter": int(maximum_iterations), "ftol": 1e-10},
    )
    theta = fitted.x if np.all(np.isfinite(fitted.x)) else initial
    omega_pct, alpha, beta = _decode_parameters(theta)
    return Garch11Parameters(
        mean=mean_decimal,
        omega=omega_pct / 10000.0,
        alpha=alpha,
        beta=beta,
    )


def forecast_average_variance(
    parameters: Garch11Parameters,
    returns,
    horizon: int,
) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 1:
        return np.nan
    residuals = values - parameters.mean
    unconditional = parameters.omega / max(
        1.0 - parameters.alpha - parameters.beta, 1e-8
    )
    filtered = _filter_variance(
        residuals,
        omega=parameters.omega,
        alpha=parameters.alpha,
        beta=parameters.beta,
        initial_variance=max(unconditional, float(np.var(residuals)), 1e-10),
    )
    next_variance = (
        parameters.omega
        + parameters.alpha * residuals[-1] ** 2
        + parameters.beta * filtered[-1]
    )
    persistence = parameters.alpha + parameters.beta
    forecasts = []
    current = max(next_variance, 1e-12)
    for _ in range(int(horizon)):
        forecasts.append(current)
        current = parameters.omega + persistence * current
    return float(np.mean(forecasts))


def add_causal_garch_features(
    features: pd.DataFrame,
    *,
    horizons: list[int],
    minimum_observations: int,
    refit_frequency: str = "monthly",
) -> pd.DataFrame:
    data = features.sort_values("trade_date").reset_index(drop=True).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    for horizon in horizons:
        data[f"garch_forecast_rv_{horizon}d"] = np.nan
        data[f"log_garch_forecast_rv_{horizon}d"] = np.nan
    data["garch_training_end"] = pd.NaT
    state: Garch11Parameters | None = None
    state_key = None
    training_end = pd.NaT
    for index, row in data.iterrows():
        date = row["trade_date"]
        history_before = data.loc[
            data["trade_date"].lt(date) & data["daily_log_return"].notna(),
            "daily_log_return",
        ]
        key = cadence_key(date, refit_frequency)
        if key != state_key:
            state_key = key
            if len(history_before) >= int(minimum_observations):
                state = fit_garch11(history_before.to_numpy())
                training_end = data.loc[history_before.index, "trade_date"].max()
        current_history = data.loc[
            data["trade_date"].le(date) & data["daily_log_return"].notna(),
            "daily_log_return",
        ]
        if state is None or len(current_history) < int(minimum_observations):
            continue
        data.loc[index, "garch_training_end"] = training_end
        for horizon in horizons:
            forecast = forecast_average_variance(
                state, current_history.to_numpy(), int(horizon)
            )
            data.loc[index, f"garch_forecast_rv_{horizon}d"] = forecast
            data.loc[index, f"log_garch_forecast_rv_{horizon}d"] = np.log(
                max(forecast, 1e-12)
            )
    return data
