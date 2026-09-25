from __future__ import annotations

import numpy as np
import pandas as pd


def _error_summary(data: pd.DataFrame, level: str) -> dict:
    residual = pd.to_numeric(data["residual"], errors="coerce").dropna()
    if residual.empty:
        return {
            "conditional_mae": np.nan,
            "conditional_rmse": np.nan,
            "conditional_residual_q10": np.nan,
            "conditional_residual_q90": np.nan,
            "conditional_sample_count": 0,
            "conditional_error_level": level,
        }
    return {
        "conditional_mae": float(residual.abs().mean()),
        "conditional_rmse": float(np.sqrt(np.mean(residual**2))),
        "conditional_residual_q10": float(residual.quantile(0.10)),
        "conditional_residual_q90": float(residual.quantile(0.90)),
        "conditional_sample_count": int(len(residual)),
        "conditional_error_level": level,
    }


def _volatility_bucket(
    current_vol: float,
    historical_vol: pd.Series,
    quantiles: list[float],
) -> str:
    clean = pd.to_numeric(historical_vol, errors="coerce").dropna()
    if clean.empty or not np.isfinite(current_vol):
        return "unknown"
    low, high = clean.quantile(quantiles).to_list()
    if current_vol <= low:
        return "low"
    if current_vol <= high:
        return "mid"
    return "high"


def conditional_error_estimate(
    mature: pd.DataFrame,
    *,
    current_forecast_vol: float,
    current_event_flag: bool,
    minimum_samples: int,
    rolling_window: int,
    volatility_quantiles: list[float],
) -> dict:
    if mature.empty:
        return _error_summary(mature, "insufficient_mature_oos_residuals")
    data = mature.copy()
    current_bucket = _volatility_bucket(
        current_forecast_vol, data["forecast_vol"], volatility_quantiles
    )
    low, high = pd.to_numeric(data["forecast_vol"], errors="coerce").quantile(
        volatility_quantiles
    )
    data["volatility_bucket"] = np.select(
        [
            data["forecast_vol"].le(low),
            data["forecast_vol"].gt(low) & data["forecast_vol"].le(high),
            data["forecast_vol"].gt(high),
        ],
        ["low", "mid", "high"],
        default="unknown",
    )
    exact = data[
        data["volatility_bucket"].eq(current_bucket)
        & data["event_window_flag"].astype(bool).eq(bool(current_event_flag))
    ]
    if len(exact) >= minimum_samples:
        return _error_summary(exact, "horizon+volatility_bucket+event_flag")
    volatility_only = data[data["volatility_bucket"].eq(current_bucket)]
    if len(volatility_only) >= minimum_samples:
        return _error_summary(volatility_only, "horizon+volatility_bucket")
    rolling = data.tail(int(rolling_window))
    if len(rolling) >= minimum_samples:
        return _error_summary(rolling, "horizon+rolling_oos")
    if len(data) >= minimum_samples:
        return _error_summary(data, "horizon+all_mature_oos")
    # Report the actual evidence count, but do not emit deceptively precise
    # error estimates before the configured minimum is reached.
    return {
        "conditional_mae": np.nan,
        "conditional_rmse": np.nan,
        "conditional_residual_q10": np.nan,
        "conditional_residual_q90": np.nan,
        "conditional_sample_count": int(len(data)),
        "conditional_error_level": "insufficient_mature_oos_residuals",
    }
