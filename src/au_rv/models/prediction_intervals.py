from __future__ import annotations

import numpy as np
import pandas as pd


def mature_residuals(
    predictions: pd.DataFrame,
    *,
    forecast_date,
    horizon: int,
    model_name: str = "full",
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    data = predictions.copy()
    maturity = pd.to_datetime(data["target_maturity_date"], errors="coerce")
    cutoff_date = pd.Timestamp(forecast_date).normalize()
    return data[
        data["horizon"].astype(int).eq(int(horizon))
        & data["model_name"].eq(model_name)
        & maturity.le(cutoff_date)
        & data["residual"].notna()
    ].sort_values("forecast_date")


def residual_quantiles(
    residuals: pd.Series,
    probabilities: list[float],
    minimum_samples: int,
) -> dict[float, float]:
    clean = pd.to_numeric(residuals, errors="coerce").dropna().to_numpy()
    if clean.size < int(minimum_samples):
        return {float(probability): np.nan for probability in probabilities}
    values = np.quantile(clean, probabilities)
    # np.quantile is monotone; maximum.accumulate also protects against any
    # downstream floating-point reordering.
    values = np.maximum.accumulate(values)
    return {
        float(probability): float(value)
        for probability, value in zip(probabilities, values)
    }


def project_quantiles(
    forecast_log_rv: float,
    residual_quantile_map: dict[float, float],
    *,
    epsilon: float,
    annualization_days: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    last_rv = -np.inf
    last_vol = -np.inf
    for probability in sorted(residual_quantile_map):
        label = f"q{int(round(probability * 100)):02d}"
        residual = residual_quantile_map[probability]
        if not np.isfinite(residual):
            output[f"forecast_log_rv_{label}"] = np.nan
            output[f"forecast_rv_{label}"] = np.nan
            output[f"forecast_vol_{label}"] = np.nan
            continue
        log_value = float(forecast_log_rv + residual)
        rv = max(float(np.exp(log_value) - epsilon), 0.0)
        vol = float(np.sqrt(annualization_days * rv))
        rv = max(rv, last_rv)
        vol = max(vol, last_vol)
        last_rv, last_vol = rv, vol
        output[f"forecast_log_rv_{label}"] = log_value
        output[f"forecast_rv_{label}"] = rv
        output[f"forecast_vol_{label}"] = vol
    return output
