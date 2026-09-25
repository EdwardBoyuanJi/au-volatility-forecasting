from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.config import resolve_path
from au_rv.evaluation.metrics import (
    diebold_mariano,
    diebold_mariano_losses,
    direction_accuracy,
    interval_coverage,
    qlike,
    qlike_losses,
)
from au_rv.io import read_parquet, write_csv


def _scope_groups(data: pd.DataFrame):
    yield "all", "all", data
    for year, group in data.groupby(pd.to_datetime(data["forecast_date"]).dt.year):
        yield "year", str(year), group
    for flag, group in data.groupby(data["event_window_flag"].astype(bool)):
        yield "event_window", "event" if flag else "no_event", group


def evaluate_predictions_frame(predictions: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    mature = predictions.dropna(
        subset=["actual_log_rv", "actual_rv", "forecast_log_rv", "forecast_rv"]
    ).copy()
    rows: list[dict] = []
    for horizon in sorted(mature["horizon"].astype(int).unique()):
        horizon_data = mature[mature["horizon"].astype(int).eq(horizon)].copy()
        for scope_type, scope_value, scoped in _scope_groups(horizon_data):
            har = scoped[scoped["model_name"].eq("har")][
                ["forecast_date", "actual_log_rv", "forecast_log_rv", "forecast_rv"]
            ].rename(
                columns={
                    "forecast_log_rv": "har_forecast_log_rv",
                    "forecast_rv": "har_forecast_rv",
                }
            )
            persistence = scoped[scoped["model_name"].eq("persistence")][
                ["forecast_date", "forecast_log_rv"]
            ].rename(columns={"forecast_log_rv": "persistence_forecast_log_rv"})
            for model_name, group in scoped.groupby("model_name"):
                aligned = group.merge(har, on=["forecast_date", "actual_log_rv"], how="inner")
                aligned = aligned.merge(persistence, on="forecast_date", how="left")
                if aligned.empty:
                    continue
                errors = aligned["actual_log_rv"] - aligned["forecast_log_rv"]
                har_errors = aligned["actual_log_rv"] - aligned["har_forecast_log_rv"]
                persistence_errors = (
                    aligned["actual_log_rv"] - aligned["persistence_forecast_log_rv"]
                )
                denominator = float(np.sum(persistence_errors.dropna() ** 2))
                oos_r2 = (
                    1.0 - float(np.sum(errors**2)) / denominator
                    if denominator > 0
                    else np.nan
                )
                dm_stat, dm_p, dm_n = diebold_mariano(
                    errors,
                    har_errors,
                    horizon=horizon,
                )
                dm_qlike_stat, dm_qlike_p, dm_qlike_n = diebold_mariano_losses(
                    qlike_losses(
                        aligned["actual_rv"], aligned["forecast_rv"], epsilon
                    ),
                    qlike_losses(
                        aligned["actual_rv"], aligned["har_forecast_rv"], epsilon
                    ),
                    horizon=horizon,
                )
                row = {
                    "horizon": horizon,
                    "model_name": model_name,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "sample_count": int(len(aligned)),
                    "qlike": qlike(
                        aligned["actual_rv"], aligned["forecast_rv"], epsilon
                    ),
                    "rmse_log_rv": float(np.sqrt(np.mean(errors**2))),
                    "mae_log_rv": float(np.mean(np.abs(errors))),
                    "oos_r2_vs_persistence": oos_r2,
                    "direction_accuracy": direction_accuracy(aligned),
                    "coverage_q05_q95": (
                        interval_coverage(
                            aligned, "forecast_rv_q05", "forecast_rv_q95"
                        )
                        if {"forecast_rv_q05", "forecast_rv_q95"}.issubset(aligned.columns)
                        else np.nan
                    ),
                    "coverage_q10_q90": (
                        interval_coverage(
                            aligned, "forecast_rv_q10", "forecast_rv_q90"
                        )
                        if {"forecast_rv_q10", "forecast_rv_q90"}.issubset(aligned.columns)
                        else np.nan
                    ),
                    "dm_stat_vs_har_squared_log_loss": dm_stat,
                    "dm_p_value_vs_har": dm_p,
                    "dm_sample_count": dm_n,
                    "dm_stat_vs_har_qlike": dm_qlike_stat,
                    "dm_p_value_vs_har_qlike": dm_qlike_p,
                    "dm_qlike_sample_count": dm_qlike_n,
                }
                rows.append(row)
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    har_reference = metrics[metrics["model_name"].eq("har")][
        [
            "horizon",
            "scope_type",
            "scope_value",
            "qlike",
            "rmse_log_rv",
            "mae_log_rv",
        ]
    ].rename(
        columns={
            "qlike": "har_qlike",
            "rmse_log_rv": "har_rmse_log_rv",
            "mae_log_rv": "har_mae_log_rv",
        }
    )
    metrics = metrics.merge(
        har_reference, on=["horizon", "scope_type", "scope_value"], how="left"
    )
    metrics["qlike_improvement_vs_har"] = metrics["har_qlike"] - metrics["qlike"]
    metrics["rmse_improvement_vs_har"] = (
        metrics["har_rmse_log_rv"] - metrics["rmse_log_rv"]
    )
    metrics["mae_improvement_vs_har"] = (
        metrics["har_mae_log_rv"] - metrics["mae_log_rv"]
    )
    return metrics.sort_values(
        ["horizon", "scope_type", "scope_value", "model_name"]
    ).reset_index(drop=True)


def evaluate_models(config: dict) -> Path:
    predictions = read_parquet(resolve_path(config, config["outputs"]["predictions_path"]))
    metrics = evaluate_predictions_frame(predictions, float(config["project"]["epsilon"]))
    path = resolve_path(config, config["outputs"]["metrics_path"])
    return write_csv(metrics, path)
