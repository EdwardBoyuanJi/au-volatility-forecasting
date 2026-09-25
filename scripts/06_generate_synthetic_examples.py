#!/usr/bin/env python3
"""Generate clearly labelled synthetic examples; never formal forecasts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.config import load_config
from au_rv.evaluation.model_comparison import evaluate_predictions_frame
from au_rv.features.targets import add_forward_targets
from au_rv.io import write_csv, write_json, write_parquet
from au_rv.models.latest_forecast import _nan_to_none, generate_latest_forecast_frame
from au_rv.models.walk_forward import run_walk_forward_frame


def synthetic_feature_table(config: dict, observations: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["project"]["random_seed"]))
    dates = pd.bdate_range("2021-01-04", periods=observations)
    shocks = rng.normal(0.0, 0.28, observations)
    log_rv = np.empty(observations)
    log_rv[0] = np.log(8.0e-5)
    for index in range(1, observations):
        log_rv[index] = (
            0.97 * log_rv[index - 1]
            + 0.03 * np.log(8.0e-5)
            + 0.12 * shocks[index]
        )
    rv = np.exp(log_rv)
    frame = pd.DataFrame({"trade_date": dates, "rv": rv})
    epsilon = float(config["project"]["epsilon"])
    frame["log_rv_1d"] = np.log(frame["rv"] + epsilon)
    frame["log_rv_5d"] = np.log(
        frame["rv"].rolling(5, min_periods=1).mean() + epsilon
    )
    frame["log_rv_22d"] = np.log(
        frame["rv"].rolling(22, min_periods=1).mean() + epsilon
    )
    frame["realized_quarticity"] = (
        frame["rv"] ** 2 * np.exp(rng.normal(0.0, 0.35, observations))
    )
    frame["sqrt_realized_quarticity_1d"] = np.sqrt(
        frame["realized_quarticity"]
    )
    frame["harq_log_rv_rq_interaction_1d"] = (
        frame["log_rv_1d"] * frame["sqrt_realized_quarticity_1d"]
    )
    jump_indicator = rng.random(observations) < 0.08
    frame["jump_var_1d"] = jump_indicator * rv * rng.uniform(0.05, 0.5, observations)
    frame["jump_var_5d"] = frame["jump_var_1d"].rolling(5, min_periods=1).mean()
    frame["log_comex_nonoverlap_rv_1d"] = (
        frame["log_rv_1d"].shift(1).fillna(frame["log_rv_1d"].iloc[0])
        + rng.normal(0, 0.18, observations)
    )
    frame["abs_broad_dollar_ret_1d"] = np.abs(rng.normal(0, 0.003, observations))
    frame["abs_us10y_real_chg_1d"] = np.abs(rng.normal(3.0, 2.0, observations))
    frame["abs_usdcny_ret_1d"] = np.abs(rng.normal(0, 0.002, observations))
    for horizon in (5, 20, 40):
        frame[f"cpi_count_{horizon}d"] = (
            (np.arange(observations) % 21) < max(1, horizon // 20)
        ).astype(int)
        frame[f"nfp_count_{horizon}d"] = (
            (np.arange(observations) % 22) < max(1, horizon // 20)
        ).astype(int)
        frame[f"fomc_count_{horizon}d"] = (
            (np.arange(observations) % 32) < max(1, horizon // 20)
        ).astype(int)
    cutoff = (
        pd.to_datetime(frame["trade_date"]).dt.tz_localize("Asia/Shanghai")
        + pd.Timedelta(hours=15, minutes=5)
    )
    frame["model_cutoff"] = cutoff
    frame["feature_available_at_max"] = cutoff
    frame["availability_cutoff_pass"] = True
    frame["feature_valid_flag"] = True
    frame["selected_au_contract"] = np.where(
        np.arange(observations) < observations // 2, "AU_SYNTH_A", "AU_SYNTH_B"
    )
    frame["contract_roll_flag"] = (
        frame["selected_au_contract"].ne(frame["selected_au_contract"].shift(1))
        & frame.index.to_series().gt(0)
    )
    frame["missing_au_bars"] = 0
    frame["missing_comex_bars"] = 0
    frame["comex_nonoverlap_bar_count"] = 60
    frame["macro_missing_flag"] = False
    frame["calendar_missing_flag"] = False
    frame["data_quality_flag"] = "synthetic_example_only_not_formal"
    frame["data_quality_notes"] = "synthetic_example_only"
    return add_forward_targets(frame, [5, 20, 40], epsilon)


def main() -> None:
    config = load_config()
    example_config = copy.deepcopy(config)
    example_config["model"].update(
        {
            "minimum_training_years": 1,
            "minimum_training_samples": 252,
            "alpha_grid": [0.1, 1.0, 10.0],
            "inner_cv_splits": 3,
            "minimum_cv_test_samples": 20,
        }
    )
    example_config["outputs"]["model_dir"] = "data/examples/models"
    examples = ROOT / "data/examples"
    features = synthetic_feature_table(example_config)
    predictions = run_walk_forward_frame(features, example_config)
    metrics = evaluate_predictions_frame(
        predictions, float(example_config["project"]["epsilon"])
    )
    latest = generate_latest_forecast_frame(
        features, predictions, example_config, save_models=True
    )
    write_parquet(features, examples / "model_features_daily.synthetic.parquet")
    write_parquet(
        predictions, examples / "walk_forward_predictions.synthetic.parquet"
    )
    write_csv(metrics, examples / "model_metrics.synthetic.csv")
    write_csv(latest, examples / "latest_forecast.synthetic.csv")
    write_json(
        _nan_to_none(latest.to_dict("records")),
        examples / "latest_forecast.synthetic.json",
    )
    print("Synthetic examples written to:", examples)


if __name__ == "__main__":
    main()
