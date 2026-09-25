from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.features.build import build_feature_table_from_frames
from au_rv.io import read_parquet, utc_now, write_csv, write_parquet
from au_rv.models.control_experiment import _fit_spec_path, _mature_training
from au_rv.models.strategy_forecasts import selected_strategy_specs
from au_rv.strategy.volatility_straddle import (
    add_strategy_signals,
    build_option_opportunities,
)


def _stitch_legacy_broad_dollar(
    fred: pd.DataFrame,
    legacy: pd.DataFrame,
    *,
    current_series_id: str,
) -> pd.DataFrame:
    """Extend the current dollar index with its causally published predecessor.

    The model uses one-day log changes, so a constant scale has no effect on
    legacy returns. Scaling only removes the artificial level jump on the day
    the current index becomes historically available.
    """
    result = fred.copy()
    current = result[result["series_id"].eq(current_series_id)].sort_values(
        "observation_date"
    )
    if current.empty:
        raise RuntimeError(f"No current broad-dollar rows for {current_series_id}.")
    first_current_date = pd.Timestamp(current["observation_date"].min()).normalize()
    prior = legacy[
        pd.to_datetime(legacy["observation_date"]).dt.normalize().lt(first_current_date)
    ].copy()
    if prior.empty:
        return result
    scale = float(current.iloc[0]["value"]) / float(prior.iloc[-1]["value"])
    prior["value"] = pd.to_numeric(prior["value"], errors="coerce") * scale
    prior["source_series_id"] = prior["series_id"]
    prior["series_id"] = current_series_id
    result["source_series_id"] = result["series_id"]
    return (
        pd.concat([result, prior], ignore_index=True)
        .sort_values(["series_id", "observation_date", "available_at"])
        .drop_duplicates(["series_id", "observation_date"], keep="first")
        .reset_index(drop=True)
    )


def generate_v6_strategy_forecasts(config: dict) -> Path:
    """Generate the same two fixed HAR paths without unrelated-block gating."""
    root = Path(config["_project_root"])
    features = read_parquet(root / config["outputs"]["features_path"])
    horizons = [int(value) for value in config["features"]["horizons"]]
    settings = config["control_experiment"]
    epsilon = float(config["project"]["epsilon"])
    data = features.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    pieces: list[pd.DataFrame] = []
    for horizon in horizons:
        specs = selected_strategy_specs(horizon)
        required_features = tuple(
            dict.fromkeys(feature for spec in specs for feature in spec.features)
        )
        data["experiment_common_valid"] = (
            data["feature_valid_flag"].astype(bool)
            & data[list(required_features)].notna().all(axis=1)
        )
        eligible = []
        for index in data.index[data["experiment_common_valid"]]:
            training = _mature_training(
                data,
                anchor=data.loc[index, "trade_date"],
                horizon=horizon,
                required_features=required_features,
            )
            if len(training) >= int(settings["minimum_training_samples"]):
                eligible.append(int(index))
        indices = np.asarray(eligible, dtype=int)
        if indices.size == 0:
            raise RuntimeError(f"No eligible v6 strategy forecasts for {horizon}d.")
        paths = [
            _fit_spec_path(
                spec,
                data,
                indices,
                horizon=horizon,
                settings=settings,
                epsilon=epsilon,
            )
            for spec in specs
        ]
        persistence_rv = data.loc[indices, "rv"].to_numpy(dtype=float)
        paths.append(
            {
                "model_name": "persistence__raw__none",
                "family": "persistence",
                "objective": "raw",
                "blocks": "none",
                "alpha": np.nan,
                "forecast_rv": np.maximum(persistence_rv, epsilon),
                "forecast_log_rv": np.log(np.maximum(persistence_rv, epsilon)),
            }
        )
        for path in paths:
            pieces.append(
                pd.DataFrame(
                    {
                        "trade_date": data.loc[indices, "trade_date"].to_numpy(),
                        "horizon": horizon,
                        "model_name": path["model_name"],
                        "family": path["family"],
                        "objective": path["objective"],
                        "blocks": path["blocks"],
                        "alpha": path["alpha"],
                        "forecast_rv": path["forecast_rv"],
                        "forecast_log_rv": path["forecast_log_rv"],
                        "actual_rv": data.loc[
                            indices, f"target_rv_{horizon}d"
                        ].to_numpy(dtype=float),
                        "maturity_date": data.loc[
                            indices, f"target_maturity_date_{horizon}d"
                        ].to_numpy(),
                        "rv_at_signal": data.loc[indices, "rv"].to_numpy(dtype=float),
                        "jump_significant": data.loc[
                            indices, "jump_significant"
                        ].fillna(False).to_numpy(dtype=bool),
                    }
                )
            )
    result = pd.concat(pieces, ignore_index=True).sort_values(
        ["trade_date", "horizon", "model_name"]
    )
    return write_parquet(result, root / config["outputs"]["strategy_forecasts_path"])


def build_extended_features_and_opportunities(
    settings: dict,
    model_config: dict,
) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    raw = root / "data/raw"
    backfill = root / settings["backfill"]["raw_dir"]
    original_au = read_parquet(raw / "tqsdk_au_5m.parquet")
    early_au = read_parquet(backfill / "tqsdk_au_main_5m_2018_2019.parquet")
    extended_au = (
        pd.concat([early_au, original_au], ignore_index=True)
        .drop_duplicates(["ts_code", "timestamp_utc"], keep="last")
        .sort_values(["timestamp_utc", "ts_code"])
        .reset_index(drop=True)
    )
    original_slv = read_parquet(raw / "databento_slv_iv_30d.parquet")
    early_slv = read_parquet(backfill / "databento_slv_iv_30d_2018_2019.parquet")
    extended_slv = (
        pd.concat([early_slv, original_slv], ignore_index=True)
        .drop_duplicates("observation_date", keep="last")
        .sort_values("observation_date")
        .reset_index(drop=True)
    )
    extended_au_path = write_parquet(
        extended_au, backfill / "tqsdk_au_5m_extended.parquet"
    )
    extended_slv_path = write_parquet(
        extended_slv, backfill / "databento_slv_iv_30d_extended.parquet"
    )
    legacy_fred = read_parquet(backfill / "fred_legacy_broad_dollar.parquet")
    extended_fred = _stitch_legacy_broad_dollar(
        read_parquet(raw / "fred_initial_release_daily.parquet"),
        legacy_fred,
        current_series_id=model_config["data_sources"]["fred"]["series"][
            "broad_dollar"
        ],
    )
    extended_fred_path = write_parquet(
        extended_fred, backfill / "fred_initial_release_daily_extended.parquet"
    )
    features, audit = build_feature_table_from_frames(
        model_config,
        au_bars=extended_au,
        shfe_calendar=read_parquet(raw / "tqsdk_shfe_trade_calendar.parquet"),
        gc_1m=read_parquet(raw / "databento_gc_1m.parquet"),
        fred=extended_fred,
        events=read_parquet(raw / "official_macro_events.parquet"),
        gpr=read_parquet(raw / "gpr_daily_recent.parquet"),
        slv_iv=extended_slv,
        asof_utc=utc_now(),
    )
    feature_path = root / settings["outputs"]["feature_path"]
    audit_path = root / settings["outputs"]["leakage_audit_path"]
    write_parquet(features, feature_path)
    write_csv(audit, audit_path)
    forecast_path = generate_v6_strategy_forecasts(model_config)
    pairs = read_parquet(raw / "tqsdk_au_option_strategy_pairs.parquet")
    option_daily = read_parquet(raw / "tqsdk_au_option_daily_strategy.parquet")
    futures_daily = read_parquet(
        raw / "tqsdk_au_futures_daily_all_contracts.parquet"
    )
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    forecasts = read_parquet(forecast_path)
    opportunities = build_option_opportunities(
        pairs,
        option_daily,
        futures_daily,
        features,
        forecasts,
        risk_free_rate=float(settings["execution"]["risk_free_rate"]),
        minimum_vrp_history=int(model_config["strategy"]["minimum_vrp_history"]),
    )
    opportunities = add_strategy_signals(
        opportunities,
        minimum_volatility_edge=float(
            settings["model"]["signal_minimum_volatility_edge"]
        ),
        minimum_total_open_interest=float(
            model_config["strategy"]["minimum_total_open_interest"]
        ),
        minimum_leg_volume=float(model_config["strategy"]["minimum_leg_volume"]),
        minimum_premium_ticks=float(
            model_config["strategy"]["minimum_premium_ticks"]
        ),
    )
    opportunities_path = write_parquet(
        opportunities, root / settings["outputs"]["opportunities_path"]
    )
    return {
        "extended_au": extended_au_path,
        "extended_slv": extended_slv_path,
        "extended_fred": extended_fred_path,
        "features": feature_path,
        "leakage_audit": audit_path,
        "forecasts": forecast_path,
        "opportunities": opportunities_path,
    }
