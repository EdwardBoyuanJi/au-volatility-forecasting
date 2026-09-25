from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.config import resolve_path
from au_rv.io import read_parquet, write_parquet
from au_rv.models.control_experiment import (
    ExperimentSpec,
    _benchmark_paths,
    _fit_spec_path,
    _mature_training,
    base_features,
    control_blocks,
)
from au_rv.models.garch import add_causal_garch_features


def selected_strategy_specs(horizon: int) -> list[ExperimentSpec]:
    blocks = control_blocks(horizon)

    def features(family: str, selected_blocks: tuple[str, ...]) -> tuple[str, ...]:
        columns = list(base_features(family, horizon))
        for block in selected_blocks:
            columns.extend(blocks[block])
        return tuple(dict.fromkeys(columns))

    main_blocks = ("macro", "gvz", "slv_iv", "us_epu")
    robust_blocks = ("macro", "us_epu")
    return [
        ExperimentSpec(
            model_name="har__mse_log__macro+gvz+slv_iv+us_epu",
            family="har",
            objective="mse_log",
            blocks=main_blocks,
            features=features("har", main_blocks),
        ),
        ExperimentSpec(
            model_name="har__qlike__macro+us_epu",
            family="har",
            objective="qlike",
            blocks=robust_blocks,
            features=features("har", robust_blocks),
        ),
    ]


def generate_strategy_forecasts(config: dict) -> Path:
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    horizons = [int(value) for value in config["features"]["horizons"]]
    settings = config["control_experiment"]
    epsilon = float(config["project"]["epsilon"])
    data = add_causal_garch_features(
        features,
        horizons=horizons,
        minimum_observations=int(settings["minimum_garch_observations"]),
        refit_frequency=settings["garch_refit_frequency"],
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    pieces: list[pd.DataFrame] = []
    for horizon in horizons:
        garch_column = f"garch_forecast_rv_{horizon}d"
        data["experiment_common_valid"] = (
            data["experiment_feature_valid_flag"].astype(bool)
            & data[garch_column].notna()
        )
        eligible = []
        for index in data.index[data["experiment_common_valid"]]:
            training = _mature_training(
                data,
                anchor=data.loc[index, "trade_date"],
                horizon=horizon,
                required_features=tuple(base_features("harq", horizon)),
            )
            if len(training) >= int(settings["minimum_training_samples"]):
                eligible.append(int(index))
        indices = np.asarray(eligible, dtype=int)
        if indices.size == 0:
            raise RuntimeError(f"No eligible strategy forecasts for {horizon}d.")
        paths = [
            _fit_spec_path(
                spec,
                data,
                indices,
                horizon=horizon,
                settings=settings,
                epsilon=epsilon,
            )
            for spec in selected_strategy_specs(horizon)
        ]
        persistence = _benchmark_paths(data, indices, horizon)[0]
        paths.append(persistence)
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
    destination = resolve_path(
        config, config["outputs"]["strategy_forecasts_path"]
    )
    return write_parquet(result, destination)
