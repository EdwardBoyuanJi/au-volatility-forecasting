from __future__ import annotations

import copy
from pathlib import Path

import yaml

from au_rv.config import load_config


def load_strategy_v6_settings(project_root: Path) -> tuple[dict, dict]:
    root = Path(project_root)
    path = root / "strategy_v6_5y_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["_project_root"] = str(root)
    settings["_config_path"] = str(path)
    model_config = copy.deepcopy(load_config(root / "config.yaml"))
    model_config["control_experiment"]["minimum_training_samples"] = int(
        settings["model"]["minimum_training_samples"]
    )
    model_config["control_experiment"]["retrain_frequency"] = settings["model"][
        "retrain_frequency"
    ]
    model_config["outputs"]["features_path"] = settings["outputs"]["feature_path"]
    model_config["outputs"]["leakage_audit_path"] = settings["outputs"][
        "leakage_audit_path"
    ]
    model_config["outputs"]["strategy_forecasts_path"] = settings["outputs"][
        "forecast_path"
    ]
    return settings, model_config
