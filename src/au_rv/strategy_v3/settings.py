from __future__ import annotations

from pathlib import Path

import yaml


def load_strategy_v3_v4_settings(project_root: Path) -> dict:
    path = Path(project_root) / "strategy_v3_v4_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["_config_path"] = str(path)
    settings["_project_root"] = str(Path(project_root))
    return settings
