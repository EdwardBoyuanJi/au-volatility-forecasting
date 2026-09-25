from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from au_rv.final_strategy.spec import validate_final_strategy_settings


def _settings() -> dict:
    with open("final_strategy_10pct_2d_config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_frozen_final_strategy_config_is_valid() -> None:
    validate_final_strategy_settings(_settings())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_budget_fraction", 0.12),
        ("hedge_interval_trading_days", 1),
        ("maximum_relative_spread", 0.30),
    ],
)
def test_frozen_final_strategy_rejects_silent_mutation(field: str, value) -> None:
    settings = deepcopy(_settings())
    settings["strategies"][0][field] = value
    with pytest.raises(ValueError):
        validate_final_strategy_settings(settings)
