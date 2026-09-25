from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.strategy_v5.backtest import (
    VariantSpec,
    _fixed_holdout_specs,
    _sizing_multiplier,
    build_variant_catalog,
)
from au_rv.strategy_v5.settings import load_strategy_v5_settings


ROOT = Path(__file__).resolve().parents[1]


def test_variant_catalog_contains_requested_capacity_depths_once() -> None:
    specs = build_variant_catalog(load_strategy_v5_settings(ROOT))
    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    for depth in (1, 3, 5, 8, 12):
        assert f"capacity_depth_{depth}x" in names
    assert "capacity_unlimited" in names


def test_edge_spread_sizing_rewards_strong_edge_and_tight_spread() -> None:
    spec = VariantSpec(
        name="test", family="test", sizing_mode="edge_spread_weighted"
    )
    weak = pd.Series(
        {"main_volatility_edge": -0.03, "robust_volatility_edge": -0.04}
    )
    strong = pd.Series(
        {"main_volatility_edge": -0.08, "robust_volatility_edge": -0.09}
    )
    assert _sizing_multiplier(spec, strong, 0.10) > _sizing_multiplier(
        spec, weak, 0.50
    )
    assert _sizing_multiplier(spec, strong, 0.10) <= 2.5


def test_fixed_holdout_selection_cannot_see_holdout_outcomes() -> None:
    specs = [
        VariantSpec(name="capacity_depth_1x", family="capacity"),
        VariantSpec(name="alternative", family="test"),
    ]
    selected = pd.DataFrame(
        {
            "opportunity_id": [f"p{i}" for i in range(4)] + ["future"],
            "expiry_date": pd.to_datetime(
                ["2025-02-01", "2025-04-01", "2025-06-01", "2025-08-01", "2026-03-01"]
            ),
        }
    )
    rows = []
    for index, opportunity_id in enumerate(selected["opportunity_id"]):
        rows.extend(
            [
                {
                    "opportunity_id": opportunity_id,
                    "variant": "capacity_depth_1x",
                    "executed": True,
                    "net_pnl": float(index + 1),
                },
                {
                    "opportunity_id": opportunity_id,
                    "variant": "alternative",
                    "executed": True,
                    "net_pnl": (
                        float(15 + index) if opportunity_id != "future" else -1e9
                    ),
                },
            ]
        )
    frozen, history = _fixed_holdout_specs(
        selected,
        pd.DataFrame(rows),
        specs,
        holdout_start=pd.Timestamp("2026-01-01"),
        minimum_executed_trades=4,
    )
    assert {spec.name.split("__", 1)[1] for spec in frozen} == {"alternative"}
    assert set(history["selected_variant"]) == {"alternative"}


def test_v5_depth_one_exactly_reproduces_frozen_v3_baseline() -> None:
    v3 = pd.read_csv(ROOT / "data/outputs/strategy_v3_real_quotes/metrics.csv")
    v5 = pd.read_csv(
        ROOT / "data/outputs/strategy_v5_profit_enhancement/metrics.csv"
    )
    expected = v3[v3["strategy"].eq("v3_old_signal_real_topbook_depth_1x")].iloc[0]
    actual = v5[v5["strategy"].eq("capacity_depth_1x")].iloc[0]
    for column in ("net_profit", "sharpe", "max_drawdown", "win_rate"):
        assert np.isclose(actual[column], expected[column])
    assert int(actual["trade_count"]) == int(expected["trade_count"])
