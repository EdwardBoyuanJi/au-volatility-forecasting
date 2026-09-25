from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.strategy_v3.execution import synchronized_naked_entry_quotes
from au_rv.strategy_v4.models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_causal_direct_pnl_predictions,
    choose_causal_policy,
)


def test_real_quote_entry_requires_synchronized_three_leg_topbook() -> None:
    base = pd.Timestamp("2025-04-24 13:00:00", tz="UTC")
    ticks = pd.DataFrame(
        [
            {
                "opportunity_id": "x",
                "window_type": "entry_night",
                "leg_role": role,
                "timestamp_utc": base + pd.Timedelta(seconds=offset),
                "bid_price1": 10.0,
                "ask_price1": 10.2,
                "bid_volume1": 3,
                "ask_volume1": 4,
            }
            for offset, role in enumerate(
                ["short_call", "short_put", "underlying_future"]
            )
        ]
    )
    synchronized = synchronized_naked_entry_quotes(
        ticks, "x", maximum_quote_age_seconds=10.0
    )
    assert synchronized is not None
    timestamp, quotes, window = synchronized
    assert timestamp == base + pd.Timedelta(seconds=2)
    assert set(quotes) == {"short_call", "short_put", "underlying_future"}
    assert window == "entry_night"
    assert (
        synchronized_naked_entry_quotes(
            ticks, "x", maximum_quote_age_seconds=1.0
        )
        is None
    )


def _synthetic_direct_pnl_rows() -> pd.DataFrame:
    rows = []
    for opportunity in range(10):
        entry = pd.Timestamp("2024-01-02") + pd.Timedelta(days=10 * opportunity)
        for side in (-1, 1):
            row = {
                "opportunity_id": f"o{opportunity}",
                "signal_date": entry - pd.Timedelta(days=1),
                "entry_date": entry,
                "expiry_date": entry + pd.Timedelta(days=5),
                "side": side,
                "target_per_lot_net_pnl": float((opportunity + 1) * -side * 100),
            }
            for feature in NUMERIC_FEATURES:
                row[feature] = float(opportunity + (side == 1))
            row.update(
                {
                    "horizon": 5,
                    "side_label": "short" if side == -1 else "long",
                    "execution_window": "entry_night",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_direct_pnl_walk_forward_never_uses_future_expiry_targets() -> None:
    original = _synthetic_direct_pnl_rows()
    changed = original.copy()
    changed.loc[changed["opportunity_id"].eq("o9"), "target_per_lot_net_pnl"] = 1e9
    first = add_causal_direct_pnl_predictions(
        original,
        families=["ridge"],
        minimum_training_rows=4,
        random_seed=1,
    )
    second = add_causal_direct_pnl_predictions(
        changed,
        families=["ridge"],
        minimum_training_rows=4,
        random_seed=1,
    )
    before_changed_outcome = first["entry_date"].lt(
        original.loc[original["opportunity_id"].eq("o9"), "entry_date"].iloc[0]
    )
    np.testing.assert_allclose(
        first.loc[before_changed_outcome, "predicted_per_lot_pnl_ridge"],
        second.loc[before_changed_outcome, "predicted_per_lot_pnl_ridge"],
        equal_nan=True,
    )


def test_policy_uses_predicted_best_side_and_positive_realized_history() -> None:
    rows = []
    for opportunity in range(10):
        for side in (-1, 1):
            prediction = 100.0 if side == -1 else -100.0
            rows.append(
                {
                    "opportunity_id": f"o{opportunity}",
                    "side": side,
                    "predicted_per_lot_pnl_ridge": prediction,
                    "target_per_lot_net_pnl": (
                        80.0 + opportunity if side == -1 else -120.0
                    ),
                    "available_lots_depth_1x": 1,
                }
            )
    policy = choose_causal_policy(
        pd.DataFrame(rows),
        families=["ridge"],
        lot_column="available_lots_depth_1x",
        objective="sharpe",
        threshold_quantiles=[0.0, 0.5],
        minimum_trades=8,
    )
    assert policy is not None
    assert policy["model_family"] == "ridge"
    assert policy["historical_net_profit"] == 845.0
