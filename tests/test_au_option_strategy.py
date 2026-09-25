from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.data.au_option_loader import select_atm_option_pairs
from au_rv.strategy.black76 import (
    black76_straddle_delta,
    black76_straddle_price,
    implied_volatility_from_straddle,
)
from au_rv.strategy.volatility_straddle import add_strategy_signals


def test_black76_straddle_iv_round_trip_and_atm_delta():
    price = black76_straddle_price(600.0, 600.0, 30.0 / 365.0, 0.22, 0.018)
    implied = implied_volatility_from_straddle(
        price, 600.0, 600.0, 30.0 / 365.0, 0.018
    )
    assert np.isclose(implied, 0.22, atol=1.0e-8)
    delta = black76_straddle_delta(600.0, 600.0, 30.0 / 365.0, 0.22, 0.018)
    assert abs(delta) < 0.05


def test_atm_selection_uses_signal_close_and_exact_horizon():
    dates = pd.Series(pd.bdate_range("2024-01-02", periods=8))
    expiry = dates.iloc[7]
    catalog = pd.DataFrame(
        {
            "ts_code": ["C600", "P600", "C610", "P610"],
            "underlying_symbol": ["SHFE.au2402"] * 4,
            "expiry_date": [expiry] * 4,
            "strike_price": [600.0, 600.0, 610.0, 610.0],
            "option_class": ["CALL", "PUT", "CALL", "PUT"],
            "price_tick": [0.02] * 4,
            "volume_multiple": [1000.0] * 4,
        }
    )
    futures = pd.DataFrame(
        {
            "trade_date": [dates.iloc[2]],
            "ts_code": ["SHFE.au2402"],
            "close": [602.0],
        }
    )
    selected = select_atm_option_pairs(
        catalog,
        futures,
        dates,
        horizons=[5],
        start_date=dates.min(),
        end_date=dates.max(),
    )
    assert len(selected) == 1
    row = selected.iloc[0]
    assert row["signal_date"] == dates.iloc[2]
    assert row["entry_date"] == dates.iloc[3]
    assert row["strike_price"] == 600.0
    assert row["call_symbol"] == "C600"
    assert row["put_symbol"] == "P600"


def test_primary_signal_requires_main_and_robust_short_agreement():
    frame = pd.DataFrame(
        {
            "signal_iv": [0.30, 0.30, 0.30],
            "entry_iv": [0.30, 0.30, 0.30],
            "entry_straddle_open": [20.0, 20.0, 20.0],
            "price_tick": [0.02, 0.02, 0.02],
            "moneyness_abs": [0.001, 0.001, 0.001],
            "vrp_estimate": [0.01, 0.01, 0.01],
            "entry_future_open": [600.0, 600.0, 600.0],
            "signal_call_volume": [10.0, 10.0, 10.0],
            "signal_put_volume": [10.0, 10.0, 10.0],
            "entry_call_volume": [10.0, 10.0, 10.0],
            "entry_put_volume": [10.0, 10.0, 10.0],
            "entry_call_open_oi": [100.0, 100.0, 100.0],
            "entry_put_open_oi": [100.0, 100.0, 100.0],
            "main_volatility_edge": [-0.05, -0.05, 0.05],
            "robust_volatility_edge": [-0.04, 0.04, 0.04],
            "persistence_volatility_edge": [-0.03, -0.03, -0.03],
            "jump_significant": [False, False, False],
            "main_forecast_rv": [0.001, 0.001, 0.001],
        }
    )
    result = add_strategy_signals(
        frame,
        minimum_volatility_edge=0.015,
        minimum_total_open_interest=20.0,
        minimum_leg_volume=1.0,
        minimum_premium_ticks=6.0,
    )
    assert result["signal_model_timed_short"].tolist() == [-1.0, 0.0, 0.0]
