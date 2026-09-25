from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.strategy_v2.data import select_protective_wings
from au_rv.strategy_v2.execution import (
    published_margin_rate,
    short_option_margin,
    synchronized_entry_quotes,
)
from au_rv.strategy_v2.models import (
    MAIN_MODEL,
    ROBUST_MODEL,
    add_causal_forecast_bounds,
    add_v2_signals,
)
from au_rv.strategy_v2.pricing import (
    black76_option_price,
    implied_volatility_from_option,
)


def test_black76_option_iv_round_trip() -> None:
    price = black76_option_price(800.0, 820.0, 40.0 / 365.0, 0.24, "call", 0.018)
    recovered = implied_volatility_from_option(
        price, 800.0, 820.0, 40.0 / 365.0, "call", 0.018
    )
    assert recovered == pytest.approx(0.24, abs=1.0e-8)


def test_select_protective_wings_uses_first_strikes_outside_fixed_band() -> None:
    opportunity = pd.DataFrame(
        [
            {
                "signal_distribution_short": True,
                "signal_date": "2025-04-23",
                "expiry_date": "2025-05-23",
                "underlying_symbol": "SHFE.au2506",
                "signal_future_close": 784.0,
                "strike_price": 784.0,
                "horizon": 20,
            }
        ]
    )
    catalog = pd.DataFrame(
        [
            {"ts_code": "P728", "underlying_symbol": "SHFE.au2506", "expiry_date": "2025-05-23", "option_class": "PUT", "strike_price": 728.0},
            {"ts_code": "P720", "underlying_symbol": "SHFE.au2506", "expiry_date": "2025-05-23", "option_class": "PUT", "strike_price": 720.0},
            {"ts_code": "C840", "underlying_symbol": "SHFE.au2506", "expiry_date": "2025-05-23", "option_class": "CALL", "strike_price": 840.0},
            {"ts_code": "C848", "underlying_symbol": "SHFE.au2506", "expiry_date": "2025-05-23", "option_class": "CALL", "strike_price": 848.0},
        ]
    )
    selected = select_protective_wings(opportunity, catalog, wing_width_fraction=0.08)
    assert selected.loc[0, "put_wing_strike"] == 720.0
    assert selected.loc[0, "call_wing_strike"] == 848.0


def test_synchronized_entry_quotes_respects_side_and_staleness() -> None:
    base = pd.Timestamp("2025-04-24 13:00:00", tz="UTC")
    rows = []
    for offset, role in enumerate(
        ["short_call", "short_put", "long_call_wing", "long_put_wing", "underlying_future"]
    ):
        rows.append(
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
        )
    synchronized = synchronized_entry_quotes(pd.DataFrame(rows), "x")
    assert synchronized is not None
    timestamp, quotes, window = synchronized
    assert timestamp == base + pd.Timedelta(seconds=4)
    assert set(quotes) == {
        "short_call",
        "short_put",
        "long_call_wing",
        "long_put_wing",
        "underlying_future",
    }
    assert window == "entry_night"


def test_exchange_margin_schedule_and_option_formula() -> None:
    assert published_margin_rate(pd.Timestamp("2024-04-25"), "SHFE.au2406") == 0.10
    assert published_margin_rate(pd.Timestamp("2024-05-24"), "SHFE.au2408") == 0.12
    assert published_margin_rate(pd.Timestamp("2025-05-30"), "SHFE.au2508") == 0.15
    assert published_margin_rate(pd.Timestamp("2026-03-24"), "SHFE.au2606") == 0.17
    margin = short_option_margin(12.0, 800.0, 820.0, "call", 1000.0, 0.13)
    assert margin == 12_000.0 + max(104_000.0 - 10_000.0, 52_000.0)


def test_forecast_bound_excludes_unmatured_history() -> None:
    opportunities = pd.DataFrame(
        [
            {
                "signal_date": "2024-01-10",
                "horizon": 5,
                "main_forecast_rv": 0.0004,
                "robust_forecast_rv": 0.0005,
                "vrp_estimate": 0.0,
            }
        ]
    )
    forecast_rows = []
    for model in (MAIN_MODEL, ROBUST_MODEL):
        for date, value in zip(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), [0.0003, 0.0004, 0.0005]):
            forecast_rows.append(
                {"trade_date": date, "horizon": 5, "model_name": model, "forecast_rv": value}
            )
    features = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "target_rv_5d": [0.0004, 0.0008, 0.0006],
            "target_maturity_date_5d": pd.to_datetime(["2024-01-05", "2024-01-11", "2024-01-08"]),
        }
    )
    result = add_causal_forecast_bounds(
        opportunities,
        pd.DataFrame(forecast_rows),
        features,
        quantile=0.90,
        residual_window=252,
        minimum_samples=2,
    )
    assert result.loc[0, "main_bound_history_count"] == 2
    assert np.isfinite(result.loc[0, "main_upper_fair_volatility"])


def test_v2_signal_requires_all_distribution_and_pnl_gates() -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_model_timed_short": -1,
                "signal_iv": 0.30,
                "main_upper_fair_volatility": 0.25,
                "robust_upper_fair_volatility": 0.26,
                "forecast_tail_imbalance": 0.60,
                "forecast_jump_share": 0.10,
                "jump_significant": False,
                "predicted_normalized_pnl_lower": 0.01,
            },
            {
                "signal_model_timed_short": -1,
                "signal_iv": 0.30,
                "main_upper_fair_volatility": 0.25,
                "robust_upper_fair_volatility": 0.26,
                "forecast_tail_imbalance": 0.80,
                "forecast_jump_share": 0.10,
                "jump_significant": False,
                "predicted_normalized_pnl_lower": 0.01,
            },
        ]
    )
    result = add_v2_signals(
        frame,
        minimum_volatility_edge=0.015,
        maximum_tail_imbalance=0.70,
        maximum_jump_share=0.25,
    )
    assert result["signal_v2_short"].tolist() == [True, False]


import pytest
