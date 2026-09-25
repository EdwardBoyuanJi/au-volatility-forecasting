from __future__ import annotations

import math


FINAL_STRATEGY_NAME = "final_narrow_rb10_hedge2d"
EXPECTED_PROJECT_VERSION = "final_narrow_rb10_hedge2d_20260830"

EXPECTED_REFERENCE_METRICS = {
    "trade_count": 12,
    "net_profit": 509760.0,
    "sharpe": 1.0849355242203031,
    "max_drawdown": -0.0090916363096299,
    "win_rate": 0.8333333333333334,
    "total_lots": 42,
    "total_hedge_contract_turnover": 72,
}


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)


def validate_final_strategy_settings(settings: dict) -> None:
    if settings.get("project", {}).get("version") != EXPECTED_PROJECT_VERSION:
        raise ValueError(
            f"Frozen version must be {EXPECTED_PROJECT_VERSION!r}; create a new version "
            "instead of mutating the final strategy."
        )
    strategies = settings.get("strategies", [])
    if len(strategies) != 1:
        raise ValueError("The final package must contain exactly one strategy.")
    strategy = strategies[0]
    exact = {
        "name": FINAL_STRATEGY_NAME,
        "family": "final_frozen_strategy",
        "displayed_depth_multiplier": 5,
        "hedge_interval_trading_days": 2,
        "delta_change_threshold_contracts": 1,
        "horizons": [5, 20, 40],
        "sizing_mode": "flat",
    }
    for field, expected in exact.items():
        if strategy.get(field) != expected:
            raise ValueError(
                f"Frozen field {field!r} must equal {expected!r}; found "
                f"{strategy.get(field)!r}."
            )
    numeric = {
        "risk_budget_fraction": 0.10,
        "minimum_edge_strength": 0.015,
        "maximum_relative_spread": 0.25,
    }
    for field, expected in numeric.items():
        if not _same(strategy.get(field), expected):
            raise ValueError(
                f"Frozen field {field!r} must equal {expected}; found "
                f"{strategy.get(field)!r}."
            )
    execution = settings.get("execution", {})
    execution_numeric = {
        "initial_capital": 10_000_000.0,
        "maximum_portfolio_risk_fraction": 0.30,
        "entry_quote_maximum_age_seconds": 10.0,
        "futures_slippage_ticks_per_trade": 0.0,
    }
    for field, expected in execution_numeric.items():
        if not _same(execution.get(field), expected):
            raise ValueError(
                f"Frozen execution field {field!r} must equal {expected}; found "
                f"{execution.get(field)!r}."
            )
    if execution.get("enforce_futures_visible_depth") is not True:
        raise ValueError("Real futures top-book visible depth must remain enforced.")
