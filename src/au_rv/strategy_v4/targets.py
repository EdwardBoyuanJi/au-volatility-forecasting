from __future__ import annotations

import math

import numpy as np
import pandas as pd

from au_rv.strategy.black76 import implied_volatility_from_straddle
from au_rv.strategy.volatility_straddle import _calendar_time
from au_rv.strategy_v3.execution import (
    RealQuoteExecutionSettings,
    expiry_future_price,
    quote_snapshot_features,
    simulate_naked_real_quote_trade,
)


def _finite(value) -> float:
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan


def _feature_row(
    row: pd.Series,
    side: int,
    snapshot: dict,
    settings: RealQuoteExecutionSettings,
) -> dict:
    multiplier = float(row["volume_multiple"])
    future_mid = float(snapshot["future_mid"])
    notional = future_mid * multiplier
    if side == -1:
        premium = float(snapshot["short_entry_premium"])
        displayed_size = min(
            int(snapshot["call_bid_size"]), int(snapshot["put_bid_size"])
        )
        per_lot_risk = settings.short_risk_capital_fraction_of_notional * notional
        crossing = (float(snapshot["mid_entry_premium"]) - premium) * multiplier
    else:
        premium = float(snapshot["long_entry_premium"])
        displayed_size = min(
            int(snapshot["call_ask_size"]), int(snapshot["put_ask_size"])
        )
        per_lot_risk = max(
            premium * multiplier,
            settings.long_minimum_risk_capital_fraction_of_notional * notional,
        )
        crossing = (premium - float(snapshot["mid_entry_premium"])) * multiplier
    risk_lot_cap = max(
        1,
        int(
            math.floor(
                settings.initial_capital
                * settings.risk_budget_fraction_per_trade
                / per_lot_risk
            )
        ),
    )
    entry_iv = implied_volatility_from_straddle(
        premium,
        future_mid,
        float(row["strike_price"]),
        _calendar_time(row["entry_date"], row["expiry_date"], at_open=True),
        settings.risk_free_rate,
    )
    if not np.isfinite(entry_iv):
        entry_iv = implied_volatility_from_straddle(
            float(snapshot["mid_entry_premium"]),
            future_mid,
            float(row["strike_price"]),
            _calendar_time(row["entry_date"], row["expiry_date"], at_open=True),
            settings.risk_free_rate,
        )
    signal_volume = float(row.get("signal_call_volume", 0) or 0) + float(
        row.get("signal_put_volume", 0) or 0
    )
    entry_volume = float(row.get("entry_call_volume", 0) or 0) + float(
        row.get("entry_put_volume", 0) or 0
    )
    open_interest = float(row.get("entry_call_open_oi", 0) or 0) + float(
        row.get("entry_put_open_oi", 0) or 0
    )
    signal_iv = _finite(row.get("signal_iv"))
    main_fair = _finite(row.get("main_fair_volatility"))
    robust_fair = _finite(row.get("robust_fair_volatility"))
    persistence_fair = _finite(row.get("persistence_fair_volatility"))
    return {
        "opportunity_id": row["opportunity_id"],
        "signal_date": row["signal_date"],
        "entry_date": row["entry_date"],
        "expiry_date": row["expiry_date"],
        "horizon": int(row["horizon"]),
        "side": int(side),
        "side_label": "short" if side == -1 else "long",
        "execution_window": snapshot["execution_window"],
        "signal_iv": signal_iv,
        "real_entry_iv": float(entry_iv) if np.isfinite(entry_iv) else np.nan,
        "main_fair_volatility": main_fair,
        "robust_fair_volatility": robust_fair,
        "persistence_fair_volatility": persistence_fair,
        "main_volatility_edge": _finite(row.get("main_volatility_edge")),
        "robust_volatility_edge": _finite(row.get("robust_volatility_edge")),
        "persistence_volatility_edge": _finite(
            row.get("persistence_volatility_edge")
        ),
        "forecast_disagreement": abs(main_fair - robust_fair)
        if np.isfinite(main_fair) and np.isfinite(robust_fair)
        else np.nan,
        "vrp_estimate": _finite(row.get("vrp_estimate")),
        "moneyness_abs": _finite(row.get("moneyness_abs")),
        "term_sqrt": math.sqrt(float(row["horizon"]) / 252.0),
        "jump_significant_numeric": int(bool(row.get("jump_significant", False))),
        "log_signal_volume": math.log1p(max(signal_volume, 0.0)),
        "log_entry_daily_volume": math.log1p(max(entry_volume, 0.0)),
        "log_open_interest": math.log1p(max(open_interest, 0.0)),
        "entry_future_mid": future_mid,
        "log_entry_notional": math.log(max(notional, 1.0)),
        "entry_premium": premium,
        "entry_mid_premium": float(snapshot["mid_entry_premium"]),
        "entry_premium_to_future": premium / future_mid,
        "entry_relative_spread": (
            float(snapshot["long_entry_premium"] - snapshot["short_entry_premium"])
            / max(float(snapshot["mid_entry_premium"]), 1.0e-8)
        ),
        "future_spread_bps": (
            float(snapshot["future_ask"] - snapshot["future_bid"])
            / future_mid
            * 10_000.0
        ),
        "signal_to_entry_iv_change": float(entry_iv - signal_iv)
        if np.isfinite(entry_iv) and np.isfinite(signal_iv)
        else np.nan,
        "displayed_size": displayed_size,
        "per_lot_risk_capital": per_lot_risk,
        "risk_budget_lot_cap": risk_lot_cap,
        "available_lots_depth_1x": min(risk_lot_cap, displayed_size),
        "available_lots_depth_3x": min(risk_lot_cap, displayed_size * 3),
        "available_lots_unlimited": risk_lot_cap,
        "embedded_entry_crossing_cost_per_lot": max(crossing, 0.0),
    }


def build_direct_pnl_targets(
    opportunities: pd.DataFrame,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    settings: RealQuoteExecutionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-lot long/short net-P&L targets and log quote exclusions."""

    option_lookup = option_daily.copy()
    option_lookup["trade_date"] = pd.to_datetime(
        option_lookup["trade_date"]
    ).dt.normalize()
    option_lookup = option_lookup.set_index(["trade_date", "ts_code"])
    target_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for _, row in opportunities.sort_values(["entry_date", "horizon"]).iterrows():
        snapshot = quote_snapshot_features(
            ticks,
            row,
            maximum_quote_age_seconds=settings.quote_maximum_age_seconds,
        )
        coverage = {
            "opportunity_id": row["opportunity_id"],
            "entry_date": row["entry_date"],
            "expiry_date": row["expiry_date"],
            "horizon": int(row["horizon"]),
            "synchronized_entry_quote_available": snapshot is not None,
            "long_target_available": False,
            "short_target_available": False,
        }
        if snapshot is None:
            coverage_rows.append(coverage)
            continue
        expiry_quote = expiry_future_price(ticks, str(row["opportunity_id"]))
        for side in (-1, 1):
            features = _feature_row(row, side, snapshot, settings)
            features.update(
                {
                    "target_per_lot_net_pnl": np.nan,
                    "target_return_on_risk_capital": np.nan,
                    "target_gross_option_pnl": np.nan,
                    "target_gross_futures_pnl": np.nan,
                    "target_transaction_cost": np.nan,
                }
            )
            simulated = simulate_naked_real_quote_trade(
                row,
                side,
                option_lookup,
                futures_daily,
                ticks,
                settings=settings,
                displayed_depth_multiplier=None,
                maximum_lots=1,
                entry_snapshot=snapshot,
                expiry_future_quote=(
                    float(expiry_quote) if np.isfinite(expiry_quote) else None
                ),
            )
            if simulated is not None:
                trade, _ = simulated
                features.update(
                    {
                        "target_per_lot_net_pnl": float(trade["net_pnl"]),
                        "target_return_on_risk_capital": float(
                            trade["return_on_risk_capital"]
                        ),
                        "target_gross_option_pnl": float(trade["gross_option_pnl"]),
                        "target_gross_futures_pnl": float(
                            trade["gross_futures_pnl"]
                        ),
                        "target_transaction_cost": float(trade["transaction_cost"]),
                    }
                )
                coverage[f"{'short' if side == -1 else 'long'}_target_available"] = True
            target_rows.append(features)
        coverage_rows.append(coverage)
    return pd.DataFrame(target_rows), pd.DataFrame(coverage_rows)
