from __future__ import annotations

import math

import numpy as np
import pandas as pd

from au_rv.strategy.black76 import (
    black76_straddle_delta,
    implied_volatility_from_straddle,
)
from au_rv.strategy.volatility_straddle import _calendar_time, _option_mark_and_iv
from au_rv.strategy_v3.execution import (
    RealQuoteExecutionSettings,
    expiry_future_price,
    quote_snapshot_features,
)


def simulate_v5_trade(
    row: pd.Series,
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    settings: RealQuoteExecutionSettings,
    displayed_depth_multiplier: int | None,
    maximum_lots: int | None,
    hedge_interval_trading_days: int = 1,
    delta_change_threshold_contracts: int = 1,
    entry_snapshot: dict | None = None,
    expiry_future_quote: float | None = None,
) -> tuple[dict, pd.DataFrame] | None:
    """Simulate the v3 short straddle with configurable delta rehedging.

    The default interval/threshold exactly reproduces v3. Entry quotes remain
    synchronized historical top-book; scheduled rehedges use the daily close
    plus the configured explicit futures slippage assumption.
    """

    snapshot = entry_snapshot or quote_snapshot_features(
        ticks,
        row,
        maximum_quote_age_seconds=settings.quote_maximum_age_seconds,
    )
    if snapshot is None:
        return None
    future_path = futures_daily[
        futures_daily["ts_code"].astype(str).eq(str(row["underlying_symbol"]))
        & futures_daily["trade_date"].between(row["entry_date"], row["expiry_date"])
    ].sort_values("trade_date")
    if future_path.empty:
        return None
    if pd.Timestamp(future_path.iloc[0]["trade_date"]).normalize() != pd.Timestamp(
        row["entry_date"]
    ).normalize():
        return None
    if pd.Timestamp(future_path.iloc[-1]["trade_date"]).normalize() != pd.Timestamp(
        row["expiry_date"]
    ).normalize():
        return None

    multiplier = float(row["volume_multiple"])
    future_mid = float(snapshot["future_mid"])
    entry_premium = float(snapshot["short_entry_premium"])
    per_lot_risk_capital = (
        settings.short_risk_capital_fraction_of_notional * future_mid * multiplier
    )
    displayed_size = min(
        int(snapshot["call_bid_size"]), int(snapshot["put_bid_size"])
    )
    risk_lots = max(
        1,
        int(
            math.floor(
                settings.initial_capital
                * settings.risk_budget_fraction_per_trade
                / per_lot_risk_capital
            )
        ),
    )
    caps = [risk_lots]
    if displayed_depth_multiplier is not None:
        caps.append(displayed_size * int(displayed_depth_multiplier))
    if maximum_lots is not None:
        caps.append(int(maximum_lots))
    lots = min(caps)
    if lots < 1:
        return None

    time_to_expiry = _calendar_time(row["entry_date"], row["expiry_date"], at_open=True)
    entry_iv = implied_volatility_from_straddle(
        entry_premium,
        future_mid,
        float(row["strike_price"]),
        time_to_expiry,
        settings.risk_free_rate,
    )
    if not np.isfinite(entry_iv):
        entry_iv = implied_volatility_from_straddle(
            float(snapshot["mid_entry_premium"]),
            future_mid,
            float(row["strike_price"]),
            time_to_expiry,
            settings.risk_free_rate,
        )
    if not np.isfinite(entry_iv):
        return None

    cumulative_cost = lots * 2.0 * settings.option_commission_per_contract_side
    cumulative_option_pnl = 0.0
    cumulative_futures_pnl = 0.0
    last_option_mark = entry_premium
    last_iv = entry_iv
    initial_delta = black76_straddle_delta(
        future_mid,
        float(row["strike_price"]),
        time_to_expiry,
        last_iv,
        settings.risk_free_rate,
    )
    hedge = int(round(lots * initial_delta))
    initial_hedge = hedge
    if hedge > 0:
        previous_future_price = float(snapshot["future_ask"])
    elif hedge < 0:
        previous_future_price = float(snapshot["future_bid"])
    else:
        previous_future_price = future_mid
    hedge_trade_count = int(hedge != 0)
    hedge_contract_turnover = abs(hedge)
    if hedge:
        cumulative_cost += abs(hedge) * settings.futures_commission_per_contract_side
    expiry_tick_price = (
        float(expiry_future_quote)
        if expiry_future_quote is not None
        else expiry_future_price(ticks, str(row["opportunity_id"]))
    )
    daily_rows: list[dict] = []
    for position, future_record in enumerate(future_path.itertuples(index=False)):
        date = pd.Timestamp(future_record.trade_date).normalize()
        future_close = float(future_record.close)
        is_expiry = date == pd.Timestamp(row["expiry_date"]).normalize()
        if is_expiry and np.isfinite(expiry_tick_price):
            future_close = expiry_tick_price
        cumulative_futures_pnl += hedge * multiplier * (
            future_close - previous_future_price
        )
        if is_expiry:
            option_mark = abs(future_close - float(row["strike_price"]))
        else:
            option_mark, last_iv = _option_mark_and_iv(
                date,
                str(row["call_symbol"]),
                str(row["put_symbol"]),
                future_close,
                float(row["strike_price"]),
                pd.Timestamp(row["expiry_date"]),
                option_lookup,
                last_iv,
                settings.risk_free_rate,
            )
        cumulative_option_pnl -= lots * multiplier * (option_mark - last_option_mark)
        last_option_mark = option_mark

        if is_expiry:
            if hedge:
                cumulative_cost += abs(hedge) * (
                    settings.futures_commission_per_contract_side
                    + settings.futures_slippage_ticks_per_trade
                    * settings.future_tick
                    * multiplier
                )
                hedge_trade_count += 1
                hedge_contract_turnover += abs(hedge)
            cumulative_cost += lots * 2.0 * settings.option_exercise_fee_per_contract
            hedge = 0
        else:
            delta = black76_straddle_delta(
                future_close,
                float(row["strike_price"]),
                _calendar_time(date, row["expiry_date"]),
                last_iv,
                settings.risk_free_rate,
            )
            if not np.isfinite(delta):
                return None
            desired_hedge = int(round(lots * delta))
            hedge_change = desired_hedge - hedge
            scheduled = (position + 1) % max(int(hedge_interval_trading_days), 1) == 0
            large_enough = abs(hedge_change) >= max(
                int(delta_change_threshold_contracts), 1
            )
            if scheduled and large_enough:
                cumulative_cost += abs(hedge_change) * (
                    settings.futures_commission_per_contract_side
                    + settings.futures_slippage_ticks_per_trade
                    * settings.future_tick
                    * multiplier
                )
                hedge_trade_count += 1
                hedge_contract_turnover += abs(hedge_change)
                hedge = desired_hedge
        net = cumulative_option_pnl + cumulative_futures_pnl - cumulative_cost
        daily_rows.append(
            {
                "trade_date": date,
                "cumulative_option_pnl": cumulative_option_pnl,
                "cumulative_futures_pnl": cumulative_futures_pnl,
                "cumulative_cost": cumulative_cost,
                "cumulative_net_pnl": net,
                "daily_hedge_contracts": hedge,
                "option_mark": option_mark,
                "mark_iv": last_iv,
            }
        )
        previous_future_price = future_close
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return None
    daily["daily_net_pnl"] = daily["cumulative_net_pnl"].diff().fillna(
        daily["cumulative_net_pnl"]
    )
    final = daily.iloc[-1]
    risk_capital = lots * per_lot_risk_capital
    crossing_per_lot = (
        float(snapshot["mid_entry_premium"]) - entry_premium
    ) * multiplier
    trade = {
        "opportunity_id": row["opportunity_id"],
        "signal_date": row["signal_date"],
        "entry_date": row["entry_date"],
        "expiry_date": row["expiry_date"],
        "horizon": int(row["horizon"]),
        "side": -1,
        "underlying_symbol": row["underlying_symbol"],
        "call_symbol": row["call_symbol"],
        "put_symbol": row["put_symbol"],
        "strike_price": float(row["strike_price"]),
        "volume_multiple": multiplier,
        "lots": lots,
        "risk_budget_lot_cap": risk_lots,
        "displayed_size": displayed_size,
        "displayed_depth_multiplier": displayed_depth_multiplier,
        "execution_timestamp_utc": snapshot["execution_timestamp_utc"],
        "execution_window": snapshot["execution_window"],
        "entry_future_mid": future_mid,
        "entry_future_bid_ask_spread": float(
            snapshot["future_ask"] - snapshot["future_bid"]
        ),
        "entry_straddle_premium": entry_premium,
        "entry_mid_straddle_premium": float(snapshot["mid_entry_premium"]),
        "entry_straddle_bid_ask_spread": float(
            snapshot["long_entry_premium"] - snapshot["short_entry_premium"]
        ),
        "entry_relative_spread": float(
            (snapshot["long_entry_premium"] - snapshot["short_entry_premium"])
            / max(snapshot["mid_entry_premium"], 1.0e-8)
        ),
        "embedded_entry_crossing_cost": lots * max(crossing_per_lot, 0.0),
        "entry_iv": entry_iv,
        "initial_straddle_delta": initial_delta,
        "initial_hedge_contracts": initial_hedge,
        "hedge_interval_trading_days": int(hedge_interval_trading_days),
        "delta_change_threshold_contracts": int(delta_change_threshold_contracts),
        "hedge_trade_count": hedge_trade_count,
        "hedge_contract_turnover": hedge_contract_turnover,
        "risk_capital": risk_capital,
        "per_lot_risk_capital": per_lot_risk_capital,
        "gross_option_pnl": float(final["cumulative_option_pnl"]),
        "gross_futures_pnl": float(final["cumulative_futures_pnl"]),
        "transaction_cost": float(final["cumulative_cost"]),
        "net_pnl": float(final["cumulative_net_pnl"]),
        "return_on_risk_capital": float(final["cumulative_net_pnl"] / risk_capital),
    }
    return trade, daily
