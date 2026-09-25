from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from au_rv.strategy.black76 import (
    black76_straddle_delta,
    implied_volatility_from_straddle,
)
from au_rv.strategy.volatility_straddle import _calendar_time, _option_mark_and_iv


@dataclass(frozen=True)
class RealQuoteExecutionSettings:
    initial_capital: float
    risk_budget_fraction_per_trade: float
    short_risk_capital_fraction_of_notional: float
    long_minimum_risk_capital_fraction_of_notional: float
    risk_free_rate: float
    option_commission_per_contract_side: float
    option_exercise_fee_per_contract: float
    futures_commission_per_contract_side: float
    futures_slippage_ticks_per_trade: float
    future_tick: float
    quote_maximum_age_seconds: float


def _valid_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["bid_price1"].gt(0)
        & frame["ask_price1"].gt(frame["bid_price1"])
        & frame["bid_volume1"].gt(0)
        & frame["ask_volume1"].gt(0)
    ].sort_values("timestamp_utc")


def synchronized_naked_entry_quotes(
    ticks: pd.DataFrame,
    opportunity: str,
    *,
    maximum_quote_age_seconds: float,
    window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
) -> tuple[pd.Timestamp, dict[str, pd.Series], str] | None:
    roles = ("short_call", "short_put", "underlying_future")
    data = ticks[ticks["opportunity_id"].astype(str).eq(str(opportunity))]
    for window in window_preference:
        frames: dict[str, pd.DataFrame] = {}
        first_times: list[pd.Timestamp] = []
        for role in roles:
            quotes = _valid_quotes(
                data[
                    data["window_type"].eq(window)
                    & data["leg_role"].eq(role)
                ]
            )
            if quotes.empty:
                break
            frames[role] = quotes
            first_times.append(pd.Timestamp(quotes["timestamp_utc"].iloc[0]))
        if len(frames) != len(roles):
            continue
        execution_time = max(first_times)
        selected: dict[str, pd.Series] = {}
        for role, quotes in frames.items():
            before = quotes[quotes["timestamp_utc"].le(execution_time)]
            quote = before.iloc[-1] if not before.empty else quotes.iloc[0]
            age = abs(
                (pd.Timestamp(quote["timestamp_utc"]) - execution_time).total_seconds()
            )
            if age > float(maximum_quote_age_seconds):
                selected = {}
                break
            selected[role] = quote
        if len(selected) == len(roles):
            return execution_time, selected, window
    return None


def quote_snapshot_features(
    ticks: pd.DataFrame,
    row: pd.Series,
    *,
    maximum_quote_age_seconds: float,
    window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
) -> dict | None:
    synchronized = synchronized_naked_entry_quotes(
        ticks,
        str(row["opportunity_id"]),
        maximum_quote_age_seconds=maximum_quote_age_seconds,
        window_preference=window_preference,
    )
    if synchronized is None:
        return None
    execution_time, quotes, window = synchronized
    call = quotes["short_call"]
    put = quotes["short_put"]
    future = quotes["underlying_future"]
    call_mid = float(call["bid_price1"] + call["ask_price1"]) / 2.0
    put_mid = float(put["bid_price1"] + put["ask_price1"]) / 2.0
    future_mid = float(future["bid_price1"] + future["ask_price1"]) / 2.0
    return {
        "execution_timestamp_utc": execution_time,
        "execution_window": window,
        "call_bid": float(call["bid_price1"]),
        "call_ask": float(call["ask_price1"]),
        "put_bid": float(put["bid_price1"]),
        "put_ask": float(put["ask_price1"]),
        "call_mid": call_mid,
        "put_mid": put_mid,
        "short_entry_premium": float(call["bid_price1"] + put["bid_price1"]),
        "long_entry_premium": float(call["ask_price1"] + put["ask_price1"]),
        "mid_entry_premium": call_mid + put_mid,
        "call_bid_size": int(call["bid_volume1"]),
        "call_ask_size": int(call["ask_volume1"]),
        "put_bid_size": int(put["bid_volume1"]),
        "put_ask_size": int(put["ask_volume1"]),
        "future_bid": float(future["bid_price1"]),
        "future_ask": float(future["ask_price1"]),
        "future_mid": future_mid,
        "future_bid_size": int(future["bid_volume1"]),
        "future_ask_size": int(future["ask_volume1"]),
    }


def expiry_future_price(ticks: pd.DataFrame, opportunity: str) -> float:
    frame = _valid_quotes(
        ticks[
            ticks["opportunity_id"].astype(str).eq(str(opportunity))
            & ticks["window_type"].eq("expiry_close")
            & ticks["leg_role"].eq("underlying_future")
        ]
    )
    if frame.empty:
        return np.nan
    quote = frame.iloc[-1]
    return float(quote["bid_price1"] + quote["ask_price1"]) / 2.0


def simulate_naked_real_quote_trade(
    row: pd.Series,
    side: int,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    settings: RealQuoteExecutionSettings,
    displayed_depth_multiplier: int | None = 1,
    maximum_lots: int | None = None,
    window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
    entry_snapshot: dict | None = None,
    expiry_future_quote: float | None = None,
) -> tuple[dict, pd.DataFrame] | None:
    if side not in (-1, 1):
        return None
    snapshot = entry_snapshot or quote_snapshot_features(
        ticks,
        row,
        maximum_quote_age_seconds=settings.quote_maximum_age_seconds,
        window_preference=window_preference,
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
    entry_premium = float(
        snapshot["short_entry_premium"]
        if side == -1
        else snapshot["long_entry_premium"]
    )
    notional = future_mid * multiplier
    if side == -1:
        per_lot_risk_capital = (
            settings.short_risk_capital_fraction_of_notional * notional
        )
        displayed_size = min(
            int(snapshot["call_bid_size"]), int(snapshot["put_bid_size"])
        )
        crossing_cost_per_lot = (
            float(snapshot["mid_entry_premium"]) - entry_premium
        ) * multiplier
    else:
        per_lot_risk_capital = max(
            entry_premium * multiplier,
            settings.long_minimum_risk_capital_fraction_of_notional * notional,
        )
        displayed_size = min(
            int(snapshot["call_ask_size"]), int(snapshot["put_ask_size"])
        )
        crossing_cost_per_lot = (
            entry_premium - float(snapshot["mid_entry_premium"])
        ) * multiplier
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

    if (
        isinstance(option_daily.index, pd.MultiIndex)
        and list(option_daily.index.names) == ["trade_date", "ts_code"]
    ):
        option_lookup = option_daily
    else:
        option_lookup = option_daily.copy()
        option_lookup["trade_date"] = pd.to_datetime(
            option_lookup["trade_date"]
        ).dt.normalize()
        option_lookup = option_lookup.set_index(["trade_date", "ts_code"])
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
    delta = black76_straddle_delta(
        future_mid,
        float(row["strike_price"]),
        time_to_expiry,
        last_iv,
        settings.risk_free_rate,
    )
    initial_delta = delta
    desired_hedge = int(round(-side * lots * delta))
    initial_hedge = desired_hedge
    if desired_hedge > 0:
        previous_future_price = float(snapshot["future_ask"])
    elif desired_hedge < 0:
        previous_future_price = float(snapshot["future_bid"])
    else:
        previous_future_price = future_mid
    if desired_hedge:
        cumulative_cost += abs(desired_hedge) * settings.futures_commission_per_contract_side
    hedge = desired_hedge
    expiry_tick_price = (
        float(expiry_future_quote)
        if expiry_future_quote is not None
        else expiry_future_price(ticks, str(row["opportunity_id"]))
    )
    daily_rows: list[dict] = []
    for future_record in future_path.itertuples(index=False):
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
        cumulative_option_pnl += side * lots * multiplier * (
            option_mark - last_option_mark
        )
        last_option_mark = option_mark

        if is_expiry:
            if hedge:
                cumulative_cost += abs(hedge) * (
                    settings.futures_commission_per_contract_side
                    + settings.futures_slippage_ticks_per_trade
                    * settings.future_tick
                    * multiplier
                )
            cumulative_cost += (
                lots * 2.0 * settings.option_exercise_fee_per_contract
            )
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
            desired_hedge = int(round(-side * lots * delta))
            hedge_change = desired_hedge - hedge
            if hedge_change:
                cumulative_cost += abs(hedge_change) * (
                    settings.futures_commission_per_contract_side
                    + settings.futures_slippage_ticks_per_trade
                    * settings.future_tick
                    * multiplier
                )
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
    trade = {
        "opportunity_id": row["opportunity_id"],
        "signal_date": row["signal_date"],
        "entry_date": row["entry_date"],
        "expiry_date": row["expiry_date"],
        "horizon": int(row["horizon"]),
        "side": int(side),
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
        "embedded_entry_crossing_cost": lots * max(crossing_cost_per_lot, 0.0),
        "entry_iv": entry_iv,
        "initial_straddle_delta": initial_delta,
        "initial_hedge_contracts": initial_hedge,
        "risk_capital": risk_capital,
        "per_lot_risk_capital": per_lot_risk_capital,
        "gross_option_pnl": float(final["cumulative_option_pnl"]),
        "gross_futures_pnl": float(final["cumulative_futures_pnl"]),
        "transaction_cost": float(final["cumulative_cost"]),
        "net_pnl": float(final["cumulative_net_pnl"]),
        "return_on_risk_capital": float(final["cumulative_net_pnl"] / risk_capital),
    }
    return trade, daily
