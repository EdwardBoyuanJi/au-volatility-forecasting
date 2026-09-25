from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from au_rv.strategy.black76 import implied_volatility_from_straddle
from au_rv.strategy_v2.pricing import iron_fly_delta, iron_fly_value


@dataclass(frozen=True)
class V2RiskSettings:
    initial_capital: float
    max_loss_fraction_per_trade: float
    max_margin_fraction_per_expiry: float
    max_portfolio_margin_fraction: float
    delta_band_per_option: float
    risk_free_rate: float
    option_exchange_fee: float
    option_broker_fee: float
    option_exercise_exchange_fee: float
    futures_broker_fee: float
    futures_hedge_slippage_ticks: float
    future_tick: float = 0.02


def _calendar_time(start, expiry, *, at_open: bool = False) -> float:
    days = (pd.Timestamp(expiry).normalize() - pd.Timestamp(start).normalize()).days
    fraction = 0.75 if at_open else 0.25
    return max((days + fraction) / 365.0, 1.0 / (365.0 * 24.0))


def _valid_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["bid_price1"].gt(0)
        & frame["ask_price1"].gt(frame["bid_price1"])
        & frame["bid_volume1"].gt(0)
        & frame["ask_volume1"].gt(0)
    ].sort_values("timestamp_utc")


def synchronized_entry_quotes(
    ticks: pd.DataFrame,
    opportunity: str,
    *,
    maximum_quote_age_seconds: float = 10.0,
    window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
) -> tuple[pd.Timestamp, dict[str, pd.Series], str] | None:
    roles = ["short_call", "short_put", "long_call_wing", "long_put_wing", "underlying_future"]
    data = ticks[ticks["opportunity_id"].eq(opportunity)].copy()
    for window in window_preference:
        by_role: dict[str, pd.DataFrame] = {}
        first_times = []
        for role in roles:
            quotes = _valid_quotes(
                data[data["window_type"].eq(window) & data["leg_role"].eq(role)]
            )
            if quotes.empty:
                break
            by_role[role] = quotes
            first_times.append(pd.Timestamp(quotes["timestamp_utc"].iloc[0]))
        if len(by_role) != len(roles):
            continue
        execution_time = max(first_times)
        selected: dict[str, pd.Series] = {}
        valid = True
        for role, quotes in by_role.items():
            before = quotes[quotes["timestamp_utc"].le(execution_time)]
            if before.empty:
                after = quotes[quotes["timestamp_utc"].gt(execution_time)]
                if after.empty:
                    valid = False
                    break
                quote = after.iloc[0]
            else:
                quote = before.iloc[-1]
            age = abs((pd.Timestamp(quote["timestamp_utc"]) - execution_time).total_seconds())
            if age > float(maximum_quote_age_seconds):
                valid = False
                break
            selected[role] = quote
        if valid:
            return execution_time, selected, window
    return None


def signal_close_mid_iv(
    ticks: pd.DataFrame,
    row: pd.Series,
    *,
    risk_free_rate: float,
) -> float:
    data = ticks[
        ticks["opportunity_id"].eq(row["opportunity_id"])
        & ticks["window_type"].eq("signal_close")
    ]
    mids = []
    for role in ("short_call", "short_put"):
        quotes = _valid_quotes(data[data["leg_role"].eq(role)])
        if quotes.empty:
            return np.nan
        quote = quotes.iloc[-1]
        mids.append(float(quote["bid_price1"] + quote["ask_price1"]) / 2.0)
    return implied_volatility_from_straddle(
        sum(mids),
        float(row["signal_future_close"]),
        float(row["strike_price"]),
        _calendar_time(row["signal_date"], row["expiry_date"]),
        risk_free_rate,
    )


def futures_exchange_fee(date: pd.Timestamp, symbol: str) -> float:
    match = re.search(r"au(\d{2})(\d{2})$", str(symbol), flags=re.IGNORECASE)
    if not match:
        return 10.0
    delivery_year = 2000 + int(match.group(1))
    delivery_month = int(match.group(2))
    date = pd.Timestamp(date).normalize()
    if str(symbol).upper() == "SHFE.AU2506" and date >= pd.Timestamp("2025-04-25"):
        return 20.0
    months_to_delivery = (delivery_year - date.year) * 12 + delivery_month - date.month
    if delivery_month in (6, 12) or months_to_delivery <= 2:
        return 10.0
    return 2.0


def published_margin_rate(date: pd.Timestamp, symbol: str) -> float:
    """Published general-position rate, with known contract/lifecycle floors.

    These are exchange rates, not a reconstruction of an individual broker's
    customer add-on. The effective date is the settlement date named by SHFE.
    """

    date = pd.Timestamp(date).normalize()
    symbol_upper = str(symbol).upper()
    if date < pd.Timestamp("2024-04-17"):
        base = 0.04
    elif date < pd.Timestamp("2024-05-23"):
        base = 0.10
    elif date < pd.Timestamp("2025-04-10"):
        base = 0.12
    elif date < pd.Timestamp("2025-10-21"):
        base = 0.13
    else:
        base = 0.16
    if pd.Timestamp("2025-05-29") <= date < pd.Timestamp("2025-06-03"):
        base = max(base, 0.15)
    if date >= pd.Timestamp("2026-01-22"):
        if symbol_upper in {"SHFE.AU2602", "SHFE.AU2603", "SHFE.AU2604"}:
            base = max(base, 0.18)
        elif symbol_upper in {
            "SHFE.AU2606",
            "SHFE.AU2608",
            "SHFE.AU2610",
            "SHFE.AU2612",
            "SHFE.AU2702",
        }:
            base = max(base, 0.17)
    if date >= pd.Timestamp("2026-06-11") and symbol_upper == "SHFE.AU2609":
        base = max(base, 0.19)
    match = re.search(r"au(\d{2})(\d{2})$", str(symbol), flags=re.IGNORECASE)
    if not match:
        return base
    delivery_year = 2000 + int(match.group(1))
    delivery_month = int(match.group(2))
    months_to_delivery = (delivery_year - date.year) * 12 + delivery_month - date.month
    if months_to_delivery == 0:
        base = max(base, 0.15)
    elif months_to_delivery == 1:
        base = max(base, 0.10)
    return base


def short_option_margin(
    option_price: float,
    future: float,
    strike: float,
    option_type: str,
    multiplier: float,
    futures_margin_rate: float,
) -> float:
    futures_margin = future * multiplier * futures_margin_rate
    out_of_money = (
        max(strike - future, 0.0) * multiplier
        if option_type == "call"
        else max(future - strike, 0.0) * multiplier
    )
    premium = option_price * multiplier
    return float(
        premium
        + max(futures_margin - 0.5 * out_of_money, 0.5 * futures_margin)
    )


def _daily_option_price(
    lookup: pd.DataFrame,
    date: pd.Timestamp,
    symbol: str,
) -> float:
    key = (pd.Timestamp(date).normalize(), str(symbol))
    if key not in lookup.index:
        return np.nan
    value = lookup.loc[key]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[-1]
    return float(value["close"])


def simulate_protected_trade(
    row: pd.Series,
    option_daily: pd.DataFrame,
    wing_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    settings: V2RiskSettings,
    maximum_lots: int | None = None,
    entry_window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
) -> tuple[dict, pd.DataFrame] | None:
    synchronized = synchronized_entry_quotes(
        ticks,
        str(row["opportunity_id"]),
        window_preference=entry_window_preference,
    )
    if synchronized is None:
        return None
    execution_time, quotes, execution_window = synchronized
    entry_credit = (
        float(quotes["short_call"]["bid_price1"])
        + float(quotes["short_put"]["bid_price1"])
        - float(quotes["long_call_wing"]["ask_price1"])
        - float(quotes["long_put_wing"]["ask_price1"])
    )
    if entry_credit <= 0:
        return None
    entry_mid_credit = (
        float(quotes["short_call"]["bid_price1"] + quotes["short_call"]["ask_price1"])
        / 2.0
        + float(quotes["short_put"]["bid_price1"] + quotes["short_put"]["ask_price1"])
        / 2.0
        - float(
            quotes["long_call_wing"]["bid_price1"]
            + quotes["long_call_wing"]["ask_price1"]
        )
        / 2.0
        - float(
            quotes["long_put_wing"]["bid_price1"]
            + quotes["long_put_wing"]["ask_price1"]
        )
        / 2.0
    )
    future_mid = float(
        quotes["underlying_future"]["bid_price1"]
        + quotes["underlying_future"]["ask_price1"]
    ) / 2.0
    multiplier = float(row["volume_multiple"])
    maximum_width = max(float(row["put_wing_width"]), float(row["call_wing_width"]))
    maximum_loss_per_lot = max(maximum_width - entry_credit, 0.0) * multiplier
    if maximum_loss_per_lot <= 0:
        return None
    rate = published_margin_rate(pd.Timestamp(row["entry_date"]), str(row["underlying_symbol"]))
    margin_per_lot = short_option_margin(
        float(quotes["short_call"]["bid_price1"]),
        future_mid,
        float(row["strike_price"]),
        "call",
        multiplier,
        rate,
    ) + short_option_margin(
        float(quotes["short_put"]["bid_price1"]),
        future_mid,
        float(row["strike_price"]),
        "put",
        multiplier,
        rate,
    )
    lots_by_loss = int(
        math.floor(
            settings.initial_capital
            * settings.max_loss_fraction_per_trade
            / maximum_loss_per_lot
        )
    )
    lots_by_margin = int(
        math.floor(
            settings.initial_capital
            * settings.max_margin_fraction_per_expiry
            / margin_per_lot
        )
    )
    displayed_size = int(
        min(
            quotes[role]["bid_volume1"] if role.startswith("short") else quotes[role]["ask_volume1"]
            for role in ("short_call", "short_put", "long_call_wing", "long_put_wing")
        )
    )
    caps = [lots_by_loss, lots_by_margin, displayed_size]
    if maximum_lots is not None:
        caps.append(int(maximum_lots))
    lots = min(caps)
    if lots < 1:
        return None

    future_path = futures_daily[
        futures_daily["ts_code"].astype(str).eq(str(row["underlying_symbol"]))
        & futures_daily["trade_date"].between(row["entry_date"], row["expiry_date"])
    ].sort_values("trade_date")
    if future_path.empty:
        return None
    if pd.Timestamp(future_path.iloc[0]["trade_date"]).normalize() != pd.Timestamp(row["entry_date"]).normalize():
        return None
    if pd.Timestamp(future_path.iloc[-1]["trade_date"]).normalize() != pd.Timestamp(row["expiry_date"]).normalize():
        return None

    combined_options = pd.concat([option_daily, wing_daily], ignore_index=True)
    combined_options["trade_date"] = pd.to_datetime(combined_options["trade_date"]).dt.normalize()
    option_lookup = combined_options.set_index(["trade_date", "ts_code"])
    entry_iv = implied_volatility_from_straddle(
        float(quotes["short_call"]["bid_price1"] + quotes["short_put"]["bid_price1"]),
        future_mid,
        float(row["strike_price"]),
        _calendar_time(row["entry_date"], row["expiry_date"], at_open=True),
        settings.risk_free_rate,
    )
    if not np.isfinite(entry_iv):
        entry_iv = float(row["entry_iv"])
    option_fee = settings.option_exchange_fee + settings.option_broker_fee
    cumulative_cost = lots * 4.0 * option_fee
    cumulative_option_pnl = 0.0
    cumulative_futures_pnl = 0.0
    previous_future = future_mid
    previous_value = entry_credit
    last_iv = entry_iv
    hedge = 0
    initial_delta = iron_fly_delta(
        future_mid,
        float(row["strike_price"]),
        float(row["put_wing_strike"]),
        float(row["call_wing_strike"]),
        _calendar_time(row["entry_date"], row["expiry_date"], at_open=True),
        entry_iv,
        settings.risk_free_rate,
    )
    desired_initial_hedge = int(round(-lots * initial_delta))
    if desired_initial_hedge > 0:
        hedge_price = float(quotes["underlying_future"]["ask_price1"])
    elif desired_initial_hedge < 0:
        hedge_price = float(quotes["underlying_future"]["bid_price1"])
    else:
        hedge_price = future_mid
    if desired_initial_hedge:
        cumulative_cost += abs(desired_initial_hedge) * (
            futures_exchange_fee(pd.Timestamp(row["entry_date"]), str(row["underlying_symbol"]))
            + settings.futures_broker_fee
        )
    hedge = desired_initial_hedge
    previous_future = hedge_price
    daily_rows: list[dict] = []
    maximum_margin = 0.0
    for future_record in future_path.itertuples(index=False):
        date = pd.Timestamp(future_record.trade_date).normalize()
        future_close = float(future_record.close)
        expiry = date == pd.Timestamp(row["expiry_date"]).normalize()
        if expiry:
            expiry_ticks = ticks[
                ticks["opportunity_id"].eq(row["opportunity_id"])
                & ticks["window_type"].eq("expiry_close")
                & ticks["leg_role"].eq("underlying_future")
            ]
            valid_expiry = _valid_quotes(expiry_ticks)
            if not valid_expiry.empty:
                last = valid_expiry.iloc[-1]
                future_close = float(last["bid_price1"] + last["ask_price1"]) / 2.0
            value = (
                abs(future_close - float(row["strike_price"]))
                - max(future_close - float(row["call_wing_strike"]), 0.0)
                - max(float(row["put_wing_strike"]) - future_close, 0.0)
            )
        else:
            call = _daily_option_price(option_lookup, date, str(row["call_symbol"]))
            put = _daily_option_price(option_lookup, date, str(row["put_symbol"]))
            wing_call = _daily_option_price(option_lookup, date, str(row["call_wing_symbol"]))
            wing_put = _daily_option_price(option_lookup, date, str(row["put_wing_symbol"]))
            observed = call + put - wing_call - wing_put
            observed_iv = implied_volatility_from_straddle(
                call + put,
                future_close,
                float(row["strike_price"]),
                _calendar_time(date, row["expiry_date"]),
                settings.risk_free_rate,
            )
            if np.isfinite(observed_iv):
                last_iv = observed_iv
            value = (
                observed
                if all(np.isfinite([call, put, wing_call, wing_put]))
                else iron_fly_value(
                    future_close,
                    float(row["strike_price"]),
                    float(row["put_wing_strike"]),
                    float(row["call_wing_strike"]),
                    _calendar_time(date, row["expiry_date"]),
                    last_iv,
                    settings.risk_free_rate,
                )
            )
        cumulative_futures_pnl += hedge * multiplier * (future_close - previous_future)
        cumulative_option_pnl += lots * multiplier * (previous_value - value)
        previous_value = value

        if expiry:
            if hedge:
                cumulative_cost += abs(hedge) * (
                    futures_exchange_fee(date, str(row["underlying_symbol"]))
                    + settings.futures_broker_fee
                    + settings.futures_hedge_slippage_ticks * settings.future_tick * multiplier
                )
            exercised_legs = int(future_close > float(row["strike_price"])) + int(
                future_close < float(row["strike_price"])
            )
            exercised_legs += int(future_close > float(row["call_wing_strike"])) + int(
                future_close < float(row["put_wing_strike"])
            )
            cumulative_cost += lots * exercised_legs * (
                settings.option_exercise_exchange_fee + settings.option_broker_fee
            )
            hedge = 0
        else:
            delta = iron_fly_delta(
                future_close,
                float(row["strike_price"]),
                float(row["put_wing_strike"]),
                float(row["call_wing_strike"]),
                _calendar_time(date, row["expiry_date"]),
                last_iv,
                settings.risk_free_rate,
            )
            desired = int(round(-lots * delta))
            minimum_change = max(1, int(math.ceil(lots * settings.delta_band_per_option)))
            change = desired - hedge
            if abs(change) >= minimum_change:
                cumulative_cost += abs(change) * (
                    futures_exchange_fee(date, str(row["underlying_symbol"]))
                    + settings.futures_broker_fee
                    + settings.futures_hedge_slippage_ticks * settings.future_tick * multiplier
                )
                hedge = desired
        if expiry:
            daily_margin = 0.0
        else:
            rate = published_margin_rate(date, str(row["underlying_symbol"]))
            call_margin_price = _daily_option_price(
                option_lookup, date, str(row["call_symbol"])
            )
            put_margin_price = _daily_option_price(
                option_lookup, date, str(row["put_symbol"])
            )
            if not np.isfinite(call_margin_price):
                call_margin_price = max(value / 2.0, 0.0)
            if not np.isfinite(put_margin_price):
                put_margin_price = max(value / 2.0, 0.0)
            call_margin_price = max(call_margin_price, 0.0)
            put_margin_price = max(put_margin_price, 0.0)
            daily_margin = lots * (
                short_option_margin(
                    call_margin_price,
                    future_close,
                    float(row["strike_price"]),
                    "call",
                    multiplier,
                    rate,
                )
                + short_option_margin(
                    put_margin_price,
                    future_close,
                    float(row["strike_price"]),
                    "put",
                    multiplier,
                    rate,
                )
            ) + abs(hedge) * future_close * multiplier * rate
        maximum_margin = max(maximum_margin, daily_margin)
        net = cumulative_option_pnl + cumulative_futures_pnl - cumulative_cost
        daily_rows.append(
            {
                "trade_date": date,
                "cumulative_option_pnl": cumulative_option_pnl,
                "cumulative_futures_pnl": cumulative_futures_pnl,
                "cumulative_cost": cumulative_cost,
                "cumulative_net_pnl": net,
                "hedge_contracts": hedge,
                "position_margin": daily_margin,
                "iron_fly_mark": value,
                "mark_iv": last_iv,
            }
        )
        previous_future = future_close
    daily = pd.DataFrame(daily_rows)
    daily["daily_net_pnl"] = daily["cumulative_net_pnl"].diff().fillna(daily["cumulative_net_pnl"])
    final = daily.iloc[-1]
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
        "call_wing_symbol": row["call_wing_symbol"],
        "put_wing_symbol": row["put_wing_symbol"],
        "strike_price": float(row["strike_price"]),
        "call_wing_strike": float(row["call_wing_strike"]),
        "put_wing_strike": float(row["put_wing_strike"]),
        "volume_multiple": multiplier,
        "lots": lots,
        "displayed_size_cap": displayed_size,
        "execution_timestamp_utc": execution_time,
        "execution_window": execution_window,
        "entry_future_mid": future_mid,
        "entry_net_credit": entry_credit,
        "entry_mid_credit": entry_mid_credit,
        "embedded_entry_crossing_cost": lots
        * multiplier
        * max(entry_mid_credit - entry_credit, 0.0),
        "entry_iv": entry_iv,
        "initial_iron_fly_delta": initial_delta,
        "initial_hedge_contracts": desired_initial_hedge,
        "maximum_loss_per_lot": maximum_loss_per_lot,
        "risk_capital": lots * maximum_loss_per_lot,
        "initial_margin_per_lot": margin_per_lot,
        "maximum_position_margin": maximum_margin,
        "gross_option_pnl": float(final["cumulative_option_pnl"]),
        "gross_futures_pnl": float(final["cumulative_futures_pnl"]),
        "transaction_cost": float(final["cumulative_cost"]),
        "net_pnl": float(final["cumulative_net_pnl"]),
        "return_on_maximum_loss": float(final["cumulative_net_pnl"] / (lots * maximum_loss_per_lot)),
        "return_on_risk_capital": float(final["cumulative_net_pnl"] / (lots * maximum_loss_per_lot)),
        "predicted_normalized_pnl": float(row["predicted_normalized_pnl"]),
        "predicted_normalized_pnl_lower": float(row["predicted_normalized_pnl_lower"]),
        "forecast_tail_imbalance": float(row["forecast_tail_imbalance"]),
        "forecast_jump_share": float(row["forecast_jump_share"]),
        "signal_tick_iv": float(row["signal_tick_iv"]),
    }
    return trade, daily


def protected_tail_stress(
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    risk_free_rate: float = 0.018,
) -> pd.DataFrame:
    rows = []
    for trade in trades.itertuples(index=False):
        defined_loss = -float(trade.maximum_loss_per_lot) * int(trade.lots)
        time_to_expiry = _calendar_time(trade.entry_date, trade.expiry_date, at_open=True)
        for underlying_shock in (-0.15, -0.08, 0.08, 0.15):
            for volatility_multiplier in (1.0, 1.75):
                shocked_future = float(trade.entry_future_mid) * (1.0 + underlying_shock)
                shocked_value = iron_fly_value(
                    shocked_future,
                    float(trade.strike_price),
                    float(trade.put_wing_strike),
                    float(trade.call_wing_strike),
                    max(time_to_expiry - 1.0 / 365.0, 1.0 / (365.0 * 24.0)),
                    float(trade.entry_iv) * volatility_multiplier,
                    risk_free_rate,
                )
                multiplier = float(trade.volume_multiple)
                option_pnl = int(trade.lots) * multiplier * (
                    float(trade.entry_net_credit) - shocked_value
                )
                hedge_pnl = (
                    int(trade.initial_hedge_contracts)
                    * multiplier
                    * (shocked_future - float(trade.entry_future_mid))
                )
                total = option_pnl + hedge_pnl
                rows.append(
                    {
                        "opportunity_id": trade.opportunity_id,
                        "entry_date": trade.entry_date,
                        "horizon": int(trade.horizon),
                        "underlying_shock": underlying_shock,
                        "volatility_multiplier": volatility_multiplier,
                        "instant_option_pnl": option_pnl,
                        "instant_initial_hedge_pnl": hedge_pnl,
                        "instant_total_pnl_before_exit_cost": total,
                        "instant_total_pnl_fraction_of_capital": total / initial_capital,
                        "defined_option_loss_floor": defined_loss,
                        "defined_option_loss_fraction_of_capital": defined_loss / initial_capital,
                        "maximum_position_margin": float(trade.maximum_position_margin),
                        "maximum_margin_fraction_of_capital": float(trade.maximum_position_margin) / initial_capital,
                    }
                )
    return pd.DataFrame(rows)
