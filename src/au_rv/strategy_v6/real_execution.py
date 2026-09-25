from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from au_rv.strategy.black76 import (
    black76_straddle_delta,
    implied_volatility_from_straddle,
)
from au_rv.strategy.volatility_straddle import _calendar_time, _option_mark_and_iv
from au_rv.strategy_v3.execution import RealQuoteExecutionSettings


def build_futures_close_snapshots(
    ticks: pd.DataFrame,
    *,
    target_time: str,
    maximum_age_seconds: float,
) -> pd.DataFrame:
    """Select the last executable top-book quote at or before the close target."""

    frame = ticks.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if "timestamp_shanghai" not in frame:
        frame["timestamp_shanghai"] = frame["timestamp_utc"].dt.tz_convert(
            "Asia/Shanghai"
        )
    else:
        frame["timestamp_shanghai"] = pd.to_datetime(
            frame["timestamp_shanghai"], utc=True
        ).dt.tz_convert("Asia/Shanghai")
    valid = frame[
        frame["bid_price1"].gt(0)
        & frame["ask_price1"].gt(frame["bid_price1"])
        & frame["bid_volume1"].gt(0)
        & frame["ask_volume1"].gt(0)
    ].copy()
    clock = datetime.strptime(target_time, "%H:%M:%S").time()
    rows: list[dict] = []
    for (day, symbol), group in valid.groupby(["trade_date", "ts_code"]):
        target = pd.Timestamp(
            datetime.combine(pd.Timestamp(day).date(), clock),
            tz="Asia/Shanghai",
        )
        eligible = group[group["timestamp_shanghai"].le(target)].sort_values(
            "timestamp_shanghai"
        )
        if eligible.empty:
            continue
        quote = eligible.iloc[-1]
        age = float((target - quote["timestamp_shanghai"]).total_seconds())
        if age < 0 or age > float(maximum_age_seconds):
            continue
        rows.append(
            {
                "trade_date": pd.Timestamp(day).normalize(),
                "ts_code": str(symbol),
                "timestamp_utc": quote["timestamp_utc"],
                "timestamp_shanghai": quote["timestamp_shanghai"],
                "target_timestamp_shanghai": target,
                "quote_age_seconds": age,
                "bid_price1": float(quote["bid_price1"]),
                "ask_price1": float(quote["ask_price1"]),
                "bid_volume1": int(quote["bid_volume1"]),
                "ask_volume1": int(quote["ask_volume1"]),
                "mid_price": float(quote["bid_price1"] + quote["ask_price1"]) / 2.0,
                "price_source": "REAL_TQ_TOPBOOK_AT_OR_BEFORE_1500",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"])


def _order(
    *,
    opportunity_id: str,
    order_date: pd.Timestamp,
    timestamp_utc,
    event: str,
    instrument_type: str,
    symbol: str,
    signed_quantity: int,
    open_close: str,
    effective_price: float,
    bid_price: float,
    ask_price: float,
    bid_volume: int,
    ask_volume: int,
    commission: float,
    embedded_crossing_cost: float,
    price_source: str,
) -> dict:
    return {
        "opportunity_id": str(opportunity_id),
        "order_date": pd.Timestamp(order_date).normalize(),
        "timestamp_utc": timestamp_utc,
        "event": event,
        "instrument_type": instrument_type,
        "symbol": str(symbol),
        "side": "BUY" if signed_quantity > 0 else "SELL",
        "open_close": open_close,
        "signed_quantity": int(signed_quantity),
        "quantity": abs(int(signed_quantity)),
        "effective_price": float(effective_price),
        "bid_price1": float(bid_price),
        "ask_price1": float(ask_price),
        "bid_volume1": int(bid_volume),
        "ask_volume1": int(ask_volume),
        "commission": float(commission),
        "embedded_crossing_cost": float(embedded_crossing_cost),
        "price_source": price_source,
    }


def simulate_real_topbook_trade(
    row: pd.Series,
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    close_snapshot_lookup: pd.DataFrame,
    *,
    settings: RealQuoteExecutionSettings,
    displayed_depth_multiplier: int | None,
    maximum_lots: int | None,
    hedge_interval_trading_days: int,
    delta_change_threshold_contracts: int,
    entry_snapshot: dict,
    enforce_futures_visible_depth: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    """Simulate a short straddle with every futures execution at real bid/ask.

    Futures cash flows are booked at the selected top-book side. Open positions
    are marked to the immediately executable liquidation side (long to bid,
    short to ask), never to a daily close or a hypothetical slipped price.
    """

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
    required_keys = [
        (pd.Timestamp(day).normalize(), str(row["underlying_symbol"]))
        for day in future_path["trade_date"]
    ]
    if any(key not in close_snapshot_lookup.index for key in required_keys):
        return None

    multiplier = float(row["volume_multiple"])
    future_mid = float(entry_snapshot["future_mid"])
    entry_premium = float(entry_snapshot["short_entry_premium"])
    per_lot_risk_capital = (
        settings.short_risk_capital_fraction_of_notional * future_mid * multiplier
    )
    displayed_size = min(
        int(entry_snapshot["call_bid_size"]), int(entry_snapshot["put_bid_size"])
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
            float(entry_snapshot["mid_entry_premium"]),
            future_mid,
            float(row["strike_price"]),
            time_to_expiry,
            settings.risk_free_rate,
        )
    if not np.isfinite(entry_iv):
        return None

    initial_delta = black76_straddle_delta(
        future_mid,
        float(row["strike_price"]),
        time_to_expiry,
        entry_iv,
        settings.risk_free_rate,
    )
    requested_initial_hedge = int(round(lots * initial_delta))
    initial_available = int(
        entry_snapshot["future_ask_size"]
        if requested_initial_hedge > 0
        else entry_snapshot["future_bid_size"]
    )
    initial_hedge = requested_initial_hedge
    if enforce_futures_visible_depth and requested_initial_hedge:
        initial_hedge = int(np.sign(requested_initial_hedge)) * min(
            abs(requested_initial_hedge), initial_available
        )

    explicit_cost = lots * 2.0 * settings.option_commission_per_contract_side
    option_crossing = lots * max(
        float(entry_snapshot["mid_entry_premium"]) - entry_premium, 0.0
    ) * multiplier
    futures_crossing = 0.0
    futures_cash = 0.0
    hedge = 0
    hedge_trade_count = 0
    hedge_contract_turnover = 0
    orders: list[dict] = []
    option_commission = lots * settings.option_commission_per_contract_side
    for symbol, bid, ask, bid_size, ask_size in (
        (
            row["call_symbol"],
            entry_snapshot["call_bid"],
            entry_snapshot["call_ask"],
            entry_snapshot["call_bid_size"],
            entry_snapshot["call_ask_size"],
        ),
        (
            row["put_symbol"],
            entry_snapshot["put_bid"],
            entry_snapshot["put_ask"],
            entry_snapshot["put_bid_size"],
            entry_snapshot["put_ask_size"],
        ),
    ):
        orders.append(
            _order(
                opportunity_id=row["opportunity_id"],
                order_date=row["entry_date"],
                timestamp_utc=entry_snapshot["execution_timestamp_utc"],
                event="OPTION_ENTRY",
                instrument_type="OPTION",
                symbol=str(symbol),
                signed_quantity=-lots,
                open_close="OPEN",
                effective_price=float(bid),
                bid_price=float(bid),
                ask_price=float(ask),
                bid_volume=int(bid_size),
                ask_volume=int(ask_size),
                commission=option_commission,
                embedded_crossing_cost=(float(ask) - float(bid)) / 2.0 * lots * multiplier,
                price_source="REAL_SYNCHRONIZED_OPTION_TOPBOOK_BID_5X_CAPACITY",
            )
        )

    if initial_hedge:
        price = float(
            entry_snapshot["future_ask"]
            if initial_hedge > 0
            else entry_snapshot["future_bid"]
        )
        signed = int(initial_hedge)
        futures_cash -= signed * price * multiplier
        hedge += signed
        commission = abs(signed) * settings.futures_commission_per_contract_side
        explicit_cost += commission
        crossing = (
            abs(signed)
            * abs(price - float(entry_snapshot["future_mid"]))
            * multiplier
        )
        futures_crossing += crossing
        hedge_trade_count += 1
        hedge_contract_turnover += abs(signed)
        orders.append(
            _order(
                opportunity_id=row["opportunity_id"],
                order_date=row["entry_date"],
                timestamp_utc=entry_snapshot["execution_timestamp_utc"],
                event="INITIAL_DELTA_HEDGE",
                instrument_type="FUTURE",
                symbol=str(row["underlying_symbol"]),
                signed_quantity=signed,
                open_close="OPEN",
                effective_price=price,
                bid_price=float(entry_snapshot["future_bid"]),
                ask_price=float(entry_snapshot["future_ask"]),
                bid_volume=int(entry_snapshot["future_bid_size"]),
                ask_volume=int(entry_snapshot["future_ask_size"]),
                commission=commission,
                embedded_crossing_cost=crossing,
                price_source="REAL_SYNCHRONIZED_FUTURES_TOPBOOK_BID_OR_ASK",
            )
        )

    last_option_mark = entry_premium
    last_iv = entry_iv
    cumulative_option_pnl = 0.0
    daily_rows: list[dict] = []
    partial_hedge_order_count = int(initial_hedge != requested_initial_hedge)
    for position, future_record in enumerate(future_path.itertuples(index=False)):
        date = pd.Timestamp(future_record.trade_date).normalize()
        quote = close_snapshot_lookup.loc[(date, str(row["underlying_symbol"]))]
        if isinstance(quote, pd.DataFrame):
            quote = quote.iloc[-1]
        bid = float(quote["bid_price1"])
        ask = float(quote["ask_price1"])
        mid = float(quote["mid_price"])
        is_expiry = date == pd.Timestamp(row["expiry_date"]).normalize()
        if is_expiry:
            option_mark = abs(mid - float(row["strike_price"]))
        else:
            option_mark, last_iv = _option_mark_and_iv(
                date,
                str(row["call_symbol"]),
                str(row["put_symbol"]),
                mid,
                float(row["strike_price"]),
                pd.Timestamp(row["expiry_date"]),
                option_lookup,
                last_iv,
                settings.risk_free_rate,
            )
        cumulative_option_pnl -= lots * multiplier * (option_mark - last_option_mark)
        last_option_mark = option_mark

        requested_change = 0
        filled_change = 0
        execution_price = np.nan
        event = None
        if is_expiry:
            requested_change = -hedge
        else:
            delta = black76_straddle_delta(
                mid,
                float(row["strike_price"]),
                _calendar_time(date, row["expiry_date"]),
                last_iv,
                settings.risk_free_rate,
            )
            if not np.isfinite(delta):
                return None
            desired = int(round(lots * delta))
            candidate = desired - hedge
            scheduled = (position + 1) % max(int(hedge_interval_trading_days), 1) == 0
            large_enough = abs(candidate) >= max(
                int(delta_change_threshold_contracts), 1
            )
            if scheduled and large_enough:
                requested_change = candidate
        if requested_change:
            available = int(quote["ask_volume1"] if requested_change > 0 else quote["bid_volume1"])
            filled_change = requested_change
            if enforce_futures_visible_depth:
                filled_change = int(np.sign(requested_change)) * min(
                    abs(requested_change), available
                )
            if is_expiry and filled_change != requested_change:
                return None
            if filled_change:
                execution_price = ask if filled_change > 0 else bid
                futures_cash -= filled_change * execution_price * multiplier
                hedge += filled_change
                commission = (
                    abs(filled_change) * settings.futures_commission_per_contract_side
                )
                explicit_cost += commission
                crossing = abs(filled_change) * abs(execution_price - mid) * multiplier
                futures_crossing += crossing
                hedge_trade_count += 1
                hedge_contract_turnover += abs(filled_change)
                partial_hedge_order_count += int(filled_change != requested_change)
                event = "EXPIRY_FUTURE_CLOSE" if is_expiry else "DELTA_REHEDGE"
                orders.append(
                    _order(
                        opportunity_id=row["opportunity_id"],
                        order_date=date,
                        timestamp_utc=quote["timestamp_utc"],
                        event=event,
                        instrument_type="FUTURE",
                        symbol=str(row["underlying_symbol"]),
                        signed_quantity=filled_change,
                        open_close="CLOSE" if is_expiry else "ADJUST",
                        effective_price=execution_price,
                        bid_price=bid,
                        ask_price=ask,
                        bid_volume=int(quote["bid_volume1"]),
                        ask_volume=int(quote["ask_volume1"]),
                        commission=commission,
                        embedded_crossing_cost=crossing,
                        price_source=str(quote["price_source"]),
                    )
                )
        liquidation_price = bid if hedge > 0 else ask if hedge < 0 else 0.0
        cumulative_futures_pnl = futures_cash + hedge * liquidation_price * multiplier
        if is_expiry:
            explicit_cost += lots * 2.0 * settings.option_exercise_fee_per_contract
            exercise_fee = lots * settings.option_exercise_fee_per_contract
            call_settlement = max(mid - float(row["strike_price"]), 0.0)
            put_settlement = max(float(row["strike_price"]) - mid, 0.0)
            for symbol, settlement in (
                (row["call_symbol"], call_settlement),
                (row["put_symbol"], put_settlement),
            ):
                orders.append(
                    _order(
                        opportunity_id=row["opportunity_id"],
                        order_date=date,
                        timestamp_utc=quote["timestamp_utc"],
                        event="OPTION_EXPIRY_SETTLEMENT",
                        instrument_type="OPTION",
                        symbol=str(symbol),
                        signed_quantity=lots,
                        open_close="SETTLEMENT",
                        effective_price=settlement,
                        bid_price=settlement,
                        ask_price=settlement,
                        bid_volume=lots,
                        ask_volume=lots,
                        commission=exercise_fee,
                        embedded_crossing_cost=0.0,
                        price_source="EXPIRY_INTRINSIC_FROM_REAL_FUTURES_TOPBOOK_MID",
                    )
                )
        net = cumulative_option_pnl + cumulative_futures_pnl - explicit_cost
        daily_rows.append(
            {
                "trade_date": date,
                "futures_quote_timestamp_utc": quote["timestamp_utc"],
                "futures_quote_age_seconds": float(quote["quote_age_seconds"]),
                "futures_bid_price1": bid,
                "futures_ask_price1": ask,
                "futures_bid_volume1": int(quote["bid_volume1"]),
                "futures_ask_volume1": int(quote["ask_volume1"]),
                "futures_liquidation_price": liquidation_price,
                "futures_price_source": str(quote["price_source"]),
                "requested_hedge_change": int(requested_change),
                "filled_hedge_change": int(filled_change),
                "hedge_execution_price": execution_price,
                "daily_hedge_contracts": int(hedge),
                "cumulative_option_pnl": cumulative_option_pnl,
                "cumulative_futures_pnl": cumulative_futures_pnl,
                "cumulative_explicit_cost": explicit_cost,
                "cumulative_embedded_futures_crossing_cost": futures_crossing,
                "cumulative_net_pnl": net,
                "option_mark": option_mark,
                "mark_iv": last_iv,
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty or hedge != 0:
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
        "execution_timestamp_utc": entry_snapshot["execution_timestamp_utc"],
        "execution_window": entry_snapshot["execution_window"],
        "entry_future_mid": future_mid,
        "entry_future_bid_ask_spread": float(
            entry_snapshot["future_ask"] - entry_snapshot["future_bid"]
        ),
        "entry_straddle_premium": entry_premium,
        "entry_mid_straddle_premium": float(entry_snapshot["mid_entry_premium"]),
        "entry_straddle_bid_ask_spread": float(
            entry_snapshot["long_entry_premium"] - entry_snapshot["short_entry_premium"]
        ),
        "entry_relative_spread": float(
            (entry_snapshot["long_entry_premium"] - entry_snapshot["short_entry_premium"])
            / max(entry_snapshot["mid_entry_premium"], 1.0e-8)
        ),
        "embedded_entry_crossing_cost": option_crossing,
        "embedded_futures_crossing_cost": futures_crossing,
        "entry_iv": entry_iv,
        "initial_straddle_delta": initial_delta,
        "requested_initial_hedge_contracts": requested_initial_hedge,
        "initial_hedge_contracts": initial_hedge,
        "partial_hedge_order_count": partial_hedge_order_count,
        "hedge_interval_trading_days": int(hedge_interval_trading_days),
        "delta_change_threshold_contracts": int(delta_change_threshold_contracts),
        "hedge_trade_count": hedge_trade_count,
        "hedge_contract_turnover": hedge_contract_turnover,
        "risk_capital": risk_capital,
        "per_lot_risk_capital": per_lot_risk_capital,
        "gross_option_pnl": float(final["cumulative_option_pnl"]),
        "gross_futures_pnl": float(final["cumulative_futures_pnl"]),
        "transaction_cost": float(final["cumulative_explicit_cost"]),
        "all_in_observed_friction": float(
            final["cumulative_explicit_cost"] + option_crossing + futures_crossing
        ),
        "net_pnl": float(final["cumulative_net_pnl"]),
        "return_on_risk_capital": float(final["cumulative_net_pnl"] / risk_capital),
        "all_futures_executions_real_topbook": True,
        "futures_visible_depth_enforced": bool(enforce_futures_visible_depth),
    }
    return trade, daily, pd.DataFrame(orders)
