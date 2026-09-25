from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from au_rv.strategy_v3.execution import RealQuoteExecutionSettings
from au_rv.strategy_v5.backtest import VariantSpec
from au_rv.strategy_v5.execution import simulate_v5_trade


def _order_row(
    *,
    strategy: str,
    trade_id: str,
    opportunity_id: str,
    order_sequence: int,
    order_date,
    timestamp_utc,
    event: str,
    instrument_type: str,
    symbol: str,
    signed_quantity: int,
    open_close: str,
    reference_price: float,
    effective_price: float,
    price_source: str,
    commission: float,
    slippage_cost: float,
    horizon: int,
) -> dict:
    return {
        "strategy": strategy,
        "trade_id": trade_id,
        "opportunity_id": opportunity_id,
        "order_sequence": order_sequence,
        "order_date": pd.Timestamp(order_date).normalize(),
        "timestamp_utc": timestamp_utc,
        "event": event,
        "instrument_type": instrument_type,
        "symbol": symbol,
        "side": "BUY" if signed_quantity > 0 else "SELL",
        "open_close": open_close,
        "signed_quantity": int(signed_quantity),
        "quantity": abs(int(signed_quantity)),
        "reference_price": float(reference_price),
        "effective_price": float(effective_price),
        "price_source": price_source,
        "commission": float(commission),
        "slippage_cost": float(slippage_cost),
        "modeled_order_cost": float(commission + slippage_cost),
        "horizon": int(horizon),
    }


def build_order_ledger(
    trades: pd.DataFrame,
    opportunities: pd.DataFrame,
    specs: dict[str, VariantSpec],
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    quote_cache: dict[str, dict | None],
    expiry_cache: dict[str, float],
    *,
    base_settings: RealQuoteExecutionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opportunity_lookup = opportunities.set_index("opportunity_id")
    rows: list[dict] = []
    checks: list[dict] = []
    for trade in trades.itertuples(index=False):
        row = opportunity_lookup.loc[str(trade.opportunity_id)].copy()
        row["opportunity_id"] = str(trade.opportunity_id)
        spec = specs[str(trade.strategy)]
        snapshot = quote_cache[str(trade.opportunity_id)]
        if snapshot is None:
            raise RuntimeError(f"Missing entry snapshot for {trade.opportunity_id}.")
        effective_budget = float(trade.effective_risk_budget_fraction)
        simulated = simulate_v5_trade(
            row,
            option_lookup,
            futures_daily,
            ticks,
            settings=replace(
                base_settings,
                risk_budget_fraction_per_trade=effective_budget,
            ),
            displayed_depth_multiplier=spec.depth_multiplier,
            maximum_lots=int(trade.lots),
            hedge_interval_trading_days=spec.hedge_interval,
            delta_change_threshold_contracts=spec.delta_threshold,
            entry_snapshot=snapshot,
            expiry_future_quote=(
                expiry_cache[str(trade.opportunity_id)]
                if np.isfinite(expiry_cache[str(trade.opportunity_id)])
                else None
            ),
        )
        if simulated is None:
            raise RuntimeError(f"Could not reconstruct orders for {trade.trade_id}.")
        reconstructed, daily = simulated
        if int(reconstructed["lots"]) != int(trade.lots):
            raise RuntimeError(f"Lot mismatch while reconstructing {trade.trade_id}.")
        sequence = 1
        option_commission = (
            int(trade.lots) * base_settings.option_commission_per_contract_side
        )
        for symbol, price in (
            (str(trade.call_symbol), float(snapshot["call_bid"])),
            (str(trade.put_symbol), float(snapshot["put_bid"])),
        ):
            rows.append(
                _order_row(
                    strategy=str(trade.strategy),
                    trade_id=str(trade.trade_id),
                    opportunity_id=str(trade.opportunity_id),
                    order_sequence=sequence,
                    order_date=trade.entry_date,
                    timestamp_utc=snapshot["execution_timestamp_utc"],
                    event="OPTION_ENTRY",
                    instrument_type="OPTION",
                    symbol=symbol,
                    signed_quantity=-int(trade.lots),
                    open_close="OPEN",
                    reference_price=price,
                    effective_price=price,
                    price_source="REAL_SYNCHRONIZED_TOPBOOK_BID",
                    commission=option_commission,
                    slippage_cost=0.0,
                    horizon=int(trade.horizon),
                )
            )
            sequence += 1
        previous_hedge = int(trade.initial_hedge_contracts)
        if previous_hedge != 0:
            initial_price = float(
                snapshot["future_ask"]
                if previous_hedge > 0
                else snapshot["future_bid"]
            )
            rows.append(
                _order_row(
                    strategy=str(trade.strategy),
                    trade_id=str(trade.trade_id),
                    opportunity_id=str(trade.opportunity_id),
                    order_sequence=sequence,
                    order_date=trade.entry_date,
                    timestamp_utc=snapshot["execution_timestamp_utc"],
                    event="INITIAL_DELTA_HEDGE",
                    instrument_type="FUTURE",
                    symbol=str(trade.underlying_symbol),
                    signed_quantity=previous_hedge,
                    open_close="OPEN",
                    reference_price=initial_price,
                    effective_price=initial_price,
                    price_source="REAL_SYNCHRONIZED_TOPBOOK_ASK_OR_BID",
                    commission=(
                        abs(previous_hedge)
                        * base_settings.futures_commission_per_contract_side
                    ),
                    slippage_cost=0.0,
                    horizon=int(trade.horizon),
                )
            )
            sequence += 1
        future_path = futures_daily[
            futures_daily["ts_code"].astype(str).eq(str(trade.underlying_symbol))
            & futures_daily["trade_date"].between(trade.entry_date, trade.expiry_date)
        ].copy()
        close_lookup = future_path.set_index("trade_date")["close"]
        expiry_quote = expiry_cache[str(trade.opportunity_id)]
        for record in daily.itertuples(index=False):
            date = pd.Timestamp(record.trade_date).normalize()
            current_hedge = int(record.daily_hedge_contracts)
            change = current_hedge - previous_hedge
            if change != 0:
                is_expiry = date == pd.Timestamp(trade.expiry_date).normalize()
                reference = float(expiry_quote if is_expiry else close_lookup.loc[date])
                slippage = (
                    abs(change)
                    * base_settings.futures_slippage_ticks_per_trade
                    * base_settings.future_tick
                    * float(trade.volume_multiple)
                )
                effective = reference + np.sign(change) * (
                    base_settings.futures_slippage_ticks_per_trade
                    * base_settings.future_tick
                )
                rows.append(
                    _order_row(
                        strategy=str(trade.strategy),
                        trade_id=str(trade.trade_id),
                        opportunity_id=str(trade.opportunity_id),
                        order_sequence=sequence,
                        order_date=date,
                        timestamp_utc=pd.NaT,
                        event="EXPIRY_FUTURE_CLOSE" if is_expiry else "DELTA_REHEDGE",
                        instrument_type="FUTURE",
                        symbol=str(trade.underlying_symbol),
                        signed_quantity=change,
                        open_close="CLOSE" if is_expiry else "ADJUST",
                        reference_price=reference,
                        effective_price=effective,
                        price_source=(
                            "REAL_EXPIRY_TOPBOOK_MID_PLUS_1_TICK"
                            if is_expiry
                            else "DAILY_CLOSE_PLUS_1_TICK_MODELED"
                        ),
                        commission=(
                            abs(change)
                            * base_settings.futures_commission_per_contract_side
                        ),
                        slippage_cost=slippage,
                        horizon=int(trade.horizon),
                    )
                )
                sequence += 1
            previous_hedge = current_hedge
        settlement_future = float(expiry_quote)
        call_settlement = max(
            settlement_future - float(trade.strike_price), 0.0
        )
        put_settlement = max(
            float(trade.strike_price) - settlement_future, 0.0
        )
        for symbol, price in (
            (str(trade.call_symbol), call_settlement),
            (str(trade.put_symbol), put_settlement),
        ):
            rows.append(
                _order_row(
                    strategy=str(trade.strategy),
                    trade_id=str(trade.trade_id),
                    opportunity_id=str(trade.opportunity_id),
                    order_sequence=sequence,
                    order_date=trade.expiry_date,
                    timestamp_utc=pd.NaT,
                    event="OPTION_EXPIRY_SETTLEMENT",
                    instrument_type="OPTION",
                    symbol=symbol,
                    signed_quantity=int(trade.lots),
                    open_close="SETTLEMENT",
                    reference_price=price,
                    effective_price=price,
                    price_source="EXPIRY_INTRINSIC_VALUE",
                    commission=(
                        int(trade.lots) * base_settings.option_exercise_fee_per_contract
                    ),
                    slippage_cost=0.0,
                    horizon=int(trade.horizon),
                )
            )
            sequence += 1
        trade_orders = [item for item in rows if item["trade_id"] == trade.trade_id]
        modeled_cost = float(sum(item["modeled_order_cost"] for item in trade_orders))
        checks.append(
            {
                "strategy": trade.strategy,
                "trade_id": trade.trade_id,
                "reported_transaction_cost": float(trade.transaction_cost),
                "order_ledger_transaction_cost": modeled_cost,
                "cost_reconciliation_error": modeled_cost
                - float(trade.transaction_cost),
                "order_count": len(trade_orders),
            }
        )
    orders = pd.DataFrame(rows).sort_values(
        ["strategy", "order_date", "trade_id", "order_sequence"]
    )
    return orders.reset_index(drop=True), pd.DataFrame(checks)
