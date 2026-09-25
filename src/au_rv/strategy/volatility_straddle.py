from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.config import resolve_path
from au_rv.io import read_parquet, write_csv, write_parquet
from au_rv.strategy.black76 import (
    black76_straddle_delta,
    black76_straddle_price,
    implied_volatility_from_straddle,
)


MAIN_MODEL = "har__mse_log__macro+gvz+slv_iv+us_epu"
ROBUST_MODEL = "har__qlike__macro+us_epu"
PERSISTENCE_MODEL = "persistence__raw__none"


@dataclass(frozen=True)
class CostAssumptions:
    option_commission_per_contract_side: float
    option_exercise_fee_per_contract: float
    futures_commission_per_contract_side: float
    option_slippage_ticks_per_side: float
    futures_slippage_ticks_per_trade: float


def _calendar_time(start, expiry, *, at_open: bool = False) -> float:
    days = (pd.Timestamp(expiry).normalize() - pd.Timestamp(start).normalize()).days
    fraction = 0.75 if at_open else 0.25
    return max((days + fraction) / 365.0, 1.0 / (365.0 * 24.0))


def _price_lookup(option_daily: pd.DataFrame, price_column: str) -> pd.Series:
    frame = option_daily[["trade_date", "ts_code", price_column]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame.set_index(["trade_date", "ts_code"])[price_column]


def build_option_opportunities(
    pairs: pd.DataFrame,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    features: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    risk_free_rate: float,
    minimum_vrp_history: int,
) -> pd.DataFrame:
    result = pairs.copy()
    for column in ("signal_date", "entry_date", "expiry_date"):
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    futures = futures_daily.copy()
    futures["trade_date"] = pd.to_datetime(futures["trade_date"]).dt.normalize()
    future_lookup = futures.set_index(["trade_date", "ts_code"])
    option_lookup = option_daily.copy()
    option_lookup["trade_date"] = pd.to_datetime(option_lookup["trade_date"]).dt.normalize()
    option_lookup = option_lookup.set_index(["trade_date", "ts_code"])

    signal_values = []
    for row in result.itertuples(index=False):
        values: dict[str, float] = {}
        for leg, symbol in (("call", row.call_symbol), ("put", row.put_symbol)):
            key_signal = (row.signal_date, symbol)
            key_entry = (row.entry_date, symbol)
            if key_signal in option_lookup.index:
                record = option_lookup.loc[key_signal]
                if isinstance(record, pd.DataFrame):
                    record = record.iloc[-1]
                for column in ("close", "volume", "close_oi"):
                    values[f"signal_{leg}_{column}"] = float(record[column])
            if key_entry in option_lookup.index:
                record = option_lookup.loc[key_entry]
                if isinstance(record, pd.DataFrame):
                    record = record.iloc[-1]
                for column in ("open", "volume", "open_oi", "close_oi"):
                    values[f"entry_{leg}_{column}"] = float(record[column])
        future_key = (row.entry_date, row.underlying_symbol)
        if future_key in future_lookup.index:
            future_record = future_lookup.loc[future_key]
            if isinstance(future_record, pd.DataFrame):
                future_record = future_record.iloc[-1]
            values["entry_future_open"] = float(future_record["open"])
        signal_values.append(values)
    result = pd.concat([result.reset_index(drop=True), pd.DataFrame(signal_values)], axis=1)
    result["signal_straddle_close"] = (
        result.get("signal_call_close") + result.get("signal_put_close")
    )
    result["entry_straddle_open"] = (
        result.get("entry_call_open") + result.get("entry_put_open")
    )
    result["signal_iv"] = [
        implied_volatility_from_straddle(
            price,
            future,
            strike,
            _calendar_time(signal, expiry),
            risk_free_rate,
        )
        for price, future, strike, signal, expiry in zip(
            result["signal_straddle_close"],
            result["signal_future_close"],
            result["strike_price"],
            result["signal_date"],
            result["expiry_date"],
        )
    ]
    result["entry_iv"] = [
        implied_volatility_from_straddle(
            price,
            future,
            strike,
            _calendar_time(entry, expiry, at_open=True),
            risk_free_rate,
        )
        for price, future, strike, entry, expiry in zip(
            result["entry_straddle_open"],
            result["entry_future_open"],
            result["strike_price"],
            result["entry_date"],
            result["expiry_date"],
        )
    ]

    feature_columns = ["trade_date"]
    for horizon in sorted(result["horizon"].unique()):
        feature_columns.extend(
            [f"target_rv_{int(horizon)}d", f"target_maturity_date_{int(horizon)}d"]
        )
    feature_data = features[list(dict.fromkeys(feature_columns))].copy()
    feature_data["trade_date"] = pd.to_datetime(feature_data["trade_date"]).dt.normalize()
    actual_rows = []
    feature_lookup = feature_data.set_index("trade_date")
    for row in result.itertuples(index=False):
        record = feature_lookup.loc[row.signal_date] if row.signal_date in feature_lookup.index else None
        actual_rows.append(
            {
                "actual_rv": (
                    float(record[f"target_rv_{row.horizon}d"])
                    if record is not None and pd.notna(record[f"target_rv_{row.horizon}d"])
                    else np.nan
                ),
                "maturity_date": (
                    pd.Timestamp(record[f"target_maturity_date_{row.horizon}d"]).normalize()
                    if record is not None
                    and pd.notna(record[f"target_maturity_date_{row.horizon}d"])
                    else pd.NaT
                ),
            }
        )
    result = pd.concat([result, pd.DataFrame(actual_rows)], axis=1)

    forecasts = forecasts.copy()
    forecasts["trade_date"] = pd.to_datetime(forecasts["trade_date"]).dt.normalize()
    forecast_wide = forecasts.pivot_table(
        index=["trade_date", "horizon"],
        columns="model_name",
        values="forecast_rv",
        aggfunc="last",
    ).reset_index()
    forecast_wide = forecast_wide.rename(
        columns={
            MAIN_MODEL: "main_forecast_rv",
            ROBUST_MODEL: "robust_forecast_rv",
            PERSISTENCE_MODEL: "persistence_forecast_rv",
        }
    )
    result = result.merge(
        forecast_wide,
        left_on=["signal_date", "horizon"],
        right_on=["trade_date", "horizon"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    jump = forecasts[forecasts["model_name"].eq(MAIN_MODEL)][
        ["trade_date", "horizon", "jump_significant"]
    ].drop_duplicates(["trade_date", "horizon"])
    result = result.merge(
        jump,
        left_on=["signal_date", "horizon"],
        right_on=["trade_date", "horizon"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    result["jump_significant"] = result["jump_significant"].fillna(False).astype(bool)
    result["signal_implied_variance"] = result["signal_iv"] ** 2
    result["actual_annual_variance"] = 252.0 * result["actual_rv"]
    result["realized_vrp"] = (
        result["signal_implied_variance"] - result["actual_annual_variance"]
    )

    result = result.sort_values(["signal_date", "horizon"]).reset_index(drop=True)
    estimates = []
    for row in result.itertuples(index=False):
        history = result[
            result["horizon"].eq(row.horizon)
            & result["maturity_date"].le(row.signal_date)
            & result["realized_vrp"].notna()
        ]["realized_vrp"]
        if len(history) >= minimum_vrp_history:
            lower, upper = history.quantile([0.1, 0.9])
            trimmed = history.clip(lower=lower, upper=upper)
            estimate = float(trimmed.median())
        else:
            estimate = np.nan
        estimates.append(
            {"vrp_estimate": estimate, "vrp_history_count": int(len(history))}
        )
    result = pd.concat([result, pd.DataFrame(estimates)], axis=1)
    for label in ("main", "robust", "persistence"):
        result[f"{label}_fair_variance"] = (
            252.0 * result[f"{label}_forecast_rv"] + result["vrp_estimate"]
        ).clip(lower=1.0e-8)
        result[f"{label}_fair_volatility"] = np.sqrt(
            result[f"{label}_fair_variance"]
        )
        result[f"{label}_volatility_edge"] = (
            result[f"{label}_fair_volatility"] - result["signal_iv"]
        )
    return result


def add_strategy_signals(
    opportunities: pd.DataFrame,
    *,
    minimum_volatility_edge: float,
    minimum_total_open_interest: float,
    minimum_leg_volume: float,
    minimum_premium_ticks: float,
) -> pd.DataFrame:
    result = opportunities.copy()
    base_valid = (
        result["signal_iv"].between(0.01, 2.0)
        & result["entry_iv"].between(0.01, 2.0)
        & result["entry_straddle_open"].ge(
            minimum_premium_ticks * result["price_tick"]
        )
        & result["moneyness_abs"].le(0.015)
        & result["vrp_estimate"].notna()
        & result["entry_future_open"].gt(0)
        & result["signal_call_volume"].ge(minimum_leg_volume)
        & result["signal_put_volume"].ge(minimum_leg_volume)
        & result["entry_call_volume"].ge(minimum_leg_volume)
        & result["entry_put_volume"].ge(minimum_leg_volume)
        & (
            result["entry_call_open_oi"].fillna(0)
            + result["entry_put_open_oi"].fillna(0)
        ).ge(minimum_total_open_interest)
    )
    main_direction = np.sign(result["main_volatility_edge"]).astype(float)
    robust_direction = np.sign(result["robust_volatility_edge"]).astype(float)
    persistence_direction = np.sign(result["persistence_volatility_edge"]).astype(float)
    main_edge_valid = result["main_volatility_edge"].abs().ge(minimum_volatility_edge)
    robust_edge_valid = result["robust_volatility_edge"].abs().ge(minimum_volatility_edge)
    persistence_edge_valid = result["persistence_volatility_edge"].abs().ge(
        minimum_volatility_edge
    )
    result["signal_main_robust"] = np.where(
        base_valid
        & main_edge_valid
        & robust_edge_valid
        & main_direction.eq(robust_direction)
        & ~(main_direction.lt(0) & result["jump_significant"]),
        main_direction,
        0.0,
    )
    result["signal_main_only"] = np.where(
        base_valid & main_edge_valid & ~(main_direction.lt(0) & result["jump_significant"]),
        main_direction,
        0.0,
    )
    result["signal_persistence"] = np.where(
        base_valid
        & persistence_edge_valid
        & ~(persistence_direction.lt(0) & result["jump_significant"]),
        persistence_direction,
        0.0,
    )
    result["signal_always_short"] = np.where(
        base_valid
        & result["main_forecast_rv"].notna()
        & ~result["jump_significant"],
        -1.0,
        0.0,
    )
    result["signal_matched_short"] = np.where(
        result["signal_main_robust"].ne(0), -1.0, 0.0
    )
    result["signal_model_timed_short"] = np.where(
        result["signal_main_robust"].lt(0), -1.0, 0.0
    )
    result["signal_model_timed_long"] = np.where(
        result["signal_main_robust"].gt(0), 1.0, 0.0
    )
    result["liquidity_and_data_valid"] = base_valid
    return result


def _option_mark_and_iv(
    date: pd.Timestamp,
    call_symbol: str,
    put_symbol: str,
    future: float,
    strike: float,
    expiry: pd.Timestamp,
    option_lookup: pd.DataFrame,
    last_iv: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    call_key = (date, call_symbol)
    put_key = (date, put_symbol)
    if call_key in option_lookup.index and put_key in option_lookup.index:
        call = option_lookup.loc[call_key]
        put = option_lookup.loc[put_key]
        if isinstance(call, pd.DataFrame):
            call = call.iloc[-1]
        if isinstance(put, pd.DataFrame):
            put = put.iloc[-1]
        price = float(call["close"] + put["close"])
        iv = implied_volatility_from_straddle(
            price,
            future,
            strike,
            _calendar_time(date, expiry),
            risk_free_rate,
        )
        if np.isfinite(iv):
            return price, iv
    return (
        black76_straddle_price(
            future,
            strike,
            _calendar_time(date, expiry),
            last_iv,
            risk_free_rate,
        ),
        last_iv,
    )


def simulate_delta_hedged_trade(
    row: pd.Series,
    side: int,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    *,
    initial_capital: float,
    risk_budget_fraction: float,
    risk_capital_fraction_of_notional: float,
    risk_free_rate: float,
    costs: CostAssumptions,
) -> tuple[dict, pd.DataFrame] | None:
    if side not in (-1, 1):
        return None
    future_path = futures_daily[
        futures_daily["ts_code"].astype(str).eq(str(row["underlying_symbol"]))
        & futures_daily["trade_date"].between(row["entry_date"], row["expiry_date"])
    ].sort_values("trade_date")
    if future_path.empty or future_path.iloc[0]["trade_date"] != row["entry_date"]:
        return None
    if future_path.iloc[-1]["trade_date"] != row["expiry_date"]:
        return None
    multiplier = float(row["volume_multiple"])
    option_tick = float(row["price_tick"])
    future_tick = 0.02
    per_lot_risk_capital = (
        risk_capital_fraction_of_notional
        * float(row["entry_future_open"])
        * multiplier
    )
    lots = int(
        max(
            1,
            np.floor(initial_capital * risk_budget_fraction / per_lot_risk_capital),
        )
    )
    risk_capital = lots * per_lot_risk_capital
    option_lookup = option_daily.copy()
    option_lookup["trade_date"] = pd.to_datetime(option_lookup["trade_date"]).dt.normalize()
    option_lookup = option_lookup.set_index(["trade_date", "ts_code"])
    entry_premium = float(row["entry_straddle_open"])
    last_option_mark = entry_premium
    last_iv = float(row["entry_iv"])
    option_entry_cost = lots * (
        2.0 * costs.option_commission_per_contract_side
        + 2.0 * costs.option_slippage_ticks_per_side * option_tick * multiplier
    )
    cumulative_cost = option_entry_cost
    cumulative_option_pnl = 0.0
    cumulative_futures_pnl = 0.0
    hedge = 0
    previous_future_price = np.nan
    daily_rows: list[dict] = []
    initial_delta = np.nan
    initial_hedge_contracts = 0
    for position, future_row in enumerate(future_path.itertuples(index=False)):
        date = pd.Timestamp(future_row.trade_date).normalize()
        future_close = float(future_row.close)
        if position == 0:
            future_reference = float(future_row.open)
            delta = black76_straddle_delta(
                future_reference,
                float(row["strike_price"]),
                _calendar_time(date, row["expiry_date"], at_open=True),
                last_iv,
                risk_free_rate,
            )
            desired_hedge = int(round(-side * lots * delta))
            initial_delta = delta
            initial_hedge_contracts = desired_hedge
            hedge_change = desired_hedge - hedge
            hedge = desired_hedge
            cumulative_cost += abs(hedge_change) * (
                costs.futures_commission_per_contract_side
                + costs.futures_slippage_ticks_per_trade * future_tick * multiplier
            )
            cumulative_futures_pnl += hedge * multiplier * (
                future_close - future_reference
            )
        else:
            cumulative_futures_pnl += hedge * multiplier * (
                future_close - previous_future_price
            )

        is_expiry = date == pd.Timestamp(row["expiry_date"]).normalize()
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
                risk_free_rate,
            )
        cumulative_option_pnl += (
            side * lots * multiplier * (option_mark - last_option_mark)
        )
        last_option_mark = option_mark

        if is_expiry:
            cumulative_cost += abs(hedge) * (
                costs.futures_commission_per_contract_side
                + costs.futures_slippage_ticks_per_trade * future_tick * multiplier
            )
            cumulative_cost += lots * 2.0 * costs.option_exercise_fee_per_contract
            hedge = 0
        else:
            delta = black76_straddle_delta(
                future_close,
                float(row["strike_price"]),
                _calendar_time(date, row["expiry_date"]),
                last_iv,
                risk_free_rate,
            )
            desired_hedge = int(round(-side * lots * delta))
            hedge_change = desired_hedge - hedge
            if hedge_change:
                cumulative_cost += abs(hedge_change) * (
                    costs.futures_commission_per_contract_side
                    + costs.futures_slippage_ticks_per_trade * future_tick * multiplier
                )
            hedge = desired_hedge
        net_pnl = cumulative_option_pnl + cumulative_futures_pnl - cumulative_cost
        daily_rows.append(
            {
                "trade_date": date,
                "cumulative_option_pnl": cumulative_option_pnl,
                "cumulative_futures_pnl": cumulative_futures_pnl,
                "cumulative_cost": cumulative_cost,
                "cumulative_net_pnl": net_pnl,
                "hedge_contracts": hedge,
                "option_mark": option_mark,
                "mark_iv": last_iv,
            }
        )
        previous_future_price = future_close
    daily = pd.DataFrame(daily_rows)
    daily["daily_net_pnl"] = daily["cumulative_net_pnl"].diff().fillna(
        daily["cumulative_net_pnl"]
    )
    final = daily.iloc[-1]
    trade = {
        "signal_date": row["signal_date"],
        "entry_date": row["entry_date"],
        "expiry_date": row["expiry_date"],
        "horizon": int(row["horizon"]),
        "underlying_symbol": row["underlying_symbol"],
        "call_symbol": row["call_symbol"],
        "put_symbol": row["put_symbol"],
        "strike_price": float(row["strike_price"]),
        "volume_multiple": multiplier,
        "side": int(side),
        "lots": lots,
        "signal_iv": float(row["signal_iv"]),
        "entry_iv": float(row["entry_iv"]),
        "entry_future_open": float(row["entry_future_open"]),
        "entry_straddle_open": entry_premium,
        "initial_straddle_delta": float(initial_delta),
        "initial_hedge_contracts": int(initial_hedge_contracts),
        "main_fair_volatility": float(row.get("main_fair_volatility", np.nan)),
        "robust_fair_volatility": float(row.get("robust_fair_volatility", np.nan)),
        "persistence_fair_volatility": float(
            row.get("persistence_fair_volatility", np.nan)
        ),
        "vrp_estimate": float(row["vrp_estimate"]),
        "actual_annual_volatility": float(np.sqrt(252.0 * row["actual_rv"]))
        if pd.notna(row["actual_rv"])
        else np.nan,
        "risk_capital": risk_capital,
        "gross_option_pnl": float(final["cumulative_option_pnl"]),
        "gross_futures_pnl": float(final["cumulative_futures_pnl"]),
        "transaction_cost": float(final["cumulative_cost"]),
        "net_pnl": float(final["cumulative_net_pnl"]),
        "return_on_risk_capital": float(final["cumulative_net_pnl"] / risk_capital),
    }
    return trade, daily


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def portfolio_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> dict:
    if daily.empty:
        return {}
    daily = daily.sort_values("trade_date").copy()
    daily["equity"] = initial_capital + daily["daily_net_pnl"].cumsum()
    daily["return"] = daily["equity"].pct_change().fillna(
        daily["daily_net_pnl"] / initial_capital
    )
    observations = len(daily)
    years = observations / 252.0
    ending_equity = float(daily["equity"].iloc[-1])
    annual_return = (
        (ending_equity / initial_capital) ** (1.0 / years) - 1.0
        if ending_equity > 0 and years > 0
        else np.nan
    )
    annual_volatility = float(daily["return"].std(ddof=1) * np.sqrt(252.0))
    sharpe = (
        float(daily["return"].mean() / daily["return"].std(ddof=1) * np.sqrt(252.0))
        if daily["return"].std(ddof=1) > 0
        else np.nan
    )
    downside = daily.loc[daily["return"].lt(0), "return"]
    sortino = (
        float(daily["return"].mean() / downside.std(ddof=1) * np.sqrt(252.0))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else np.nan
    )
    drawdown = _max_drawdown(daily["equity"])
    gross_profit = float(trades.loc[trades["net_pnl"].gt(0), "net_pnl"].sum())
    gross_loss = abs(float(trades.loc[trades["net_pnl"].lt(0), "net_pnl"].sum()))
    expiry_clusters = trades.groupby("expiry_date")["net_pnl"].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(20260730)
    if len(expiry_clusters) >= 2:
        samples = rng.choice(
            expiry_clusters,
            size=(10000, len(expiry_clusters)),
            replace=True,
        ).sum(axis=1)
        bootstrap_lower, bootstrap_upper = np.quantile(samples, [0.025, 0.975])
        bootstrap_positive = float(np.mean(samples > 0))
    else:
        bootstrap_lower = bootstrap_upper = bootstrap_positive = np.nan
    capital_events = []
    for trade in trades.itertuples(index=False):
        capital_events.append((pd.Timestamp(trade.entry_date), float(trade.risk_capital)))
        capital_events.append(
            (pd.Timestamp(trade.expiry_date) + pd.Timedelta(nanoseconds=1), -float(trade.risk_capital))
        )
    active_capital = 0.0
    maximum_active_capital = 0.0
    for _, change in sorted(capital_events, key=lambda item: item[0]):
        active_capital += change
        maximum_active_capital = max(maximum_active_capital, active_capital)
    return {
        "start_date": daily["trade_date"].min(),
        "end_date": daily["trade_date"].max(),
        "trading_days": observations,
        "trade_count": int(len(trades)),
        "long_trade_count": int(trades["side"].gt(0).sum()),
        "short_trade_count": int(trades["side"].lt(0).sum()),
        "net_profit": float(ending_equity - initial_capital),
        "total_return": float(ending_equity / initial_capital - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": float(annual_return / abs(drawdown)) if drawdown < 0 else np.nan,
        "win_rate": float(trades["net_pnl"].gt(0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "average_trade_pnl": float(trades["net_pnl"].mean()),
        "median_trade_pnl": float(trades["net_pnl"].median()),
        "total_transaction_cost": float(trades["transaction_cost"].sum()),
        "cost_to_gross_abs_pnl": float(
            trades["transaction_cost"].sum()
            / max(
                (
                    trades["gross_option_pnl"] + trades["gross_futures_pnl"]
                ).abs().sum(),
                1.0,
            )
        ),
        "daily_var_95": float(daily["return"].quantile(0.05)),
        "daily_expected_shortfall_95": float(
            daily.loc[daily["return"].le(daily["return"].quantile(0.05)), "return"].mean()
        ),
        "expiry_cluster_count": int(len(expiry_clusters)),
        "bootstrap_net_profit_ci_95_lower": float(bootstrap_lower),
        "bootstrap_net_profit_ci_95_upper": float(bootstrap_upper),
        "bootstrap_probability_net_profit_positive": bootstrap_positive,
        "maximum_active_risk_capital": maximum_active_capital,
        "maximum_active_risk_capital_fraction": maximum_active_capital / initial_capital,
    }


def run_strategy_variant(
    opportunities: pd.DataFrame,
    signal_column: str,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    *,
    strategy_name: str,
    initial_capital: float,
    risk_budget_fraction: float,
    risk_capital_fraction_of_notional: float,
    risk_free_rate: float,
    costs: CostAssumptions,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trade_rows = []
    daily_pieces = []
    for _, row in opportunities[opportunities[signal_column].ne(0)].iterrows():
        simulated = simulate_delta_hedged_trade(
            row,
            int(row[signal_column]),
            option_daily,
            futures_daily,
            initial_capital=initial_capital,
            risk_budget_fraction=risk_budget_fraction,
            risk_capital_fraction_of_notional=risk_capital_fraction_of_notional,
            risk_free_rate=risk_free_rate,
            costs=costs,
        )
        if simulated is None:
            continue
        trade, daily = simulated
        trade_id = f"{strategy_name}_{len(trade_rows) + 1:04d}"
        trade["strategy"] = strategy_name
        trade["trade_id"] = trade_id
        daily["strategy"] = strategy_name
        daily["trade_id"] = trade_id
        trade_rows.append(trade)
        daily_pieces.append(daily)
    trades = pd.DataFrame(trade_rows)
    if not daily_pieces:
        return trades, pd.DataFrame(), {}
    trade_daily = pd.concat(daily_pieces, ignore_index=True)
    calendar = pd.DatetimeIndex(
        sorted(
            futures_daily.loc[
                futures_daily["trade_date"].between(
                    trade_daily["trade_date"].min(), trade_daily["trade_date"].max()
                ),
                "trade_date",
            ].unique()
        )
    )
    daily = (
        trade_daily.groupby("trade_date", as_index=False)["daily_net_pnl"].sum()
        .set_index("trade_date")
        .reindex(calendar, fill_value=0.0)
        .rename_axis("trade_date")
        .reset_index()
    )
    daily["strategy"] = strategy_name
    daily["equity"] = initial_capital + daily["daily_net_pnl"].cumsum()
    daily["return"] = daily["equity"].pct_change().fillna(
        daily["daily_net_pnl"] / initial_capital
    )
    metrics = portfolio_metrics(daily, trades, initial_capital=initial_capital)
    metrics["strategy"] = strategy_name
    return trades, daily, metrics


def trade_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = trades.copy()
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    data["year"] = data["entry_date"].dt.year
    groupings = {
        "horizon_side": ["strategy", "horizon", "side"],
        "year": ["strategy", "year"],
    }
    for scope, columns in groupings.items():
        for keys, group in data.groupby(columns):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"scope": scope}
            row.update(dict(zip(columns, keys)))
            row.update(
                {
                    "trade_count": int(len(group)),
                    "net_profit": float(group["net_pnl"].sum()),
                    "average_trade_pnl": float(group["net_pnl"].mean()),
                    "win_rate": float(group["net_pnl"].gt(0).mean()),
                    "average_return_on_risk_capital": float(
                        group["return_on_risk_capital"].mean()
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def one_day_tail_stress(
    trades: pd.DataFrame,
    *,
    risk_free_rate: float,
    underlying_shock: float,
    volatility_multiplier: float,
    volatility_additive: float,
    initial_capital: float,
) -> pd.DataFrame:
    rows = []
    for trade in trades.itertuples(index=False):
        future = float(trade.entry_future_open)
        stressed_iv = max(
            float(trade.entry_iv) * volatility_multiplier,
            float(trade.entry_iv) + volatility_additive,
        )
        remaining = max(
            _calendar_time(trade.entry_date, trade.expiry_date, at_open=True)
            - 1.0 / 365.0,
            1.0 / 365.0,
        )
        scenario_values = []
        for direction in (-1.0, 1.0):
            stressed_future = future * (1.0 + direction * underlying_shock)
            stressed_option = black76_straddle_price(
                stressed_future,
                float(trade.strike_price),
                remaining,
                stressed_iv,
                risk_free_rate,
            )
            option_pnl = (
                int(trade.side)
                * int(trade.lots)
                * float(trade.volume_multiple)
                * (stressed_option - float(trade.entry_straddle_open))
            )
            hedge_pnl = (
                int(trade.initial_hedge_contracts)
                * float(trade.volume_multiple)
                * (stressed_future - future)
            )
            scenario_values.append(option_pnl + hedge_pnl)
        worst = float(min(scenario_values))
        rows.append(
            {
                "strategy": trade.strategy,
                "trade_id": trade.trade_id,
                "entry_date": trade.entry_date,
                "horizon": int(trade.horizon),
                "side": int(trade.side),
                "underlying_shock": underlying_shock,
                "stressed_iv": stressed_iv,
                "worst_one_day_pnl": worst,
                "worst_one_day_return_on_initial_capital": worst / initial_capital,
                "worst_one_day_return_on_trade_risk_capital": worst
                / float(trade.risk_capital),
            }
        )
    return pd.DataFrame(rows)


def run_volatility_strategy_backtest(config: dict) -> dict[str, Path]:
    settings = config["strategy"]
    root = Path(config["_project_root"])
    pairs = read_parquet(root / "data/raw/tqsdk_au_option_strategy_pairs.parquet")
    option_daily = read_parquet(root / "data/raw/tqsdk_au_option_daily_strategy.parquet")
    futures_daily = read_parquet(root / "data/raw/tqsdk_au_futures_daily_all_contracts.parquet")
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    forecasts = read_parquet(resolve_path(config, config["outputs"]["strategy_forecasts_path"]))
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    opportunities = build_option_opportunities(
        pairs,
        option_daily,
        futures_daily,
        features,
        forecasts,
        risk_free_rate=float(settings["risk_free_rate"]),
        minimum_vrp_history=int(settings["minimum_vrp_history"]),
    )
    opportunities = add_strategy_signals(
        opportunities,
        minimum_volatility_edge=float(settings["minimum_volatility_edge"]),
        minimum_total_open_interest=float(settings["minimum_total_open_interest"]),
        minimum_leg_volume=float(settings["minimum_leg_volume"]),
        minimum_premium_ticks=float(settings["minimum_premium_ticks"]),
    )
    costs = CostAssumptions(
        option_commission_per_contract_side=float(
            settings["costs"]["option_commission_per_contract_side"]
        ),
        option_exercise_fee_per_contract=float(
            settings["costs"]["option_exercise_fee_per_contract"]
        ),
        futures_commission_per_contract_side=float(
            settings["costs"]["futures_commission_per_contract_side"]
        ),
        option_slippage_ticks_per_side=float(
            settings["costs"]["option_slippage_ticks_per_side"]
        ),
        futures_slippage_ticks_per_trade=float(
            settings["costs"]["futures_slippage_ticks_per_trade"]
        ),
    )
    variants = {
        "model_timed_short": "signal_model_timed_short",
        "main_plus_robust": "signal_main_robust",
        "model_timed_long": "signal_model_timed_long",
        "main_only": "signal_main_only",
        "persistence": "signal_persistence",
        "always_short": "signal_always_short",
        "matched_always_short": "signal_matched_short",
    }
    trade_pieces = []
    daily_pieces = []
    metric_rows = []
    for strategy_name, signal_column in variants.items():
        trades, daily, metrics = run_strategy_variant(
            opportunities,
            signal_column,
            option_daily,
            futures_daily,
            strategy_name=strategy_name,
            initial_capital=float(settings["initial_capital"]),
            risk_budget_fraction=float(settings["risk_budget_fraction_per_trade"]),
            risk_capital_fraction_of_notional=float(
                settings["risk_capital_fraction_of_notional"]
            ),
            risk_free_rate=float(settings["risk_free_rate"]),
            costs=costs,
        )
        if not trades.empty:
            trade_pieces.append(trades)
        if not daily.empty:
            daily_pieces.append(daily)
        if metrics:
            metric_rows.append(metrics)
    all_trades = pd.concat(trade_pieces, ignore_index=True)
    all_daily = pd.concat(daily_pieces, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    breakdown = trade_breakdown(all_trades)
    primary_trades = all_trades[all_trades["strategy"].eq("model_timed_short")]
    stress = one_day_tail_stress(
        primary_trades,
        risk_free_rate=float(settings["risk_free_rate"]),
        underlying_shock=float(settings["stress"]["underlying_shock"]),
        volatility_multiplier=float(settings["stress"]["volatility_multiplier"]),
        volatility_additive=float(settings["stress"]["volatility_additive"]),
        initial_capital=float(settings["initial_capital"]),
    )

    sensitivity_rows = []
    for threshold in settings["sensitivity"]["volatility_edges"]:
        candidate = add_strategy_signals(
            opportunities.drop(
                columns=[
                    "signal_main_robust",
                    "signal_main_only",
                    "signal_persistence",
                    "signal_always_short",
                    "signal_matched_short",
                    "signal_model_timed_short",
                    "signal_model_timed_long",
                    "liquidity_and_data_valid",
                ],
                errors="ignore",
            ),
            minimum_volatility_edge=float(threshold),
            minimum_total_open_interest=float(settings["minimum_total_open_interest"]),
            minimum_leg_volume=float(settings["minimum_leg_volume"]),
            minimum_premium_ticks=float(settings["minimum_premium_ticks"]),
        )
        _, _, metrics = run_strategy_variant(
            candidate,
            "signal_model_timed_short",
            option_daily,
            futures_daily,
            strategy_name=f"edge_{float(threshold):.3f}",
            initial_capital=float(settings["initial_capital"]),
            risk_budget_fraction=float(settings["risk_budget_fraction_per_trade"]),
            risk_capital_fraction_of_notional=float(
                settings["risk_capital_fraction_of_notional"]
            ),
            risk_free_rate=float(settings["risk_free_rate"]),
            costs=costs,
        )
        if metrics:
            metrics["minimum_volatility_edge"] = float(threshold)
            sensitivity_rows.append(metrics)
    sensitivity = pd.DataFrame(sensitivity_rows)
    cost_sensitivity_rows = []
    for multiplier in settings["sensitivity"]["cost_multipliers"]:
        cost_multiplier = float(multiplier)
        stressed_costs = CostAssumptions(
            option_commission_per_contract_side=(
                costs.option_commission_per_contract_side * cost_multiplier
            ),
            option_exercise_fee_per_contract=(
                costs.option_exercise_fee_per_contract * cost_multiplier
            ),
            futures_commission_per_contract_side=(
                costs.futures_commission_per_contract_side * cost_multiplier
            ),
            option_slippage_ticks_per_side=(
                costs.option_slippage_ticks_per_side * cost_multiplier
            ),
            futures_slippage_ticks_per_trade=(
                costs.futures_slippage_ticks_per_trade * cost_multiplier
            ),
        )
        _, _, metrics = run_strategy_variant(
            opportunities,
            "signal_model_timed_short",
            option_daily,
            futures_daily,
            strategy_name=f"cost_{cost_multiplier:.1f}x",
            initial_capital=float(settings["initial_capital"]),
            risk_budget_fraction=float(settings["risk_budget_fraction_per_trade"]),
            risk_capital_fraction_of_notional=float(
                settings["risk_capital_fraction_of_notional"]
            ),
            risk_free_rate=float(settings["risk_free_rate"]),
            costs=stressed_costs,
        )
        if metrics:
            metrics["cost_multiplier"] = cost_multiplier
            cost_sensitivity_rows.append(metrics)
    cost_sensitivity = pd.DataFrame(cost_sensitivity_rows)
    outputs = {
        "opportunities": write_parquet(
            opportunities, resolve_path(config, settings["opportunities_path"])
        ),
        "trades": write_csv(
            all_trades, resolve_path(config, settings["trades_path"])
        ),
        "daily": write_csv(
            all_daily, resolve_path(config, settings["daily_path"])
        ),
        "metrics": write_csv(
            all_metrics, resolve_path(config, settings["metrics_path"])
        ),
        "breakdown": write_csv(
            breakdown, resolve_path(config, settings["breakdown_path"])
        ),
        "stress": write_csv(
            stress, resolve_path(config, settings["stress_path"])
        ),
        "sensitivity": write_csv(
            sensitivity, resolve_path(config, settings["sensitivity_path"])
        ),
        "cost_sensitivity": write_csv(
            cost_sensitivity,
            resolve_path(config, settings["cost_sensitivity_path"]),
        ),
    }
    return outputs
