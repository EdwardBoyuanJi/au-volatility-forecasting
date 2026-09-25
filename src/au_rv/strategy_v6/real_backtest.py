from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.io import read_parquet, utc_now, write_csv, write_json, write_parquet
from au_rv.strategy.volatility_straddle import portfolio_metrics, trade_breakdown
from au_rv.strategy_v3.data import load_real_quote_ticks, prepare_real_quote_opportunities
from au_rv.strategy_v3.execution import quote_snapshot_features
from au_rv.strategy_v5.backtest import VariantSpec, _combine_daily
from au_rv.strategy_v6.backtest import (
    _annual_summary,
    _execution_settings,
    _monthly_summary,
    _strategy_specs,
)
from au_rv.strategy_v6.real_execution import (
    build_futures_close_snapshots,
    simulate_real_topbook_trade,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_strength(row: pd.Series) -> float:
    values = [-float(row["main_volatility_edge"]), -float(row["robust_volatility_edge"])]
    return float(min(values)) if all(np.isfinite(values)) else np.nan


def _sizing_multiplier(spec: VariantSpec, row: pd.Series, spread: float) -> float:
    edge = _edge_strength(row)
    if spec.sizing_mode == "flat":
        return 1.0
    if spec.sizing_mode == "edge_linear":
        return float(np.clip(edge / 0.05, 0.50, 2.00))
    if spec.sizing_mode == "edge_tiered":
        if edge < 0.03:
            return 0.50
        if edge >= 0.07:
            return 1.50
        return 1.00
    if spec.sizing_mode == "horizon_weighted":
        return {5: 0.50, 20: 1.00, 40: 1.25}.get(int(row["horizon"]), 1.0)
    if spec.sizing_mode == "edge_spread_weighted":
        edge_multiplier = float(np.clip(edge / 0.05, 0.50, 2.00))
        spread_multiplier = float(np.clip(0.25 / max(spread, 0.01), 0.50, 1.25))
        return edge_multiplier * spread_multiplier
    raise ValueError(f"Unsupported v6 real-topbook sizing mode: {spec.sizing_mode}")


def _simulate_strategy(
    selected: pd.DataFrame,
    spec: VariantSpec,
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    close_lookup: pd.DataFrame,
    entry_cache: dict[str, dict | None],
    *,
    base_settings,
    maximum_portfolio_risk_fraction: float,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    enforce_futures_visible_depth: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    trades: list[dict] = []
    daily_pieces: list[pd.DataFrame] = []
    order_pieces: list[pd.DataFrame] = []
    audits: list[dict] = []
    active_risk: list[tuple[pd.Timestamp, float]] = []
    maximum_risk = base_settings.initial_capital * maximum_portfolio_risk_fraction
    for _, row in selected.sort_values(["entry_date", "horizon"]).iterrows():
        oid = str(row["opportunity_id"])
        entry_date = pd.Timestamp(row["entry_date"]).normalize()
        expiry_date = pd.Timestamp(row["expiry_date"]).normalize()
        active_risk = [(expiry, risk) for expiry, risk in active_risk if expiry >= entry_date]
        audit = {
            "strategy": spec.name,
            "opportunity_id": oid,
            "entry_date": entry_date,
            "expiry_date": expiry_date,
            "horizon": int(row["horizon"]),
            "executed": False,
            "skip_reason": None,
        }
        if int(row["horizon"]) not in spec.horizons:
            audit["skip_reason"] = "horizon_filter"
            audits.append(audit)
            continue
        edge = _edge_strength(row)
        audit["edge_strength"] = edge
        if not np.isfinite(edge) or edge < spec.minimum_edge_strength:
            audit["skip_reason"] = "edge_filter"
            audits.append(audit)
            continue
        snapshot = entry_cache.get(oid)
        if snapshot is None:
            audit["skip_reason"] = "synchronized_entry_quote_unavailable"
            audits.append(audit)
            continue
        relative_spread = float(
            (snapshot["long_entry_premium"] - snapshot["short_entry_premium"])
            / max(snapshot["mid_entry_premium"], 1.0e-8)
        )
        audit["relative_spread"] = relative_spread
        if spec.maximum_relative_spread is not None and relative_spread > spec.maximum_relative_spread:
            audit["skip_reason"] = "spread_filter"
            audits.append(audit)
            continue
        multiplier = _sizing_multiplier(spec, row, relative_spread)
        effective_budget = min(
            spec.risk_budget_fraction * multiplier,
            maximum_portfolio_risk_fraction,
        )
        per_lot_risk = (
            base_settings.short_risk_capital_fraction_of_notional
            * float(snapshot["future_mid"])
            * float(row["volume_multiple"])
        )
        active_before = float(sum(value for _, value in active_risk))
        maximum_lots = int(math.floor(max(maximum_risk - active_before, 0.0) / per_lot_risk))
        audit.update(
            {
                "sizing_multiplier": multiplier,
                "effective_risk_budget_fraction": effective_budget,
                "active_risk_before": active_before,
                "maximum_lots_from_portfolio_cap": maximum_lots,
            }
        )
        if maximum_lots < 1:
            audit["skip_reason"] = "portfolio_risk_cap"
            audits.append(audit)
            continue
        risk_lots = max(
            1,
            int(
                math.floor(
                    base_settings.initial_capital
                    * effective_budget
                    / per_lot_risk
                )
            ),
        )
        option_depth_cap = (
            risk_lots
            if spec.depth_multiplier is None
            else min(
                int(snapshot["call_bid_size"]), int(snapshot["put_bid_size"])
            )
            * int(spec.depth_multiplier)
        )
        planned_lots = min(risk_lots, option_depth_cap, maximum_lots)
        simulated = None
        executable_lot_cap = planned_lots
        while executable_lot_cap >= 1 and simulated is None:
            simulated = simulate_real_topbook_trade(
                row,
                option_lookup,
                futures_daily,
                close_lookup,
                settings=replace(
                    base_settings,
                    risk_budget_fraction_per_trade=effective_budget,
                ),
                displayed_depth_multiplier=spec.depth_multiplier,
                maximum_lots=executable_lot_cap,
                hedge_interval_trading_days=spec.hedge_interval,
                delta_change_threshold_contracts=spec.delta_threshold,
                entry_snapshot=snapshot,
                enforce_futures_visible_depth=enforce_futures_visible_depth,
            )
            if simulated is None:
                executable_lot_cap -= 1
        audit["planned_lots_before_futures_depth"] = planned_lots
        audit["executable_lot_cap_after_futures_depth"] = max(executable_lot_cap, 0)
        audit["lots_reduced_for_futures_topbook"] = max(
            planned_lots - max(executable_lot_cap, 0), 0
        )
        if simulated is None:
            audit["skip_reason"] = "incomplete_or_non_executable_real_topbook_path"
            audits.append(audit)
            continue
        trade, daily, orders = simulated
        trade_id = f"{spec.name}_{len(trades) + 1:04d}"
        trade.update(
            {
                "strategy": spec.name,
                "trade_id": trade_id,
                "variant_family": spec.family,
                "sizing_mode": spec.sizing_mode,
                "edge_strength": edge,
                "effective_risk_budget_fraction": effective_budget,
            }
        )
        daily["strategy"] = spec.name
        daily["trade_id"] = trade_id
        orders["strategy"] = spec.name
        orders["trade_id"] = trade_id
        orders["horizon"] = int(row["horizon"])
        orders["order_sequence"] = np.arange(1, len(orders) + 1)
        trades.append(trade)
        daily_pieces.append(daily)
        order_pieces.append(orders)
        active_risk.append((expiry_date, float(trade["risk_capital"])))
        audit.update(
            {
                "executed": True,
                "lots": int(trade["lots"]),
                "risk_capital": float(trade["risk_capital"]),
                "net_pnl": float(trade["net_pnl"]),
                "partial_hedge_order_count": int(trade["partial_hedge_order_count"]),
            }
        )
        audits.append(audit)
    trade_frame = pd.DataFrame(trades)
    order_frame = pd.concat(order_pieces, ignore_index=True) if order_pieces else pd.DataFrame()
    daily_frame = _combine_daily(
        daily_pieces,
        futures_daily,
        initial_capital=base_settings.initial_capital,
        strategy_name=spec.name,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    if trade_frame.empty:
        metrics = {"strategy": spec.name, "trade_count": 0, "net_profit": 0.0}
    else:
        metrics = portfolio_metrics(
            daily_frame,
            trade_frame,
            initial_capital=base_settings.initial_capital,
        )
        metrics.update(
            {
                "strategy": spec.name,
                "total_lots": int(trade_frame["lots"].sum()),
                "total_hedge_contract_turnover": int(
                    trade_frame["hedge_contract_turnover"].sum()
                ),
                "embedded_entry_crossing_cost": float(
                    trade_frame["embedded_entry_crossing_cost"].sum()
                ),
                "embedded_futures_crossing_cost": float(
                    trade_frame["embedded_futures_crossing_cost"].sum()
                ),
                "all_in_observed_friction": float(
                    trade_frame["all_in_observed_friction"].sum()
                ),
            }
        )
    metrics.update(asdict(spec))
    return trade_frame, daily_frame, order_frame, pd.DataFrame(audits), metrics


def _validate_futures_orders(orders: pd.DataFrame) -> pd.DataFrame:
    futures = orders[orders["instrument_type"].eq("FUTURE")].copy()
    if futures.empty:
        raise RuntimeError("No futures hedge orders were produced.")
    futures["side_price_correct"] = np.where(
        futures["side"].eq("BUY"),
        np.isclose(futures["effective_price"], futures["ask_price1"]),
        np.isclose(futures["effective_price"], futures["bid_price1"]),
    )
    futures["within_visible_depth"] = np.where(
        futures["side"].eq("BUY"),
        futures["quantity"].le(futures["ask_volume1"]),
        futures["quantity"].le(futures["bid_volume1"]),
    )
    futures["timestamp_present"] = futures["timestamp_utc"].notna()
    futures["real_topbook_source"] = futures["price_source"].str.startswith("REAL_")
    futures["execution_valid"] = futures[
        [
            "side_price_correct",
            "within_visible_depth",
            "timestamp_present",
            "real_topbook_source",
        ]
    ].all(axis=1)
    if not bool(futures["execution_valid"].all()):
        bad = futures[~futures["execution_valid"]]
        raise RuntimeError(f"Found {len(bad)} non-compliant futures executions.")
    forbidden = futures["price_source"].str.contains(
        "MODELED|DAILY_CLOSE|PLUS_1_TICK|SLIPPAGE", case=False, regex=True
    )
    if bool(forbidden.any()):
        raise RuntimeError("A modeled futures execution source leaked into real-topbook results.")
    return futures


def run_real_topbook_backtest(settings: dict) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    output_dir = root / settings["outputs"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities = prepare_real_quote_opportunities(
        read_parquet(root / settings["inputs"]["opportunities_path"])
    )
    option_daily = read_parquet(root / settings["inputs"]["option_daily_path"])
    futures_daily = read_parquet(root / settings["inputs"]["futures_daily_path"])
    entry_ticks = load_real_quote_ticks(root, settings["inputs"]["entry_quote_dir"])
    hedge_ticks = read_parquet(root / settings["inputs"]["hedge_tick_path"])
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    for column in ("signal_date", "entry_date", "expiry_date"):
        opportunities[column] = pd.to_datetime(opportunities[column]).dt.normalize()
    start = pd.Timestamp(settings["sample"]["requested_backtest_start"]).normalize()
    end = pd.Timestamp(settings["sample"]["evaluation_end"]).normalize()
    calendar_years = ((end - start).days + 1) / 365.2425
    if calendar_years < float(settings["sample"]["minimum_calendar_years"]):
        raise RuntimeError("Configured evaluation window is shorter than five years.")
    selected = opportunities[
        opportunities["signal_model_timed_short"].eq(-1)
        & opportunities["entry_date"].between(start, end)
        & opportunities["expiry_date"].le(end)
    ].copy()
    if selected.empty:
        raise RuntimeError("No completed model-timed short signals are available.")
    base_settings = _execution_settings(
        {
            "execution": {
                **settings["execution"],
                "quote_maximum_age_seconds": settings["execution"][
                    "entry_quote_maximum_age_seconds"
                ],
            }
        }
    )
    entry_cache = {
        str(row["opportunity_id"]): quote_snapshot_features(
            entry_ticks,
            row,
            maximum_quote_age_seconds=base_settings.quote_maximum_age_seconds,
        )
        for _, row in selected.iterrows()
    }
    snapshots = build_futures_close_snapshots(
        hedge_ticks,
        target_time=settings["download"]["close_target_time"],
        maximum_age_seconds=float(settings["download"]["maximum_quote_age_seconds"]),
    )
    if snapshots.empty:
        raise RuntimeError("No valid futures close top-book snapshots are available.")
    close_lookup = snapshots.set_index(["trade_date", "ts_code"])
    option_lookup = option_daily.set_index(["trade_date", "ts_code"])
    specs = _strategy_specs(settings)
    trade_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    order_pieces: list[pd.DataFrame] = []
    audit_pieces: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    for spec in specs:
        trades, daily, orders, audit, metrics = _simulate_strategy(
            selected,
            spec,
            option_lookup,
            futures_daily,
            close_lookup,
            entry_cache,
            base_settings=base_settings,
            maximum_portfolio_risk_fraction=float(
                settings["execution"]["maximum_portfolio_risk_fraction"]
            ),
            evaluation_start=start,
            evaluation_end=end,
            enforce_futures_visible_depth=bool(
                settings["execution"]["enforce_futures_visible_depth"]
            ),
        )
        if not trades.empty:
            trade_pieces.append(trades)
        if not orders.empty:
            order_pieces.append(orders)
        daily["peak_equity"] = daily["equity"].cummax()
        daily["drawdown"] = daily["equity"] / daily["peak_equity"] - 1.0
        daily_pieces.append(daily)
        audit_pieces.append(audit)
        metrics["evaluation_calendar_years"] = calendar_years
        metrics_rows.append(metrics)
    trades = pd.concat(trade_pieces, ignore_index=True)
    daily = pd.concat(daily_pieces, ignore_index=True)
    orders = pd.concat(order_pieces, ignore_index=True).sort_values(
        ["strategy", "order_date", "trade_id", "order_sequence"]
    )
    audit = pd.concat(audit_pieces, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    futures_execution_audit = _validate_futures_orders(orders)
    if not bool(trades["all_futures_executions_real_topbook"].all()):
        raise RuntimeError("Not every completed trade has real top-book futures execution.")
    reconciliation = (
        orders.groupby(["strategy", "trade_id"], as_index=False)["commission"]
        .sum()
        .rename(columns={"commission": "order_ledger_explicit_cost"})
        .merge(
            trades[["strategy", "trade_id", "transaction_cost"]],
            on=["strategy", "trade_id"],
            how="left",
        )
    )
    reconciliation["cost_reconciliation_error"] = (
        reconciliation["order_ledger_explicit_cost"] - reconciliation["transaction_cost"]
    )
    if reconciliation["cost_reconciliation_error"].abs().max() > 1.0e-6:
        raise RuntimeError("Order ledger does not reconcile to explicit costs.")
    coverage = pd.DataFrame(
        [
            {
                "evaluation_start": start,
                "evaluation_end": end,
                "calendar_days": int((end - start).days + 1),
                "calendar_years": calendar_years,
                "model_forecast_first_available": opportunities.loc[
                    opportunities[["main_forecast_rv", "robust_forecast_rv"]]
                    .notna()
                    .all(axis=1),
                    "signal_date",
                ].min(),
                "short_model_signals": int(len(selected)),
                "first_signal_entry": selected["entry_date"].min(),
                "last_signal_entry": selected["entry_date"].max(),
                "signals_with_synchronized_entry_quote": int(
                    sum(value is not None for value in entry_cache.values())
                ),
                "unique_required_futures_close_snapshots": int(
                    len(
                        {
                            (pd.Timestamp(day).normalize(), str(row.underlying_symbol))
                            for row in selected.itertuples(index=False)
                            for day in futures_daily.loc[
                                futures_daily["ts_code"].astype(str).eq(str(row.underlying_symbol))
                                & futures_daily["trade_date"].between(row.entry_date, row.expiry_date),
                                "trade_date",
                            ]
                        }
                    )
                ),
                "valid_real_futures_close_snapshots": int(len(snapshots)),
                "futures_orders": int(len(futures_execution_audit)),
                "invalid_futures_orders": int(
                    (~futures_execution_audit["execution_valid"]).sum()
                ),
            }
        ]
    )
    annual = _annual_summary(daily, trades, initial_capital=base_settings.initial_capital)
    monthly = _monthly_summary(daily, initial_capital=base_settings.initial_capital)
    breakdown = trade_breakdown(trades)
    outputs = {
        "close_snapshots": write_parquet(
            snapshots, output_dir / "futures_close_topbook_snapshots.parquet"
        ),
        "strategy_catalog": write_csv(
            pd.DataFrame([asdict(spec) for spec in specs]),
            output_dir / "strategy_catalog.csv",
        ),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "trades": write_csv(trades, output_dir / "trades.csv"),
        "orders": write_csv(orders, output_dir / "orders.csv"),
        "futures_orders": write_csv(
            futures_execution_audit,
            output_dir / "futures_orders_real_bid_ask.csv",
        ),
        "order_reconciliation": write_csv(
            reconciliation, output_dir / "order_reconciliation.csv"
        ),
        "daily": write_csv(daily, output_dir / "daily.csv"),
        "annual": write_csv(annual, output_dir / "annual.csv"),
        "monthly": write_csv(monthly, output_dir / "monthly.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
        "execution_audit": write_csv(audit, output_dir / "execution_audit.csv"),
        "futures_execution_audit": write_csv(
            futures_execution_audit, output_dir / "futures_execution_audit.csv"
        ),
        "coverage": write_csv(coverage, output_dir / "coverage.csv"),
    }
    for strategy, strategy_orders in orders.groupby("strategy", sort=True):
        outputs[f"orders_{strategy}"] = write_csv(
            strategy_orders.sort_values(["order_date", "trade_id", "order_sequence"]),
            output_dir / "orders_by_strategy" / f"{strategy}.csv",
        )
    extra_manifest_paths = [
        root / str(path) for path in settings.get("_manifest_extra_paths", [])
    ]
    manifest = {
        "strategy_version": settings["project"]["version"],
        "created_at_utc": utc_now(),
        "parent_versions_untouched": [
            "strategy_v3",
            "strategy_v4",
            "strategy_v5",
            "strategy_v6_five_year_20260830",
        ],
        "hard_guarantees": {
            "futures_orders_use_real_topbook_bid_or_ask": True,
            "futures_orders_within_visible_topbook_size": True,
            "modeled_daily_close_execution_forbidden": True,
            "hypothetical_tick_slippage_execution_forbidden": True,
        },
        "files": {
            str(path.relative_to(root)): _sha256(path)
            for path in [
                Path(settings["_config_path"]),
                root / settings["inputs"]["hedge_tick_path"],
                root / "src/au_rv/strategy_v6/real_execution.py",
                root / "src/au_rv/strategy_v6/real_backtest.py",
                *extra_manifest_paths,
                *outputs.values(),
            ]
            if path.exists() and path.is_file()
        },
    }
    outputs["freeze_manifest"] = write_json(
        manifest,
        output_dir / f"{settings['project']['version']}_freeze_manifest.json",
    )
    return outputs
