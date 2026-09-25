from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.io import read_parquet, utc_now, write_csv, write_json
from au_rv.strategy.volatility_straddle import portfolio_metrics, trade_breakdown
from au_rv.strategy_v3.data import load_real_quote_ticks, prepare_real_quote_opportunities
from au_rv.strategy_v3.execution import (
    RealQuoteExecutionSettings,
    expiry_future_price,
    quote_snapshot_features,
)
from au_rv.strategy_v5.execution import simulate_v5_trade


@dataclass(frozen=True)
class VariantSpec:
    name: str
    family: str
    depth_multiplier: int | None = 5
    risk_budget_fraction: float = 0.075
    hedge_interval: int = 1
    delta_threshold: int = 1
    minimum_edge_strength: float = 0.015
    maximum_relative_spread: float | None = None
    horizons: tuple[int, ...] = (5, 20, 40)
    sizing_mode: str = "flat"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_settings(settings: dict) -> RealQuoteExecutionSettings:
    values = settings["execution"]
    return RealQuoteExecutionSettings(
        initial_capital=float(values["initial_capital"]),
        risk_budget_fraction_per_trade=float(
            values["base_risk_budget_fraction_per_trade"]
        ),
        short_risk_capital_fraction_of_notional=float(
            values["short_risk_capital_fraction_of_notional"]
        ),
        long_minimum_risk_capital_fraction_of_notional=float(
            values["long_minimum_risk_capital_fraction_of_notional"]
        ),
        risk_free_rate=float(values["risk_free_rate"]),
        option_commission_per_contract_side=float(
            values["option_commission_per_contract_side"]
        ),
        option_exercise_fee_per_contract=float(
            values["option_exercise_fee_per_contract"]
        ),
        futures_commission_per_contract_side=float(
            values["futures_commission_per_contract_side"]
        ),
        futures_slippage_ticks_per_trade=float(
            values["futures_slippage_ticks_per_trade"]
        ),
        future_tick=float(values["future_tick"]),
        quote_maximum_age_seconds=float(
            settings["data"]["quote_maximum_age_seconds"]
        ),
    )


def build_variant_catalog(settings: dict) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for depth in settings["experiments"]["capacity_depths"]:
        specs.append(
            VariantSpec(
                name=f"capacity_depth_{int(depth)}x",
                family="capacity",
                depth_multiplier=int(depth),
            )
        )
    if bool(settings["experiments"]["include_unlimited_depth"]):
        specs.append(
            VariantSpec(
                name="capacity_unlimited",
                family="capacity",
                depth_multiplier=None,
            )
        )
    for interval in settings["experiments"]["hedge_intervals"]:
        interval = int(interval)
        if interval == 1:
            continue
        specs.append(
            VariantSpec(
                name=(
                    "hedge_hold_initial"
                    if interval >= 999
                    else f"hedge_every_{interval}d"
                ),
                family="hedge_frequency",
                hedge_interval=interval,
            )
        )
    for threshold in settings["experiments"]["delta_change_thresholds"]:
        threshold = int(threshold)
        if threshold == 1:
            continue
        specs.append(
            VariantSpec(
                name=f"delta_band_{threshold}_contracts",
                family="delta_band",
                delta_threshold=threshold,
            )
        )
    for threshold in settings["experiments"]["edge_thresholds"]:
        threshold = float(threshold)
        if math.isclose(threshold, 0.015):
            continue
        specs.append(
            VariantSpec(
                name=f"edge_min_{threshold:.3f}",
                family="signal_filter",
                minimum_edge_strength=threshold,
            )
        )
    for threshold in settings["experiments"]["maximum_relative_spreads"]:
        threshold = float(threshold)
        specs.append(
            VariantSpec(
                name=f"spread_max_{threshold:.2f}",
                family="liquidity_filter",
                maximum_relative_spread=threshold,
            )
        )
    horizon_sets = {
        "horizons_5_20": (5, 20),
        "horizons_20_40": (20, 40),
        "horizon_5_only": (5,),
        "horizon_20_only": (20,),
        "horizon_40_only": (40,),
    }
    for name, horizons in horizon_sets.items():
        specs.append(
            VariantSpec(name=name, family="horizon_allocation", horizons=horizons)
        )
    for risk in settings["experiments"]["risk_budget_fractions"]:
        risk = float(risk)
        if math.isclose(risk, 0.075):
            continue
        specs.append(
            VariantSpec(
                name=f"risk_budget_{risk:.3f}",
                family="risk_budget",
                depth_multiplier=12,
                risk_budget_fraction=risk,
            )
        )
    for mode in (
        "edge_linear",
        "edge_tiered",
        "horizon_weighted",
        "edge_spread_weighted",
    ):
        specs.append(
            VariantSpec(
                name=f"sizing_{mode}",
                family="dynamic_sizing",
                sizing_mode=mode,
            )
        )
    specs.extend(
        [
            VariantSpec(
                name="combined_depth5_hedge3_edge030_spread050",
                family="combined",
                hedge_interval=3,
                minimum_edge_strength=0.030,
                maximum_relative_spread=0.50,
            ),
            VariantSpec(
                name="combined_depth5_hedge3_edge050_spread050",
                family="combined",
                hedge_interval=3,
                minimum_edge_strength=0.050,
                maximum_relative_spread=0.50,
            ),
            VariantSpec(
                name="combined_depth8_hedge3_edge030_spread100_linear",
                family="combined",
                depth_multiplier=8,
                hedge_interval=3,
                minimum_edge_strength=0.030,
                maximum_relative_spread=1.00,
                sizing_mode="edge_linear",
            ),
            VariantSpec(
                name="combined_depth12_hedge5_edge030_spread050_tiered",
                family="combined",
                depth_multiplier=12,
                hedge_interval=5,
                minimum_edge_strength=0.030,
                maximum_relative_spread=0.50,
                sizing_mode="edge_tiered",
            ),
            VariantSpec(
                name="combined_20_40_depth8_hedge3_linear",
                family="combined",
                depth_multiplier=8,
                hedge_interval=3,
                horizons=(20, 40),
                sizing_mode="edge_linear",
            ),
            VariantSpec(
                name="proposed_balanced_depth5_daily_edge030_spread050_tiered",
                family="proposed_combined",
                depth_multiplier=5,
                minimum_edge_strength=0.030,
                maximum_relative_spread=0.50,
                sizing_mode="edge_tiered",
            ),
            VariantSpec(
                name="proposed_profit_depth5_hedge5_spread050_edge_spread",
                family="proposed_combined",
                depth_multiplier=5,
                hedge_interval=5,
                maximum_relative_spread=0.50,
                sizing_mode="edge_spread_weighted",
            ),
            VariantSpec(
                name="proposed_profit_20_40_depth5_hedge5_spread050_edge_spread",
                family="proposed_combined",
                depth_multiplier=5,
                hedge_interval=5,
                maximum_relative_spread=0.50,
                horizons=(20, 40),
                sizing_mode="edge_spread_weighted",
            ),
        ]
    )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate v5 variant names.")
    return specs


def _edge_strength(row: pd.Series) -> float:
    values = [-float(row["main_volatility_edge"]), -float(row["robust_volatility_edge"])]
    return float(min(values)) if all(np.isfinite(values)) else np.nan


def _sizing_multiplier(
    spec: VariantSpec,
    row: pd.Series,
    relative_spread: float,
) -> float:
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
        spread_multiplier = float(np.clip(0.25 / max(relative_spread, 0.01), 0.50, 1.25))
        return edge_multiplier * spread_multiplier
    raise ValueError(f"Unknown sizing mode: {spec.sizing_mode}")


def _combine_daily(
    pieces: list[pd.DataFrame],
    futures_daily: pd.DataFrame,
    *,
    initial_capital: float,
    strategy_name: str,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        sorted(
            futures_daily.loc[
                futures_daily["trade_date"].between(evaluation_start, evaluation_end),
                "trade_date",
            ].unique()
        )
    )
    if pieces:
        grouped = (
            pd.concat(pieces, ignore_index=True)
            .groupby("trade_date", as_index=False)["daily_net_pnl"]
            .sum()
            .set_index("trade_date")
        )
        daily = (
            grouped.reindex(calendar, fill_value=0.0)
            .rename_axis("trade_date")
            .reset_index()
        )
    else:
        daily = pd.DataFrame({"trade_date": calendar, "daily_net_pnl": 0.0})
    daily["strategy"] = strategy_name
    daily["equity"] = initial_capital + daily["daily_net_pnl"].cumsum()
    daily["return"] = daily["equity"].pct_change().fillna(
        daily["daily_net_pnl"] / initial_capital
    )
    return daily


def _simulate_spec(
    selected: pd.DataFrame,
    spec: VariantSpec,
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    quote_cache: dict[str, dict | None],
    expiry_cache: dict[str, float],
    *,
    base_settings: RealQuoteExecutionSettings,
    maximum_portfolio_risk_fraction: float,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    trades: list[dict] = []
    daily_pieces: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    active_risk: list[tuple[pd.Timestamp, float]] = []
    maximum_risk = base_settings.initial_capital * maximum_portfolio_risk_fraction
    for _, row in selected.sort_values(["entry_date", "horizon"]).iterrows():
        oid = str(row["opportunity_id"])
        entry_date = pd.Timestamp(row["entry_date"]).normalize()
        expiry_date = pd.Timestamp(row["expiry_date"]).normalize()
        active_risk = [(expiry, risk) for expiry, risk in active_risk if expiry >= entry_date]
        audit = {
            "variant": spec.name,
            "opportunity_id": oid,
            "entry_date": entry_date,
            "expiry_date": expiry_date,
            "horizon": int(row["horizon"]),
            "executed": False,
            "skip_reason": None,
        }
        if int(row["horizon"]) not in spec.horizons:
            audit["skip_reason"] = "horizon_filter"
            audit_rows.append(audit)
            continue
        edge = _edge_strength(row)
        audit["edge_strength"] = edge
        if not np.isfinite(edge) or edge < spec.minimum_edge_strength:
            audit["skip_reason"] = "edge_filter"
            audit_rows.append(audit)
            continue
        snapshot = quote_cache.get(oid)
        if snapshot is None:
            audit["skip_reason"] = "synchronized_quote_unavailable"
            audit_rows.append(audit)
            continue
        relative_spread = float(
            (snapshot["long_entry_premium"] - snapshot["short_entry_premium"])
            / max(snapshot["mid_entry_premium"], 1.0e-8)
        )
        audit["relative_spread"] = relative_spread
        if (
            spec.maximum_relative_spread is not None
            and relative_spread > spec.maximum_relative_spread
        ):
            audit["skip_reason"] = "spread_filter"
            audit_rows.append(audit)
            continue
        sizing_multiplier = _sizing_multiplier(spec, row, relative_spread)
        effective_risk_budget = min(
            spec.risk_budget_fraction * sizing_multiplier,
            maximum_portfolio_risk_fraction,
        )
        per_lot_risk = (
            base_settings.short_risk_capital_fraction_of_notional
            * float(snapshot["future_mid"])
            * float(row["volume_multiple"])
        )
        active_before = float(sum(risk for _, risk in active_risk))
        available_risk = max(maximum_risk - active_before, 0.0)
        maximum_lots = int(math.floor(available_risk / per_lot_risk))
        audit.update(
            {
                "sizing_multiplier": sizing_multiplier,
                "effective_risk_budget_fraction": effective_risk_budget,
                "active_risk_before": active_before,
                "maximum_lots_from_portfolio_cap": maximum_lots,
            }
        )
        if maximum_lots < 1:
            audit["skip_reason"] = "portfolio_risk_cap"
            audit_rows.append(audit)
            continue
        trade_settings = replace(
            base_settings,
            risk_budget_fraction_per_trade=effective_risk_budget,
        )
        simulated = simulate_v5_trade(
            row,
            option_lookup,
            futures_daily,
            ticks,
            settings=trade_settings,
            displayed_depth_multiplier=spec.depth_multiplier,
            maximum_lots=maximum_lots,
            hedge_interval_trading_days=spec.hedge_interval,
            delta_change_threshold_contracts=spec.delta_threshold,
            entry_snapshot=snapshot,
            expiry_future_quote=(
                expiry_cache[oid] if np.isfinite(expiry_cache[oid]) else None
            ),
        )
        if simulated is None:
            audit["skip_reason"] = "incomplete_trade_path"
            audit_rows.append(audit)
            continue
        trade, daily = simulated
        trade_id = f"{spec.name}_{len(trades) + 1:04d}"
        trade.update(
            {
                "strategy": spec.name,
                "trade_id": trade_id,
                "variant_family": spec.family,
                "sizing_mode": spec.sizing_mode,
                "edge_strength": edge,
                "effective_risk_budget_fraction": effective_risk_budget,
            }
        )
        daily["strategy"] = spec.name
        daily["trade_id"] = trade_id
        trades.append(trade)
        daily_pieces.append(daily)
        active_risk.append((expiry_date, float(trade["risk_capital"])))
        audit.update(
            {
                "executed": True,
                "lots": int(trade["lots"]),
                "risk_capital": float(trade["risk_capital"]),
                "net_pnl": float(trade["net_pnl"]),
            }
        )
        audit_rows.append(audit)
    trade_frame = pd.DataFrame(trades)
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
            }
        )
    metrics.update(asdict(spec))
    return trade_frame, daily_frame, metrics, pd.DataFrame(audit_rows)


def _opportunity_outcome_grid(
    selected: pd.DataFrame,
    all_trades: pd.DataFrame,
    specs: list[VariantSpec],
) -> pd.DataFrame:
    base = selected[
        ["opportunity_id", "entry_date", "expiry_date", "horizon"]
    ].copy()
    pieces = []
    for spec in specs:
        trade = all_trades[all_trades["strategy"].eq(spec.name)][
            ["opportunity_id", "net_pnl", "risk_capital", "lots"]
        ].copy()
        frame = base.merge(trade, on="opportunity_id", how="left")
        frame["variant"] = spec.name
        frame["executed"] = frame["net_pnl"].notna()
        frame[["net_pnl", "risk_capital", "lots"]] = frame[
            ["net_pnl", "risk_capital", "lots"]
        ].fillna(0.0)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def _choose_variant(
    history: pd.DataFrame,
    specs: list[VariantSpec],
    *,
    objective: str,
    minimum_executed_trades: int,
) -> tuple[VariantSpec, dict]:
    scores = []
    for spec in specs:
        data = history[history["variant"].eq(spec.name)]
        executed = int(data["executed"].sum())
        if executed < minimum_executed_trades:
            continue
        pnl = data["net_pnl"].to_numpy(dtype=float)
        net_profit = float(pnl.sum())
        standard_deviation = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else np.nan
        mean_over_std = (
            float(np.mean(pnl) / standard_deviation)
            if standard_deviation > 0
            else np.nan
        )
        if net_profit <= 0 or not np.isfinite(mean_over_std):
            continue
        primary = mean_over_std if objective == "sharpe" else net_profit
        secondary = net_profit if objective == "sharpe" else mean_over_std
        scores.append(
            {
                "spec": spec,
                "primary": primary,
                "secondary": secondary,
                "historical_net_profit": net_profit,
                "historical_mean_over_std": mean_over_std,
                "historical_executed_trades": executed,
            }
        )
    if not scores:
        baseline = next(spec for spec in specs if spec.name == "capacity_depth_1x")
        return baseline, {
            "historical_net_profit": np.nan,
            "historical_mean_over_std": np.nan,
            "historical_executed_trades": 0,
        }
    winner = max(scores, key=lambda item: (item["primary"], item["secondary"]))
    return winner.pop("spec"), winner


def _causal_selector_specs(
    selected: pd.DataFrame,
    outcome_grid: pd.DataFrame,
    specs: list[VariantSpec],
    *,
    objective: str,
    minimum_matured_signals: int,
    minimum_executed_trades: int,
) -> tuple[dict[str, VariantSpec], pd.DataFrame]:
    choices: dict[str, VariantSpec] = {}
    rows: list[dict] = []
    baseline = next(spec for spec in specs if spec.name == "capacity_depth_1x")
    for _, opportunity in selected.sort_values(["entry_date", "horizon"]).iterrows():
        entry_date = pd.Timestamp(opportunity["entry_date"]).normalize()
        matured_ids = selected[
            pd.to_datetime(selected["expiry_date"]).dt.normalize().lt(entry_date)
        ]["opportunity_id"]
        if len(matured_ids) < minimum_matured_signals:
            spec = baseline
            stats = {
                "historical_net_profit": np.nan,
                "historical_mean_over_std": np.nan,
                "historical_executed_trades": 0,
            }
            selection_state = "warmup_baseline"
        else:
            history = outcome_grid[
                outcome_grid["opportunity_id"].isin(matured_ids)
            ]
            spec, stats = _choose_variant(
                history,
                specs,
                objective=objective,
                minimum_executed_trades=minimum_executed_trades,
            )
            selection_state = "causal_optimization"
        oid = str(opportunity["opportunity_id"])
        choices[oid] = spec
        rows.append(
            {
                "opportunity_id": oid,
                "entry_date": entry_date,
                "horizon": int(opportunity["horizon"]),
                "objective": objective,
                "selected_variant": spec.name,
                "selection_state": selection_state,
                "matured_signal_count": len(matured_ids),
                **stats,
            }
        )
    return choices, pd.DataFrame(rows)


def _fixed_holdout_specs(
    selected: pd.DataFrame,
    outcome_grid: pd.DataFrame,
    specs: list[VariantSpec],
    *,
    holdout_start: pd.Timestamp,
    minimum_executed_trades: int,
) -> tuple[list[VariantSpec], pd.DataFrame]:
    """Choose once on completed pre-holdout trades, then freeze for holdout.

    This is stricter than the rolling selector: no holdout outcome can change
    the selected configuration. It remains a pseudo-holdout because the
    research menu was developed after inspecting the broader project sample.
    """
    matured_ids = selected.loc[
        pd.to_datetime(selected["expiry_date"]).dt.normalize().lt(holdout_start),
        "opportunity_id",
    ]
    history = outcome_grid[outcome_grid["opportunity_id"].isin(matured_ids)]
    frozen_specs: list[VariantSpec] = []
    rows: list[dict] = []
    for objective in ("sharpe", "net_profit"):
        winner, stats = _choose_variant(
            history,
            specs,
            objective=objective,
            minimum_executed_trades=minimum_executed_trades,
        )
        frozen = replace(
            winner,
            name=f"fixed_2026_holdout_{objective}__{winner.name}",
            family="fixed_2026_holdout",
        )
        frozen_specs.append(frozen)
        rows.append(
            {
                "holdout_start_date": holdout_start,
                "objective": objective,
                "selected_variant": winner.name,
                "pre_holdout_matured_signal_count": int(len(matured_ids)),
                **stats,
                "selection_score_note": (
                    "sharpe objective uses mean trade PnL divided by trade-PnL "
                    "standard deviation; net_profit objective uses summed PnL"
                ),
                "research_design_warning": (
                    "pseudo-holdout: holdout outcomes are excluded from selection, "
                    "but the candidate menu was designed after broader project inspection"
                ),
            }
        )
    return frozen_specs, pd.DataFrame(rows)


def _simulate_dynamic_selector(
    selected: pd.DataFrame,
    choices: dict[str, VariantSpec],
    selector_name: str,
    option_lookup: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    quote_cache: dict[str, dict | None],
    expiry_cache: dict[str, float],
    *,
    base_settings: RealQuoteExecutionSettings,
    maximum_portfolio_risk_fraction: float,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trade_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    audit_pieces: list[pd.DataFrame] = []
    # Run one opportunity at a time so the selected configuration is known at
    # that date. The single-opportunity runs cannot enforce cross-trade risk, so
    # a final conservative 30% active-risk check is applied below.
    provisional: list[tuple[pd.Series, VariantSpec, dict, pd.DataFrame]] = []
    for _, row in selected.sort_values(["entry_date", "horizon"]).iterrows():
        spec = choices[str(row["opportunity_id"])]
        trades, _, _, audit = _simulate_spec(
            pd.DataFrame([row]),
            spec,
            option_lookup,
            futures_daily,
            ticks,
            quote_cache,
            expiry_cache,
            base_settings=base_settings,
            maximum_portfolio_risk_fraction=maximum_portfolio_risk_fraction,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
        audit_pieces.append(audit)
        if trades.empty:
            continue
        # Re-simulate below if the selector's own active-risk book requires a cap.
        trade = trades.iloc[0]
        provisional.append((row, spec, trade.to_dict(), pd.DataFrame()))
    active: list[tuple[pd.Timestamp, float]] = []
    final_trades: list[dict] = []
    for row, spec, _, _ in provisional:
        entry_date = pd.Timestamp(row["entry_date"]).normalize()
        active = [(expiry, risk) for expiry, risk in active if expiry >= entry_date]
        snapshot = quote_cache[str(row["opportunity_id"])]
        per_lot_risk = (
            base_settings.short_risk_capital_fraction_of_notional
            * float(snapshot["future_mid"])
            * float(row["volume_multiple"])
        )
        maximum_risk = base_settings.initial_capital * maximum_portfolio_risk_fraction
        maximum_lots = int(
            math.floor(max(maximum_risk - sum(risk for _, risk in active), 0.0) / per_lot_risk)
        )
        if maximum_lots < 1:
            continue
        relative_spread = float(
            (snapshot["long_entry_premium"] - snapshot["short_entry_premium"])
            / max(snapshot["mid_entry_premium"], 1.0e-8)
        )
        effective_budget = min(
            spec.risk_budget_fraction * _sizing_multiplier(spec, row, relative_spread),
            maximum_portfolio_risk_fraction,
        )
        simulated = simulate_v5_trade(
            row,
            option_lookup,
            futures_daily,
            ticks,
            settings=replace(
                base_settings, risk_budget_fraction_per_trade=effective_budget
            ),
            displayed_depth_multiplier=spec.depth_multiplier,
            maximum_lots=maximum_lots,
            hedge_interval_trading_days=spec.hedge_interval,
            delta_change_threshold_contracts=spec.delta_threshold,
            entry_snapshot=snapshot,
            expiry_future_quote=(
                expiry_cache[str(row["opportunity_id"])]
                if np.isfinite(expiry_cache[str(row["opportunity_id"])] )
                else None
            ),
        )
        if simulated is None:
            continue
        trade, daily = simulated
        trade_id = f"{selector_name}_{len(final_trades) + 1:04d}"
        trade.update(
            {
                "strategy": selector_name,
                "trade_id": trade_id,
                "selected_variant": spec.name,
                "variant_family": "causal_selector",
            }
        )
        daily["strategy"] = selector_name
        daily["trade_id"] = trade_id
        final_trades.append(trade)
        daily_pieces.append(daily)
        active.append(
            (pd.Timestamp(trade["expiry_date"]).normalize(), float(trade["risk_capital"]))
        )
    trade_frame = pd.DataFrame(final_trades)
    daily_frame = _combine_daily(
        daily_pieces,
        futures_daily,
        initial_capital=base_settings.initial_capital,
        strategy_name=selector_name,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    if trade_frame.empty:
        metrics = {"strategy": selector_name, "trade_count": 0, "net_profit": 0.0}
    else:
        metrics = portfolio_metrics(
            daily_frame, trade_frame, initial_capital=base_settings.initial_capital
        )
        metrics.update(
            {
                "strategy": selector_name,
                "family": "causal_selector",
                "total_lots": int(trade_frame["lots"].sum()),
            }
        )
    return trade_frame, daily_frame, metrics


def _reality_check(outcome_grid: pd.DataFrame, specs: list[VariantSpec]) -> pd.DataFrame:
    matrix = outcome_grid.pivot(
        index="opportunity_id", columns="variant", values="net_pnl"
    ).fillna(0.0)
    baseline = matrix["capacity_depth_1x"].to_numpy(dtype=float)
    alternatives = [spec.name for spec in specs if spec.name != "capacity_depth_1x"]
    differences = matrix[alternatives].to_numpy(dtype=float) - baseline[:, None]
    observed_improvements = differences.sum(axis=0)
    best_index = int(np.argmax(observed_improvements))
    observed_best = float(observed_improvements[best_index])
    centered = differences - differences.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(20260830)
    bootstrap_max = np.empty(20000, dtype=float)
    for sample in range(len(bootstrap_max)):
        indices = rng.integers(0, len(centered), size=len(centered))
        bootstrap_max[sample] = centered[indices].sum(axis=0).max()
    return pd.DataFrame(
        [
            {
                "baseline": "capacity_depth_1x",
                "best_in_sample_variant": alternatives[best_index],
                "observed_net_profit_improvement": observed_best,
                "white_reality_check_p_value": float(
                    np.mean(bootstrap_max >= observed_best)
                ),
                "variant_count_in_familywise_test": len(alternatives),
                "signal_count": len(matrix),
            }
        ]
    )


def _write_manifest(
    settings: dict,
    output_dir: Path,
    outputs: dict[str, Path],
    sources: list[Path],
) -> Path:
    root = Path(settings["_project_root"])
    paths = [Path(settings["_config_path"]), *sources, *outputs.values()]
    manifest = {
        "strategy_version": settings["project"]["version"],
        "created_at_utc": utc_now(),
        "parent_versions_untouched": [
            "strategy_v1",
            "strategy_v2",
            "strategy_v3",
            "strategy_v4",
        ],
        "research_warning": "many variants tested on only 21 matured signals",
        "files": {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path): _sha256(path)
            for path in paths
            if path.exists() and path.is_file()
        },
    }
    return write_json(
        manifest,
        output_dir / f"{settings['project']['version']}_freeze_manifest.json",
    )


def run_strategy_v5_backtest(settings: dict) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    output_dir = root / settings["outputs"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities_path = root / settings["data"]["opportunities_path"]
    option_path = root / settings["data"]["option_daily_path"]
    futures_path = root / settings["data"]["futures_daily_path"]
    ticks_path = root / settings["data"]["raw_dir"] / "real_quote_ticks.parquet"
    opportunities = prepare_real_quote_opportunities(read_parquet(opportunities_path))
    selected = opportunities[opportunities["signal_model_timed_short"].eq(-1)].copy()
    option_daily = read_parquet(option_path)
    futures_daily = read_parquet(futures_path)
    ticks = load_real_quote_ticks(root, str(settings["data"]["raw_dir"]))
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    option_lookup = option_daily.copy().set_index(["trade_date", "ts_code"])
    base_settings = _execution_settings(settings)
    quote_cache = {
        str(row["opportunity_id"]): quote_snapshot_features(
            ticks,
            row,
            maximum_quote_age_seconds=base_settings.quote_maximum_age_seconds,
        )
        for _, row in selected.iterrows()
    }
    expiry_cache = {
        oid: expiry_future_price(ticks, oid) for oid in quote_cache
    }
    # The last v3 signal is still open on the research cutoff date. Common
    # evaluation dates are inferred from completed futures paths.
    completed = selected[
        selected["expiry_date"].le(futures_daily["trade_date"].max())
    ].copy()
    evaluation_start = pd.Timestamp(completed["entry_date"].min()).normalize()
    evaluation_end = pd.Timestamp(completed["expiry_date"].max()).normalize()
    specs = build_variant_catalog(settings)
    trade_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    audit_pieces: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for spec in specs:
        trades, daily, metrics, audit = _simulate_spec(
            completed,
            spec,
            option_lookup,
            futures_daily,
            ticks,
            quote_cache,
            expiry_cache,
            base_settings=base_settings,
            maximum_portfolio_risk_fraction=float(
                settings["execution"]["maximum_portfolio_risk_fraction"]
            ),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
        if not trades.empty:
            trade_pieces.append(trades)
        daily_pieces.append(daily)
        audit_pieces.append(audit)
        metric_rows.append(metrics)
    all_trades = pd.concat(trade_pieces, ignore_index=True) if trade_pieces else pd.DataFrame()
    all_daily = pd.concat(daily_pieces, ignore_index=True)
    all_audit = pd.concat(audit_pieces, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    outcome_grid = _opportunity_outcome_grid(completed, all_trades, specs)
    selection_rows: list[pd.DataFrame] = []
    for objective in ("sharpe", "net_profit"):
        choices, selection_history = _causal_selector_specs(
            completed,
            outcome_grid,
            specs,
            objective=objective,
            minimum_matured_signals=int(
                settings["experiments"]["causal_selector_minimum_matured_signals"]
            ),
            minimum_executed_trades=int(
                settings["experiments"]["causal_selector_minimum_executed_trades"]
            ),
        )
        selection_rows.append(selection_history)
        selector_name = f"causal_selector_{objective}"
        selector_trades, selector_daily, selector_metrics = _simulate_dynamic_selector(
            completed,
            choices,
            selector_name,
            option_lookup,
            futures_daily,
            ticks,
            quote_cache,
            expiry_cache,
            base_settings=base_settings,
            maximum_portfolio_risk_fraction=float(
                settings["execution"]["maximum_portfolio_risk_fraction"]
            ),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
        if not selector_trades.empty:
            all_trades = pd.concat([all_trades, selector_trades], ignore_index=True)
        all_daily = pd.concat([all_daily, selector_daily], ignore_index=True)
        metric_rows.append(selector_metrics)
    selection_history = pd.concat(selection_rows, ignore_index=True)
    holdout_start = pd.Timestamp(
        settings["experiments"]["holdout_start_date"]
    ).normalize()
    holdout_specs, holdout_selection = _fixed_holdout_specs(
        completed,
        outcome_grid,
        specs,
        holdout_start=holdout_start,
        minimum_executed_trades=int(
            settings["experiments"]["causal_selector_minimum_executed_trades"]
        ),
    )
    holdout_opportunities = completed[
        pd.to_datetime(completed["entry_date"]).dt.normalize().ge(holdout_start)
    ].copy()
    holdout_diagnostic_rows: list[dict] = []
    if not holdout_opportunities.empty:
        holdout_end = pd.Timestamp(
            holdout_opportunities["expiry_date"].max()
        ).normalize()
        for spec in holdout_specs:
            trades, daily, holdout_metrics, audit = _simulate_spec(
                holdout_opportunities,
                spec,
                option_lookup,
                futures_daily,
                ticks,
                quote_cache,
                expiry_cache,
                base_settings=base_settings,
                maximum_portfolio_risk_fraction=float(
                    settings["execution"]["maximum_portfolio_risk_fraction"]
                ),
                evaluation_start=holdout_start,
                evaluation_end=holdout_end,
            )
            if not trades.empty:
                all_trades = pd.concat([all_trades, trades], ignore_index=True)
            all_daily = pd.concat([all_daily, daily], ignore_index=True)
            all_audit = pd.concat([all_audit, audit], ignore_index=True)
            metric_rows.append(holdout_metrics)
        spec_lookup = {spec.name: spec for spec in specs}
        diagnostic_names = (
            "capacity_depth_1x",
            "capacity_depth_5x",
            "capacity_depth_8x",
            "hedge_every_5d",
            "edge_min_0.030",
            "spread_max_0.25",
            "spread_max_0.50",
            "horizons_20_40",
            "sizing_edge_spread_weighted",
            "proposed_balanced_depth5_daily_edge030_spread050_tiered",
            "proposed_profit_depth5_hedge5_spread050_edge_spread",
            "proposed_profit_20_40_depth5_hedge5_spread050_edge_spread",
        )
        for original_name in diagnostic_names:
            original = spec_lookup[original_name]
            diagnostic = replace(
                original,
                name=f"diagnostic_2026__{original.name}",
                family="diagnostic_2026_ex_post",
            )
            trades, daily, diagnostic_metrics, audit = _simulate_spec(
                holdout_opportunities,
                diagnostic,
                option_lookup,
                futures_daily,
                ticks,
                quote_cache,
                expiry_cache,
                base_settings=base_settings,
                maximum_portfolio_risk_fraction=float(
                    settings["execution"]["maximum_portfolio_risk_fraction"]
                ),
                evaluation_start=holdout_start,
                evaluation_end=holdout_end,
            )
            diagnostic_metrics["original_variant"] = original_name
            diagnostic_metrics["research_design_warning"] = (
                "ex-post 2026 slice for regime diagnosis only; not a clean holdout"
            )
            holdout_diagnostic_rows.append(diagnostic_metrics)
            if not trades.empty:
                all_trades = pd.concat([all_trades, trades], ignore_index=True)
            all_daily = pd.concat([all_daily, daily], ignore_index=True)
            all_audit = pd.concat([all_audit, audit], ignore_index=True)
            metric_rows.append(diagnostic_metrics)
    metrics = pd.DataFrame(metric_rows)
    reality_check = _reality_check(outcome_grid, specs)
    breakdown = trade_breakdown(all_trades) if not all_trades.empty else pd.DataFrame()
    catalog = pd.DataFrame([asdict(spec) for spec in specs])
    outputs = {
        "variant_catalog": write_csv(catalog, output_dir / "variant_catalog.csv"),
        "trades": write_csv(all_trades, output_dir / "trades.csv"),
        "daily": write_csv(all_daily, output_dir / "daily.csv"),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
        "execution_audit": write_csv(all_audit, output_dir / "execution_audit.csv"),
        "opportunity_outcome_grid": write_csv(
            outcome_grid, output_dir / "opportunity_outcome_grid.csv"
        ),
        "causal_selection_history": write_csv(
            selection_history, output_dir / "causal_selection_history.csv"
        ),
        "holdout_selection": write_csv(
            holdout_selection, output_dir / "holdout_selection.csv"
        ),
        "holdout_diagnostics": write_csv(
            pd.DataFrame(holdout_diagnostic_rows),
            output_dir / "holdout_diagnostic_metrics.csv",
        ),
        "reality_check": write_csv(
            reality_check, output_dir / "multiple_testing_reality_check.csv"
        ),
    }
    outputs["freeze_manifest"] = _write_manifest(
        settings,
        output_dir,
        outputs,
        [
            opportunities_path,
            option_path,
            futures_path,
            ticks_path,
            root / "src/au_rv/strategy_v5/execution.py",
            root / "src/au_rv/strategy_v5/backtest.py",
        ],
    )
    return outputs
