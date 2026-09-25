from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.config import resolve_path
from au_rv.io import read_parquet, write_csv, write_parquet
from au_rv.strategy.volatility_straddle import (
    CostAssumptions,
    portfolio_metrics,
    simulate_delta_hedged_trade,
    trade_breakdown,
)
from au_rv.strategy_v2.data import load_strategy_v2_market_data, opportunity_id
from au_rv.strategy_v2.execution import (
    V2RiskSettings,
    protected_tail_stress,
    signal_close_mid_iv,
    simulate_protected_trade,
)
from au_rv.strategy_v2.models import (
    add_causal_forecast_bounds,
    add_causal_pnl_forecasts,
    add_causal_tail_forecasts,
    add_v2_signals,
    build_semivariance_daily,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_enhanced_opportunities(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(config["_project_root"])
    settings = config["strategy_v2"]
    opportunities = read_parquet(
        resolve_path(config, config["strategy"]["opportunities_path"])
    )
    forecasts = read_parquet(resolve_path(config, config["outputs"]["strategy_forecasts_path"]))
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    bars = read_parquet(root / "data/raw/tqsdk_au_5m.parquet")
    calendar = read_parquet(root / "data/raw/tqsdk_shfe_trade_calendar.parquet")
    enhanced = add_causal_forecast_bounds(
        opportunities,
        forecasts,
        features,
        quantile=float(settings["forecast_upper_quantile"]),
        residual_window=int(settings["forecast_residual_window"]),
        minimum_samples=int(settings["minimum_forecast_residual_samples"]),
    )
    semivariance = build_semivariance_daily(bars, calendar, features)
    enhanced = add_causal_tail_forecasts(
        enhanced,
        semivariance,
        minimum_samples=int(settings["tail_model_minimum_samples"]),
        ridge_alpha=float(settings["tail_model_ridge_alpha"]),
    )
    enhanced["normalized_short_pnl_target"] = np.nan
    option_daily = read_parquet(root / "data/raw/tqsdk_au_option_daily_strategy.parquet")
    futures_daily = read_parquet(root / "data/raw/tqsdk_au_futures_daily_all_contracts.parquet")
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    target_costs = CostAssumptions(
        option_commission_per_contract_side=5.0,
        option_exercise_fee_per_contract=5.0,
        futures_commission_per_contract_side=5.0,
        option_slippage_ticks_per_side=1.0,
        futures_slippage_ticks_per_trade=1.0,
    )
    for index, row in enhanced.iterrows():
        try:
            simulated = simulate_delta_hedged_trade(
                row,
                -1,
                option_daily,
                futures_daily,
                initial_capital=10_000_000.0,
                risk_budget_fraction=0.001,
                risk_capital_fraction_of_notional=0.20,
                risk_free_rate=float(config["strategy"]["risk_free_rate"]),
                costs=target_costs,
            )
        except (OverflowError, ValueError):
            # Some catalog opportunities lack a finite entry IV or complete
            # daily hedge path. They are unavailable targets, not zero P&L.
            continue
        if simulated is None:
            continue
        trade, _ = simulated
        enhanced.loc[index, "normalized_short_pnl_target"] = (
            float(trade["net_pnl"])
            / int(trade["lots"])
            / (float(trade["entry_future_open"]) * float(trade["volume_multiple"]))
        )
    enhanced = add_causal_pnl_forecasts(
        enhanced,
        minimum_samples=int(settings["pnl_model_minimum_samples"]),
        ridge_alpha=float(settings["pnl_model_ridge_alpha"]),
        residual_quantile=float(settings["pnl_residual_quantile"]),
    )
    enhanced = add_v2_signals(
        enhanced,
        minimum_volatility_edge=float(settings["minimum_volatility_edge"]),
        maximum_tail_imbalance=float(settings["maximum_tail_imbalance"]),
        maximum_jump_share=float(settings["maximum_jump_share"]),
    )
    enhanced["opportunity_id"] = [opportunity_id(row) for _, row in enhanced.iterrows()]
    return enhanced, semivariance


def _combine_daily(
    pieces: list[pd.DataFrame],
    futures_daily: pd.DataFrame,
    *,
    initial_capital: float,
    strategy_name: str,
) -> pd.DataFrame:
    trade_daily = pd.concat(pieces, ignore_index=True)
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
        trade_daily.groupby("trade_date", as_index=False)
        .agg(daily_net_pnl=("daily_net_pnl", "sum"), position_margin=("position_margin", "sum"))
        .set_index("trade_date")
        .reindex(calendar, fill_value=0.0)
        .rename_axis("trade_date")
        .reset_index()
    )
    daily["strategy"] = strategy_name
    daily["equity"] = initial_capital + daily["daily_net_pnl"].cumsum()
    daily["return"] = daily["equity"].pct_change().fillna(daily["daily_net_pnl"] / initial_capital)
    daily["margin_fraction"] = daily["position_margin"] / daily["equity"].clip(lower=1.0)
    return daily


def _comparable_naked_trades(
    selected: pd.DataFrame,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old = config["strategy"]
    costs = CostAssumptions(
        option_commission_per_contract_side=float(old["costs"]["option_commission_per_contract_side"]),
        option_exercise_fee_per_contract=float(old["costs"]["option_exercise_fee_per_contract"]),
        futures_commission_per_contract_side=float(old["costs"]["futures_commission_per_contract_side"]),
        option_slippage_ticks_per_side=float(old["costs"]["option_slippage_ticks_per_side"]),
        futures_slippage_ticks_per_trade=float(old["costs"]["futures_slippage_ticks_per_trade"]),
    )
    trades = []
    daily_pieces = []
    for _, row in selected.iterrows():
        simulated = simulate_delta_hedged_trade(
            row,
            -1,
            option_daily,
            futures_daily,
            initial_capital=float(old["initial_capital"]),
            risk_budget_fraction=float(old["risk_budget_fraction_per_trade"]),
            risk_capital_fraction_of_notional=float(old["risk_capital_fraction_of_notional"]),
            risk_free_rate=float(old["risk_free_rate"]),
            costs=costs,
        )
        if simulated is None:
            continue
        trade, daily = simulated
        trade["strategy"] = "v2_dates_naked_v1_execution"
        trade["trade_id"] = f"v2_dates_naked_{len(trades) + 1:04d}"
        daily["strategy"] = trade["strategy"]
        daily["trade_id"] = trade["trade_id"]
        daily["position_margin"] = 0.0
        trades.append(trade)
        daily_pieces.append(daily)
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return trade_frame, pd.DataFrame(), {}
    daily = _combine_daily(
        daily_pieces,
        futures_daily,
        initial_capital=float(old["initial_capital"]),
        strategy_name="v2_dates_naked_v1_execution",
    )
    metrics = portfolio_metrics(daily, trade_frame, initial_capital=float(old["initial_capital"]))
    metrics["strategy"] = "v2_dates_naked_v1_execution"
    return trade_frame, daily, metrics


def _protected_variant(
    selected: pd.DataFrame,
    option_daily: pd.DataFrame,
    wing_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    settings: V2RiskSettings,
    variant: str,
    entry_window_preference: tuple[str, ...] = ("entry_night", "entry_day"),
) -> tuple[dict, pd.DataFrame]:
    trades: list[dict] = []
    daily_pieces: list[pd.DataFrame] = []
    for _, row in selected.sort_values("entry_date").iterrows():
        simulated = simulate_protected_trade(
            row,
            option_daily,
            wing_daily,
            futures_daily,
            ticks,
            settings=settings,
            entry_window_preference=entry_window_preference,
        )
        if simulated is None:
            continue
        trade, daily = simulated
        trade["variant"] = variant
        trades.append(trade)
        daily["variant"] = variant
        daily_pieces.append(daily)
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return {"variant": variant, "trade_count": 0}, trade_frame
    daily = _combine_daily(
        daily_pieces,
        futures_daily,
        initial_capital=settings.initial_capital,
        strategy_name=variant,
    )
    metrics = portfolio_metrics(daily, trade_frame, initial_capital=settings.initial_capital)
    metrics.update(
        {
            "variant": variant,
            "maximum_reconstructed_margin": float(daily["position_margin"].max()),
            "maximum_reconstructed_margin_fraction": float(daily["margin_fraction"].max()),
            "embedded_entry_crossing_cost": float(
                trade_frame["embedded_entry_crossing_cost"].sum()
            ),
        }
    )
    return metrics, trade_frame


def _write_freeze_manifest(config: dict, output_dir: Path) -> Path:
    root = Path(config["_project_root"])
    version = str(config["strategy_v2"]["strategy_version"])
    manifest_path = output_dir / f"{version}_freeze_manifest.json"
    tracked = [
        root / "config.yaml",
        root / "src/au_rv/strategy_v2/data.py",
        root / "src/au_rv/strategy_v2/pricing.py",
        root / "src/au_rv/strategy_v2/models.py",
        root / "src/au_rv/strategy_v2/execution.py",
        root / "src/au_rv/strategy_v2/backtest.py",
        resolve_path(config, config["outputs"]["strategy_forecasts_path"]),
        root / "data/raw/strategy_v2/strategy_v2_candidates.parquet",
        root / "data/raw/strategy_v2/strategy_v2_wing_daily.parquet",
        root / "data/raw/strategy_v2/strategy_v2_execution_ticks.parquet",
    ]
    manifest = {
        "strategy_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_effective_trade_date": str(config["strategy_v2"]["forward_effective_trade_date"]),
        "historical_research_data_cutoff": "2026-08-24",
        "rule": "Never revise this manifest or its files when evaluating forward observations.",
        "files": {str(path.relative_to(root)): _sha256(path) for path in tracked},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("files") != manifest["files"]:
            raise RuntimeError(
                "Frozen strategy-v2 files changed. Create a new strategy_version; "
                "do not silently rewrite a forward-test specification."
            )
        return manifest_path
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run_strategy_v2_backtest(config: dict) -> dict[str, Path]:
    root = Path(config["_project_root"])
    settings = config["strategy_v2"]
    output_dir = resolve_path(config, settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced, semivariance = build_enhanced_opportunities(config)
    raw_candidates, wing_daily, ticks = load_strategy_v2_market_data(root)
    wing_columns = [
        "opportunity_id",
        "put_wing_symbol",
        "put_wing_strike",
        "call_wing_symbol",
        "call_wing_strike",
        "put_wing_width",
        "call_wing_width",
    ]
    enhanced = enhanced.merge(
        raw_candidates[wing_columns].drop_duplicates("opportunity_id"),
        on="opportunity_id",
        how="left",
    )
    enhanced["signal_tick_iv"] = np.nan
    for index, row in enhanced[enhanced["put_wing_symbol"].notna()].iterrows():
        enhanced.loc[index, "signal_tick_iv"] = signal_close_mid_iv(
            ticks,
            row,
            risk_free_rate=float(settings["risk_free_rate"]),
        )
    tick_bound = (
        enhanced["signal_tick_iv"].sub(enhanced["main_upper_fair_volatility"])
        .ge(float(settings["minimum_volatility_edge"]))
        & enhanced["signal_tick_iv"].sub(enhanced["robust_upper_fair_volatility"])
        .ge(float(settings["minimum_volatility_edge"]))
    )
    enhanced["signal_tick_bound_pass"] = tick_bound
    enhanced["signal_v2_short"] = enhanced["signal_v2_short"] & tick_bound

    option_daily = read_parquet(root / "data/raw/tqsdk_au_option_daily_strategy.parquet")
    futures_daily = read_parquet(root / "data/raw/tqsdk_au_futures_daily_all_contracts.parquet")
    for frame in (option_daily, wing_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    risk = V2RiskSettings(
        initial_capital=float(settings["initial_capital"]),
        max_loss_fraction_per_trade=float(settings["max_loss_fraction_per_trade"]),
        max_margin_fraction_per_expiry=float(settings["max_margin_fraction_per_expiry"]),
        max_portfolio_margin_fraction=float(settings["max_portfolio_margin_fraction"]),
        delta_band_per_option=float(settings["delta_band_per_option"]),
        risk_free_rate=float(settings["risk_free_rate"]),
        option_exchange_fee=float(settings["costs"]["option_exchange_fee"]),
        option_broker_fee=float(settings["costs"]["option_broker_fee"]),
        option_exercise_exchange_fee=float(settings["costs"]["option_exercise_exchange_fee"]),
        futures_broker_fee=float(settings["costs"]["futures_broker_fee"]),
        futures_hedge_slippage_ticks=float(settings["costs"]["futures_hedge_slippage_ticks"]),
    )
    trade_rows = []
    daily_pieces = []
    selected = enhanced[enhanced["signal_v2_short"]].sort_values("entry_date")
    for _, row in selected.iterrows():
        simulated = simulate_protected_trade(
            row,
            option_daily,
            wing_daily,
            futures_daily,
            ticks,
            settings=risk,
        )
        if simulated is None:
            continue
        trade, daily = simulated
        if daily_pieces:
            existing_margin = (
                pd.concat(daily_pieces, ignore_index=True)
                .groupby("trade_date")["position_margin"]
                .sum()
            )
            proposed = daily.set_index("trade_date")["position_margin"]
            used = existing_margin.reindex(proposed.index, fill_value=0.0)
            per_lot = proposed / max(int(trade["lots"]), 1)
            available = (
                risk.initial_capital * risk.max_portfolio_margin_fraction - used
            ).clip(lower=0.0)
            positive = per_lot.gt(0.0)
            if positive.any():
                portfolio_lot_cap = int(
                    np.floor((available[positive] / per_lot[positive]).min())
                )
                if portfolio_lot_cap < int(trade["lots"]):
                    simulated = simulate_protected_trade(
                        row,
                        option_daily,
                        wing_daily,
                        futures_daily,
                        ticks,
                        settings=risk,
                        maximum_lots=portfolio_lot_cap,
                    )
                    if simulated is None:
                        continue
                    trade, daily = simulated
        trade["strategy"] = "strategy_v2_protected"
        trade["trade_id"] = f"strategy_v2_protected_{len(trade_rows) + 1:04d}"
        daily["strategy"] = trade["strategy"]
        daily["trade_id"] = trade["trade_id"]
        trade_rows.append(trade)
        daily_pieces.append(daily)
    protected_trades = pd.DataFrame(trade_rows)
    if protected_trades.empty:
        protected_daily = pd.DataFrame()
        protected_metrics = {"strategy": "strategy_v2_protected"}
    else:
        protected_daily = _combine_daily(
            daily_pieces,
            futures_daily,
            initial_capital=risk.initial_capital,
            strategy_name="strategy_v2_protected",
        )
        protected_metrics = portfolio_metrics(
            protected_daily, protected_trades, initial_capital=risk.initial_capital
        )
        protected_metrics.update(
            {
                "strategy": "strategy_v2_protected",
                "maximum_reconstructed_margin": float(protected_daily["position_margin"].max()),
                "maximum_reconstructed_margin_fraction": float(protected_daily["margin_fraction"].max()),
                "maximum_defined_option_loss": float(
                    (protected_trades["maximum_loss_per_lot"] * protected_trades["lots"]).max()
                ),
                "embedded_entry_crossing_cost": float(
                    protected_trades["embedded_entry_crossing_cost"].sum()
                ),
            }
        )
    naked_trades, naked_daily, naked_metrics = _comparable_naked_trades(
        selected, option_daily, futures_daily, config
    )
    old_metrics = pd.read_csv(resolve_path(config, config["strategy"]["metrics_path"]))
    old_primary = old_metrics[old_metrics["strategy"].eq("model_timed_short")].copy()
    if not old_primary.empty:
        old_primary.loc[:, "strategy"] = "strategy_v1_original"
    metric_rows = [protected_metrics]
    if naked_metrics:
        metric_rows.append(naked_metrics)
    metrics = pd.concat([pd.DataFrame(metric_rows), old_primary], ignore_index=True, sort=False)
    all_trades = pd.concat([protected_trades, naked_trades], ignore_index=True, sort=False)
    all_daily = pd.concat([protected_daily, naked_daily], ignore_index=True, sort=False)
    breakdown = trade_breakdown(all_trades) if not all_trades.empty else pd.DataFrame()
    stress = protected_tail_stress(
        protected_trades,
        initial_capital=risk.initial_capital,
        risk_free_rate=risk.risk_free_rate,
    )
    sensitivity_rows: list[dict] = []
    sensitivity_trade_pieces: list[pd.DataFrame] = []
    variants: list[tuple[str, pd.DataFrame, V2RiskSettings, tuple[str, ...]]] = [
        ("primary_top_book_night_first", selected, risk, ("entry_night", "entry_day")),
        ("entry_day_only", selected, risk, ("entry_day",)),
        (
            "distribution_without_pnl_gate",
            enhanced[
                enhanced["signal_distribution_short"]
                & enhanced["signal_tick_bound_pass"]
            ],
            risk,
            ("entry_night", "entry_day"),
        ),
        (
            "no_daily_rehedge_after_entry",
            selected,
            replace(risk, delta_band_per_option=100.0),
            ("entry_night", "entry_day"),
        ),
    ]
    for multiplier_value in (0.0, 2.0, 3.0):
        variants.append(
            (
                f"cost_{multiplier_value:.0f}x_top_book",
                selected,
                replace(
                    risk,
                    option_exchange_fee=risk.option_exchange_fee * multiplier_value,
                    option_broker_fee=risk.option_broker_fee * multiplier_value,
                    option_exercise_exchange_fee=(
                        risk.option_exercise_exchange_fee * multiplier_value
                    ),
                    futures_broker_fee=risk.futures_broker_fee * multiplier_value,
                    futures_hedge_slippage_ticks=(
                        risk.futures_hedge_slippage_ticks * multiplier_value
                    ),
                ),
                ("entry_night", "entry_day"),
            )
        )
    for variant_name, variant_selected, variant_risk, window_preference in variants:
        variant_metrics, variant_trades = _protected_variant(
            variant_selected,
            option_daily,
            wing_daily,
            futures_daily,
            ticks,
            settings=variant_risk,
            variant=variant_name,
            entry_window_preference=window_preference,
        )
        sensitivity_rows.append(variant_metrics)
        if not variant_trades.empty:
            sensitivity_trade_pieces.append(variant_trades)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_trades = (
        pd.concat(sensitivity_trade_pieces, ignore_index=True, sort=False)
        if sensitivity_trade_pieces
        else pd.DataFrame()
    )
    forecasts_for_bounds = read_parquet(
        resolve_path(config, config["outputs"]["strategy_forecasts_path"])
    )
    features_for_bounds = read_parquet(
        resolve_path(config, config["outputs"]["features_path"])
    )
    bound_sensitivity_rows: list[dict] = []
    for quantile in (0.75, 0.80, 0.85, 0.90, 0.95):
        bounded = add_causal_forecast_bounds(
            enhanced,
            forecasts_for_bounds,
            features_for_bounds,
            quantile=quantile,
            residual_window=int(settings["forecast_residual_window"]),
            minimum_samples=int(settings["minimum_forecast_residual_samples"]),
        )
        bounded = add_v2_signals(
            bounded,
            minimum_volatility_edge=float(settings["minimum_volatility_edge"]),
            maximum_tail_imbalance=float(settings["maximum_tail_imbalance"]),
            maximum_jump_share=float(settings["maximum_jump_share"]),
        )
        bound_sensitivity_rows.append(
            {
                "forecast_upper_quantile": quantile,
                "original_timed_short_count": int(
                    bounded["signal_model_timed_short"].eq(-1).sum()
                ),
                "distribution_gate_count_using_daily_close_iv": int(
                    bounded["signal_distribution_short"].sum()
                ),
                "pnl_gate_count_using_daily_close_iv": int(
                    bounded["signal_v2_short"].sum()
                ),
                "historical_tick_execution_retested": quantile == 0.90,
            }
        )
    bound_sensitivity = pd.DataFrame(bound_sensitivity_rows)
    forward_tracker = output_dir / "strategy_v2_forward_tracker.csv"
    if not forward_tracker.exists():
        pd.DataFrame(
            columns=[
                "trade_date",
                "model_version",
                "signal_status",
                "entry_status",
                "maturity_status",
                "realized_net_pnl",
                "notes",
            ]
        ).to_csv(forward_tracker, index=False)
    outputs = {
        "enhanced_opportunities": write_parquet(enhanced, output_dir / "enhanced_opportunities.parquet"),
        "semivariance_daily": write_parquet(semivariance, output_dir / "semivariance_daily.parquet"),
        "trades": write_csv(all_trades, output_dir / "trades.csv"),
        "daily": write_csv(all_daily, output_dir / "daily.csv"),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
        "tail_stress": write_csv(stress, output_dir / "tail_stress.csv"),
        "sensitivity": write_csv(sensitivity, output_dir / "sensitivity.csv"),
        "sensitivity_trades": write_csv(
            sensitivity_trades, output_dir / "sensitivity_trades.csv"
        ),
        "forecast_bound_sensitivity": write_csv(
            bound_sensitivity, output_dir / "forecast_bound_sensitivity.csv"
        ),
        "forward_tracker": forward_tracker,
        "freeze_manifest": _write_freeze_manifest(config, output_dir),
    }
    return outputs
