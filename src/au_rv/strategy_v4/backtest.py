from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.io import read_parquet, utc_now, write_csv, write_json, write_parquet
from au_rv.strategy.volatility_straddle import portfolio_metrics, trade_breakdown
from au_rv.strategy_v3.backtest import (
    combine_trade_daily,
    real_quote_execution_settings,
)
from au_rv.strategy_v3.data import load_real_quote_ticks, prepare_real_quote_opportunities
from au_rv.strategy_v3.execution import (
    expiry_future_price,
    quote_snapshot_features,
    simulate_naked_real_quote_trade,
)
from au_rv.strategy_v4.models import (
    add_causal_direct_pnl_predictions,
    add_causal_policy_decisions,
    direct_pnl_prediction_metrics,
)
from au_rv.strategy_v4.targets import build_direct_pnl_targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _depth_labels(settings: dict) -> list[str]:
    labels = [
        f"depth_{int(value)}x"
        for value in settings["v4"]["displayed_depth_multipliers"]
    ]
    if bool(settings["v4"]["include_unlimited_depth_sensitivity"]):
        labels.append("unlimited")
    return labels


def _depth_multiplier(depth_label: str) -> int | None:
    if depth_label == "unlimited":
        return None
    return int(depth_label.removeprefix("depth_").removesuffix("x"))


def _run_decision_variant(
    opportunities: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    execution_settings,
    objective: str,
    depth_label: str,
    maximum_portfolio_risk_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    strategy_name = f"v4_direct_pnl_{objective}_{depth_label}"
    chosen = decisions[
        decisions["objective"].eq(objective)
        & decisions["depth_label"].eq(depth_label)
        & decisions["trade"].eq(True)
    ].copy()
    if chosen.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"strategy": strategy_name, "trade_count": 0},
            chosen,
        )
    chosen = chosen.sort_values(
        ["entry_date", "predicted_sized_pnl"], ascending=[True, False]
    )
    opportunity_lookup = opportunities.set_index("opportunity_id", drop=False)
    prediction_lookup = predictions.set_index(["opportunity_id", "side"])
    option_lookup = option_daily.copy()
    option_lookup["trade_date"] = pd.to_datetime(
        option_lookup["trade_date"]
    ).dt.normalize()
    option_lookup = option_lookup.set_index(["trade_date", "ts_code"])
    quote_cache: dict[str, dict | None] = {}
    expiry_cache: dict[str, float] = {}
    active_risk: list[tuple[pd.Timestamp, float]] = []
    trade_rows: list[dict] = []
    daily_pieces: list[pd.DataFrame] = []
    execution_rows: list[dict] = []
    maximum_risk = execution_settings.initial_capital * maximum_portfolio_risk_fraction
    for decision in chosen.itertuples(index=False):
        oid = str(decision.opportunity_id)
        entry_date = pd.Timestamp(decision.entry_date).normalize()
        active_risk = [
            (expiry, risk) for expiry, risk in active_risk if expiry >= entry_date
        ]
        active_before = float(sum(risk for _, risk in active_risk))
        available_risk = max(maximum_risk - active_before, 0.0)
        side = int(decision.selected_side)
        prediction_row = prediction_lookup.loc[(oid, side)]
        if isinstance(prediction_row, pd.DataFrame):
            prediction_row = prediction_row.iloc[-1]
        per_lot_risk = float(prediction_row["per_lot_risk_capital"])
        maximum_lots = int(math.floor(available_risk / per_lot_risk))
        execution_record = {
            "opportunity_id": oid,
            "strategy": strategy_name,
            "entry_date": entry_date,
            "expiry_date": decision.expiry_date,
            "selected_side": side,
            "predicted_per_lot_pnl": decision.predicted_per_lot_pnl,
            "predicted_sized_pnl": decision.predicted_sized_pnl,
            "active_risk_before": active_before,
            "available_portfolio_risk": available_risk,
            "maximum_lots_from_portfolio_cap": maximum_lots,
            "executed": False,
            "skip_reason": None,
        }
        if maximum_lots < 1:
            execution_record["skip_reason"] = "portfolio_risk_cap"
            execution_rows.append(execution_record)
            continue
        row = opportunity_lookup.loc[oid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if oid not in quote_cache:
            quote_cache[oid] = quote_snapshot_features(
                ticks,
                row,
                maximum_quote_age_seconds=execution_settings.quote_maximum_age_seconds,
            )
            expiry_cache[oid] = expiry_future_price(ticks, oid)
        simulated = simulate_naked_real_quote_trade(
            row,
            side,
            option_lookup,
            futures_daily,
            ticks,
            settings=execution_settings,
            displayed_depth_multiplier=_depth_multiplier(depth_label),
            maximum_lots=maximum_lots,
            entry_snapshot=quote_cache[oid],
            expiry_future_quote=(
                expiry_cache[oid] if np.isfinite(expiry_cache[oid]) else None
            ),
        )
        if simulated is None:
            execution_record["skip_reason"] = "incomplete_trade_path_or_quote"
            execution_rows.append(execution_record)
            continue
        trade, daily = simulated
        trade_id = f"{strategy_name}_{len(trade_rows) + 1:04d}"
        trade.update(
            {
                "strategy": strategy_name,
                "trade_id": trade_id,
                "optimization_objective": objective,
                "depth_label": depth_label,
                "selected_model_family": decision.model_family,
                "selected_threshold": decision.threshold,
                "predicted_per_lot_pnl": decision.predicted_per_lot_pnl,
                "predicted_sized_pnl_before_portfolio_cap": (
                    decision.predicted_sized_pnl
                ),
                "historical_policy_net_profit": decision.historical_net_profit,
                "historical_policy_trade_mean_over_std": (
                    decision.historical_trade_mean_over_std
                ),
            }
        )
        daily["strategy"] = strategy_name
        daily["trade_id"] = trade_id
        trade_rows.append(trade)
        daily_pieces.append(daily)
        active_risk.append((pd.Timestamp(trade["expiry_date"]).normalize(), trade["risk_capital"]))
        execution_record.update(
            {
                "executed": True,
                "executed_lots": int(trade["lots"]),
                "executed_risk_capital": float(trade["risk_capital"]),
            }
        )
        execution_rows.append(execution_record)
    trade_frame = pd.DataFrame(trade_rows)
    execution_frame = pd.DataFrame(execution_rows)
    daily_frame = combine_trade_daily(
        daily_pieces,
        futures_daily,
        initial_capital=execution_settings.initial_capital,
        strategy_name=strategy_name,
    )
    if trade_frame.empty:
        return (
            trade_frame,
            daily_frame,
            {"strategy": strategy_name, "trade_count": 0},
            execution_frame,
        )
    metrics = portfolio_metrics(
        daily_frame,
        trade_frame,
        initial_capital=execution_settings.initial_capital,
    )
    metrics.update(
        {
            "strategy": strategy_name,
            "optimization_objective": objective,
            "depth_label": depth_label,
            "embedded_entry_crossing_cost": float(
                trade_frame["embedded_entry_crossing_cost"].sum()
            ),
            "portfolio_risk_rejection_count": int(
                execution_frame["skip_reason"].eq("portfolio_risk_cap").sum()
            ),
        }
    )
    return trade_frame, daily_frame, metrics, execution_frame


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
        "target": "realized one-lot delta-hedged strategy net P&L in RMB",
        "causality": "each fit and policy choice uses only outcomes expired before entry",
        "parent_versions_untouched": ["strategy_v1", "strategy_v2", "strategy_v3"],
        "files": {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path): _sha256(path)
            for path in paths
            if path.exists() and path.is_file()
        },
    }
    return write_json(
        manifest,
        output_dir / f"{settings['project']['version']}_v4_freeze_manifest.json",
    )


def run_strategy_v4_backtest(settings: dict) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    output_dir = root / settings["v4"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities_path = root / "data/outputs/volatility_strategy_opportunities.parquet"
    option_path = root / "data/raw/tqsdk_au_option_daily_strategy.parquet"
    futures_path = root / "data/raw/tqsdk_au_futures_daily_all_contracts.parquet"
    ticks_path = root / settings["data"]["raw_dir"] / "real_quote_ticks.parquet"
    opportunities = prepare_real_quote_opportunities(read_parquet(opportunities_path))
    option_daily = read_parquet(option_path)
    futures_daily = read_parquet(futures_path)
    ticks = load_real_quote_ticks(root, str(settings["data"]["raw_dir"]))
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    execution_settings = real_quote_execution_settings(settings)
    targets, target_coverage = build_direct_pnl_targets(
        opportunities,
        option_daily,
        futures_daily,
        ticks,
        settings=execution_settings,
    )
    families = [str(value) for value in settings["v4"]["model_families"]]
    predictions = add_causal_direct_pnl_predictions(
        targets,
        families=families,
        minimum_training_rows=int(settings["v4"]["minimum_model_training_rows"]),
        random_seed=int(settings["v4"]["random_seed"]),
    )
    prediction_metrics = direct_pnl_prediction_metrics(
        predictions, families=families
    )
    depth_labels = _depth_labels(settings)
    objectives = [str(value) for value in settings["v4"]["optimization_objectives"]]
    decisions, policy_history = add_causal_policy_decisions(
        predictions,
        families=families,
        objectives=objectives,
        depth_labels=depth_labels,
        threshold_quantiles=[
            float(value)
            for value in settings["v4"]["threshold_prediction_quantiles"]
        ],
        minimum_trades=int(settings["v4"]["minimum_threshold_selection_trades"]),
    )
    trade_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    execution_pieces: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for objective in objectives:
        for depth_label in depth_labels:
            trades, daily, metrics, execution = _run_decision_variant(
                opportunities,
                predictions,
                decisions,
                option_daily,
                futures_daily,
                ticks,
                execution_settings=execution_settings,
                objective=objective,
                depth_label=depth_label,
                maximum_portfolio_risk_fraction=float(
                    settings["execution"]["maximum_portfolio_risk_fraction"]
                ),
            )
            if not trades.empty:
                trade_pieces.append(trades)
            if not daily.empty:
                daily_pieces.append(daily)
            if not execution.empty:
                execution_pieces.append(execution)
            metric_rows.append(metrics)
    all_trades = pd.concat(trade_pieces, ignore_index=True) if trade_pieces else pd.DataFrame()
    all_daily = pd.concat(daily_pieces, ignore_index=True) if daily_pieces else pd.DataFrame()
    all_execution = (
        pd.concat(execution_pieces, ignore_index=True)
        if execution_pieces
        else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    breakdown = trade_breakdown(all_trades) if not all_trades.empty else pd.DataFrame()
    outputs = {
        "targets": write_parquet(targets, output_dir / "direct_pnl_targets.parquet"),
        "target_coverage": write_csv(target_coverage, output_dir / "target_coverage.csv"),
        "predictions": write_parquet(
            predictions, output_dir / "causal_oos_predictions.parquet"
        ),
        "prediction_metrics": write_csv(
            prediction_metrics, output_dir / "prediction_metrics.csv"
        ),
        "decisions": write_csv(decisions, output_dir / "decisions.csv"),
        "policy_history": write_csv(
            policy_history, output_dir / "policy_history.csv"
        ),
        "execution_audit": write_csv(
            all_execution, output_dir / "execution_audit.csv"
        ),
        "trades": write_csv(all_trades, output_dir / "trades.csv"),
        "daily": write_csv(all_daily, output_dir / "daily.csv"),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
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
            root / "src/au_rv/strategy_v3/execution.py",
            root / "src/au_rv/strategy_v4/targets.py",
            root / "src/au_rv/strategy_v4/models.py",
            root / "src/au_rv/strategy_v4/backtest.py",
        ],
    )
    return outputs
