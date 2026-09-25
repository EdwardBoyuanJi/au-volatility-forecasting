from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from au_rv.io import read_parquet, utc_now, write_csv, write_json
from au_rv.strategy.volatility_straddle import trade_breakdown
from au_rv.strategy_v3.data import load_real_quote_ticks, prepare_real_quote_opportunities
from au_rv.strategy_v3.execution import (
    RealQuoteExecutionSettings,
    expiry_future_price,
    quote_snapshot_features,
)
from au_rv.strategy_v5.backtest import VariantSpec, _simulate_spec
from au_rv.strategy_v6.orders import build_order_ledger
from au_rv.strategy_v6.pipeline import build_extended_features_and_opportunities


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
        quote_maximum_age_seconds=float(values["quote_maximum_age_seconds"]),
    )


def _strategy_specs(settings: dict) -> list[VariantSpec]:
    specs = []
    for item in settings["strategies"]:
        specs.append(
            VariantSpec(
                name=str(item["name"]),
                family=str(item.get("family", "five_year_candidate")),
                depth_multiplier=int(item["displayed_depth_multiplier"]),
                risk_budget_fraction=float(
                    item.get(
                        "risk_budget_fraction",
                        settings["execution"]["base_risk_budget_fraction_per_trade"],
                    )
                ),
                hedge_interval=int(item["hedge_interval_trading_days"]),
                delta_threshold=int(
                    item.get("delta_change_threshold_contracts", 1)
                ),
                minimum_edge_strength=float(item["minimum_edge_strength"]),
                maximum_relative_spread=(
                    None
                    if item["maximum_relative_spread"] is None
                    else float(item["maximum_relative_spread"])
                ),
                horizons=tuple(int(value) for value in item.get("horizons", [5, 20, 40])),
                sizing_mode=str(item["sizing_mode"]),
            )
        )
    return specs


def _annual_summary(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    pieces = []
    for strategy, data in daily.groupby("strategy"):
        frame = data.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["year"] = frame["trade_date"].dt.year
        grouped = frame.groupby("year")["daily_net_pnl"].agg(["sum", "count"])
        grouped = grouped.rename(columns={"sum": "net_profit", "count": "trading_days"})
        grouped["return_on_initial_capital"] = grouped["net_profit"] / initial_capital
        trade_data = trades[trades["strategy"].eq(strategy)].copy()
        trade_data["year"] = pd.to_datetime(trade_data["entry_date"]).dt.year
        trade_counts = trade_data.groupby("year").agg(
            trade_count=("trade_id", "size"),
            total_lots=("lots", "sum"),
            win_rate=("net_pnl", lambda values: float(values.gt(0).mean())),
        )
        grouped = grouped.join(trade_counts, how="left").fillna(
            {"trade_count": 0, "total_lots": 0, "win_rate": 0.0}
        )
        grouped["strategy"] = strategy
        pieces.append(grouped.reset_index())
    return pd.concat(pieces, ignore_index=True)


def _monthly_summary(daily: pd.DataFrame, *, initial_capital: float) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["month"] = frame["trade_date"].dt.to_period("M").astype(str)
    result = (
        frame.groupby(["strategy", "month"], as_index=False)["daily_net_pnl"]
        .sum()
        .rename(columns={"daily_net_pnl": "net_profit"})
    )
    result["return_on_initial_capital"] = result["net_profit"] / initial_capital
    return result


def _coverage_summary(
    opportunities: pd.DataFrame,
    selected: pd.DataFrame,
    quote_cache: dict[str, dict | None],
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> pd.DataFrame:
    all_in_window = opportunities[
        opportunities["entry_date"].between(evaluation_start, evaluation_end)
    ]
    return pd.DataFrame(
        [
            {
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "calendar_days": int((evaluation_end - evaluation_start).days + 1),
                "calendar_years": float(
                    ((evaluation_end - evaluation_start).days + 1) / 365.2425
                ),
                "all_option_opportunities": int(len(all_in_window)),
                "short_model_signals": int(len(selected)),
                "signals_with_synchronized_entry_quote": int(
                    sum(value is not None for value in quote_cache.values())
                ),
                "first_signal_entry": selected["entry_date"].min(),
                "last_signal_entry": selected["entry_date"].max(),
                "model_forecast_first_available": opportunities.loc[
                    opportunities[["main_forecast_rv", "robust_forecast_rv"]]
                    .notna()
                    .all(axis=1),
                    "signal_date",
                ].min(),
            }
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_strategy_v6_five_year_backtest(
    settings: dict,
    model_config: dict,
    *,
    rebuild_pipeline: bool = True,
) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    if rebuild_pipeline:
        pipeline_outputs = build_extended_features_and_opportunities(
            settings, model_config
        )
    else:
        pipeline_outputs = {
            "extended_au": root
            / settings["backfill"]["raw_dir"]
            / "tqsdk_au_5m_extended.parquet",
            "extended_slv": root
            / settings["backfill"]["raw_dir"]
            / "databento_slv_iv_30d_extended.parquet",
            "extended_fred": root
            / settings["backfill"]["raw_dir"]
            / "fred_initial_release_daily_extended.parquet",
            "features": root / settings["outputs"]["feature_path"],
            "leakage_audit": root / settings["outputs"]["leakage_audit_path"],
            "forecasts": root / settings["outputs"]["forecast_path"],
            "opportunities": root / settings["outputs"]["opportunities_path"],
        }
        missing = [path for path in pipeline_outputs.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Cannot reuse v6 pipeline; missing: "
                + ", ".join(str(path) for path in missing)
            )
    output_dir = root / settings["outputs"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities = prepare_real_quote_opportunities(
        read_parquet(pipeline_outputs["opportunities"])
    )
    option_daily = read_parquet(
        root / "data/raw/tqsdk_au_option_daily_strategy.parquet"
    )
    futures_daily = read_parquet(
        root / "data/raw/tqsdk_au_futures_daily_all_contracts.parquet"
    )
    ticks = load_real_quote_ticks(root, "data/raw/strategy_real_quotes")
    for frame in (option_daily, futures_daily):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    for column in ("signal_date", "entry_date", "expiry_date"):
        opportunities[column] = pd.to_datetime(opportunities[column]).dt.normalize()
    evaluation_start = pd.Timestamp(
        settings["sample"]["requested_backtest_start"]
    ).normalize()
    evaluation_end = pd.Timestamp(settings["sample"]["evaluation_end"]).normalize()
    calendar_years = ((evaluation_end - evaluation_start).days + 1) / 365.2425
    if calendar_years < float(settings["sample"]["minimum_calendar_years"]):
        raise RuntimeError("Configured v6 evaluation window is shorter than five years.")
    selected = opportunities[
        opportunities["signal_model_timed_short"].eq(-1)
        & opportunities["entry_date"].between(evaluation_start, evaluation_end)
        & opportunities["expiry_date"].le(evaluation_end)
    ].copy()
    if selected.empty:
        raise RuntimeError("The extended model generated no completed five-year signals.")
    base_settings = _execution_settings(settings)
    option_lookup = option_daily.set_index(["trade_date", "ts_code"])
    quote_cache = {
        str(row["opportunity_id"]): quote_snapshot_features(
            ticks,
            row,
            maximum_quote_age_seconds=base_settings.quote_maximum_age_seconds,
        )
        for _, row in selected.iterrows()
    }
    expiry_cache = {
        opportunity_id: expiry_future_price(ticks, opportunity_id)
        for opportunity_id in quote_cache
    }
    specs = _strategy_specs(settings)
    trades_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    audit_pieces: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    for spec in specs:
        trades, daily, metrics, audit = _simulate_spec(
            selected,
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
            trades_pieces.append(trades)
        daily["peak_equity"] = daily["equity"].cummax()
        daily["drawdown"] = daily["equity"] / daily["peak_equity"] - 1.0
        daily_pieces.append(daily)
        audit_pieces.append(audit)
        metrics["evaluation_calendar_years"] = calendar_years
        metrics_rows.append(metrics)
    trades = pd.concat(trades_pieces, ignore_index=True)
    daily = pd.concat(daily_pieces, ignore_index=True)
    audit = pd.concat(audit_pieces, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    spec_lookup = {spec.name: spec for spec in specs}
    orders, order_reconciliation = build_order_ledger(
        trades,
        selected,
        spec_lookup,
        option_lookup,
        futures_daily,
        ticks,
        quote_cache,
        expiry_cache,
        base_settings=base_settings,
    )
    tolerance = 1.0e-6
    if order_reconciliation["cost_reconciliation_error"].abs().max() > tolerance:
        raise RuntimeError("Order ledger does not reconcile to modeled transaction costs.")
    annual = _annual_summary(
        daily, trades, initial_capital=base_settings.initial_capital
    )
    monthly = _monthly_summary(daily, initial_capital=base_settings.initial_capital)
    breakdown = trade_breakdown(trades)
    coverage = _coverage_summary(
        opportunities,
        selected,
        quote_cache,
        evaluation_start,
        evaluation_end,
    )
    catalog = pd.DataFrame([asdict(spec) for spec in specs])
    outputs = {
        **pipeline_outputs,
        "strategy_catalog": write_csv(catalog, output_dir / "strategy_catalog.csv"),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "trades": write_csv(trades, output_dir / "trades.csv"),
        "orders": write_csv(orders, output_dir / "orders.csv"),
        "order_reconciliation": write_csv(
            order_reconciliation, output_dir / "order_reconciliation.csv"
        ),
        "daily": write_csv(daily, output_dir / "daily.csv"),
        "annual": write_csv(annual, output_dir / "annual.csv"),
        "monthly": write_csv(monthly, output_dir / "monthly.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
        "execution_audit": write_csv(audit, output_dir / "execution_audit.csv"),
        "coverage": write_csv(coverage, output_dir / "coverage.csv"),
    }
    manifest = {
        "strategy_version": settings["project"]["version"],
        "created_at_utc": utc_now(),
        "parent_versions_untouched": ["strategy_v3", "strategy_v4", "strategy_v5"],
        "files": {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path): _sha256(path)
            for path in [
                Path(settings["_config_path"]),
                root / "config.yaml",
                root / "src/au_rv/strategy_v6/pipeline.py",
                root / "src/au_rv/strategy_v6/orders.py",
                root / "src/au_rv/strategy_v6/backtest.py",
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
