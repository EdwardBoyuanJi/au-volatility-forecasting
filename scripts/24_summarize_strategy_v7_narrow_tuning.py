#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.io import write_csv, write_json


def main() -> int:
    output_dir = ROOT / "data/outputs/strategy_v7_narrow_tuning"
    metrics = pd.read_csv(output_dir / "metrics.csv")
    annual = pd.read_csv(output_dir / "annual.csv")
    orders = pd.read_csv(output_dir / "orders.csv")
    futures_orders = pd.read_csv(output_dir / "futures_orders_real_bid_ask.csv")
    baseline = metrics.loc[metrics["strategy"].eq("narrow_base_25")].iloc[0]
    comparison = metrics.copy()
    comparison["net_profit_change_vs_baseline"] = (
        comparison["net_profit"] - float(baseline["net_profit"])
    )
    comparison["net_profit_change_pct_vs_baseline"] = (
        comparison["net_profit"] / float(baseline["net_profit"]) - 1.0
    )
    comparison["sharpe_change_vs_baseline"] = (
        comparison["sharpe"] - float(baseline["sharpe"])
    )
    comparison["max_drawdown_change_vs_baseline"] = (
        comparison["max_drawdown"] - float(baseline["max_drawdown"])
    )
    comparison["improves_net_profit"] = comparison["net_profit"].gt(
        float(baseline["net_profit"])
    )
    comparison["improves_sharpe"] = comparison["sharpe"].gt(
        float(baseline["sharpe"])
    )
    comparison["improves_both"] = comparison[
        ["improves_net_profit", "improves_sharpe"]
    ].all(axis=1)
    active_annual = annual[annual["trade_count"].gt(0)].copy()
    consistency = active_annual.groupby("strategy").agg(
        active_years=("year", "size"),
        profitable_active_years=("net_profit", lambda values: int(values.gt(0).sum())),
        worst_active_year_profit=("net_profit", "min"),
    )
    comparison = comparison.merge(
        consistency.reset_index(), on="strategy", how="left"
    )
    order_counts = orders.groupby("strategy").size().rename("all_order_rows")
    future_counts = futures_orders.groupby("strategy").size().rename(
        "futures_order_count"
    )
    comparison = comparison.merge(
        order_counts.reset_index(), on="strategy", how="left"
    ).merge(future_counts.reset_index(), on="strategy", how="left")
    pareto = []
    for row in comparison.itertuples(index=False):
        dominated = comparison[
            comparison["net_profit"].ge(float(row.net_profit))
            & comparison["sharpe"].ge(float(row.sharpe))
            & (
                comparison["net_profit"].gt(float(row.net_profit))
                | comparison["sharpe"].gt(float(row.sharpe))
            )
        ]
        if dominated.empty:
            pareto.append(row.strategy)
    comparison["pareto_efficient_net_profit_sharpe"] = comparison[
        "strategy"
    ].isin(pareto)
    comparison = comparison.sort_values(
        ["improves_both", "sharpe", "net_profit"],
        ascending=[False, False, False],
    )
    comparison_path = write_csv(
        comparison, output_dir / "comparison_to_narrow_base_25.csv"
    )
    pareto_path = write_csv(
        comparison[comparison["pareto_efficient_net_profit_sharpe"]],
        output_dir / "pareto_front_net_profit_sharpe.csv",
    )
    decisions = {
        "baseline": "narrow_base_25",
        "highest_net_profit": comparison.loc[
            comparison["net_profit"].idxmax(), "strategy"
        ],
        "highest_sharpe": comparison.loc[comparison["sharpe"].idxmax(), "strategy"],
        "strategies_improving_both": comparison.loc[
            comparison["improves_both"], "strategy"
        ].tolist(),
        "pareto_front": pareto,
        "total_order_rows": int(len(orders)),
        "total_futures_orders": int(len(futures_orders)),
        "invalid_futures_orders": int((~futures_orders["execution_valid"]).sum()),
    }
    decision_path = write_json(decisions, output_dir / "selection_summary.json")
    print(f"comparison={comparison_path}")
    print(f"pareto={pareto_path}")
    print(f"selection_summary={decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
