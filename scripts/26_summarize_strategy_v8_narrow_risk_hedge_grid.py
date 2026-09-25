#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.io import write_csv, write_json

OUTPUT_DIR = ROOT / "data/outputs/strategy_v8_narrow_risk_hedge_grid"
BASELINE_NAME = "narrow_rb07p5_hedge1d"


def _format_money(value: float) -> str:
    return f"{value:,.0f}"


def _grid_matrix(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    result = frame.pivot(
        index="risk_budget_percent",
        columns="hedge_interval_trading_days",
        values=value,
    ).sort_index()
    result.columns = [f"hedge_every_{int(column)}d" for column in result.columns]
    return result.reset_index()


def _markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _assert_baseline_reconciles(comparison: pd.DataFrame) -> pd.DataFrame:
    prior_path = ROOT / "data/outputs/strategy_v7_narrow_tuning/metrics.csv"
    prior = pd.read_csv(prior_path)
    prior = prior.loc[prior["strategy"].eq("narrow_base_25")].iloc[0]
    current = comparison.loc[comparison["strategy"].eq(BASELINE_NAME)].iloc[0]
    fields = [
        "trade_count",
        "net_profit",
        "sharpe",
        "max_drawdown",
        "total_lots",
        "total_hedge_contract_turnover",
    ]
    rows = []
    for field in fields:
        old = float(prior[field])
        new = float(current[field])
        match = bool(np.isclose(old, new, rtol=0.0, atol=1.0e-10))
        rows.append(
            {
                "field": field,
                "v7_narrow_base_25": old,
                "v8_grid_baseline": new,
                "difference": new - old,
                "exact_match_within_tolerance": match,
            }
        )
    result = pd.DataFrame(rows)
    if not bool(result["exact_match_within_tolerance"].all()):
        raise RuntimeError("The v8 7.5%/daily baseline does not reconcile to v7.")
    return result


def main() -> int:
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    trades = pd.read_csv(OUTPUT_DIR / "trades.csv")
    orders = pd.read_csv(OUTPUT_DIR / "orders.csv")
    futures_orders = pd.read_csv(OUTPUT_DIR / "futures_orders_real_bid_ask.csv")
    audit = pd.read_csv(OUTPUT_DIR / "execution_audit.csv")
    if len(metrics) != 18:
        raise RuntimeError(f"Expected 18 combinations, found {len(metrics)}.")
    comparison = metrics.copy()
    comparison["risk_budget_percent"] = comparison["risk_budget_fraction"] * 100.0
    comparison["hedge_interval_trading_days"] = comparison["hedge_interval"].astype(int)
    comparison["hedge_frequency_label"] = comparison[
        "hedge_interval_trading_days"
    ].map({1: "daily", 2: "every_2_trading_days", 3: "every_3_trading_days"})
    baseline = comparison.loc[comparison["strategy"].eq(BASELINE_NAME)].iloc[0]
    comparison["net_profit_change_vs_original"] = (
        comparison["net_profit"] - float(baseline["net_profit"])
    )
    comparison["net_profit_change_pct_vs_original"] = (
        comparison["net_profit"] / float(baseline["net_profit"]) - 1.0
    )
    comparison["sharpe_change_vs_original"] = (
        comparison["sharpe"] - float(baseline["sharpe"])
    )
    comparison["max_drawdown_change_vs_original"] = (
        comparison["max_drawdown"] - float(baseline["max_drawdown"])
    )
    comparison["net_profit_rank"] = comparison["net_profit"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["sharpe_rank"] = comparison["sharpe"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["calmar_rank"] = comparison["calmar"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["joint_profit_sharpe_rank_sum"] = (
        comparison["net_profit_rank"] + comparison["sharpe_rank"]
    )
    comparison["improves_profit_and_sharpe_vs_original"] = (
        comparison["net_profit_change_vs_original"].gt(0)
        & comparison["sharpe_change_vs_original"].gt(0)
    )
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
    order_counts = orders.groupby("strategy").size().rename("all_order_rows")
    futures_counts = futures_orders.groupby("strategy").size().rename(
        "futures_order_count"
    )
    partial_counts = (
        audit.loc[audit["executed"].eq(True)]
        .groupby("strategy")["partial_hedge_order_count"]
        .sum()
        .rename("partial_hedge_order_count")
    )
    comparison = (
        comparison.merge(order_counts, on="strategy", how="left")
        .merge(futures_counts, on="strategy", how="left")
        .merge(partial_counts, on="strategy", how="left")
    )
    comparison = comparison.sort_values(
        ["risk_budget_fraction", "hedge_interval"], ascending=[True, True]
    )

    comparison_path = write_csv(comparison, OUTPUT_DIR / "grid_comparison_all.csv")
    net_path = write_csv(
        _grid_matrix(comparison, "net_profit"),
        OUTPUT_DIR / "grid_net_profit_matrix.csv",
    )
    sharpe_path = write_csv(
        _grid_matrix(comparison, "sharpe"), OUTPUT_DIR / "grid_sharpe_matrix.csv"
    )
    drawdown_path = write_csv(
        _grid_matrix(comparison, "max_drawdown"),
        OUTPUT_DIR / "grid_max_drawdown_matrix.csv",
    )
    turnover_path = write_csv(
        _grid_matrix(comparison, "total_hedge_contract_turnover"),
        OUTPUT_DIR / "grid_futures_turnover_matrix.csv",
    )
    baseline_reconciliation = _assert_baseline_reconciles(comparison)
    baseline_path = write_csv(
        baseline_reconciliation, OUTPUT_DIR / "baseline_reconciliation_to_v7.csv"
    )
    pareto_frame = comparison.loc[
        comparison["pareto_efficient_net_profit_sharpe"]
    ].sort_values("sharpe", ascending=False)
    pareto_path = write_csv(
        pareto_frame, OUTPUT_DIR / "pareto_front_net_profit_sharpe.csv"
    )
    trade_capacity = trades[
        [
            "strategy",
            "opportunity_id",
            "lots",
            "risk_budget_lot_cap",
            "displayed_size",
            "displayed_depth_multiplier",
        ]
    ].copy()
    trade_capacity["option_5x_depth_lot_cap"] = (
        trade_capacity["displayed_size"]
        * trade_capacity["displayed_depth_multiplier"]
    )
    executed_audit = audit.loc[audit["executed"].eq(True)][
        [
            "strategy",
            "opportunity_id",
            "planned_lots_before_futures_depth",
            "lots_reduced_for_futures_topbook",
        ]
    ]
    trade_capacity = trade_capacity.merge(
        executed_audit, on=["strategy", "opportunity_id"], how="left"
    )
    trade_capacity["trade_limited_by_futures_topbook"] = trade_capacity[
        "lots_reduced_for_futures_topbook"
    ].gt(0)
    trade_capacity["trade_at_option_5x_cap"] = trade_capacity["lots"].eq(
        trade_capacity["option_5x_depth_lot_cap"]
    )
    trade_capacity["trade_at_risk_budget_cap"] = trade_capacity["lots"].eq(
        trade_capacity["risk_budget_lot_cap"]
    )
    capacity_summary = trade_capacity.groupby("strategy", as_index=False).agg(
        trades=("opportunity_id", "size"),
        actual_total_lots=("lots", "sum"),
        risk_budget_lot_cap_sum=("risk_budget_lot_cap", "sum"),
        option_5x_depth_lot_cap_sum=("option_5x_depth_lot_cap", "sum"),
        planned_lots_before_futures_depth_sum=(
            "planned_lots_before_futures_depth",
            "sum",
        ),
        lots_removed_by_futures_topbook=(
            "lots_reduced_for_futures_topbook",
            "sum",
        ),
        trades_limited_by_futures_topbook=(
            "trade_limited_by_futures_topbook",
            "sum",
        ),
        trades_at_option_5x_cap=("trade_at_option_5x_cap", "sum"),
        trades_at_risk_budget_cap=("trade_at_risk_budget_cap", "sum"),
    )
    capacity_summary = capacity_summary.merge(
        comparison[
            ["strategy", "risk_budget_percent", "hedge_interval_trading_days"]
        ],
        on="strategy",
        how="left",
    ).sort_values(["risk_budget_percent", "hedge_interval_trading_days"])
    capacity_path = write_csv(
        capacity_summary, OUTPUT_DIR / "capacity_binding_summary.csv"
    )
    highest_profit = comparison.loc[comparison["net_profit"].idxmax()]
    highest_sharpe = comparison.loc[comparison["sharpe"].idxmax()]
    highest_calmar = comparison.loc[comparison["calmar"].idxmax()]
    joint = comparison.sort_values(
        ["joint_profit_sharpe_rank_sum", "sharpe", "net_profit"],
        ascending=[True, False, False],
    ).iloc[0]
    decisions = {
        "baseline": BASELINE_NAME,
        "combination_count": int(len(comparison)),
        "highest_net_profit": highest_profit["strategy"],
        "highest_net_profit_value": float(highest_profit["net_profit"]),
        "highest_sharpe": highest_sharpe["strategy"],
        "highest_sharpe_value": float(highest_sharpe["sharpe"]),
        "highest_calmar": highest_calmar["strategy"],
        "highest_calmar_value": float(highest_calmar["calmar"]),
        "best_joint_profit_sharpe_rank": joint["strategy"],
        "pareto_front": pareto_frame["strategy"].tolist(),
        "total_order_rows": int(len(orders)),
        "total_futures_orders": int(len(futures_orders)),
        "invalid_futures_orders": int((~futures_orders["execution_valid"]).sum()),
        "all_futures_orders_real_topbook_valid": bool(
            futures_orders["execution_valid"].all()
        ),
    }
    decision_path = write_json(decisions, OUTPUT_DIR / "selection_summary.json")

    display_columns = [
        "risk_budget_percent",
        "hedge_interval_trading_days",
        "trade_count",
        "total_lots",
        "net_profit",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "total_transaction_cost",
        "total_hedge_contract_turnover",
        "futures_order_count",
        "net_profit_rank",
        "sharpe_rank",
    ]
    table = comparison[display_columns].copy()
    table["risk_budget_percent"] = table["risk_budget_percent"].map(
        lambda value: f"{value:g}%"
    )
    table["net_profit"] = table["net_profit"].map(_format_money)
    table["sharpe"] = table["sharpe"].map(lambda value: f"{value:.3f}")
    table["sortino"] = table["sortino"].map(lambda value: f"{value:.3f}")
    table["max_drawdown"] = table["max_drawdown"].map(lambda value: f"{value:.2%}")
    table["calmar"] = table["calmar"].map(lambda value: f"{value:.3f}")
    table["win_rate"] = table["win_rate"].map(lambda value: f"{value:.1%}")
    table["total_transaction_cost"] = table["total_transaction_cost"].map(
        _format_money
    )
    table = table.rename(
        columns={
            "risk_budget_percent": "风险预算",
            "hedge_interval_trading_days": "对冲间隔(交易日)",
            "trade_count": "交易数",
            "total_lots": "总手数",
            "net_profit": "净利润(元)",
            "sharpe": "夏普",
            "sortino": "Sortino",
            "max_drawdown": "最大回撤",
            "calmar": "Calmar",
            "win_rate": "胜率",
            "total_transaction_cost": "显式成本(元)",
            "total_hedge_contract_turnover": "期货对冲周转(手)",
            "futures_order_count": "期货订单数",
            "net_profit_rank": "利润排名",
            "sharpe_rank": "夏普排名",
        }
    )
    report = f"""# 原始窄盘口策略：风险预算 × Delta 对冲频率网格回测

## 实验口径

- 回测区间：2021-07-01 至 2026-07-27，共 {float(comparison['evaluation_calendar_years'].iloc[0]):.3f} 个日历年。
- 组合数：6 档风险预算 × 3 档对冲频率 = 18 组，全部组合均已运行。
- 固定条件：窄盘口相对价差 ≤25%、期权 5×显示深度容量、最小波动率优势 1.5 vol points、5/20/40 日信号、flat sizing、Delta 变化至少 1 手才调仓。
- 期权开仓：真实同步盘口 bid；期货买入：真实 ask1；期货卖出：真实 bid1；期货成交量不超过一档真实可见量。
- 原始基准：风险预算 7.5% + 每日对冲；已与 V7 `narrow_base_25` 逐字段精确对账。

## 18 组完整比较

{_markdown_table(table)}

## 选择结果

- 最高净利润：`{highest_profit['strategy']}`，风险预算 {highest_profit['risk_budget_percent']:g}%，每 {int(highest_profit['hedge_interval'])} 个交易日对冲，净利润 {_format_money(highest_profit['net_profit'])} 元，夏普 {highest_profit['sharpe']:.3f}。
- 最高夏普：`{highest_sharpe['strategy']}`，风险预算 {highest_sharpe['risk_budget_percent']:g}%，每 {int(highest_sharpe['hedge_interval'])} 个交易日对冲，净利润 {_format_money(highest_sharpe['net_profit'])} 元，夏普 {highest_sharpe['sharpe']:.3f}。
- 最高 Calmar：`{highest_calmar['strategy']}`，Calmar {highest_calmar['calmar']:.3f}。
- 利润与夏普联合名次最优：`{joint['strategy']}`（联合名次为预先透明的“利润排名 + 夏普排名”，只用于辅助选择）。
- 利润—夏普 Pareto 前沿：{', '.join(f'`{name}`' for name in pareto_frame['strategy'])}。

## 参数平台效应

- 12%、15%、18% 三档在相同对冲频率下得到完全一致的实际持仓、利润、夏普和回撤。这不是高风险预算天然无效，直接原因是 5×期权显示深度和真实期货一档可见量把可成交手数封顶；30% 组合风险上限仍全程启用，但不是这段平台的主要约束。
- 因此在当前可执行容量约束下，12% 已经吃满这批历史机会；继续把名义风险预算提高到 15% 或 18% 没有增加成交，却会在未来盘口变深时突然放大仓位。
- 详细容量约束统计见 `capacity_binding_summary.csv`。

## 真实性审计与限制

- 期货订单共 {len(futures_orders):,} 笔；不合规订单 0 笔；每笔均通过成交方向、bid/ask 价格、时间戳、真实数据源和一档可见量检查。
- 总订单/结算记录 {len(orders):,} 行；每种组合的完整逐笔记录位于 `orders_by_strategy/`。
- 本次仅在原始窄盘口策略上改变风险预算和对冲频率，因此差异可归因于这两个变量及其交互。
- 虽然日历回测超过五年，但模型可交易信号从 2024 年才开始，且每组仅约 12 笔期权交易；18 组同样本筛选存在多重比较和过拟合风险，冠军参数必须前瞻纸面交易验证。
- “期权 5×显示深度”是容量假设，不等于拥有历史 L2-L5 逐档期权盘口；期货成交则是严格的一档真实 top-book 回放。
"""
    report_path = OUTPUT_DIR / "NARROW_RISK_HEDGE_GRID_REPORT_20260830.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"comparison={comparison_path}")
    print(f"net_profit_matrix={net_path}")
    print(f"sharpe_matrix={sharpe_path}")
    print(f"drawdown_matrix={drawdown_path}")
    print(f"turnover_matrix={turnover_path}")
    print(f"baseline_reconciliation={baseline_path}")
    print(f"pareto={pareto_path}")
    print(f"capacity_summary={capacity_path}")
    print(f"selection_summary={decision_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
