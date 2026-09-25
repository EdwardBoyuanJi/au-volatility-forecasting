from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from au_rv.io import read_parquet, utc_now, write_csv, write_json
from au_rv.strategy.volatility_straddle import portfolio_metrics, trade_breakdown
from au_rv.strategy_v3.data import load_real_quote_ticks, prepare_real_quote_opportunities
from au_rv.strategy_v3.execution import (
    RealQuoteExecutionSettings,
    quote_snapshot_features,
    simulate_naked_real_quote_trade,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def real_quote_execution_settings(settings: dict) -> RealQuoteExecutionSettings:
    values = settings["execution"]
    return RealQuoteExecutionSettings(
        initial_capital=float(values["initial_capital"]),
        risk_budget_fraction_per_trade=float(values["risk_budget_fraction_per_trade"]),
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


def combine_trade_daily(
    pieces: list[pd.DataFrame],
    futures_daily: pd.DataFrame,
    *,
    initial_capital: float,
    strategy_name: str,
) -> pd.DataFrame:
    if not pieces:
        return pd.DataFrame()
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
        trade_daily.groupby("trade_date", as_index=False)["daily_net_pnl"]
        .sum()
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
    return daily


def run_real_quote_variant(
    opportunities: pd.DataFrame,
    option_daily: pd.DataFrame,
    futures_daily: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    execution_settings: RealQuoteExecutionSettings,
    strategy_name: str,
    displayed_depth_multiplier: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trades: list[dict] = []
    daily_pieces: list[pd.DataFrame] = []
    selected = opportunities[opportunities["signal_model_timed_short"].eq(-1)].copy()
    for _, row in selected.sort_values(["entry_date", "horizon"]).iterrows():
        simulated = simulate_naked_real_quote_trade(
            row,
            -1,
            option_daily,
            futures_daily,
            ticks,
            settings=execution_settings,
            displayed_depth_multiplier=displayed_depth_multiplier,
        )
        if simulated is None:
            continue
        trade, daily = simulated
        trade_id = f"{strategy_name}_{len(trades) + 1:04d}"
        trade["strategy"] = strategy_name
        trade["trade_id"] = trade_id
        daily["strategy"] = strategy_name
        daily["trade_id"] = trade_id
        trades.append(trade)
        daily_pieces.append(daily)
    trade_frame = pd.DataFrame(trades)
    daily_frame = combine_trade_daily(
        daily_pieces,
        futures_daily,
        initial_capital=execution_settings.initial_capital,
        strategy_name=strategy_name,
    )
    if trade_frame.empty:
        return trade_frame, daily_frame, {"strategy": strategy_name, "trade_count": 0}
    metrics = portfolio_metrics(
        daily_frame,
        trade_frame,
        initial_capital=execution_settings.initial_capital,
    )
    metrics.update(
        {
            "strategy": strategy_name,
            "displayed_depth_multiplier": (
                displayed_depth_multiplier
                if displayed_depth_multiplier is not None
                else "unlimited"
            ),
            "embedded_entry_crossing_cost": float(
                trade_frame["embedded_entry_crossing_cost"].sum()
            ),
            "displayed_depth_limited_trade_count": int(
                trade_frame["lots"].lt(trade_frame["risk_budget_lot_cap"]).sum()
            ),
        }
    )
    return trade_frame, daily_frame, metrics


def _quote_coverage(
    opportunities: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    maximum_quote_age_seconds: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in opportunities.iterrows():
        snapshot = quote_snapshot_features(
            ticks,
            row,
            maximum_quote_age_seconds=maximum_quote_age_seconds,
        )
        rows.append(
            {
                "opportunity_id": row["opportunity_id"],
                "signal_date": row["signal_date"],
                "entry_date": row["entry_date"],
                "expiry_date": row["expiry_date"],
                "horizon": int(row["horizon"]),
                "matured": bool(pd.notna(row.get("actual_rv"))),
                "synchronized_entry_quote_available": snapshot is not None,
                "execution_window": snapshot["execution_window"] if snapshot else None,
                "entry_straddle_bid_ask_spread": (
                    snapshot["long_entry_premium"] - snapshot["short_entry_premium"]
                    if snapshot
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_freeze_manifest(
    settings: dict,
    output_dir: Path,
    output_paths: dict[str, Path],
    source_paths: list[Path],
) -> Path:
    paths = [Path(settings["_config_path"]), *source_paths, *output_paths.values()]
    manifest = {
        "strategy_version": settings["project"]["version"],
        "created_at_utc": utc_now(),
        "parent_versions_untouched": ["strategy_v1", "strategy_v2"],
        "entry_execution": "synchronized historical top-book bid/ask",
        "daily_delta_hedge_execution": "daily close plus explicit one-tick slippage",
        "files": {
            str(path.relative_to(Path(settings["_project_root"])))
            if path.is_relative_to(Path(settings["_project_root"]))
            else str(path): _sha256(path)
            for path in paths
            if path.exists() and path.is_file()
        },
    }
    return write_json(
        manifest,
        output_dir / f"{settings['project']['version']}_v3_freeze_manifest.json",
    )


def run_strategy_v3_backtest(settings: dict) -> dict[str, Path]:
    root = Path(settings["_project_root"])
    output_dir = root / settings["v3"]["output_dir"]
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
    selected = opportunities[opportunities["signal_model_timed_short"].eq(-1)].copy()
    coverage = _quote_coverage(
        selected,
        ticks,
        maximum_quote_age_seconds=execution_settings.quote_maximum_age_seconds,
    )
    depths: list[int | None] = [
        int(value) for value in settings["v3"]["displayed_depth_multipliers"]
    ]
    if bool(settings["v3"]["include_unlimited_depth_sensitivity"]):
        depths.append(None)
    trade_pieces: list[pd.DataFrame] = []
    daily_pieces: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for depth in depths:
        suffix = "unlimited" if depth is None else f"depth_{depth}x"
        strategy_name = f"v3_old_signal_real_topbook_{suffix}"
        trades, daily, metrics = run_real_quote_variant(
            selected,
            option_daily,
            futures_daily,
            ticks,
            execution_settings=execution_settings,
            strategy_name=strategy_name,
            displayed_depth_multiplier=depth,
        )
        if not trades.empty:
            trade_pieces.append(trades)
        if not daily.empty:
            daily_pieces.append(daily)
        metric_rows.append(metrics)
    all_trades = pd.concat(trade_pieces, ignore_index=True) if trade_pieces else pd.DataFrame()
    all_daily = pd.concat(daily_pieces, ignore_index=True) if daily_pieces else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    breakdown = trade_breakdown(all_trades) if not all_trades.empty else pd.DataFrame()
    outputs = {
        "trades": write_csv(all_trades, output_dir / "trades.csv"),
        "daily": write_csv(all_daily, output_dir / "daily.csv"),
        "metrics": write_csv(metrics, output_dir / "metrics.csv"),
        "breakdown": write_csv(breakdown, output_dir / "breakdown.csv"),
        "quote_coverage": write_csv(coverage, output_dir / "quote_coverage.csv"),
    }
    outputs["freeze_manifest"] = _write_freeze_manifest(
        settings,
        output_dir,
        outputs,
        [
            opportunities_path,
            option_path,
            futures_path,
            ticks_path,
            root / "src/au_rv/strategy_v3/execution.py",
            root / "src/au_rv/strategy_v3/backtest.py",
        ],
    )
    return outputs
