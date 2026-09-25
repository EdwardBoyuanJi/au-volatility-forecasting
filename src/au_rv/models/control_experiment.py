from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from au_rv.config import resolve_path
from au_rv.evaluation.metrics import (
    diebold_mariano_losses,
    qlike_losses,
)
from au_rv.io import read_parquet, write_csv, write_parquet
from au_rv.models.garch import add_causal_garch_features
from au_rv.models.ridge_har_x import (
    make_model_pipeline,
    prediction_to_log_and_rv,
    select_alpha_purged,
)


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: str
    family: str
    objective: str
    blocks: tuple[str, ...]
    features: tuple[str, ...]


def control_blocks(horizon: int) -> dict[str, list[str]]:
    return {
        "jump": ["jump_var_1d", "jump_var_5d"],
        "macro": [
            "log_comex_nonoverlap_rv_1d",
            "abs_broad_dollar_ret_1d",
            "abs_us10y_real_chg_1d",
            "abs_usdcny_ret_1d",
        ],
        "events": [
            f"cpi_count_{horizon}d",
            f"nfp_count_{horizon}d",
            f"fomc_count_{horizon}d",
        ],
        "gvz": [
            "log_gvz_implied_var_daily",
            "gvz_log_change_1d",
            "gvz_log_change_5d",
        ],
        "slv_iv": [
            "log_slv_implied_var_daily",
            "slv_iv_log_change_1d",
            "slv_iv_log_change_5d",
        ],
        "us_epu": ["log_us_epu", "us_epu_log_change_5d"],
        "gepu": ["log_gepu", "gepu_log_change_22d"],
        "gpr": ["log_gpr", "gpr_log_change_5d", "gpr_log_mean_5d"],
    }


def base_features(family: str, horizon: int) -> list[str]:
    if family == "har":
        return ["log_rv_1d", "log_rv_5d", "log_rv_22d"]
    if family == "harq":
        return [
            "log_rv_1d",
            "log_rv_5d",
            "log_rv_22d",
            "harq_log_rv_rq_interaction_1d",
        ]
    if family == "garch":
        return [f"log_garch_forecast_rv_{horizon}d"]
    if family == "iv_gvz":
        return ["log_gvz_implied_var_daily"]
    if family == "iv_slv":
        return ["log_slv_implied_var_daily"]
    if family == "iv_both":
        return ["log_gvz_implied_var_daily", "log_slv_implied_var_daily"]
    raise ValueError(f"Unknown model family: {family}")


def all_block_subsets(block_names: list[str]) -> list[tuple[str, ...]]:
    return [
        tuple(combination)
        for size in range(len(block_names) + 1)
        for combination in itertools.combinations(block_names, size)
    ]


def experiment_specs(config: dict, horizon: int) -> list[ExperimentSpec]:
    settings = config["control_experiment"]
    blocks = control_blocks(horizon)
    block_names = [name for name in settings["blocks"] if name in blocks]
    specs: list[ExperimentSpec] = []
    for family in settings["base_families"]:
        for objective in settings["objectives"]:
            for subset in all_block_subsets(block_names):
                columns = list(base_features(family, horizon))
                for block in subset:
                    columns.extend(blocks[block])
                block_label = "+".join(subset) if subset else "none"
                specs.append(
                    ExperimentSpec(
                        model_name=f"{family}__{objective}__{block_label}",
                        family=family,
                        objective=objective,
                        blocks=subset,
                        features=tuple(dict.fromkeys(columns)),
                    )
                )
    # Calibrated pure-IV baselines are intentionally not augmented with the
    # controls.  They answer whether IV alone predicts RV, while the exhaustive
    # HAR/HARQ/GARCH experiment answers the conditional-information question.
    for family in ("iv_gvz", "iv_slv", "iv_both"):
        for objective in settings["objectives"]:
            specs.append(
                ExperimentSpec(
                    model_name=f"{family}__{objective}__none",
                    family=family,
                    objective=objective,
                    blocks=(),
                    features=tuple(base_features(family, horizon)),
                )
            )
    return specs


def _mature_training(
    data: pd.DataFrame,
    *,
    anchor,
    horizon: int,
    required_features: tuple[str, ...],
) -> pd.DataFrame:
    maturity = pd.to_datetime(
        data[f"target_maturity_date_{horizon}d"], errors="coerce"
    ).dt.normalize()
    anchor = pd.Timestamp(anchor).normalize()
    mask = (
        data["trade_date"].lt(anchor)
        & maturity.le(anchor)
        & data[f"target_rv_{horizon}d"].notna()
        & data[f"target_log_rv_{horizon}d"].notna()
        & data["experiment_common_valid"].astype(bool)
        & data[list(required_features)].notna().all(axis=1)
    )
    return data.loc[mask].sort_values("trade_date")


def _fit_spec_path(
    spec: ExperimentSpec,
    data: pd.DataFrame,
    forecast_indices: np.ndarray,
    *,
    horizon: int,
    settings: dict,
    epsilon: float,
) -> dict:
    dates = data.loc[forecast_indices, "trade_date"]
    periods = dates.dt.to_period("M")
    unique_periods = periods.drop_duplicates().tolist()
    forecast_rv = np.full(len(forecast_indices), np.nan, dtype=float)
    forecast_log = np.full(len(forecast_indices), np.nan, dtype=float)
    selected_alpha = np.nan
    first_training_samples = 0
    last_training_samples = 0
    local_positions = {int(index): pos for pos, index in enumerate(forecast_indices)}
    for period_index, period in enumerate(unique_periods):
        month_indices = forecast_indices[periods.to_numpy() == period]
        anchor = data.loc[month_indices[0], "trade_date"]
        training = _mature_training(
            data,
            anchor=anchor,
            horizon=horizon,
            required_features=spec.features,
        )
        if len(training) < int(settings["minimum_training_samples"]):
            continue
        if period_index == 0 or not np.isfinite(selected_alpha):
            target = (
                f"target_rv_{horizon}d"
                if spec.objective == "qlike"
                else f"target_log_rv_{horizon}d"
            )
            selected_alpha, _ = select_alpha_purged(
                training[list(spec.features)],
                training[target],
                horizon=horizon,
                alpha_grid=[float(value) for value in settings["alpha_grid"]],
                n_splits=int(settings["alpha_cv_splits"]),
                minimum_test_samples=int(settings["minimum_cv_test_samples"]),
                objective=spec.objective,
                epsilon=epsilon,
            )
            first_training_samples = len(training)
        pipeline = make_model_pipeline(
            float(selected_alpha), objective=spec.objective
        )
        target = (
            training[f"target_rv_{horizon}d"].clip(lower=epsilon)
            if spec.objective == "qlike"
            else training[f"target_log_rv_{horizon}d"]
        )
        pipeline.fit(training[list(spec.features)], target)
        current = data.loc[month_indices, list(spec.features)]
        log_values, rv_values = prediction_to_log_and_rv(
            pipeline, current, objective=spec.objective, epsilon=epsilon
        )
        for source_position, data_index in enumerate(month_indices):
            destination = local_positions[int(data_index)]
            forecast_rv[destination] = rv_values[source_position]
            forecast_log[destination] = log_values[source_position]
        last_training_samples = len(training)
    return {
        "model_name": spec.model_name,
        "family": spec.family,
        "objective": spec.objective,
        "blocks": "+".join(spec.blocks) if spec.blocks else "none",
        "feature_count": len(spec.features),
        "alpha": selected_alpha,
        "first_training_samples": first_training_samples,
        "last_training_samples": last_training_samples,
        "forecast_rv": forecast_rv,
        "forecast_log_rv": forecast_log,
    }


def _benchmark_paths(data: pd.DataFrame, indices: np.ndarray, horizon: int) -> list[dict]:
    subset = data.loc[indices]
    epsilon = 1e-12
    benchmarks = {
        "persistence__raw__none": subset["rv"].to_numpy(dtype=float),
        "iv_gvz__raw__none": subset["gvz_implied_var_daily"].to_numpy(dtype=float),
        "iv_slv__raw__none": subset["slv_implied_var_daily"].to_numpy(dtype=float),
        "iv_average__raw__none": subset[
            ["gvz_implied_var_daily", "slv_implied_var_daily"]
        ].mean(axis=1).to_numpy(dtype=float),
        "garch__raw__none": subset[
            f"garch_forecast_rv_{horizon}d"
        ].to_numpy(dtype=float),
    }
    results = []
    for name, rv in benchmarks.items():
        family = name.split("__", 1)[0]
        results.append(
            {
                "model_name": name,
                "family": family,
                "objective": "raw",
                "blocks": "none",
                "feature_count": 1,
                "alpha": np.nan,
                "first_training_samples": 0,
                "last_training_samples": 0,
                "forecast_rv": np.maximum(rv, epsilon),
                "forecast_log_rv": np.log(np.maximum(rv, epsilon) + epsilon),
            }
        )
    return results


def _scope_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    years = pd.to_datetime(dates).dt.year.to_numpy()
    return {
        "all": np.ones(len(dates), dtype=bool),
        "selection_through_2025": years <= 2025,
        "2026_temporal_validation": years == 2026,
    }


def _evaluate_paths(
    paths: list[dict],
    data: pd.DataFrame,
    indices: np.ndarray,
    *,
    horizon: int,
    epsilon: float,
) -> pd.DataFrame:
    actual_rv = data.loc[indices, f"target_rv_{horizon}d"].to_numpy(dtype=float)
    actual_log = data.loc[indices, f"target_log_rv_{horizon}d"].to_numpy(dtype=float)
    rv_t = data.loc[indices, "rv"].to_numpy(dtype=float)
    dates = data.loc[indices, "trade_date"].reset_index(drop=True)
    persistence = next(path for path in paths if path["model_name"] == "persistence__raw__none")
    rows = []
    for scope, scope_mask in _scope_masks(dates).items():
        benchmark_rv = persistence["forecast_rv"]
        benchmark_log = persistence["forecast_log_rv"]
        for path in paths:
            forecast_rv = np.asarray(path["forecast_rv"], dtype=float)
            forecast_log = np.asarray(path["forecast_log_rv"], dtype=float)
            valid = (
                scope_mask
                & np.isfinite(actual_rv)
                & np.isfinite(actual_log)
                & np.isfinite(rv_t)
                & np.isfinite(forecast_rv)
                & np.isfinite(forecast_log)
                & np.isfinite(benchmark_rv)
                & np.isfinite(benchmark_log)
            )
            if valid.sum() == 0:
                continue
            errors = actual_log[valid] - forecast_log[valid]
            benchmark_errors = actual_log[valid] - benchmark_log[valid]
            denominator = float(np.sum(benchmark_errors**2))
            model_losses = qlike_losses(actual_rv[valid], forecast_rv[valid], epsilon)
            benchmark_losses = qlike_losses(
                actual_rv[valid], benchmark_rv[valid], epsilon
            )
            dm_stat, dm_p, dm_n = diebold_mariano_losses(
                model_losses, benchmark_losses, horizon=horizon
            )
            rows.append(
                {
                    "horizon": horizon,
                    "scope": scope,
                    "model_name": path["model_name"],
                    "family": path["family"],
                    "objective": path["objective"],
                    "blocks": path["blocks"],
                    "feature_count": path["feature_count"],
                    "alpha": path["alpha"],
                    "first_training_samples": path["first_training_samples"],
                    "last_training_samples": path["last_training_samples"],
                    "sample_count": int(valid.sum()),
                    "qlike": float(np.mean(model_losses)),
                    "rmse_log_rv": float(np.sqrt(np.mean(errors**2))),
                    "mae_log_rv": float(np.mean(np.abs(errors))),
                    "oos_r2_vs_persistence": (
                        1.0 - float(np.sum(errors**2)) / denominator
                        if denominator > 0
                        else np.nan
                    ),
                    "direction_accuracy": float(
                        np.mean(
                            np.sign(forecast_rv[valid] - rv_t[valid])
                            == np.sign(actual_rv[valid] - rv_t[valid])
                        )
                    ),
                    "dm_qlike_stat_vs_persistence": dm_stat,
                    "dm_qlike_p_value_vs_persistence": dm_p,
                    "dm_qlike_sample_count": dm_n,
                }
            )
    return pd.DataFrame(rows)


def incremental_control_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    data = metrics[
        metrics["family"].isin(["har", "harq", "garch"])
    ].copy()
    rows = []
    for (scope, horizon, family, objective), group in data.groupby(
        ["scope", "horizon", "family", "objective"]
    ):
        lookup = {
            frozenset(() if row.blocks == "none" else str(row.blocks).split("+")): row
            for row in group.itertuples(index=False)
        }
        all_blocks = sorted(set().union(*lookup.keys())) if lookup else []
        for block in all_blocks:
            improvements = []
            for subset, row in lookup.items():
                if block in subset:
                    continue
                augmented = lookup.get(frozenset(set(subset) | {block}))
                if augmented is not None:
                    improvements.append(float(row.qlike - augmented.qlike))
            if not improvements:
                continue
            values = np.asarray(improvements)
            rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "family": family,
                    "objective": objective,
                    "block": block,
                    "pair_count": len(values),
                    "mean_qlike_improvement": float(values.mean()),
                    "median_qlike_improvement": float(np.median(values)),
                    "win_rate": float(np.mean(values > 0)),
                    "best_qlike_improvement": float(values.max()),
                    "worst_qlike_improvement": float(values.min()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "horizon", "family", "objective", "mean_qlike_improvement"],
        ascending=[True, True, True, True, False],
    )


def unified_model_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for scope, scoped in metrics.groupby("scope"):
        persistence = scoped[
            scoped["model_name"].eq("persistence__raw__none")
        ][["horizon", "qlike"]].rename(columns={"qlike": "persistence_qlike"})
        merged = scoped.merge(persistence, on="horizon", how="left")
        merged["qlike_ratio_vs_persistence"] = (
            merged["qlike"] / merged["persistence_qlike"]
        )
        ranking = (
            merged.groupby(
                ["model_name", "family", "objective", "blocks"], as_index=False
            )
            .agg(
                horizon_count=("horizon", "nunique"),
                mean_qlike_ratio=("qlike_ratio_vs_persistence", "mean"),
                worst_qlike_ratio=("qlike_ratio_vs_persistence", "max"),
                mean_qlike=("qlike", "mean"),
            )
        )
        ranking = ranking[ranking["horizon_count"].eq(3)].copy()
        ranking.insert(0, "scope", scope)
        ranking["unified_rank"] = ranking["mean_qlike_ratio"].rank(
            method="min", ascending=True
        )
        pieces.append(ranking)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["scope", "mean_qlike_ratio", "worst_qlike_ratio", "model_name"]
    ).reset_index(drop=True)


def add_fdr_adjustment(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values within each horizon and test scope."""

    result = metrics.copy()
    column = "dm_qlike_fdr_q_value_vs_persistence"
    result[column] = np.nan
    for _, group in result.groupby(["scope", "horizon"]):
        p_values = group["dm_qlike_p_value_vs_persistence"].dropna().astype(float)
        if p_values.empty:
            continue
        order = np.argsort(p_values.to_numpy())
        ordered = p_values.to_numpy()[order]
        count = len(ordered)
        adjusted = ordered * count / np.arange(1, count + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        ordered_index = p_values.index.to_numpy()[order]
        result.loc[ordered_index, column] = adjusted
    return result


def family_distribution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, horizon), scoped in metrics.groupby(["scope", "horizon"]):
        persistence = float(
            scoped.loc[
                scoped["model_name"].eq("persistence__raw__none"), "qlike"
            ].iloc[0]
        )
        for (family, objective), group in scoped.groupby(["family", "objective"]):
            losses = group["qlike"].to_numpy(dtype=float)
            significant = (
                group["dm_qlike_p_value_vs_persistence"].lt(0.05)
                & group["qlike"].lt(persistence)
            )
            fdr_significant = (
                group["dm_qlike_fdr_q_value_vs_persistence"].lt(0.05)
                & group["qlike"].lt(persistence)
            )
            rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "family": family,
                    "objective": objective,
                    "model_count": len(group),
                    "best_qlike": float(np.min(losses)),
                    "median_qlike": float(np.median(losses)),
                    "worst_qlike": float(np.max(losses)),
                    "persistence_qlike": persistence,
                    "beat_persistence_count": int(np.sum(losses < persistence)),
                    "beat_persistence_rate": float(np.mean(losses < persistence)),
                    "significant_beat_count": int(significant.sum()),
                    "significant_beat_rate": float(significant.mean()),
                    "fdr_significant_beat_count": int(fdr_significant.sum()),
                    "fdr_significant_beat_rate": float(fdr_significant.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "horizon", "best_qlike", "family", "objective"]
    ).reset_index(drop=True)


def selection_test_champions(metrics: pd.DataFrame) -> pd.DataFrame:
    selection = metrics[metrics["scope"].eq("selection_through_2025")]
    rows = []
    for horizon, group in selection.groupby("horizon"):
        chosen = group.nsmallest(1, "qlike").iloc[0]
        for scope in ("selection_through_2025", "2026_temporal_validation", "all"):
            row = metrics[
                metrics["horizon"].eq(horizon)
                & metrics["scope"].eq(scope)
                & metrics["model_name"].eq(chosen["model_name"])
            ].iloc[0]
            record = row.to_dict()
            record["selection_model_name"] = chosen["model_name"]
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["horizon", "scope"]).reset_index(drop=True)


def _champion_prediction_frame(
    metrics: pd.DataFrame,
    horizon_paths: dict[int, tuple[pd.DataFrame, np.ndarray, list[dict]]],
) -> pd.DataFrame:
    pieces = []
    all_metrics = metrics[metrics["scope"].eq("all")]
    for horizon, (data, indices, paths) in horizon_paths.items():
        ranked = all_metrics[all_metrics["horizon"].eq(horizon)].nsmallest(20, "qlike")
        keep = set(ranked["model_name"])
        keep.update(
            {
                "persistence__raw__none",
                "iv_gvz__raw__none",
                "iv_slv__raw__none",
                "iv_average__raw__none",
                "garch__raw__none",
            }
        )
        actual = data.loc[indices, f"target_rv_{horizon}d"].to_numpy(dtype=float)
        maturity = data.loc[indices, f"target_maturity_date_{horizon}d"].to_numpy()
        dates = data.loc[indices, "trade_date"].to_numpy()
        for path in paths:
            if path["model_name"] not in keep:
                continue
            pieces.append(
                pd.DataFrame(
                    {
                        "forecast_date": dates,
                        "horizon": horizon,
                        "model_name": path["model_name"],
                        "forecast_rv": path["forecast_rv"],
                        "forecast_log_rv": path["forecast_log_rv"],
                        "actual_rv": actual,
                        "target_maturity_date": maturity,
                    }
                )
            )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def generate_control_report(
    config: dict,
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    unified: pd.DataFrame,
) -> Path:
    lines = [
        "# 沪金波动率：HAR、GARCH、IV 与控制变量完整实验",
        "",
        "本报告使用共同有效特征日、成熟目标、扩展训练窗和月度重训。HAR/HARQ/GARCH "
        "分别对8个控制变量块的全部256个子集进行 MSE-log 与 QLIKE 两种目标比较。",
        "",
        "## 总体最佳模型",
        "",
        "|期限|模型|QLIKE|log-RV RMSE|OOS R²|方向准确率|样本|",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    all_scope = metrics[metrics["scope"].eq("all")]
    for horizon in sorted(all_scope["horizon"].unique()):
        best = all_scope[all_scope["horizon"].eq(horizon)].nsmallest(1, "qlike").iloc[0]
        lines.append(
            f"|{int(horizon)}日|`{best.model_name}`|{best.qlike:.6f}|"
            f"{best.rmse_log_rv:.6f}|{best.oos_r2_vs_persistence:.4f}|"
            f"{best.direction_accuracy:.2%}|{int(best.sample_count)}|"
        )
    lines.extend(
        [
            "",
            "## 各期限前15名",
            "",
        ]
    )
    for horizon in sorted(all_scope["horizon"].unique()):
        lines.extend(
            [
                f"### {int(horizon)}日",
                "",
                "|排名|模型|QLIKE|RMSE|R²|方向|",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(
            all_scope[all_scope["horizon"].eq(horizon)].nsmallest(15, "qlike").itertuples(),
            start=1,
        ):
            lines.append(
                f"|{rank}|`{row.model_name}`|{row.qlike:.6f}|{row.rmse_log_rv:.6f}|"
                f"{row.oos_r2_vs_persistence:.4f}|{row.direction_accuracy:.2%}|"
            )
        lines.append("")
    lines.extend(
        [
            "## 统一模型候选（同一结构，三期限不同系数）",
            "",
            "|排名|模型|平均QLIKE/持久性|最差期限比率|",
            "|---:|---|---:|---:|",
        ]
    )
    for row in unified.head(20).itertuples(index=False):
        lines.append(
            f"|{int(row.unified_rank)}|`{row.model_name}`|{row.mean_qlike_ratio:.4f}|"
            f"{row.worst_qlike_ratio:.4f}|"
        )
    lines.extend(
        [
            "",
            "## 控制变量的平均条件增益",
            "",
            "正数表示在所有不包含该变量块的配对模型中，加入该块平均降低了QLIKE。",
            "",
            "|期限|基座|目标|变量块|平均QLIKE改善|胜率|配对数|",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for row in incremental.itertuples(index=False):
        lines.append(
            f"|{int(row.horizon)}日|{row.family}|{row.objective}|{row.block}|"
            f"{row.mean_qlike_improvement:.6f}|{row.win_rate:.1%}|{int(row.pair_count)}|"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 2026段是时间切片验证，但此前旧模型已查看过2026表现，因此不是从未触碰的最终测试集。",
            "- GPR使用作者当前工作簿并施加7日保守发布滞后；历史修订无法完全还原，结果需做去GPR稳健性对照。",
            "- SLV IV由OPRA近30日、近ATM期权日收盘成交价自行计算，成本较低但噪声高于收盘NBBO或完整模型无关方差。",
            "- 全子集排名用于产生候选模型，最终生产升级仍需2026-08-25之后的前瞻纸面交易确认。",
        ]
    )
    path = resolve_path(config, config["outputs"]["control_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_detailed_control_report(
    config: dict,
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    unified: pd.DataFrame,
    family_summary: pd.DataFrame,
    selection_test: pd.DataFrame,
) -> Path:
    all_scope = metrics[metrics["scope"].eq("all")]
    selection_scope = metrics[metrics["scope"].eq("selection_through_2025")]
    test_scope = metrics[metrics["scope"].eq("2026_temporal_validation")]
    lines = [
        "# 沪金波动率：Persistence、IV、HAR/HARQ、GARCH全组合实验",
        "",
        "## 执行摘要",
        "",
        "本实验使用共同有效特征日、成熟目标、扩展训练窗和月度重训。"
        "HAR、HARQ、GARCH三种基座对8个变量块的256个子集分别用MSE-log和QLIKE训练；"
        "每个期限1,542个估计模型，另加5个原始基准，三期限共4,626条训练路径。",
        "",
        "- 严格时间切分下，5日选中`HAR-QLIKE + Macro + GVZ + US EPU`；"
        "20日选中`HAR-QLIKE + Macro + US EPU`；40日选中原始`GARCH(1,1)`。",
        "- 截至2025年选出的三期限统一结构是`HAR-QLIKE + Macro + US EPU`，"
        "5/20/40日各自估计系数。",
        "- 全样本描述性最优统一结构是`HAR-MSE-log + Macro + GVZ + SLV IV + US EPU`。"
        "它利用了2026已知结果排名，是当前候选，不是未见测试冠军。",
        "- US EPU最稳定；GVZ对5日很强；SLV IV在2026和中期预测中明显增强；"
        "GEPU与GPR在当前样本的平均条件贡献为负，不建议进入生产冠军。",
        "- 原始GARCH在40日QLIKE上有独立价值；GARCH-X全面替代HAR-X则没有获得更好总体表现。",
        "",
        "## 严格选模段与2026测试段",
        "",
        "|期限|截至2025选中模型|选模QLIKE|2026 QLIKE|2026 RMSE|2026 OOS R²|2026方向|测试样本|DM p值|FDR q值|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in sorted(selection_scope["horizon"].unique()):
        chosen = selection_scope[selection_scope["horizon"].eq(horizon)].nsmallest(
            1, "qlike"
        ).iloc[0]
        test = test_scope[
            test_scope["horizon"].eq(horizon)
            & test_scope["model_name"].eq(chosen.model_name)
        ].iloc[0]
        lines.append(
            f"|{int(horizon)}日|`{chosen.model_name}`|{chosen.qlike:.6f}|"
            f"{test.qlike:.6f}|{test.rmse_log_rv:.6f}|{test.oos_r2_vs_persistence:.4f}|"
            f"{test.direction_accuracy:.2%}|{int(test.sample_count)}|"
            f"{test.dm_qlike_p_value_vs_persistence:.4g}|"
            f"{test.dm_qlike_fdr_q_value_vs_persistence:.4g}|"
        )
    lines.extend(
        [
            "",
            "2026段仅有105/97/84个5/20/40日成熟预测；它是独立时间切片，"
            "但此前项目已查看过2026表现，所以不应宣称为从未触碰的最终测试集。",
            "",
            "## 全样本描述性冠军",
            "",
            "|期限|模型|QLIKE|log-RV RMSE|OOS R²|方向|样本|DM p值|FDR q值|",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(all_scope["horizon"].unique()):
        best = all_scope[all_scope["horizon"].eq(horizon)].nsmallest(1, "qlike").iloc[0]
        lines.append(
            f"|{int(horizon)}日|`{best.model_name}`|{best.qlike:.6f}|"
            f"{best.rmse_log_rv:.6f}|{best.oos_r2_vs_persistence:.4f}|"
            f"{best.direction_accuracy:.2%}|{int(best.sample_count)}|"
            f"{best.dm_qlike_p_value_vs_persistence:.4g}|"
            f"{best.dm_qlike_fdr_q_value_vs_persistence:.4g}|"
        )

    baseline_names = [
        "persistence__raw__none",
        "garch__raw__none",
        "har__mse_log__none",
        "har__qlike__none",
        "harq__mse_log__none",
        "harq__qlike__none",
        "garch__mse_log__none",
        "garch__qlike__none",
        "iv_gvz__raw__none",
        "iv_gvz__mse_log__none",
        "iv_gvz__qlike__none",
        "iv_slv__raw__none",
        "iv_slv__mse_log__none",
        "iv_slv__qlike__none",
        "iv_average__raw__none",
        "iv_both__mse_log__none",
        "iv_both__qlike__none",
    ]
    lines.extend(
        [
            "",
            "## Persistence、IV、HAR与GARCH核心基准（全样本）",
            "",
            "|期限|模型|QLIKE|RMSE|OOS R²|方向|DM p值|",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(all_scope["horizon"].unique()):
        rows = all_scope[
            all_scope["horizon"].eq(horizon)
            & all_scope["model_name"].isin(baseline_names)
        ].sort_values("qlike")
        for row in rows.itertuples(index=False):
            p_value = row.dm_qlike_p_value_vs_persistence
            p_text = "-" if not np.isfinite(p_value) else f"{p_value:.4g}"
            lines.append(
                f"|{int(horizon)}日|`{row.model_name}`|{row.qlike:.6f}|"
                f"{row.rmse_log_rv:.6f}|{row.oos_r2_vs_persistence:.4f}|"
                f"{row.direction_accuracy:.2%}|{p_text}|"
            )

    lines.extend(
        [
            "",
            "## 256组合的模型族分布（全样本）",
            "",
            "|期限|基座|目标|数量|最好QLIKE|中位QLIKE|最差QLIKE|击败Persistence|原始p显著|FDR显著|",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    summaries = family_summary[
        family_summary["scope"].eq("all")
        & family_summary["family"].isin(["har", "harq", "garch"])
        & family_summary["objective"].isin(["mse_log", "qlike"])
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"|{int(row.horizon)}日|{row.family}|{row.objective}|{int(row.model_count)}|"
            f"{row.best_qlike:.6f}|{row.median_qlike:.6f}|{row.worst_qlike:.6f}|"
            f"{row.beat_persistence_rate:.1%}|{row.significant_beat_rate:.1%}|"
            f"{row.fdr_significant_beat_rate:.1%}|"
        )

    lines.extend(
        [
            "",
            "## 各模型族的最优组合（全样本）",
            "",
            "|期限|基座|目标|最优模型|QLIKE|RMSE|OOS R²|方向|",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    combo = all_scope[
        all_scope["family"].isin(["har", "harq", "garch"])
        & all_scope["objective"].isin(["mse_log", "qlike"])
    ]
    for _, group in combo.groupby(["horizon", "family", "objective"], sort=True):
        row = group.nsmallest(1, "qlike").iloc[0]
        lines.append(
            f"|{int(row.horizon)}日|{row.family}|{row.objective}|`{row.model_name}`|"
            f"{row.qlike:.6f}|{row.rmse_log_rv:.6f}|{row.oos_r2_vs_persistence:.4f}|"
            f"{row.direction_accuracy:.2%}|"
        )

    lines.extend(
        [
            "",
            "## 统一模型排名（同结构，三期限不同系数）",
            "",
            "### 截至2025年选择",
            "",
            "|排名|模型|选模段平均QLIKE/Persistence|最差期限比率|2026平均比率|",
            "|---:|---|---:|---:|---:|",
        ]
    )
    selection_rank = unified[unified["scope"].eq("selection_through_2025")]
    test_rank = unified[unified["scope"].eq("2026_temporal_validation")].set_index(
        "model_name"
    )
    for row in selection_rank.nsmallest(20, "mean_qlike_ratio").itertuples(index=False):
        test_ratio = float(test_rank.loc[row.model_name, "mean_qlike_ratio"])
        lines.append(
            f"|{int(row.unified_rank)}|`{row.model_name}`|{row.mean_qlike_ratio:.4f}|"
            f"{row.worst_qlike_ratio:.4f}|{test_ratio:.4f}|"
        )
    lines.extend(
        [
            "",
            "### 全样本描述性排名",
            "",
            "|排名|模型|平均QLIKE/Persistence|最差期限比率|",
            "|---:|---|---:|---:|",
        ]
    )
    for row in unified[unified["scope"].eq("all")].nsmallest(
        20, "mean_qlike_ratio"
    ).itertuples(index=False):
        lines.append(
            f"|{int(row.unified_rank)}|`{row.model_name}`|{row.mean_qlike_ratio:.4f}|"
            f"{row.worst_qlike_ratio:.4f}|"
        )

    lines.extend(
        [
            "",
            "## 控制变量平均条件增益",
            "",
            "正数表示加入该块后平均降低QLIKE。先对所有不含/含该块的配对子集比较，"
            "再对HAR/HARQ/GARCH及两种目标取平均。",
            "",
            "|区间|期限|变量块|平均QLIKE改善|平均胜率|",
            "|---|---:|---|---:|---:|",
        ]
    )
    block_summary = (
        incremental.groupby(["scope", "horizon", "block"], as_index=False)
        .agg(
            mean_qlike_improvement=("mean_qlike_improvement", "mean"),
            win_rate=("win_rate", "mean"),
        )
        .sort_values(
            ["scope", "horizon", "mean_qlike_improvement"],
            ascending=[True, True, False],
        )
    )
    scope_labels = {
        "selection_through_2025": "截至2025选模段",
        "2026_temporal_validation": "2026测试段",
        "all": "全样本",
    }
    for scope in ("selection_through_2025", "2026_temporal_validation", "all"):
        for row in block_summary[block_summary["scope"].eq(scope)].itertuples(index=False):
            lines.append(
                f"|{scope_labels[scope]}|{int(row.horizon)}日|{row.block}|"
                f"{row.mean_qlike_improvement:.6f}|{row.win_rate:.1%}|"
            )

    lines.extend(["", "## 各期限全样本前15名", ""])
    for horizon in sorted(all_scope["horizon"].unique()):
        lines.extend(
            [
                f"### {int(horizon)}日",
                "",
                "|排名|模型|QLIKE|RMSE|OOS R²|方向|",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(
            all_scope[all_scope["horizon"].eq(horizon)]
            .nsmallest(15, "qlike")
            .itertuples(index=False),
            start=1,
        ):
            lines.append(
                f"|{rank}|`{row.model_name}`|{row.qlike:.6f}|{row.rmse_log_rv:.6f}|"
                f"{row.oos_r2_vs_persistence:.4f}|{row.direction_accuracy:.2%}|"
            )
        lines.append("")

    lines.extend(
        [
            "## 数据与口径",
            "",
            "- `macro`：COMEX非重叠时段波动、美元广义指数、美国10年实际利率、USD/CNY。",
            "- `gvz`：GVZ隐含方差水平及1/5日变化。",
            "- `slv_iv`：OPRA的SLV近30日、近平值看涨/看跌日收盘成交价，用Black-76反解IV。",
            "- `us_epu`为日频美国EPU；`gepu`为月频全球EPU；`gpr`为作者官网日频GPR。",
            "- 原始IV与沪金15:05前日内RV存在测量窗口差，且IV含方差风险溢价；"
            "因此原始IV数值表现差而校准IV有效，不是数据错误。",
            "",
            "## 限制与生产建议",
            "",
            "- GPR使用作者当前工作簿并施加7日保守滞后；历史修订无法完全还原。",
            "- SLV IV使用最近30日到期和最近平值行权价的成交日线，受成交不同步、"
            "美式行权和到期滚动影响，噪声高于收盘NBBO曲面。",
            "- OPRA标记的2024-06-03和2025-10-22降级日IV位于常规分布内，未单独构成异常值。",
            "- 推荐保留两套候选：稳健主模型`HAR-QLIKE + Macro + US EPU`；"
            "适应2026状态的挑战模型`HAR-MSE-log + Macro + GVZ + SLV IV + US EPU`。",
            "- 40日额外保留原始GARCH预测，作为组合或风险上沿参考；不用GARCH-X全面替换HAR-X。",
            "- 生产升级前需用2026-08-25之后新数据做前瞻纸面运行，避免追逐2026事后最优组合。",
            "- 每个期限与时间区间内的DM检验已做Benjamini-Hochberg FDR校正；"
            "完整子集搜索仍然存在模型选择偏差，不能用校正后p值替代前瞻验证。",
        ]
    )
    path = resolve_path(config, config["outputs"]["control_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_control_experiment(config: dict) -> dict[str, Path]:
    features = read_parquet(resolve_path(config, config["outputs"]["features_path"]))
    horizons = [int(value) for value in config["features"]["horizons"]]
    settings = config["control_experiment"]
    epsilon = float(config["project"]["epsilon"])
    data = add_causal_garch_features(
        features,
        horizons=horizons,
        minimum_observations=int(settings["minimum_garch_observations"]),
        refit_frequency=settings["garch_refit_frequency"],
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    metrics_pieces = []
    horizon_paths: dict[int, tuple[pd.DataFrame, np.ndarray, list[dict]]] = {}
    for horizon in horizons:
        garch_column = f"garch_forecast_rv_{horizon}d"
        data["experiment_common_valid"] = (
            data["experiment_feature_valid_flag"].astype(bool)
            & data[garch_column].notna()
        )
        possible = data.index[data["experiment_common_valid"]].to_numpy(dtype=int)
        eligible = []
        for index in possible:
            training = _mature_training(
                data,
                anchor=data.loc[index, "trade_date"],
                horizon=horizon,
                required_features=tuple(base_features("harq", horizon)),
            )
            if len(training) >= int(settings["minimum_training_samples"]):
                eligible.append(index)
        forecast_indices = np.asarray(eligible, dtype=int)
        if forecast_indices.size == 0:
            raise RuntimeError(f"No eligible common-sample forecasts for {horizon}d.")
        specs = experiment_specs(config, horizon)
        fitted = Parallel(
            n_jobs=int(settings.get("parallel_jobs", 1)),
            prefer="threads",
            verbose=5,
        )(
            delayed(_fit_spec_path)(
                spec,
                data,
                forecast_indices,
                horizon=horizon,
                settings=settings,
                epsilon=epsilon,
            )
            for spec in specs
        )
        paths = _benchmark_paths(data, forecast_indices, horizon) + fitted
        horizon_paths[horizon] = (data.copy(), forecast_indices, paths)
        metrics_pieces.append(
            _evaluate_paths(
                paths,
                data,
                forecast_indices,
                horizon=horizon,
                epsilon=epsilon,
            )
        )
    metrics = pd.concat(metrics_pieces, ignore_index=True).sort_values(
        ["horizon", "scope", "qlike", "model_name"]
    )
    metrics = add_fdr_adjustment(metrics)
    incremental = incremental_control_effects(metrics)
    unified = unified_model_ranking(metrics)
    family_summary = family_distribution_summary(metrics)
    selection_test = selection_test_champions(metrics)
    champions = _champion_prediction_frame(metrics, horizon_paths)
    paths = {
        "predictions": write_parquet(
            champions,
            resolve_path(config, config["outputs"]["control_predictions_path"]),
        ),
        "metrics": write_csv(
            metrics, resolve_path(config, config["outputs"]["control_metrics_path"])
        ),
        "incremental": write_csv(
            incremental,
            resolve_path(config, config["outputs"]["control_incremental_path"]),
        ),
        "unified": write_csv(
            unified, resolve_path(config, config["outputs"]["control_unified_path"])
        ),
        "family_summary": write_csv(
            family_summary,
            resolve_path(config, config["outputs"]["control_family_summary_path"]),
        ),
        "selection_test": write_csv(
            selection_test,
            resolve_path(config, config["outputs"]["control_selection_test_path"]),
        ),
    }
    paths["report"] = generate_detailed_control_report(
        config,
        metrics,
        incremental,
        unified,
        family_summary,
        selection_test,
    )
    return paths
