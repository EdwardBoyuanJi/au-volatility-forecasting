from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "signal_iv",
    "real_entry_iv",
    "main_fair_volatility",
    "robust_fair_volatility",
    "persistence_fair_volatility",
    "main_volatility_edge",
    "robust_volatility_edge",
    "persistence_volatility_edge",
    "forecast_disagreement",
    "vrp_estimate",
    "moneyness_abs",
    "term_sqrt",
    "jump_significant_numeric",
    "log_signal_volume",
    "log_entry_daily_volume",
    "log_open_interest",
    "entry_future_mid",
    "log_entry_notional",
    "entry_premium",
    "entry_mid_premium",
    "entry_premium_to_future",
    "entry_relative_spread",
    "future_spread_bps",
    "signal_to_entry_iv_change",
    "displayed_size",
    "per_lot_risk_capital",
    "embedded_entry_crossing_cost_per_lot",
]
CATEGORICAL_FEATURES = ["horizon", "side_label", "execution_window"]


def _pipeline(
    family: str,
    *,
    random_seed: int,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    selected_numeric = numeric_features or NUMERIC_FEATURES
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                selected_numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if family == "ridge":
        estimator = Ridge(alpha=10.0)
    elif family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            max_features=0.75,
            random_state=random_seed,
            n_jobs=-1,
        )
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unknown direct-P&L model family: {family}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def add_causal_direct_pnl_predictions(
    targets: pd.DataFrame,
    *,
    families: list[str],
    minimum_training_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    result = targets.copy().sort_values(
        ["entry_date", "opportunity_id", "side"]
    ).reset_index(drop=True)
    for column in ("signal_date", "entry_date", "expiry_date"):
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    for family in families:
        result[f"predicted_per_lot_pnl_{family}"] = np.nan
        result[f"training_rows_{family}"] = 0
    for entry_date in sorted(result["entry_date"].unique()):
        current_mask = result["entry_date"].eq(entry_date)
        training = result[
            result["expiry_date"].lt(entry_date)
            & result["target_per_lot_net_pnl"].notna()
        ]
        for family in families:
            result.loc[current_mask, f"training_rows_{family}"] = len(training)
            if len(training) < int(minimum_training_rows):
                continue
            available_numeric = [
                feature for feature in NUMERIC_FEATURES if training[feature].notna().any()
            ]
            feature_columns = available_numeric + CATEGORICAL_FEATURES
            model = _pipeline(
                family,
                random_seed=random_seed,
                numeric_features=available_numeric,
            )
            model.fit(training[feature_columns], training["target_per_lot_net_pnl"])
            result.loc[current_mask, f"predicted_per_lot_pnl_{family}"] = model.predict(
                result.loc[current_mask, feature_columns]
            )
    return result


def direct_pnl_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    families: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for family in families:
        prediction_column = f"predicted_per_lot_pnl_{family}"
        valid = predictions.dropna(
            subset=[prediction_column, "target_per_lot_net_pnl"]
        )
        if valid.empty:
            continue
        actual = valid["target_per_lot_net_pnl"].to_numpy(dtype=float)
        predicted = valid[prediction_column].to_numpy(dtype=float)
        rows.append(
            {
                "model_family": family,
                "oos_row_count": len(valid),
                "oos_opportunity_count": valid["opportunity_id"].nunique(),
                "rmse_rmb_per_lot": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae_rmb_per_lot": float(mean_absolute_error(actual, predicted)),
                "oos_r2": float(r2_score(actual, predicted)),
                "pnl_sign_accuracy": float(
                    np.mean(np.sign(actual) == np.sign(predicted))
                ),
                "pearson_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def _best_side(history: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    valid = history.dropna(subset=[prediction_column]).copy()
    if valid.empty:
        return valid
    indices = valid.groupby("opportunity_id")[prediction_column].idxmax()
    return valid.loc[indices].copy()


def choose_causal_policy(
    history: pd.DataFrame,
    *,
    families: list[str],
    lot_column: str,
    objective: str,
    threshold_quantiles: list[float],
    minimum_trades: int,
) -> dict | None:
    candidates: list[dict] = []
    for family in families:
        prediction_column = f"predicted_per_lot_pnl_{family}"
        best = _best_side(history, prediction_column)
        if best.empty:
            continue
        thresholds = [("zero", 0.0)]
        thresholds.extend(
            [
                (f"q{quantile:.2f}", float(best[prediction_column].quantile(quantile)))
                for quantile in threshold_quantiles
            ]
        )
        seen: set[float] = set()
        for threshold_label, threshold in thresholds:
            rounded = round(threshold, 10)
            if rounded in seen:
                continue
            seen.add(rounded)
            selected = best[best[prediction_column].gt(threshold)].copy()
            if len(selected) < int(minimum_trades):
                continue
            pnl = (
                selected["target_per_lot_net_pnl"].to_numpy(dtype=float)
                * selected[lot_column].to_numpy(dtype=float)
            )
            net_profit = float(np.sum(pnl))
            standard_deviation = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else np.nan
            trade_mean_over_std = (
                float(np.mean(pnl) / standard_deviation)
                if standard_deviation > 0
                else np.nan
            )
            if not np.isfinite(trade_mean_over_std) or net_profit <= 0:
                continue
            primary_score = (
                trade_mean_over_std if objective == "sharpe" else net_profit
            )
            secondary_score = (
                net_profit if objective == "sharpe" else trade_mean_over_std
            )
            candidates.append(
                {
                    "model_family": family,
                    "threshold_label": threshold_label,
                    "threshold": threshold,
                    "historical_selected_trade_count": int(len(selected)),
                    "historical_net_profit": net_profit,
                    "historical_trade_mean_over_std": trade_mean_over_std,
                    "primary_score": primary_score,
                    "secondary_score": secondary_score,
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate["primary_score"], candidate["secondary_score"]
        ),
    )


def add_causal_policy_decisions(
    predictions: pd.DataFrame,
    *,
    families: list[str],
    objectives: list[str],
    depth_labels: list[str],
    threshold_quantiles: list[float],
    minimum_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions: list[dict] = []
    policy_rows: list[dict] = []
    data = predictions.copy()
    lot_columns = {
        "depth_1x": "available_lots_depth_1x",
        "depth_3x": "available_lots_depth_3x",
        "unlimited": "available_lots_unlimited",
    }
    for entry_date in sorted(data["entry_date"].unique()):
        current = data[data["entry_date"].eq(entry_date)]
        history = data[
            data["expiry_date"].lt(entry_date)
            & data["target_per_lot_net_pnl"].notna()
        ]
        for objective in objectives:
            for depth_label in depth_labels:
                lot_column = lot_columns[depth_label]
                policy = choose_causal_policy(
                    history,
                    families=families,
                    lot_column=lot_column,
                    objective=objective,
                    threshold_quantiles=threshold_quantiles,
                    minimum_trades=minimum_trades,
                )
                policy_record = {
                    "entry_date": entry_date,
                    "objective": objective,
                    "depth_label": depth_label,
                    "matured_oos_history_rows": int(
                        history[
                            [f"predicted_per_lot_pnl_{family}" for family in families]
                        ].notna().any(axis=1).sum()
                    ),
                    **(policy or {}),
                }
                policy_rows.append(policy_record)
                for opportunity_id, candidates in current.groupby("opportunity_id"):
                    decision = {
                        "opportunity_id": opportunity_id,
                        "entry_date": entry_date,
                        "expiry_date": candidates["expiry_date"].iloc[0],
                        "horizon": int(candidates["horizon"].iloc[0]),
                        "objective": objective,
                        "depth_label": depth_label,
                        "trade": False,
                        "selected_side": np.nan,
                        "predicted_per_lot_pnl": np.nan,
                        "predicted_sized_pnl": np.nan,
                        "policy_available": policy is not None,
                    }
                    if policy is not None:
                        family = str(policy["model_family"])
                        prediction_column = f"predicted_per_lot_pnl_{family}"
                        available = candidates.dropna(subset=[prediction_column])
                        if not available.empty:
                            chosen = available.loc[available[prediction_column].idxmax()]
                            predicted = float(chosen[prediction_column])
                            lots = int(chosen[lot_column])
                            decision.update(
                                {
                                    "model_family": family,
                                    "threshold_label": policy["threshold_label"],
                                    "threshold": float(policy["threshold"]),
                                    "historical_selected_trade_count": int(
                                        policy["historical_selected_trade_count"]
                                    ),
                                    "historical_net_profit": float(
                                        policy["historical_net_profit"]
                                    ),
                                    "historical_trade_mean_over_std": float(
                                        policy["historical_trade_mean_over_std"]
                                    ),
                                    "selected_side": int(chosen["side"]),
                                    "predicted_per_lot_pnl": predicted,
                                    "available_lots": lots,
                                    "predicted_sized_pnl": predicted * lots,
                                    "trade": bool(
                                        predicted > float(policy["threshold"])
                                        and lots >= 1
                                    ),
                                }
                            )
                    decisions.append(decision)
    return pd.DataFrame(decisions), pd.DataFrame(policy_rows)
