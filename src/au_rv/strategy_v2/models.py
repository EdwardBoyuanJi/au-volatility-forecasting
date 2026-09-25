from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from au_rv.features.realized_variance import five_minute_log_returns, prepare_au_bars


MAIN_MODEL = "har__mse_log__macro+gvz+slv_iv+us_epu"
ROBUST_MODEL = "har__qlike__macro+us_epu"


def _forward_mean(values: pd.Series, horizon: int) -> pd.Series:
    pieces = [values.shift(-offset) for offset in range(1, int(horizon) + 1)]
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=False)


def add_causal_forecast_bounds(
    opportunities: pd.DataFrame,
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    *,
    quantile: float,
    residual_window: int,
    minimum_samples: int,
) -> pd.DataFrame:
    """Add maturity-filtered one-sided realized-variance forecast bounds."""

    result = opportunities.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"]).dt.normalize()
    forecasts = forecasts.copy()
    forecasts["trade_date"] = pd.to_datetime(forecasts["trade_date"]).dt.normalize()
    features = features.copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    models = {"main": MAIN_MODEL, "robust": ROBUST_MODEL}
    for label, model_name in models.items():
        result[f"{label}_log_residual_quantile"] = np.nan
        result[f"{label}_upper_rv"] = np.nan
        result[f"{label}_upper_fair_volatility"] = np.nan
        result[f"{label}_bound_history_count"] = 0
        for horizon in sorted(result["horizon"].dropna().astype(int).unique()):
            actual = features[
                [
                    "trade_date",
                    f"target_rv_{horizon}d",
                    f"target_maturity_date_{horizon}d",
                ]
            ].rename(
                columns={
                    f"target_rv_{horizon}d": "actual_rv_for_bound",
                    f"target_maturity_date_{horizon}d": "actual_maturity_for_bound",
                }
            )
            actual["actual_maturity_for_bound"] = pd.to_datetime(
                actual["actual_maturity_for_bound"]
            ).dt.normalize()
            history = forecasts[
                forecasts["horizon"].eq(horizon)
                & forecasts["model_name"].eq(model_name)
            ].merge(actual, on="trade_date", how="left")
            history = history.dropna(
                subset=["forecast_rv", "actual_rv_for_bound", "actual_maturity_for_bound"]
            ).sort_values("trade_date")
            history["log_residual_for_bound"] = np.log(
                history["actual_rv_for_bound"].clip(lower=1.0e-12)
            ) - np.log(history["forecast_rv"].clip(lower=1.0e-12))
            mask = result["horizon"].eq(horizon)
            for index, row in result.loc[mask].iterrows():
                eligible = history[
                    history["trade_date"].lt(row["signal_date"])
                    & history["actual_maturity_for_bound"].le(row["signal_date"])
                ].tail(int(residual_window))
                count = len(eligible)
                result.loc[index, f"{label}_bound_history_count"] = count
                if count < int(minimum_samples):
                    continue
                residual_quantile = float(
                    eligible["log_residual_for_bound"].quantile(float(quantile))
                )
                forecast_rv = float(row[f"{label}_forecast_rv"])
                upper_rv = forecast_rv * float(np.exp(residual_quantile))
                upper_fair_variance = max(
                    252.0 * upper_rv + float(row["vrp_estimate"]), 1.0e-8
                )
                result.loc[index, f"{label}_log_residual_quantile"] = residual_quantile
                result.loc[index, f"{label}_upper_rv"] = upper_rv
                result.loc[index, f"{label}_upper_fair_volatility"] = float(
                    np.sqrt(upper_fair_variance)
                )
    return result


def build_semivariance_daily(
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Construct causal positive/negative semivariance and jump histories."""

    prepared = prepare_au_bars(bars, calendar)
    feature_map = features[
        ["trade_date", "selected_au_contract", "jump_var_1d", "rv"]
    ].copy()
    feature_map["trade_date"] = pd.to_datetime(feature_map["trade_date"]).dt.normalize()
    feature_map = feature_map.dropna(subset=["selected_au_contract"]).rename(
        columns={"selected_au_contract": "ts_code"}
    )
    feature_map["ts_code"] = feature_map["ts_code"].astype(str)
    prepared["ts_code"] = prepared["ts_code"].astype(str)
    selected_bars = prepared.merge(
        feature_map[["trade_date", "ts_code"]],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    metadata = feature_map.set_index(["trade_date", "ts_code"])
    rows: list[dict] = []
    for (trade_date, ts_code), day in selected_bars.groupby(
        ["trade_date", "ts_code"], sort=False
    ):
        record = metadata.loc[(trade_date, ts_code)]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[-1]
        day = day.copy()
        day["return_5m"] = five_minute_log_returns(day, bar_minutes=5)
        returns = day["return_5m"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date).normalize(),
                "up_semivariance": float(np.sum(np.square(returns[returns >= 0]))),
                "down_semivariance": float(np.sum(np.square(returns[returns < 0]))),
                "jump_variance": float(record["jump_var_1d"])
                if pd.notna(record["jump_var_1d"])
                else np.nan,
                "rv": float(record["rv"]) if pd.notna(record["rv"]) else np.nan,
            }
        )
    result = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    epsilon = 1.0e-12
    for name in ("up_semivariance", "down_semivariance", "jump_variance"):
        result[f"log_{name}_1d"] = np.log(result[name].clip(lower=0.0) + epsilon)
        for window in (5, 22):
            result[f"log_{name}_{window}d"] = np.log(
                result[name].rolling(window, min_periods=window).mean() + epsilon
            )
    return result


def add_causal_tail_forecasts(
    opportunities: pd.DataFrame,
    semivariance: pd.DataFrame,
    *,
    minimum_samples: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Forecast upside, downside and jump components with expanding HAR regressions."""

    result = opportunities.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"]).dt.normalize()
    data = semivariance.copy().sort_values("trade_date").reset_index(drop=True)
    epsilon = 1.0e-12
    for component in ("up_semivariance", "down_semivariance", "jump_variance"):
        result[f"forecast_{component}"] = np.nan
    result["forecast_tail_imbalance"] = np.nan
    result["forecast_jump_share"] = np.nan
    for horizon in sorted(result["horizon"].dropna().astype(int).unique()):
        horizon_data = data.copy()
        maturity_dates = horizon_data["trade_date"].shift(-horizon)
        for component in ("up_semivariance", "down_semivariance", "jump_variance"):
            target = _forward_mean(horizon_data[component], horizon)
            horizon_data[f"target_log_{component}"] = np.log(
                target.clip(lower=0.0) + epsilon
            )
        horizon_data["target_maturity_date"] = maturity_dates
        feature_columns = [
            f"log_{component}_{window}d"
            for component in ("up_semivariance", "down_semivariance", "jump_variance")
            for window in (1, 5, 22)
        ]
        for index, row in result[result["horizon"].eq(horizon)].iterrows():
            training = horizon_data[
                horizon_data["trade_date"].lt(row["signal_date"])
                & horizon_data["target_maturity_date"].le(row["signal_date"])
            ].dropna(subset=feature_columns + [
                "target_log_up_semivariance",
                "target_log_down_semivariance",
                "target_log_jump_variance",
            ])
            current = horizon_data[horizon_data["trade_date"].eq(row["signal_date"])]
            if len(training) < int(minimum_samples) or current.empty:
                continue
            predictor = Pipeline(
                [("scale", StandardScaler()), ("ridge", Ridge(alpha=float(ridge_alpha)))]
            )
            component_predictions: dict[str, float] = {}
            for component in ("up_semivariance", "down_semivariance", "jump_variance"):
                predictor.fit(training[feature_columns], training[f"target_log_{component}"])
                prediction = float(
                    np.exp(predictor.predict(current[feature_columns].iloc[[-1]])[0])
                    - epsilon
                )
                component_predictions[component] = max(prediction, 0.0)
                result.loc[index, f"forecast_{component}"] = max(prediction, 0.0)
            semi_total = (
                component_predictions["up_semivariance"]
                + component_predictions["down_semivariance"]
            )
            result.loc[index, "forecast_tail_imbalance"] = (
                max(
                    component_predictions["up_semivariance"],
                    component_predictions["down_semivariance"],
                )
                / max(semi_total, epsilon)
            )
            result.loc[index, "forecast_jump_share"] = (
                component_predictions["jump_variance"]
                / max(float(row["main_forecast_rv"]), epsilon)
            )
    return result


def add_causal_pnl_forecasts(
    opportunities: pd.DataFrame,
    *,
    minimum_samples: int,
    ridge_alpha: float,
    residual_quantile: float,
) -> pd.DataFrame:
    """Forecast normalized delta-hedged P&L using only already expired opportunities."""

    result = opportunities.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"]).dt.normalize()
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.normalize()
    result["term_sqrt"] = np.sqrt(result["horizon"].astype(float) / 252.0)
    result["log_signal_iv"] = np.log(result["signal_iv"].clip(lower=1.0e-8))
    result["log_open_interest"] = np.log1p(
        result["entry_call_open_oi"].fillna(0)
        + result["entry_put_open_oi"].fillna(0)
    )
    result["log_signal_volume"] = np.log1p(
        result["signal_call_volume"].fillna(0)
        + result["signal_put_volume"].fillna(0)
    )
    numeric = [
        "log_signal_iv",
        "vrp_estimate",
        "moneyness_abs",
        "term_sqrt",
        "log_open_interest",
        "log_signal_volume",
        "forecast_tail_imbalance",
        "forecast_jump_share",
    ]
    categorical = ["horizon"]
    result["predicted_normalized_pnl"] = np.nan
    result["predicted_normalized_pnl_lower"] = np.nan
    result["pnl_model_history_count"] = 0
    for index, row in result.iterrows():
        training = result[
            result["expiry_date"].le(row["signal_date"])
            & result["normalized_short_pnl_target"].notna()
        ].copy()
        result.loc[index, "pnl_model_history_count"] = len(training)
        if len(training) < int(minimum_samples):
            continue
        available_numeric = [
            column for column in numeric if training[column].notna().any()
        ]
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
                    available_numeric,
                ),
                ("horizon", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        model = Pipeline(
            [("preprocess", preprocessor), ("ridge", Ridge(alpha=float(ridge_alpha)))]
        )
        model_features = available_numeric + categorical
        model.fit(training[model_features], training["normalized_short_pnl_target"])
        prediction = float(model.predict(result.loc[[index], model_features])[0])
        residuals = training["normalized_short_pnl_target"].to_numpy(dtype=float) - model.predict(
            training[model_features]
        )
        lower = prediction + float(np.quantile(residuals, float(residual_quantile)))
        result.loc[index, "predicted_normalized_pnl"] = prediction
        result.loc[index, "predicted_normalized_pnl_lower"] = lower
    return result


def add_v2_signals(
    opportunities: pd.DataFrame,
    *,
    minimum_volatility_edge: float,
    maximum_tail_imbalance: float,
    maximum_jump_share: float,
) -> pd.DataFrame:
    result = opportunities.copy()
    bound_pass = (
        result["signal_iv"].sub(result["main_upper_fair_volatility"])
        .ge(float(minimum_volatility_edge))
        & result["signal_iv"].sub(result["robust_upper_fair_volatility"])
        .ge(float(minimum_volatility_edge))
    )
    tail_pass = (
        result["forecast_tail_imbalance"].le(float(maximum_tail_imbalance))
        & result["forecast_jump_share"].le(float(maximum_jump_share))
        & ~result["jump_significant"].fillna(False).astype(bool)
    )
    result["signal_distribution_short"] = (
        result["signal_model_timed_short"].eq(-1) & bound_pass & tail_pass
    )
    result["signal_v2_short"] = (
        result["signal_distribution_short"]
        & result["predicted_normalized_pnl_lower"].gt(0.0)
    )
    return result
