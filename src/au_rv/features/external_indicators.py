from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.time_utils import model_cutoff


def align_external_indicator(
    source: pd.DataFrame,
    trade_dates: pd.Series,
    *,
    prefix: str,
    value_column: str,
    cutoff_time: str,
    maximum_staleness_days: int,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {"trade_date": sorted(pd.to_datetime(trade_dates).dropna().unique())}
    )
    base["model_cutoff"] = [model_cutoff(day, cutoff_time) for day in base["trade_date"]]
    base["_cutoff_utc"] = pd.to_datetime(base["model_cutoff"], utc=True)
    if source is None or source.empty:
        base[prefix] = np.nan
        base[f"{prefix}_observation_date"] = pd.NaT
        base[f"{prefix}_available_at"] = pd.NaT
        return base.drop(columns=["_cutoff_utc"])
    right = source.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True, errors="coerce")
    right["observation_date"] = pd.to_datetime(
        right["observation_date"], errors="coerce"
    ).dt.normalize()
    right[value_column] = pd.to_numeric(right[value_column], errors="coerce")
    right = right.dropna(subset=["available_at", "observation_date", value_column])
    right = right.sort_values("available_at")
    subset = right[["available_at", "observation_date", value_column]].rename(
        columns={
            "available_at": f"{prefix}_available_at",
            "observation_date": f"{prefix}_observation_date",
            value_column: prefix,
        }
    )
    result = pd.merge_asof(
        base.sort_values("_cutoff_utc"),
        subset.sort_values(f"{prefix}_available_at"),
        left_on="_cutoff_utc",
        right_on=f"{prefix}_available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (
        result["trade_date"].dt.normalize()
        - pd.to_datetime(result[f"{prefix}_observation_date"]).dt.normalize()
    ).dt.days
    stale = age.gt(int(maximum_staleness_days)) | age.lt(0)
    result.loc[stale, prefix] = np.nan
    return result.drop(columns=["_cutoff_utc"])


def add_gpr_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log_gpr"] = np.log1p(result["gpr"].clip(lower=0.0))
    result["gpr_log_change_5d"] = result["log_gpr"].diff(5)
    result["gpr_log_mean_5d"] = result["log_gpr"].rolling(5, min_periods=5).mean()
    return result


def add_slv_iv_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["slv_implied_var_daily"] = (result["slv_iv30_pct"] / 100.0) ** 2 / 252.0
    result["log_slv_implied_var_daily"] = np.log(
        result["slv_implied_var_daily"].clip(lower=1e-12)
    )
    result["slv_iv_log_change_1d"] = result["log_slv_implied_var_daily"].diff()
    result["slv_iv_log_change_5d"] = result["log_slv_implied_var_daily"].diff(5)
    return result
