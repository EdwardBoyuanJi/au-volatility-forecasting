from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.time_utils import model_cutoff


def align_fred_to_cutoffs(
    fred: pd.DataFrame,
    trade_dates: pd.Series,
    *,
    series_map: dict[str, str],
    cutoff_time: str,
    maximum_staleness_days: int | dict[str, int],
) -> pd.DataFrame:
    base = pd.DataFrame({"trade_date": sorted(pd.to_datetime(trade_dates).dropna().unique())})
    base["model_cutoff"] = [
        model_cutoff(day, cutoff_time) for day in base["trade_date"]
    ]
    left = base.copy()
    left["_cutoff_utc"] = pd.to_datetime(left["model_cutoff"], utc=True)
    for logical_name, series_id in series_map.items():
        source = fred[fred["series_id"].eq(series_id)].copy()
        source["available_at"] = pd.to_datetime(source["available_at"], utc=True)
        source = source.sort_values("available_at")
        subset = source[["available_at", "observation_date", "value"]].rename(
            columns={
                "available_at": f"{logical_name}_available_at",
                "observation_date": f"{logical_name}_observation_date",
                "value": logical_name,
            }
        )
        left = pd.merge_asof(
            left.sort_values("_cutoff_utc"),
            subset.sort_values(f"{logical_name}_available_at"),
            left_on="_cutoff_utc",
            right_on=f"{logical_name}_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        age = (
            left["trade_date"].dt.normalize()
            - pd.to_datetime(left[f"{logical_name}_observation_date"]).dt.normalize()
        ).dt.days
        staleness = (
            int(maximum_staleness_days.get(logical_name, 10))
            if isinstance(maximum_staleness_days, dict)
            else int(maximum_staleness_days)
        )
        stale = age.gt(staleness) | age.lt(0)
        left.loc[stale, logical_name] = np.nan
    left["abs_broad_dollar_ret_1d"] = np.abs(
        np.log(left["broad_dollar"] / left["broad_dollar"].shift(1))
    )
    # FRED DFII10 is percentage points; multiply by 100 to store basis points.
    left["abs_us10y_real_chg_1d"] = (
        left["us10y_real"].diff().abs() * 100.0
    )
    left["abs_usdcny_ret_1d"] = np.abs(np.log(left["usdcny"] / left["usdcny"].shift(1)))
    if "gvz" in left:
        left["gvz_implied_var_daily"] = (left["gvz"] / 100.0) ** 2 / 252.0
        left["log_gvz_implied_var_daily"] = np.log(
            left["gvz_implied_var_daily"].clip(lower=1e-12)
        )
        left["gvz_log_change_1d"] = left["log_gvz_implied_var_daily"].diff()
        left["gvz_log_change_5d"] = left["log_gvz_implied_var_daily"].diff(5)
    if "us_epu" in left:
        left["log_us_epu"] = np.log1p(left["us_epu"].clip(lower=0.0))
        left["us_epu_log_change_5d"] = left["log_us_epu"].diff(5)
    if "gepu" in left:
        left["log_gepu"] = np.log1p(left["gepu"].clip(lower=0.0))
        left["gepu_log_change_22d"] = left["log_gepu"].diff(22)
    required = [
        name for name in ("broad_dollar", "us10y_real", "usdcny") if name in left
    ]
    left["macro_missing_flag"] = left[required].isna().any(axis=1)
    core_availability = [f"{name}_available_at" for name in required]
    all_availability = [f"{name}_available_at" for name in series_map]
    left["macro_available_at"] = left[core_availability].max(axis=1)
    left["fred_indicator_available_at"] = left[all_availability].max(axis=1)
    return left.drop(columns=["_cutoff_utc"])
