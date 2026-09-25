from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma
from scipy.stats import norm

MU_1 = math.sqrt(2.0 / math.pi)
MU_4_3 = (2.0 ** (2.0 / 3.0)) * gamma(7.0 / 6.0) / gamma(0.5)
BNS_THETA = (math.pi / 2.0) ** 2 + math.pi - 5.0


def realized_variance(returns) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sum(values**2)) if values.size else np.nan


def bipower_variation(returns, finite_sample: bool = True) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return np.nan
    base = (MU_1**-2) * np.sum(np.abs(values[1:]) * np.abs(values[:-1]))
    if finite_sample:
        base *= n / (n - 1)
    return float(base)


def tripower_quarticity(returns, finite_sample: bool = True) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        return np.nan
    powers = np.abs(values) ** (4.0 / 3.0)
    products = powers[2:] * powers[1:-1] * powers[:-2]
    estimate = n * (MU_4_3**-3) * np.sum(products)
    if finite_sample:
        estimate *= n / (n - 2)
    return float(estimate)


def bns_jump_statistic(rv: float, bpv: float, tpq: float, n: int) -> float:
    if n < 3 or not np.isfinite([rv, bpv, tpq]).all() or rv <= 0 or bpv <= 0:
        return np.nan
    relative_jump = 1.0 - bpv / rv
    quarticity_ratio = max(1.0, tpq / (bpv**2))
    denominator = math.sqrt(BNS_THETA * quarticity_ratio / n)
    if denominator <= 0:
        return np.nan
    return float(relative_jump / denominator)


def bns_jump_measures(
    returns,
    alpha: float = 0.01,
    finite_sample: bool = True,
) -> dict[str, float | bool | int]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    rv = realized_variance(values)
    bpv = bipower_variation(values, finite_sample=finite_sample)
    tpq = tripower_quarticity(values, finite_sample=finite_sample)
    statistic = bns_jump_statistic(rv, bpv, tpq, len(values))
    critical = float(norm.ppf(1.0 - alpha))
    significant = bool(np.isfinite(statistic) and statistic > critical)
    raw_jump = max(float(rv - bpv), 0.0) if np.isfinite(rv) and np.isfinite(bpv) else np.nan
    jump = raw_jump if significant else (0.0 if np.isfinite(raw_jump) else np.nan)
    return {
        "rv": rv,
        "bpv": bpv,
        "tripower_quarticity": tpq,
        "bns_statistic": statistic,
        "bns_critical_value": critical,
        "jump_significant": significant,
        "jump_var_1d": jump,
        "return_count": int(len(values)),
    }
