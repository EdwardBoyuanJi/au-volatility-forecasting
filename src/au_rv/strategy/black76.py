from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76_prices(
    future: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> tuple[float, float]:
    if future <= 0 or strike <= 0:
        return np.nan, np.nan
    time_to_expiry = max(float(time_to_expiry), 0.0)
    discount = math.exp(-float(risk_free_rate) * time_to_expiry)
    if time_to_expiry == 0 or volatility <= 0:
        return (
            discount * max(future - strike, 0.0),
            discount * max(strike - future, 0.0),
        )
    scale = float(volatility) * math.sqrt(time_to_expiry)
    d1 = (math.log(future / strike) + 0.5 * scale * scale) / scale
    d2 = d1 - scale
    call = discount * (future * norm.cdf(d1) - strike * norm.cdf(d2))
    put = discount * (strike * norm.cdf(-d2) - future * norm.cdf(-d1))
    return float(call), float(put)


def black76_straddle_price(
    future: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    call, put = black76_prices(
        future, strike, time_to_expiry, volatility, risk_free_rate
    )
    return float(call + put)


def implied_volatility_from_straddle(
    straddle_price: float,
    future: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.0,
    *,
    minimum_volatility: float = 1.0e-4,
    maximum_volatility: float = 3.0,
) -> float:
    values = (straddle_price, future, strike, time_to_expiry)
    if not all(np.isfinite(values)) or min(straddle_price, future, strike) <= 0:
        return np.nan
    if time_to_expiry <= 0:
        return np.nan
    intrinsic = math.exp(-risk_free_rate * time_to_expiry) * abs(future - strike)
    if straddle_price <= intrinsic + 1.0e-10:
        return np.nan

    def error(volatility: float) -> float:
        return black76_straddle_price(
            future,
            strike,
            time_to_expiry,
            volatility,
            risk_free_rate,
        ) - straddle_price

    lower_error = error(minimum_volatility)
    upper_error = error(maximum_volatility)
    if lower_error > 0 or upper_error < 0:
        return np.nan
    return float(
        brentq(error, minimum_volatility, maximum_volatility, maxiter=200)
    )


def black76_straddle_delta(
    future: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    if future <= 0 or strike <= 0:
        return np.nan
    if time_to_expiry <= 0 or volatility <= 0:
        if future > strike:
            return 1.0
        if future < strike:
            return -1.0
        return 0.0
    scale = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(future / strike) + 0.5 * scale * scale) / scale
    return float(math.exp(-risk_free_rate * time_to_expiry) * (2.0 * norm.cdf(d1) - 1.0))
