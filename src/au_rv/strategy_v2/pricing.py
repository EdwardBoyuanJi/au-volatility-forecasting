from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from au_rv.strategy.black76 import black76_prices


def black76_option_price(
    future: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.0,
) -> float:
    call, put = black76_prices(
        future, strike, time_to_expiry, volatility, risk_free_rate
    )
    return float(call if option_type.lower() == "call" else put)


def black76_option_delta(
    future: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.0,
) -> float:
    if future <= 0 or strike <= 0:
        return np.nan
    discount = math.exp(-risk_free_rate * max(time_to_expiry, 0.0))
    if time_to_expiry <= 0 or volatility <= 0:
        if option_type.lower() == "call":
            return discount * float(future > strike)
        return -discount * float(future < strike)
    scale = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(future / strike) + 0.5 * scale * scale) / scale
    call_delta = discount * norm.cdf(d1)
    return float(call_delta if option_type.lower() == "call" else call_delta - discount)


def implied_volatility_from_option(
    price: float,
    future: float,
    strike: float,
    time_to_expiry: float,
    option_type: str,
    risk_free_rate: float = 0.0,
    *,
    minimum_volatility: float = 1.0e-4,
    maximum_volatility: float = 3.0,
) -> float:
    values = (price, future, strike, time_to_expiry)
    if not all(np.isfinite(values)) or min(price, future, strike) <= 0 or time_to_expiry <= 0:
        return np.nan
    discount = math.exp(-risk_free_rate * time_to_expiry)
    intrinsic = discount * (
        max(future - strike, 0.0)
        if option_type.lower() == "call"
        else max(strike - future, 0.0)
    )
    if price <= intrinsic + 1.0e-10:
        return np.nan

    def error(volatility: float) -> float:
        return black76_option_price(
            future,
            strike,
            time_to_expiry,
            volatility,
            option_type,
            risk_free_rate,
        ) - price

    if error(minimum_volatility) > 0 or error(maximum_volatility) < 0:
        return np.nan
    return float(brentq(error, minimum_volatility, maximum_volatility, maxiter=200))


def iron_fly_value(
    future: float,
    atm_strike: float,
    put_wing_strike: float,
    call_wing_strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    short_call = black76_option_price(
        future, atm_strike, time_to_expiry, volatility, "call", risk_free_rate
    )
    short_put = black76_option_price(
        future, atm_strike, time_to_expiry, volatility, "put", risk_free_rate
    )
    long_call = black76_option_price(
        future, call_wing_strike, time_to_expiry, volatility, "call", risk_free_rate
    )
    long_put = black76_option_price(
        future, put_wing_strike, time_to_expiry, volatility, "put", risk_free_rate
    )
    return float(short_call + short_put - long_call - long_put)


def iron_fly_delta(
    future: float,
    atm_strike: float,
    put_wing_strike: float,
    call_wing_strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    return float(
        -black76_option_delta(
            future, atm_strike, time_to_expiry, volatility, "call", risk_free_rate
        )
        - black76_option_delta(
            future, atm_strike, time_to_expiry, volatility, "put", risk_free_rate
        )
        + black76_option_delta(
            future, call_wing_strike, time_to_expiry, volatility, "call", risk_free_rate
        )
        + black76_option_delta(
            future, put_wing_strike, time_to_expiry, volatility, "put", risk_free_rate
        )
    )
