from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_targets(
    frame: pd.DataFrame,
    horizons: list[int],
    epsilon: float,
) -> pd.DataFrame:
    result = frame.sort_values("trade_date").reset_index(drop=True).copy()
    rv = result["rv"].astype(float)
    for horizon in horizons:
        target = rv.shift(-1).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1))
        result[f"target_rv_{horizon}d"] = target
        result[f"target_log_rv_{horizon}d"] = np.log(target + epsilon)
        result[f"target_maturity_date_{horizon}d"] = result["trade_date"].shift(-horizon)
    return result
