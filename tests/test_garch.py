import unittest

import numpy as np
import pandas as pd

from au_rv.models.garch import (
    add_causal_garch_features,
    fit_garch11,
    forecast_average_variance,
)


class GarchTests(unittest.TestCase):
    def test_fit_is_stationary_and_forecast_positive(self):
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0, 0.01, 800)
        parameters = fit_garch11(returns)
        self.assertGreaterEqual(parameters.alpha, 0.0)
        self.assertGreaterEqual(parameters.beta, 0.0)
        self.assertLess(parameters.alpha + parameters.beta, 1.0)
        self.assertGreater(forecast_average_variance(parameters, returns, 20), 0.0)

    def test_causal_feature_does_not_change_when_future_returns_change(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2020-01-01", periods=330),
                "daily_log_return": rng.normal(0.0, 0.01, 330),
            }
        )
        first = add_causal_garch_features(
            frame, horizons=[5], minimum_observations=252
        )
        changed = frame.copy()
        changed.loc[changed.index[-20:], "daily_log_return"] *= 20.0
        second = add_causal_garch_features(
            changed, horizons=[5], minimum_observations=252
        )
        cutoff = frame.index[-21]
        self.assertAlmostEqual(
            first.loc[cutoff, "garch_forecast_rv_5d"],
            second.loc[cutoff, "garch_forecast_rv_5d"],
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
