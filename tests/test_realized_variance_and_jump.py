import math
import unittest

import numpy as np
import pandas as pd

from au_rv.features.jump import (
    BNS_THETA,
    MU_1,
    MU_4_3,
    bipower_variation,
    bns_jump_measures,
    bns_jump_statistic,
    realized_variance,
    tripower_quarticity,
)
from au_rv.features.realized_variance import (
    add_har_features,
    five_minute_log_returns,
    realized_quarticity,
)


class RealizedVarianceAndJumpTests(unittest.TestCase):
    def test_five_minute_returns_and_rv_are_hand_calculable(self):
        bars = pd.DataFrame(
            {
                "ts_code": ["AU1"] * 3,
                "session_id": ["s"] * 3,
                "timestamp_utc": pd.date_range(
                    "2024-01-02 01:00", periods=3, freq="5min", tz="UTC"
                ),
                "close": [100.0, 100.0 * math.exp(0.1), 100.0 * math.exp(-0.1)],
            }
        )
        returns = five_minute_log_returns(bars)
        self.assertTrue(np.isnan(returns.iloc[0]))
        self.assertAlmostEqual(returns.iloc[1], 0.1, places=12)
        self.assertAlmostEqual(returns.iloc[2], -0.2, places=12)
        self.assertAlmostEqual(realized_variance(returns), 0.05, places=12)

    def test_returns_do_not_cross_a_session_gap(self):
        bars = pd.DataFrame(
            {
                "ts_code": ["AU1", "AU1", "AU1"],
                "session_id": ["morning1", "morning1", "morning2"],
                "timestamp_utc": pd.to_datetime(
                    ["2024-01-02 01:00Z", "2024-01-02 01:05Z", "2024-01-02 02:30Z"]
                ),
                "close": [100.0, 101.0, 150.0],
            }
        )
        returns = five_minute_log_returns(bars)
        self.assertTrue(np.isnan(returns.iloc[2]))

    def test_har_means_raw_rv_before_logging(self):
        rv = np.arange(1.0, 23.0) * 1e-4
        frame = pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2024-01-01", periods=22),
                "rv": rv,
                "realized_quarticity": np.linspace(1e-10, 22e-10, 22),
                "jump_var_1d": np.arange(22.0),
            }
        )
        result = add_har_features(frame, 1e-12)
        self.assertAlmostEqual(
            result.iloc[-1]["log_rv_5d"], math.log(np.mean(rv[-5:]) + 1e-12)
        )
        self.assertAlmostEqual(
            result.iloc[-1]["log_rv_22d"], math.log(np.mean(rv) + 1e-12)
        )
        self.assertNotAlmostEqual(
            result.iloc[-1]["log_rv_5d"],
            np.log(rv[-5:] + 1e-12).mean(),
        )
        self.assertAlmostEqual(
            result.iloc[-1]["harq_log_rv_rq_interaction_1d"],
            result.iloc[-1]["log_rv_1d"]
            * math.sqrt(result.iloc[-1]["realized_quarticity"]),
        )

    def test_realized_quarticity_matches_harq_formula(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        expected = len(returns) / 3.0 * np.sum(returns**4)
        self.assertAlmostEqual(realized_quarticity(returns), expected, places=18)

    def test_bpv_and_tripower_quarticity_match_formulas(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        n = len(returns)
        expected_bpv = (
            MU_1**-2
            * n
            / (n - 1)
            * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
        )
        powers = np.abs(returns) ** (4 / 3)
        expected_tpq = (
            n
            * MU_4_3**-3
            * n
            / (n - 2)
            * np.sum(powers[2:] * powers[1:-1] * powers[:-2])
        )
        self.assertAlmostEqual(bipower_variation(returns), expected_bpv, places=15)
        self.assertAlmostEqual(tripower_quarticity(returns), expected_tpq, places=15)

    def test_bns_statistic_matches_ratio_formula(self):
        rv, bpv, tpq, n = 0.05, 0.03, 0.002, 100
        expected = (1 - bpv / rv) / math.sqrt(
            BNS_THETA / n * max(1.0, tpq / bpv**2)
        )
        self.assertAlmostEqual(bns_jump_statistic(rv, bpv, tpq, n), expected)

    def test_jump_significant_and_not_significant(self):
        smooth = np.repeat(0.001, 100)
        jumped = smooth.copy()
        jumped[50] = 0.10
        no_jump = bns_jump_measures(smooth, alpha=0.01)
        jump = bns_jump_measures(jumped, alpha=0.01)
        self.assertFalse(no_jump["jump_significant"])
        self.assertEqual(no_jump["jump_var_1d"], 0.0)
        self.assertTrue(jump["jump_significant"])
        self.assertGreater(jump["jump_var_1d"], 0.0)


if __name__ == "__main__":
    unittest.main()
