import math
import unittest

import numpy as np
import pandas as pd

from au_rv.models.conditional_error import conditional_error_estimate
from au_rv.models.latest_forecast import generate_latest_forecast_frame
from au_rv.models.prediction_intervals import (
    mature_residuals,
    project_quantiles,
    residual_quantiles,
)
from au_rv.models.walk_forward import mature_training_rows


class LeakageIntervalsAndErrorsTests(unittest.TestCase):
    def test_latest_forecast_never_falls_back_to_a_stale_valid_date(self):
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "feature_valid_flag": [True, False],
                "data_quality_notes": ["ok", "missing_macro"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "no stale fallback"):
            generate_latest_forecast_frame(frame, pd.DataFrame(), {}, save_models=False)

    def test_training_excludes_targets_not_yet_mature(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "feature_valid_flag": True,
                "x": np.arange(10.0),
                "target_log_rv_5d": np.arange(10.0),
                "target_maturity_date_5d": dates + pd.offsets.BDay(5),
            }
        )
        training = mature_training_rows(
            frame,
            forecast_date=dates[7],
            horizon=5,
            feature_order=["x"],
            training_window="expanding",
            rolling_training_years=5,
        )
        self.assertTrue(
            pd.to_datetime(training["target_maturity_date_5d"]).le(dates[7]).all()
        )
        self.assertEqual(training["trade_date"].max(), dates[2])

    def test_interval_library_uses_only_mature_oos_residuals(self):
        predictions = pd.DataFrame(
            {
                "forecast_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "target_maturity_date": pd.to_datetime(["2024-01-08", "2024-01-09"]),
                "horizon": [5, 5],
                "model_name": ["full", "full"],
                "residual": [0.1, 9.9],
            }
        )
        mature = mature_residuals(
            predictions,
            forecast_date="2024-01-08",
            horizon=5,
        )
        self.assertEqual(mature["residual"].tolist(), [0.1])

    def test_prediction_quantiles_are_monotone_and_annualized(self):
        quantiles = residual_quantiles(
            pd.Series([-2, -1, 0, 1, 2] * 10),
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            minimum_samples=30,
        )
        projected = project_quantiles(
            math.log(1e-4),
            quantiles,
            epsilon=0.0,
            annualization_days=252,
        )
        rv_values = [
            projected[f"forecast_rv_q{label}"]
            for label in ("05", "10", "25", "50", "75", "90", "95")
        ]
        self.assertTrue(np.all(np.diff(rv_values) >= 0))
        self.assertAlmostEqual(
            projected["forecast_vol_q50"],
            math.sqrt(252 * projected["forecast_rv_q50"]),
        )

    def test_conditional_error_fallback_for_small_exact_group(self):
        data = pd.DataFrame(
            {
                "residual": np.linspace(-0.2, 0.2, 40),
                "forecast_vol": np.linspace(0.10, 0.20, 40),
                "event_window_flag": [False] * 20 + [True] * 20,
            }
        )
        result = conditional_error_estimate(
            data,
            current_forecast_vol=0.15,
            current_event_flag=True,
            minimum_samples=30,
            rolling_window=252,
            volatility_quantiles=[1 / 3, 2 / 3],
        )
        self.assertIn(
            result["conditional_error_level"],
            {"horizon+rolling_oos", "horizon+all_mature_oos"},
        )
        self.assertEqual(result["conditional_sample_count"], 40)

    def test_conditional_error_stays_missing_when_all_history_is_too_small(self):
        data = pd.DataFrame(
            {
                "residual": np.linspace(-0.1, 0.1, 10),
                "forecast_vol": np.linspace(0.10, 0.20, 10),
                "event_window_flag": [False] * 10,
            }
        )
        result = conditional_error_estimate(
            data,
            current_forecast_vol=0.15,
            current_event_flag=False,
            minimum_samples=30,
            rolling_window=252,
            volatility_quantiles=[1 / 3, 2 / 3],
        )
        self.assertEqual(
            result["conditional_error_level"],
            "insufficient_mature_oos_residuals",
        )
        self.assertEqual(result["conditional_sample_count"], 10)
        self.assertTrue(np.isnan(result["conditional_mae"]))


if __name__ == "__main__":
    unittest.main()
