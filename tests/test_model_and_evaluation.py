import unittest
import re

import numpy as np
import pandas as pd

from au_rv.config import git_commit
from au_rv.evaluation.metrics import diebold_mariano, qlike
from au_rv.models.ridge_har_x import (
    feature_sets,
    make_qlike_pipeline,
    make_ridge_pipeline,
    prediction_to_log_and_rv,
    select_alpha_purged,
)
from au_rv.models.latest_forecast import select_champion_models
from au_rv.models.walk_forward import cadence_key


class ModelAndEvaluationTests(unittest.TestCase):
    def test_champion_selection_uses_formal_all_scope_qlike(self):
        metrics = pd.DataFrame(
            {
                "horizon": [5, 5, 20, 20],
                "model_name": ["har", "full", "har", "har_jump_macro"],
                "scope_type": ["all"] * 4,
                "scope_value": ["all"] * 4,
                "sample_count": [100] * 4,
                "qlike": [0.2, 0.3, 0.4, 0.35],
            }
        )
        self.assertEqual(
            select_champion_models(metrics, [5, 20]),
            {5: "har", 20: "har_jump_macro"},
        )

    def test_code_version_is_git_commit_or_source_fingerprint(self):
        value = git_commit()
        self.assertRegex(value, r"^(?:[0-9a-f]{40}|source-sha256:[0-9a-f]{16})$")

    def test_walk_forward_cadences_are_calendar_stable(self):
        first = pd.Timestamp("2024-01-31")
        second = pd.Timestamp("2024-02-01")
        self.assertNotEqual(cadence_key(first, "daily"), cadence_key(second, "daily"))
        self.assertNotEqual(
            cadence_key(first, "monthly"), cadence_key(second, "monthly")
        )
        self.assertEqual(
            cadence_key(pd.Timestamp("2024-01-02"), "weekly"),
            cadence_key(pd.Timestamp("2024-01-05"), "weekly"),
        )
        with self.assertRaises(ValueError):
            cadence_key(first, "random")

    def test_scaler_is_inside_pipeline(self):
        pipeline = make_ridge_pipeline(1.0)
        self.assertEqual(list(pipeline.named_steps), ["scaler", "ridge"])

    def test_purged_time_series_alpha_selection_runs(self):
        x = pd.DataFrame({"x": np.arange(120.0), "z": np.sin(np.arange(120.0))})
        y = 0.1 * x["x"] + 0.2 * x["z"]
        alpha, scores = select_alpha_purged(
            x,
            y,
            horizon=5,
            alpha_grid=[0.1, 1.0, 10.0],
            n_splits=3,
            minimum_test_samples=10,
        )
        self.assertIn(alpha, [0.1, 1.0, 10.0])
        self.assertGreater(scores["fold_count"].max(), 0)

    def test_qlike_pipeline_is_positive_and_uses_qlike_cv(self):
        x = pd.DataFrame(
            {
                "log_rv_1d": np.linspace(-10.0, -7.0, 120),
                "log_rv_5d": np.linspace(-9.8, -7.2, 120),
                "log_rv_22d": np.linspace(-9.5, -7.5, 120),
            }
        )
        y = np.exp(0.4 * x["log_rv_1d"] + 0.3 * x["log_rv_5d"] - 3.0)
        alpha, scores = select_alpha_purged(
            x,
            y,
            horizon=5,
            alpha_grid=[0.01, 0.1, 1.0],
            n_splits=3,
            minimum_test_samples=10,
            objective="qlike",
        )
        self.assertIn(alpha, [0.01, 0.1, 1.0])
        self.assertTrue(scores["mean_validation_qlike"].notna().any())
        pipeline = make_qlike_pipeline(alpha)
        pipeline.fit(x, y)
        forecast_log, forecast_rv = prediction_to_log_and_rv(
            pipeline,
            x.iloc[[-1]],
            objective="qlike",
            epsilon=1e-12,
        )
        self.assertGreater(forecast_rv[0], 0.0)
        self.assertTrue(np.isfinite(forecast_log[0]))

    def test_harq_feature_sets_include_measurement_error_interaction(self):
        sets = feature_sets(20)
        interaction = "harq_log_rv_rq_interaction_1d"
        self.assertIn(interaction, sets["harq"])
        self.assertIn(interaction, sets["qlike_harq"])
        self.assertNotIn(interaction, sets["qlike_har_jump_macro"])
        self.assertIn(interaction, sets["qlike_harq_jump_macro"])

    def test_qlike_is_zero_for_perfect_variance_forecast(self):
        actual = np.array([1e-4, 2e-4, 3e-4])
        self.assertAlmostEqual(qlike(actual, actual), 0.0, places=14)

    def test_diebold_mariano_detects_large_loss_difference(self):
        actual = np.linspace(0, 1, 100)
        good_errors = np.sin(actual) * 0.01
        bad_errors = np.ones(100) * 0.5 + np.linspace(0, 0.1, 100)
        statistic, p_value, count = diebold_mariano(
            good_errors, bad_errors, horizon=5
        )
        self.assertEqual(count, 100)
        self.assertLess(statistic, 0)
        self.assertLess(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
