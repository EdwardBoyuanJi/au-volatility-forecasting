import tempfile
import unittest
from pathlib import Path

import pandas as pd

from au_rv.features.build import _completed_model_cutoffs, build_feature_table


class DataPreflightTests(unittest.TestCase):
    def test_future_model_cutoff_is_removed_before_target_construction(self):
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2026-08-21", "2026-08-24"]),
                "model_cutoff": pd.to_datetime(
                    ["2026-08-21 07:05Z", "2026-08-24 07:05Z"]
                ),
            }
        )
        result = _completed_model_cutoffs(frame, "2026-08-22 12:00Z")
        self.assertEqual(result["trade_date"].tolist(), [pd.Timestamp("2026-08-21")])

    def test_feature_build_lists_all_missing_formal_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"_project_root": directory}
            with self.assertRaisesRegex(RuntimeError, "TqSdk AU 5-minute bars") as caught:
                build_feature_table(config)
            message = str(caught.exception)
            self.assertIn("Databento GC 1-minute bars", message)
            self.assertIn("complete official event calendar", message)
            self.assertFalse((Path(directory) / "data/processed").exists())


if __name__ == "__main__":
    unittest.main()
