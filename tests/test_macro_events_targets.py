import math
import unittest

import numpy as np
import pandas as pd

from au_rv.features.events import build_event_counts
from au_rv.features.macro import align_fred_to_cutoffs
from au_rv.features.targets import add_forward_targets


class MacroEventsTargetsTests(unittest.TestCase):
    def test_three_macro_absolute_changes_and_units(self):
        trade_dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        values = {
            "D": [100.0, 101.0, 99.0],
            "R": [2.00, 2.10, 2.05],
            "C": [7.10, 7.20, 7.15],
        }
        rows = []
        for series, sequence in values.items():
            for index, value in enumerate(sequence):
                available = (
                    pd.Timestamp("2024-01-01 00:00Z")
                    if index == 0
                    else pd.Timestamp("2024-01-01 10:00Z")
                    + pd.Timedelta(days=index)
                )
                rows.append(
                    {
                        "series_id": series,
                        "value": value,
                        "observation_date": pd.Timestamp("2024-01-01")
                        + pd.Timedelta(days=index),
                        "available_at": available,
                    }
                )
        result = align_fred_to_cutoffs(
            pd.DataFrame(rows),
            trade_dates,
            series_map={"broad_dollar": "D", "us10y_real": "R", "usdcny": "C"},
            cutoff_time="15:05",
            maximum_staleness_days=10,
        )
        self.assertAlmostEqual(
            result.iloc[1]["abs_broad_dollar_ret_1d"], abs(math.log(101 / 100))
        )
        self.assertAlmostEqual(result.iloc[1]["abs_us10y_real_chg_1d"], 10.0)
        self.assertAlmostEqual(
            result.iloc[1]["abs_usdcny_ret_1d"], abs(math.log(7.2 / 7.1))
        )

    def test_event_counts_use_future_shfe_trade_days_and_known_schedules(self):
        open_dates = pd.Series(pd.bdate_range("2024-01-02", periods=8))
        known = pd.Timestamp("2024-01-01 00:00Z")
        events = pd.DataFrame(
            {
                "event_type": ["cpi", "nfp", "fomc"],
                "timestamp_utc": pd.to_datetime(
                    ["2024-01-03 13:30Z", "2024-01-04 13:30Z", "2024-01-08 19:00Z"]
                ),
                "available_at": [known, known, known],
            }
        )
        result = build_event_counts(
            events,
            open_dates,
            pd.Series([pd.Timestamp("2024-01-02")]),
            horizons=[5],
            cutoff_time="15:05",
        )
        self.assertEqual(result.iloc[0]["cpi_count_5d"], 1)
        self.assertEqual(result.iloc[0]["nfp_count_5d"], 1)
        self.assertEqual(result.iloc[0]["fomc_count_5d"], 1)
        self.assertFalse(result.iloc[0]["calendar_missing_flag"])

    def test_unknown_calendar_revision_is_not_used(self):
        open_dates = pd.Series(pd.bdate_range("2024-01-02", periods=8))
        events = pd.DataFrame(
            {
                "event_type": ["cpi", "nfp", "fomc"],
                "timestamp_utc": pd.to_datetime(
                    ["2024-01-03 13:30Z", "2024-02-02 13:30Z", "2024-03-20 18:00Z"]
                ),
                "available_at": pd.to_datetime(
                    ["2024-01-03 15:00Z", "2024-01-01 00:00Z", "2024-01-01 00:00Z"]
                ),
            }
        )
        result = build_event_counts(
            events,
            open_dates,
            pd.Series([pd.Timestamp("2024-01-02")]),
            horizons=[5],
            cutoff_time="15:05",
        )
        self.assertEqual(result.iloc[0]["cpi_count_5d"], 0)

    def test_event_after_final_au_close_is_outside_target_window(self):
        open_dates = pd.Series(pd.bdate_range("2024-01-02", periods=8))
        known = pd.Timestamp("2024-01-01 00:00Z")
        # The 5th future trade day is Jan 9. 06:00 UTC is 14:00 Shanghai and
        # 13:00 UTC is 21:00 Shanghai, after that AU trade day has ended.
        events = pd.DataFrame(
            {
                "event_type": ["cpi", "cpi", "nfp", "fomc"],
                "timestamp_utc": pd.to_datetime(
                    [
                        "2024-01-09 06:00Z",
                        "2024-01-09 13:00Z",
                        "2024-01-03 13:30Z",
                        "2024-01-08 19:00Z",
                    ]
                ),
                "available_at": [known, known, known, known],
            }
        )
        result = build_event_counts(
            events,
            open_dates,
            pd.Series([pd.Timestamp("2024-01-02")]),
            horizons=[5],
            cutoff_time="15:05",
        )
        self.assertEqual(result.iloc[0]["cpi_count_5d"], 1)

    def test_targets_start_at_t_plus_one(self):
        frame = pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2024-01-01", periods=41),
                "rv": np.arange(1.0, 42.0),
            }
        )
        result = add_forward_targets(frame, [5, 20, 40], epsilon=1e-12)
        self.assertAlmostEqual(result.iloc[0]["target_rv_5d"], 4.0)
        self.assertAlmostEqual(result.iloc[0]["target_rv_20d"], 11.5)
        self.assertAlmostEqual(result.iloc[0]["target_rv_40d"], 21.5)
        self.assertEqual(
            result.iloc[0]["target_maturity_date_5d"], frame.iloc[5]["trade_date"]
        )
        self.assertEqual(
            result.iloc[0]["target_maturity_date_40d"], frame.iloc[40]["trade_date"]
        )


if __name__ == "__main__":
    unittest.main()
