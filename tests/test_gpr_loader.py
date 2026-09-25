import unittest

import pandas as pd

from au_rv.data.gpr_loader import normalise_gpr_frame


class GprLoaderTests(unittest.TestCase):
    def test_conservative_weekly_availability(self):
        raw = pd.DataFrame(
            {"date": ["2024-01-01", "2024-01-02"], "GPRD": [100.0, 200.0]}
        )
        result = normalise_gpr_frame(
            raw, release_lag_days=7, conservative_time_et="23:59:59"
        )
        self.assertEqual(result.loc[0, "release_assumption_date"], pd.Timestamp("2024-01-08"))
        self.assertGreater(result.loc[0, "available_at"], pd.Timestamp("2024-01-08", tz="UTC"))
        self.assertEqual(
            result.loc[0, "vintage_mode"],
            "current_workbook_conservative_weekly_lag",
        )


if __name__ == "__main__":
    unittest.main()
