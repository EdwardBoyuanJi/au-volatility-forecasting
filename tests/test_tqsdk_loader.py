import unittest

import pandas as pd

from au_rv.data.tqsdk_loader import (
    _eligible_contracts,
    _normalise_bars,
    _normalise_calendar,
    _normalise_contracts,
)


class TqSdkLoaderTests(unittest.TestCase):
    def test_kline_start_timestamp_is_available_only_after_five_minutes(self):
        start = pd.Timestamp("2024-01-02 09:00:00", tz="Asia/Shanghai")
        frame = pd.DataFrame(
            {
                "datetime": [start.tz_convert("UTC").value],
                "open": [480.0],
                "high": [481.0],
                "low": [479.0],
                "close": [480.5],
                "volume": [12],
                "close_oi": [3456],
            }
        )
        result = _normalise_bars(frame, "SHFE.au2406", duration_seconds=300)
        self.assertEqual(result.iloc[0]["timestamp_utc"], pd.Timestamp("2024-01-02 01:00Z"))
        self.assertEqual(result.iloc[0]["available_at"], pd.Timestamp("2024-01-02 01:05Z"))
        self.assertEqual(result.iloc[0]["timestamp_shanghai"], start)
        self.assertEqual(result.iloc[0]["vol"], 12)
        self.assertEqual(result.iloc[0]["oi"], 3456)

    def test_calendar_is_converted_to_feature_contract(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=4, freq="D"),
                "trading": [True, False, False, True],
            }
        )
        result = _normalise_calendar(frame)
        self.assertEqual(result["is_open"].tolist(), [1, 0, 0, 1])
        self.assertEqual(result.iloc[-1]["pretrade_date"], pd.Timestamp("2024-01-05"))
        self.assertEqual(
            result.iloc[-1]["timestamp_utc"], pd.Timestamp("2024-01-07 16:00Z")
        )

    def test_only_true_au_futures_near_the_sample_are_downloaded(self):
        info = pd.DataFrame(
            {
                "instrument_id": [
                    "SHFE.au2406",
                    "SHFE.au2506",
                    "SHFE.au2706",
                    "SHFE.au2406C500",
                    "SHFE.cu2406",
                ],
                "expire_datetime": [
                    pd.Timestamp("2024-06-20", tz="Asia/Shanghai").timestamp(),
                    pd.Timestamp("2025-06-20", tz="Asia/Shanghai").timestamp(),
                    pd.Timestamp("2027-06-20", tz="Asia/Shanghai").timestamp(),
                    pd.Timestamp("2024-05-20", tz="Asia/Shanghai").timestamp(),
                    pd.Timestamp("2024-06-20", tz="Asia/Shanghai").timestamp(),
                ],
                "delivery_year": [2024, 2025, 2027, 2024, 2024],
                "delivery_month": [6, 6, 6, 6, 6],
            }
        )
        contracts = _normalise_contracts(info)
        selected = _eligible_contracts(
            contracts,
            "2024-01-01",
            "2024-12-31",
            listing_lead_buffer_days=370,
        )
        self.assertEqual(selected, ["SHFE.au2406", "SHFE.au2506"])


if __name__ == "__main__":
    unittest.main()
