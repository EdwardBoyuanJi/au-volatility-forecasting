import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from au_rv.features.comex_nonoverlap import (
    aggregate_ohlcv_1m_to_5m,
    comex_nonoverlap_mask,
)
from au_rv.features.realized_variance import (
    five_minute_log_returns,
    select_daily_au_contracts,
)
from au_rv.time_utils import assign_shfe_trade_date


class TimeContractAndComexTests(unittest.TestCase):
    def test_night_session_maps_to_next_open_trade_date(self):
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2024-01-05 21:05:00+08:00",  # Friday night
                    "2024-01-06 01:00:00+08:00",  # Saturday early morning
                    "2024-01-08 09:05:00+08:00",  # Monday day session
                ]
            )
        )
        open_dates = pd.Series(pd.to_datetime(["2024-01-05", "2024-01-08"]))
        mapped = assign_shfe_trade_date(timestamps, open_dates)
        expected = pd.Timestamp("2024-01-08")
        self.assertTrue(mapped.eq(expected).all())

    def test_contract_selection_uses_previous_day_oi_then_volume(self):
        rows = []
        for day, values in [
            ("2024-01-02", [("AU1", 100, 50), ("AU2", 90, 200)]),
            ("2024-01-03", [("AU1", 80, 10), ("AU2", 120, 20)]),
            ("2024-01-04", [("AU1", 70, 10), ("AU2", 130, 20)]),
        ]:
            for contract, oi, volume in values:
                rows.append(
                    {
                        "trade_date": pd.Timestamp(day),
                        "ts_code": contract,
                        "timestamp_utc": pd.Timestamp(day, tz="UTC"),
                        "oi": oi,
                        "vol": volume,
                        "close": 100,
                    }
                )
        selected = select_daily_au_contracts(pd.DataFrame(rows))
        self.assertEqual(selected.iloc[1]["selected_au_contract"], "AU1")
        self.assertEqual(selected.iloc[2]["selected_au_contract"], "AU2")
        self.assertTrue(selected.iloc[2]["contract_roll_flag"])

    def test_contract_roll_cannot_create_a_return(self):
        bars = pd.DataFrame(
            {
                "ts_code": ["AU1", "AU2"],
                "session_id": ["s", "s"],
                "timestamp_utc": pd.to_datetime(
                    ["2024-01-02 01:00Z", "2024-01-02 01:05Z"]
                ),
                "close": [100.0, 150.0],
            }
        )
        returns = five_minute_log_returns(bars)
        self.assertTrue(returns.isna().all())

    def test_comex_nonoverlap_mask_excludes_au_slots(self):
        gc = pd.Series(
            pd.to_datetime(
                ["2024-01-02 01:00Z", "2024-01-02 01:05Z", "2024-01-02 01:10Z"]
            )
        )
        au = pd.Series(pd.to_datetime(["2024-01-02 01:05Z"]))
        mask = comex_nonoverlap_mask(gc, au)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_comex_aggregation_rejects_partial_five_minute_bins(self):
        timestamps = pd.to_datetime(
            [
                "2024-01-02 00:01Z",
                "2024-01-02 00:02Z",
                "2024-01-02 00:03Z",
                "2024-01-02 00:04Z",
                "2024-01-02 00:05Z",
                "2024-01-02 00:06Z",
                "2024-01-02 00:07Z",
                "2024-01-02 00:08Z",
                "2024-01-02 00:09Z",
            ]
        )
        frame = pd.DataFrame(
            {
                "symbol": "GCG4",
                "timestamp_utc": timestamps,
                "available_at": timestamps + pd.Timedelta(minutes=1),
                "open": np.arange(9.0),
                "high": np.arange(9.0) + 1,
                "low": np.arange(9.0) - 1,
                "close": np.arange(9.0) + 0.5,
                "volume": 1,
            }
        )
        result = aggregate_ohlcv_1m_to_5m(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["source_minute_count"], 5)
        self.assertEqual(result.iloc[0]["timestamp_utc"], pd.Timestamp("2024-01-02 00:10Z"))

    def test_comex_aggregation_does_not_turn_missing_volume_into_zero(self):
        timestamps = pd.date_range("2024-01-02 00:05Z", periods=5, freq="1min")
        frame = pd.DataFrame(
            {
                "symbol": "GCG4",
                "timestamp_utc": timestamps,
                "available_at": timestamps + pd.Timedelta(minutes=1),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": [1.0, 1.0, np.nan, 1.0, 1.0],
            }
        )
        self.assertTrue(aggregate_ohlcv_1m_to_5m(frame).empty)

    def test_us_dst_conversion_is_not_fixed_offset(self):
        zone = ZoneInfo("America/New_York")
        before = pd.Timestamp("2024-03-08 08:30").tz_localize(zone).tz_convert("UTC")
        after = pd.Timestamp("2024-03-11 08:30").tz_localize(zone).tz_convert("UTC")
        self.assertEqual(before.hour, 13)
        self.assertEqual(after.hour, 12)


if __name__ == "__main__":
    unittest.main()
