import unittest
from unittest.mock import patch

import pandas as pd

from au_rv.data.event_calendar_loader import (
    parse_bls_year,
    parse_fomc_historical_year,
    download_fred_release_events,
    validate_event_calendar_coverage,
)


class _FakeFredResponse:
    ok = True
    status_code = 200

    def __init__(self, dates):
        self._dates = dates

    def json(self):
        return {"release_dates": [{"date": value} for value in self._dates]}


class EventParserTests(unittest.TestCase):
    def test_bls_parser_uses_last_modified_availability(self):
        html = """
        <html><body><table>
        <tr><th>Date</th><th>Time</th><th>Release</th></tr>
        <tr><td>Friday, January 05, 2024</td><td>08:30 AM</td>
        <td>Employment Situation for December 2023</td></tr>
        <tr><td>Thursday, January 11, 2024</td><td>08:30 AM</td>
        <td>Consumer Price Index for December 2023</td></tr>
        </table><p>NOTE: All times Eastern. Last Modified Date: November 17, 2023</p>
        </body></html>
        """
        frame = parse_bls_year(html, "https://www.bls.gov/test")
        self.assertEqual(set(frame["event_type"]), {"cpi", "nfp"})
        self.assertTrue(
            frame["availability_note"].str.contains("November 17, 2023").all()
        )

    def test_fomc_parser_excludes_unscheduled_meeting(self):
        html = """
        <html><body>
        <h5>January 29-30 Meeting - 2019</h5>
        <h5>October 4 (unscheduled) Meeting - 2019</h5>
        </body></html>
        """
        frame = parse_fomc_historical_year(
            html, "https://www.federalreserve.gov/test", 2019
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["event_type"], "fomc")

    def test_incomplete_calendar_cannot_silently_write_fomc_only(self):
        frame = pd.DataFrame(
            {
                "event_type": ["fomc"],
                "timestamp_utc": [pd.Timestamp("2024-01-31 19:00Z")],
                "available_at": [pd.Timestamp("2024-01-02 04:59:59Z")],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing all cpi events"):
            validate_event_calendar_coverage(frame, "2024-01-01", "2024-12-31")

    def test_fred_fallback_is_conservative_for_past_and_future_events(self):
        with (
            patch.dict("os.environ", {"FRED_API_KEY": "secret"}),
            patch(
                "au_rv.data.event_calendar_loader.requests.get",
                side_effect=[
                    _FakeFredResponse(["2024-01-11", "2027-01-13"]),
                    _FakeFredResponse(["2024-01-05", "2027-01-08"]),
                ],
            ),
            patch(
                "au_rv.data.event_calendar_loader.utc_now",
                return_value=pd.Timestamp("2026-08-22 12:00Z"),
            ),
        ):
            frame = download_fred_release_events(
                {"cpi": 10, "nfp": 50}, "2024-01-01", "2027-12-31"
            )
        past = frame[pd.to_datetime(frame["timestamp_utc"], utc=True).dt.year.eq(2024)]
        future = frame[pd.to_datetime(frame["timestamp_utc"], utc=True).dt.year.eq(2027)]
        self.assertTrue(
            (pd.to_datetime(past["available_at"], utc=True) == pd.to_datetime(past["timestamp_utc"], utc=True)).all()
        )
        self.assertTrue(
            (pd.to_datetime(future["available_at"], utc=True) == pd.Timestamp("2026-08-22 12:00Z")).all()
        )


if __name__ == "__main__":
    unittest.main()
