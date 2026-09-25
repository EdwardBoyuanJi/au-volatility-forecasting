import unittest
from unittest.mock import patch

import pandas as pd

from au_rv.data.fred_loader import _date_chunks, _download_initial_release_series


class _FakeResponse:
    def __init__(self, payload, *, ok=True, status_code=200):
        self.payload = payload
        self.ok = ok
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FredLoaderTests(unittest.TestCase):
    def test_vintage_chunks_do_not_exceed_four_calendar_years(self):
        chunks = list(_date_chunks("2016-01-01", "2026-08-22", 4))
        self.assertEqual(chunks[0], (pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31")))
        self.assertEqual(chunks[-1], (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-08-22")))

    def test_initial_release_request_is_paginated_and_timestamped_conservatively(self):
        pages = [
            _FakeResponse(
                {
                    "count": 3,
                    "observations": [
                        {
                            "date": "2024-01-02",
                            "realtime_start": "2024-01-03",
                            "value": "100",
                        },
                        {
                            "date": "2024-01-03",
                            "realtime_start": "2024-01-04",
                            "value": "101",
                        },
                    ],
                }
            ),
            _FakeResponse(
                {
                    "count": 3,
                    "observations": [
                        {
                            "date": "2024-01-04",
                            "realtime_start": "2024-01-05",
                            "value": "102",
                        }
                    ],
                }
            ),
        ]
        with patch("au_rv.data.fred_loader.requests.get", side_effect=pages) as get:
            frame = _download_initial_release_series(
                "TEST",
                "secret",
                "2024-01-01",
                "2024-01-31",
                "23:59:59",
            )

        self.assertEqual(len(frame), 3)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["output_type"], 4)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["offset"], 0)
        self.assertEqual(get.call_args_list[1].kwargs["params"]["offset"], 2)
        self.assertEqual(
            frame.iloc[0]["available_at"],
            pd.Timestamp("2024-01-04 04:59:59Z"),
        )

    def test_http_errors_do_not_expose_api_key(self):
        response = _FakeResponse(
            {"error_message": "Bad Request"}, ok=False, status_code=400
        )
        with patch("au_rv.data.fred_loader.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "FRED TEST request failed") as caught:
                _download_initial_release_series(
                    "TEST", "super-secret-key", "2024-01-01", "2024-01-31", "23:59:59"
                )
        self.assertNotIn("super-secret-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
