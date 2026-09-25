import unittest
from unittest.mock import patch

import pandas as pd

from au_rv.data.databento_loader import (
    _normalise_ohlcv,
    _request_parameters,
    estimate_databento_cost,
)


class DatabentoLoaderTests(unittest.TestCase):
    def test_parent_symbol_spreads_are_excluded(self):
        frame = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(
                    ["2024-01-02 00:00Z", "2024-01-02 00:00Z"]
                ),
                "symbol": ["GCM4", "GCM4-GCZ4"],
                "open": [2000.0, -3.0],
                "high": [2001.0, -2.0],
                "low": [1999.0, -4.0],
                "close": [2000.5, -3.5],
                "volume": [100, 500],
            }
        )
        result = _normalise_ohlcv(frame)
        self.assertEqual(result["symbol"].tolist(), ["GCM4"])

    def test_continuous_contract_instrument_ids_are_retained(self):
        frame = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(["2024-01-02 00:00Z"]),
                "instrument_id": [12345],
                "symbol": ["GC.v.0"],
                "open": [2000.0],
                "high": [2001.0],
                "low": [1999.0],
                "close": [2000.5],
                "volume": [100],
            }
        )
        result = _normalise_ohlcv(frame)
        self.assertEqual(result["symbol"].tolist(), ["instrument_id:12345"])

    def test_cost_request_omits_timeseries_only_output_symbology(self):
        class Metadata:
            def __init__(self):
                self.parameters = None

            def get_cost(self, **parameters):
                self.parameters = parameters
                return 1.25

            def get_dataset_range(self, dataset):
                return {
                    "start": "2010-06-06T00:00:00.000000000Z",
                    "end": "2026-08-22T14:30:00.000000000Z",
                }

        class Client:
            def __init__(self):
                self.metadata = Metadata()

        client = Client()
        config = {
            "_project_root": ".",
            "data_sources": {
                "databento": {
                    "dataset": "GLBX.MDP3",
                    "symbols": ["GC.FUT"],
                    "stype_in": "parent",
                    "stype_out": "raw_symbol",
                    "schema": "ohlcv-1m",
                }
            },
        }
        with (
            patch("au_rv.data.databento_loader._client", return_value=client),
            patch("au_rv.data.databento_loader.write_json"),
        ):
            estimate = estimate_databento_cost(
                config,
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-02T00:00:00Z"),
            )
        self.assertNotIn("stype_out", client.metadata.parameters)
        self.assertEqual(
            estimate["download_request"]["stype_out"],
            "raw_symbol",
        )

    def test_request_end_is_capped_at_dataset_availability(self):
        config = {
            "data_sources": {
                "databento": {
                    "dataset": "GLBX.MDP3",
                    "symbols": ["GC.FUT"],
                    "stype_in": "parent",
                    "stype_out": "raw_symbol",
                    "schema": "ohlcv-1m",
                }
            }
        }
        parameters = _request_parameters(
            config,
            "2026-08-01",
            "2026-08-22",
            available_end="2026-08-22T14:30:00Z",
        )
        self.assertEqual(parameters["end"], "2026-08-22T14:30:00+00:00")


if __name__ == "__main__":
    unittest.main()
