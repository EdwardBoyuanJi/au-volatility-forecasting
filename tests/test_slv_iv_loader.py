import math
import unittest

import pandas as pd

from au_rv.data.slv_iv_loader import (
    black76_price,
    implied_volatility_black76,
    normalise_daily_ohlcv,
    normalise_slv_definitions,
    select_near_30d_symbols,
)


class SlvIvLoaderTests(unittest.TestCase):
    def test_daily_bar_uses_utc_session_label_without_prior_day_shift(self):
        raw = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(["2024-01-02 00:00:00Z"]),
                "symbol": ["SLV"],
                "close": [22.0],
            }
        ).set_index("ts_event")
        result = normalise_daily_ohlcv(raw, option_data=False)
        self.assertEqual(result.loc[0, "observation_date"], pd.Timestamp("2024-01-02"))
        self.assertEqual(
            result.loc[0, "available_at"], pd.Timestamp("2024-01-02 21:15:00Z")
        )

    def test_missing_activation_uses_definition_timestamp(self):
        raw = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(["2024-01-02 14:31:00Z"]),
                "raw_symbol": ["SLV   240216C00022000"],
                "instrument_class": ["C"],
                "strike_price": [22.0],
                "expiration": pd.to_datetime(["2024-02-16"], utc=True),
                "activation": [pd.NaT],
            }
        ).set_index("ts_event")
        result = normalise_slv_definitions(raw)
        self.assertEqual(result.loc[0, "activation"], result.loc[0, "ts_event"])

    def test_black76_inversion(self):
        price = black76_price(25.0, 25.0, 30 / 365, 0.35, 0.03, True)
        sigma = implied_volatility_black76(
            price, 25.0, 25.0, 30 / 365, 0.03, True
        )
        self.assertAlmostEqual(sigma, 0.35, places=8)

    def test_symbol_selection_uses_bracketing_expiries_and_nearby_strikes(self):
        rows = []
        for dte in (20, 40):
            for strike in (20.0, 24.0, 25.0, 26.0, 30.0):
                for cp in ("C", "P"):
                    rows.append(
                        {
                            "raw_symbol": f"SLV   x{dte}{cp}{strike}",
                            "instrument_class": cp,
                            "strike_price": strike,
                            "expiration": pd.Timestamp("2024-01-01", tz="UTC")
                            + pd.Timedelta(days=dte),
                            "activation": pd.Timestamp("2023-01-01", tz="UTC"),
                        }
                    )
        definitions = pd.DataFrame(rows)
        underlying = pd.DataFrame(
            {"observation_date": [pd.Timestamp("2024-01-01")], "close": [25.0]}
        )
        symbols = select_near_30d_symbols(
            definitions,
            underlying,
            target_dte=30,
            minimum_dte=14,
            maximum_dte=60,
            strikes_each_side=1,
            strike_band_fraction=0.2,
        )
        self.assertEqual(len(symbols), 12)
        self.assertTrue(all("24.0" in s or "25.0" in s or "26.0" in s for s in symbols))


if __name__ == "__main__":
    unittest.main()
