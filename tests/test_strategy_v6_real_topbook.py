from __future__ import annotations

import pandas as pd

from au_rv.strategy_v6.real_backtest import _validate_futures_orders
from au_rv.strategy_v6.real_execution import build_futures_close_snapshots


def test_close_snapshot_uses_last_quote_before_target() -> None:
    ticks = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 3,
            "ts_code": ["SHFE.au2602"] * 3,
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05 06:59:58+00:00",
                    "2026-01-05 06:59:59+00:00",
                    "2026-01-05 07:00:01+00:00",
                ],
                utc=True,
            ),
            "bid_price1": [999.0, 1000.0, 1001.0],
            "ask_price1": [1000.0, 1001.0, 1002.0],
            "bid_volume1": [5, 6, 7],
            "ask_volume1": [8, 9, 10],
        }
    )
    result = build_futures_close_snapshots(
        ticks, target_time="15:00:00", maximum_age_seconds=60
    )
    assert len(result) == 1
    assert result.iloc[0]["bid_price1"] == 1000.0
    assert result.iloc[0]["ask_price1"] == 1001.0
    assert result.iloc[0]["quote_age_seconds"] == 1.0


def test_futures_order_validation_requires_real_side_and_visible_size() -> None:
    orders = pd.DataFrame(
        [
            {
                "instrument_type": "FUTURE",
                "side": "BUY",
                "effective_price": 1001.0,
                "bid_price1": 1000.0,
                "ask_price1": 1001.0,
                "quantity": 2,
                "bid_volume1": 3,
                "ask_volume1": 2,
                "timestamp_utc": "2026-01-05 07:00:00+00:00",
                "price_source": "REAL_TQ_TOPBOOK_AT_OR_BEFORE_1500",
            },
            {
                "instrument_type": "FUTURE",
                "side": "SELL",
                "effective_price": 1000.0,
                "bid_price1": 1000.0,
                "ask_price1": 1001.0,
                "quantity": 3,
                "bid_volume1": 3,
                "ask_volume1": 2,
                "timestamp_utc": "2026-01-05 07:00:00+00:00",
                "price_source": "REAL_TQ_TOPBOOK_AT_OR_BEFORE_1500",
            },
        ]
    )
    audit = _validate_futures_orders(orders)
    assert bool(audit["execution_valid"].all())
