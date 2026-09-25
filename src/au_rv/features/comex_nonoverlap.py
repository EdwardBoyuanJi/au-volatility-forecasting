from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.time_utils import model_cutoff


def aggregate_ohlcv_1m_to_5m(frame: pd.DataFrame) -> pd.DataFrame:
    """Strictly aggregate within a symbol and contiguous one-minute segment."""

    pieces: list[pd.DataFrame] = []
    bars = frame.sort_values(["symbol", "timestamp_utc"]).copy()
    for symbol, group in bars.groupby("symbol", sort=False):
        group = group.copy()
        gap = pd.to_datetime(group["timestamp_utc"], utc=True).diff()
        group["segment"] = gap.gt(pd.Timedelta(minutes=1)).cumsum()
        for _, segment in group.groupby("segment", sort=False):
            segment = segment.copy()
            segment["_complete_ohlcv"] = segment[
                ["open", "high", "low", "close", "volume"]
            ].notna().all(axis=1)
            indexed = segment.set_index(pd.to_datetime(segment["timestamp_utc"], utc=True))
            aggregated = indexed.resample("5min", label="right", closed="left", origin="epoch").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", lambda values: values.sum(min_count=5)),
                source_minute_count=("_complete_ohlcv", "sum"),
                available_at=("available_at", "max"),
            )
            aggregated = aggregated.dropna(
                subset=["open", "high", "low", "close", "volume"]
            )
            # Never treat a partial bin as a five-minute bar. Segmentation
            # prevents filling gaps; this equality also removes edge fragments
            # when a continuous segment starts or ends inside a clock bin.
            aggregated = aggregated[aggregated["source_minute_count"].eq(5)]
            aggregated = aggregated.reset_index(names="timestamp_utc")
            aggregated["symbol"] = symbol
            pieces.append(aggregated)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    result["timestamp_original"] = result["timestamp_utc"].astype(str)
    result["timezone_original"] = "UTC"
    result["timestamp_shanghai"] = pd.to_datetime(
        result["timestamp_utc"], utc=True
    ).dt.tz_convert("Asia/Shanghai")
    result["available_at"] = pd.concat(
        [
            pd.to_datetime(result["available_at"], utc=True),
            pd.to_datetime(result["timestamp_utc"], utc=True),
        ],
        axis=1,
    ).max(axis=1)
    return result.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def comex_nonoverlap_mask(
    gc_bar_ends_utc: pd.Series,
    au_bar_ends_utc: pd.Series,
) -> pd.Series:
    gc = pd.to_datetime(gc_bar_ends_utc, utc=True).dt.floor("5min")
    au_set = set(pd.to_datetime(au_bar_ends_utc, utc=True).dt.floor("5min"))
    return ~gc.isin(au_set)


def _select_observable_gc_contract(
    gc_5m: pd.DataFrame,
    selection_cutoff_utc: pd.Timestamp,
) -> str | None:
    window_start = selection_cutoff_utc - pd.Timedelta(hours=24)
    candidates = gc_5m[
        pd.to_datetime(gc_5m["available_at"], utc=True).le(selection_cutoff_utc)
        & pd.to_datetime(gc_5m["timestamp_utc"], utc=True).gt(window_start)
    ]
    if candidates.empty:
        return None
    volumes = (
        candidates.groupby("symbol", as_index=False)
        .agg(volume=("volume", lambda values: values.sum(min_count=1)))
        .dropna(subset=["volume"])
        .sort_values(["volume", "symbol"], ascending=[False, True])
    )
    return str(volumes.iloc[0]["symbol"]) if not volumes.empty else None


def compute_comex_nonoverlap_daily(
    gc_5m: pd.DataFrame,
    all_au_bars: pd.DataFrame,
    trade_dates: pd.Series,
    *,
    cutoff_time: str,
    epsilon: float,
    minimum_bars: int,
) -> pd.DataFrame:
    dates = pd.Index(sorted(pd.to_datetime(trade_dates).dropna().unique()))
    rows: list[dict] = []
    previous_contract: str | None = None
    for position, trade_date in enumerate(dates):
        current_cutoff = model_cutoff(trade_date, cutoff_time).tz_convert("UTC")
        if position == 0:
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "selected_gc_contract": None,
                    "gc_contract_roll_flag": False,
                    "comex_nonoverlap_bar_count": 0,
                    "missing_comex_bars": 0,
                    "comex_nonoverlap_rv": np.nan,
                    "log_comex_nonoverlap_rv_1d": np.nan,
                    "comex_quality_flag": False,
                    "comex_available_at": pd.NaT,
                }
            )
            continue
        previous_cutoff = model_cutoff(dates[position - 1], cutoff_time).tz_convert("UTC")
        selected = _select_observable_gc_contract(gc_5m, previous_cutoff)
        window = gc_5m[
            pd.to_datetime(gc_5m["timestamp_utc"], utc=True).gt(previous_cutoff)
            & pd.to_datetime(gc_5m["timestamp_utc"], utc=True).le(current_cutoff)
            & pd.to_datetime(gc_5m["available_at"], utc=True).le(current_cutoff)
        ].copy()
        au_window = all_au_bars[
            pd.to_datetime(all_au_bars["timestamp_utc"], utc=True).gt(previous_cutoff)
            & pd.to_datetime(all_au_bars["timestamp_utc"], utc=True).le(current_cutoff)
        ]
        window["is_nonoverlap"] = comex_nonoverlap_mask(
            window["timestamp_utc"], au_window["timestamp_utc"]
        )
        eligible_slots = set(
            pd.to_datetime(window.loc[window["is_nonoverlap"], "timestamp_utc"], utc=True)
        )
        chosen = window[
            window["symbol"].astype(str).eq(str(selected)) & window["is_nonoverlap"]
        ].sort_values("timestamp_utc")
        previous_time = pd.to_datetime(chosen["timestamp_utc"], utc=True).shift(1)
        previous_close = chosen["close"].shift(1)
        previous_nonoverlap = chosen["is_nonoverlap"].shift(1).fillna(False).astype(bool)
        valid = (
            previous_close.gt(0)
            & chosen["close"].gt(0)
            & (pd.to_datetime(chosen["timestamp_utc"], utc=True) - previous_time).eq(
                pd.Timedelta(minutes=5)
            )
            & previous_nonoverlap
        )
        returns = np.log(chosen.loc[valid, "close"] / previous_close.loc[valid])
        rv = float(np.sum(returns**2)) if len(returns) else np.nan
        chosen_slots = set(pd.to_datetime(chosen["timestamp_utc"], utc=True))
        missing = max(len(eligible_slots.difference(chosen_slots)), 0)
        quality = selected is not None and len(returns) >= minimum_bars and np.isfinite(rv)
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "selected_gc_contract": selected,
                "gc_contract_roll_flag": bool(
                    selected is not None
                    and previous_contract is not None
                    and selected != previous_contract
                ),
                "comex_nonoverlap_bar_count": int(len(returns)),
                "missing_comex_bars": int(missing),
                "comex_nonoverlap_rv": rv,
                "log_comex_nonoverlap_rv_1d": (
                    float(np.log(rv + epsilon)) if np.isfinite(rv) else np.nan
                ),
                "comex_quality_flag": bool(quality),
                "comex_available_at": current_cutoff if quality else pd.NaT,
            }
        )
        if selected is not None:
            previous_contract = selected
    return pd.DataFrame(rows)
