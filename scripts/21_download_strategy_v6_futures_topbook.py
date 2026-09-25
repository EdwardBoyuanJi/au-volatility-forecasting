#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.data.tqsdk_loader import _create_api
from au_rv.io import read_parquet, utc_now, write_csv, write_parquet
from au_rv.strategy_v3.data import prepare_real_quote_opportunities


def _load_settings() -> dict:
    path = ROOT / "strategy_v6_real_topbook_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["_config_path"] = str(path)
    return settings


def _normalise_ticks(frame: pd.DataFrame, symbol: str, trade_date: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(
        pd.to_numeric(result["datetime"], errors="coerce"),
        unit="ns",
        utc=True,
        errors="coerce",
    )
    result["timestamp_shanghai"] = result["timestamp_utc"].dt.tz_convert(
        "Asia/Shanghai"
    )
    result["trade_date"] = pd.Timestamp(trade_date).normalize()
    result["ts_code"] = str(symbol)
    result["retrieved_at_utc"] = utc_now()
    numeric = [
        "last_price",
        "bid_price1",
        "bid_volume1",
        "ask_price1",
        "ask_volume1",
        "volume",
        "open_interest",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[
        [
            "trade_date",
            "ts_code",
            "timestamp_utc",
            "timestamp_shanghai",
            *numeric,
            "retrieved_at_utc",
        ]
    ].dropna(subset=["timestamp_utc"])


def _required_pairs(settings: dict) -> pd.DataFrame:
    opportunities = prepare_real_quote_opportunities(
        read_parquet(ROOT / settings["inputs"]["opportunities_path"])
    )
    futures = read_parquet(ROOT / settings["inputs"]["futures_daily_path"])
    futures["trade_date"] = pd.to_datetime(futures["trade_date"]).dt.normalize()
    start = pd.Timestamp(settings["sample"]["requested_backtest_start"]).normalize()
    end = pd.Timestamp(settings["sample"]["evaluation_end"]).normalize()
    selected = opportunities[
        opportunities["signal_model_timed_short"].eq(-1)
        & opportunities["entry_date"].between(start, end)
        & opportunities["expiry_date"].le(end)
    ]
    pieces = []
    for row in selected.itertuples(index=False):
        path = futures[
            futures["ts_code"].astype(str).eq(str(row.underlying_symbol))
            & futures["trade_date"].between(row.entry_date, row.expiry_date)
        ][["trade_date", "ts_code"]].copy()
        pieces.append(path)
    if not pieces:
        raise RuntimeError("No futures quote dates are required by the selected signals.")
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["trade_date", "ts_code"])
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )


def _save(
    tick_path: Path,
    coverage_path: Path,
    existing: pd.DataFrame,
    ticks: list[pd.DataFrame],
    coverage: list[dict],
) -> pd.DataFrame:
    pieces = ([existing] if not existing.empty else []) + ticks
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not combined.empty:
        combined = (
            combined.drop_duplicates(["trade_date", "ts_code", "timestamp_utc"], keep="last")
            .sort_values(["trade_date", "ts_code", "timestamp_utc"])
            .reset_index(drop=True)
        )
        write_parquet(combined, tick_path)
    old = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    summary = pd.concat([old, pd.DataFrame(coverage)], ignore_index=True)
    if not summary.empty:
        summary = summary.drop_duplicates(["trade_date", "ts_code"], keep="last")
        write_csv(summary, coverage_path)
    return combined


def main() -> int:
    load_dotenv(ROOT / ".env")
    settings = _load_settings()
    required = _required_pairs(settings)
    tick_path = ROOT / settings["inputs"]["hedge_tick_path"]
    coverage_path = ROOT / settings["inputs"]["hedge_coverage_path"]
    tick_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_parquet(tick_path) if tick_path.exists() else pd.DataFrame()
    attempted: set[tuple[str, str]] = set()
    if coverage_path.exists():
        prior = pd.read_csv(coverage_path)
        attempted = set(
            zip(prior["trade_date"].astype(str), prior["ts_code"].astype(str))
        )
    start_clock = datetime.strptime(
        settings["download"]["close_window_start"], "%H:%M:%S"
    ).time()
    end_clock = datetime.strptime(
        settings["download"]["close_window_end"], "%H:%M:%S"
    ).time()
    api = _create_api()
    buffer: list[pd.DataFrame] = []
    coverage_buffer: list[dict] = []
    completed = 0
    try:
        for row in required.itertuples(index=False):
            day = pd.Timestamp(row.trade_date).normalize()
            symbol = str(row.ts_code)
            key = (str(day.date()), symbol)
            if key in attempted:
                continue
            frame = api.get_tick_data_series(
                symbol=symbol,
                start_dt=datetime.combine(day.date(), start_clock),
                end_dt=datetime.combine(day.date(), end_clock),
            )
            count = 0 if frame is None else len(frame)
            if frame is not None and not frame.empty:
                buffer.append(_normalise_ticks(frame, symbol, day))
            coverage_buffer.append(
                {
                    "trade_date": day,
                    "ts_code": symbol,
                    "tick_count": int(count),
                    "downloaded_at_utc": utc_now(),
                }
            )
            completed += 1
            if completed % int(settings["download"]["flush_every"]) == 0:
                existing = _save(
                    tick_path, coverage_path, existing, buffer, coverage_buffer
                )
                buffer = []
                coverage_buffer = []
                print(f"saved {completed} newly attempted futures close windows", flush=True)
    finally:
        if buffer or coverage_buffer:
            existing = _save(
                tick_path, coverage_path, existing, buffer, coverage_buffer
            )
        api.close()
    print(f"required_pairs={len(required)}", flush=True)
    print(f"cached_tick_rows={len(existing)}", flush=True)
    print(f"ticks={tick_path}", flush=True)
    print(f"coverage={coverage_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
