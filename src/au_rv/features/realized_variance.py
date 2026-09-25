from __future__ import annotations

import numpy as np
import pandas as pd

from au_rv.features.jump import bns_jump_measures
from au_rv.time_utils import assign_shfe_trade_date, model_cutoff, shfe_session_label


def realized_quarticity(returns) -> float:
    """Five-minute realized quarticity, ``M / 3 * sum(r_i**4)``.

    The normalization follows Bollerslev, Patton, and Quaedvlieg (2016).
    It estimates integrated quarticity and therefore the time-varying
    measurement-error variance of realized variance used by HARQ.
    """

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    count = values.size
    if count < 1:
        return np.nan
    return float((count / 3.0) * np.sum(values**4))


def five_minute_log_returns(
    bars: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp_utc",
    price_column: str = "close",
    contract_column: str = "ts_code",
    session_column: str = "session_id",
    bar_minutes: int = 5,
) -> pd.Series:
    """Return NaN at every contract, session, or time-gap boundary."""

    ordered = bars.sort_values([contract_column, timestamp_column]).copy()
    grouped = ordered.groupby([contract_column, session_column], dropna=False, sort=False)
    previous_price = grouped[price_column].shift(1)
    previous_time = grouped[timestamp_column].shift(1)
    gap = pd.to_datetime(ordered[timestamp_column], utc=True) - pd.to_datetime(
        previous_time, utc=True
    )
    valid = (
        previous_price.gt(0)
        & ordered[price_column].gt(0)
        & gap.eq(pd.Timedelta(minutes=bar_minutes))
        & ordered[session_column].notna()
    )
    result = pd.Series(np.nan, index=ordered.index, dtype=float)
    result.loc[valid] = np.log(
        ordered.loc[valid, price_column].astype(float) / previous_price.loc[valid].astype(float)
    )
    return result.reindex(bars.index)


def prepare_au_bars(bars: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    open_dates = calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"]
    result["trade_date"] = assign_shfe_trade_date(result["timestamp_shanghai"], open_dates)
    result["session_id"] = [
        shfe_session_label(stamp, day)
        if pd.notna(stamp) and pd.notna(day)
        else None
        for stamp, day in zip(result["timestamp_shanghai"], result["trade_date"])
    ]
    return result[result["session_id"].notna() & result["trade_date"].notna()].copy()


def select_daily_au_contracts(prepared_bars: pd.DataFrame) -> pd.DataFrame:
    bars = prepared_bars.sort_values(["trade_date", "ts_code", "timestamp_utc"]).copy()
    daily = (
        bars.groupby(["trade_date", "ts_code"], as_index=False)
        .agg(
            previous_close_oi=("oi", "last"),
            previous_volume=("vol", lambda values: values.sum(min_count=1)),
            bar_count=("close", "count"),
        )
    )
    dates = pd.Index(sorted(bars["trade_date"].dropna().unique()))
    selections: list[dict] = []
    for position, trade_date in enumerate(dates):
        if position == 0:
            selections.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "selected_au_contract": None,
                    "selection_source_date": pd.NaT,
                }
            )
            continue
        previous_date = pd.Timestamp(dates[position - 1])
        present_today = set(
            daily.loc[daily["trade_date"].eq(trade_date), "ts_code"].astype(str)
        )
        candidates = daily[
            daily["trade_date"].eq(previous_date)
            & daily["ts_code"].astype(str).isin(present_today)
        ].copy()
        candidates = candidates.dropna(subset=["previous_close_oi", "previous_volume"])
        candidates = candidates.sort_values(
            ["previous_close_oi", "previous_volume", "ts_code"],
            ascending=[False, False, True],
        )
        selected = None if candidates.empty else str(candidates.iloc[0]["ts_code"])
        selections.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "selected_au_contract": selected,
                "selection_source_date": previous_date,
            }
        )
    result = pd.DataFrame(selections)
    result["contract_roll_flag"] = (
        result["selected_au_contract"].notna()
        & result["selected_au_contract"].shift(1).notna()
        & result["selected_au_contract"].ne(result["selected_au_contract"].shift(1))
    )
    return result


def compute_au_daily_measures(
    prepared_bars: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    cutoff_time: str,
    bar_minutes: int,
    minimum_bars: int,
    jump_alpha: float,
    finite_sample: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    selected_bar_pieces: list[pd.DataFrame] = []
    previous_close_by_contract: dict[str, float] = {}
    for selection in selections.itertuples(index=False):
        day_bars = prepared_bars[prepared_bars["trade_date"].eq(selection.trade_date)].copy()
        cutoff = model_cutoff(selection.trade_date, cutoff_time)
        cutoff_utc = cutoff.tz_convert("UTC")
        day_bars = day_bars[
            pd.to_datetime(day_bars["available_at"], utc=True).le(cutoff_utc)
        ].copy()
        market_slots = set(pd.to_datetime(day_bars["timestamp_utc"], utc=True))
        if selection.selected_au_contract is None:
            chosen = day_bars.iloc[0:0].copy()
        else:
            chosen = day_bars[
                day_bars["ts_code"].astype(str).eq(str(selection.selected_au_contract))
            ].copy()
        chosen = chosen.sort_values("timestamp_utc")
        daily_close = (
            float(chosen["close"].iloc[-1])
            if not chosen.empty and float(chosen["close"].iloc[-1]) > 0
            else np.nan
        )
        previous_close = previous_close_by_contract.get(
            str(selection.selected_au_contract), np.nan
        )
        daily_log_return = (
            float(np.log(daily_close / previous_close))
            if np.isfinite(daily_close)
            and np.isfinite(previous_close)
            and previous_close > 0
            else np.nan
        )
        chosen["log_return_5m"] = five_minute_log_returns(
            chosen,
            bar_minutes=bar_minutes,
        )
        selected_slots = set(pd.to_datetime(chosen["timestamp_utc"], utc=True))
        missing_bars = max(len(market_slots.difference(selected_slots)), 0)
        valid_returns = chosen["log_return_5m"].dropna().to_numpy()
        measures = bns_jump_measures(
            valid_returns,
            alpha=jump_alpha,
            finite_sample=finite_sample,
        )
        quality = (
            selection.selected_au_contract is not None
            and len(chosen) >= minimum_bars
            and measures["return_count"] >= 3
            and np.isfinite(measures["rv"])
        )
        rows.append(
            {
                "trade_date": pd.Timestamp(selection.trade_date),
                "model_cutoff": cutoff,
                "selected_au_contract": selection.selected_au_contract,
                "selection_source_date": selection.selection_source_date,
                "contract_roll_flag": bool(selection.contract_roll_flag),
                "au_bar_count": int(len(chosen)),
                "missing_au_bars": int(missing_bars),
                "rv": measures["rv"],
                "daily_close": daily_close,
                "daily_log_return": daily_log_return,
                "bpv": measures["bpv"],
                "realized_quarticity": realized_quarticity(valid_returns),
                "tripower_quarticity": measures["tripower_quarticity"],
                "bns_statistic": measures["bns_statistic"],
                "bns_critical_value": measures["bns_critical_value"],
                "jump_significant": measures["jump_significant"],
                "jump_var_1d": measures["jump_var_1d"],
                "au_quality_flag": bool(quality),
                "au_available_at": cutoff_utc,
            }
        )
        if not chosen.empty:
            selected_bar_pieces.append(chosen)
        # Update every contract only after the current feature row has consumed
        # its previous close.  On a roll date this yields a same-contract return
        # for the newly selected contract and never a synthetic cross-contract
        # jump.
        valid_day_bars = day_bars.sort_values("timestamp_utc").dropna(subset=["close"])
        for code, group in valid_day_bars.groupby("ts_code"):
            close = float(group["close"].iloc[-1])
            if close > 0:
                previous_close_by_contract[str(code)] = close
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    selected = (
        pd.concat(selected_bar_pieces, ignore_index=True)
        if selected_bar_pieces
        else prepared_bars.iloc[0:0].copy()
    )
    return daily, selected


def add_har_features(daily: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    result = daily.sort_values("trade_date").copy()
    result["log_rv_1d"] = np.log(result["rv"] + epsilon)
    result["log_rv_5d"] = np.log(
        result["rv"].rolling(5, min_periods=5).mean() + epsilon
    )
    result["log_rv_22d"] = np.log(
        result["rv"].rolling(22, min_periods=22).mean() + epsilon
    )
    result["sqrt_realized_quarticity_1d"] = np.sqrt(
        result["realized_quarticity"].clip(lower=0.0)
    )
    # Log-scale analogue of the simple HARQ interaction.  Since log_rv_1d is
    # also present in every HARQ feature set, centering sqrt(RQ) within each
    # training sample changes only the main log-RV coefficient and not the
    # model span; this avoids a full-sample (look-ahead) demeaning operation.
    result["harq_log_rv_rq_interaction_1d"] = (
        result["log_rv_1d"] * result["sqrt_realized_quarticity_1d"]
    )
    result["jump_var_5d"] = result["jump_var_1d"].rolling(5, min_periods=5).mean()
    return result
