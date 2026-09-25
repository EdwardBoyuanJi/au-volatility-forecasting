from __future__ import annotations

import math
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from au_rv.config import resolve_path
from au_rv.io import utc_now, write_json, write_parquet


def _client():
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is missing.")
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements.txt before using Databento.") from exc
    return db.Historical(key)


def _normalise_index_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index()
    timestamp = next(
        (name for name in ("ts_event", "ts_recv", "timestamp", "index") if name in result),
        None,
    )
    if timestamp is None:
        raise ValueError("Databento response has no event timestamp.")
    result["ts_event"] = pd.to_datetime(result[timestamp], utc=True, errors="coerce")
    return result


def normalise_slv_definitions(frame: pd.DataFrame) -> pd.DataFrame:
    result = _normalise_index_timestamp(frame)
    required = {"raw_symbol", "instrument_class", "strike_price", "expiration"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"SLV definitions missing fields: {sorted(missing)}")
    result["raw_symbol"] = result["raw_symbol"].astype(str)
    result = result[result["raw_symbol"].str.startswith("SLV   ")].copy()
    result["expiration"] = pd.to_datetime(result["expiration"], utc=True, errors="coerce")
    result["activation"] = pd.to_datetime(
        result.get("activation", result["ts_event"]), utc=True, errors="coerce"
    ).fillna(result["ts_event"])
    result["strike_price"] = pd.to_numeric(result["strike_price"], errors="coerce")
    result = result[
        result["instrument_class"].astype(str).isin(["C", "P"])
        & result["strike_price"].gt(0)
        & result["expiration"].notna()
    ].copy()
    result["retrieved_at_utc"] = utc_now()
    keep = [
        "raw_symbol",
        "instrument_class",
        "strike_price",
        "expiration",
        "activation",
        "ts_event",
        "retrieved_at_utc",
    ]
    return (
        result[keep]
        .sort_values(["raw_symbol", "ts_event"])
        .drop_duplicates("raw_symbol", keep="last")
        .reset_index(drop=True)
    )


def normalise_daily_ohlcv(frame: pd.DataFrame, *, option_data: bool) -> pd.DataFrame:
    result = _normalise_index_timestamp(frame)
    symbol = "symbol" if "symbol" in result else "raw_symbol"
    if symbol not in result or "close" not in result:
        raise ValueError("Databento daily OHLCV response lacks symbol or close.")
    result["raw_symbol"] = result[symbol].astype(str)
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = _label_daily_bar_dates(result)
    result = result[result["close"].gt(0)].copy()
    result["retrieved_at_utc"] = utc_now()
    keep = [
        "raw_symbol",
        "observation_date",
        "ts_event",
        "available_at",
        "close",
        "retrieved_at_utc",
    ]
    if not option_data:
        result = result.sort_values("ts_event").drop_duplicates(
            "observation_date", keep="last"
        )
    else:
        result = result.sort_values("ts_event").drop_duplicates(
            ["observation_date", "raw_symbol"], keep="last"
        )
    return result[keep].reset_index(drop=True)


def _label_daily_bar_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["ts_event"], utc=True, errors="coerce")
    # Databento's ohlcv-1d interval is labelled by the UTC calendar date at the
    # start of the bar. Converting midnight UTC to New York before taking the
    # date would shift every observation to the prior day.
    result["observation_date"] = timestamp.dt.tz_localize(None).dt.normalize()
    close_text = result["observation_date"].dt.strftime("%Y-%m-%d") + " 16:15:00"
    local = pd.to_datetime(close_text).dt.tz_localize(
        ZoneInfo("America/New_York"), ambiguous="infer", nonexistent="shift_forward"
    )
    result["available_at"] = local.dt.tz_convert("UTC")
    return result


def select_near_30d_symbols(
    definitions: pd.DataFrame,
    underlying: pd.DataFrame,
    *,
    target_dte: int,
    minimum_dte: int,
    maximum_dte: int,
    strikes_each_side: int,
    strike_band_fraction: float,
    use_bracketing_expiries: bool = True,
) -> list[str]:
    defs = definitions.copy()
    defs["expiration_date"] = pd.to_datetime(defs["expiration"], utc=True).dt.tz_localize(
        None
    ).dt.normalize()
    defs["activation_date"] = pd.to_datetime(defs["activation"], utc=True).dt.tz_localize(
        None
    ).dt.normalize()
    symbols: set[str] = set()
    for row in underlying.itertuples(index=False):
        date = pd.Timestamp(row.observation_date).normalize()
        spot = float(row.close)
        candidates = defs[
            defs["activation_date"].le(date)
            & defs["expiration_date"].gt(date)
        ].copy()
        candidates["dte"] = (candidates["expiration_date"] - date).dt.days
        candidates = candidates[
            candidates["dte"].between(int(minimum_dte), int(maximum_dte))
            & candidates["strike_price"].between(
                spot * (1.0 - float(strike_band_fraction)),
                spot * (1.0 + float(strike_band_fraction)),
            )
        ]
        if candidates.empty:
            continue
        dtes = sorted(candidates["dte"].unique())
        chosen_dtes = set()
        if use_bracketing_expiries:
            lower = [value for value in dtes if value <= target_dte]
            upper = [value for value in dtes if value >= target_dte]
            if lower:
                chosen_dtes.add(max(lower))
            if upper:
                chosen_dtes.add(min(upper))
        if not chosen_dtes:
            chosen_dtes.add(min(dtes, key=lambda value: abs(value - target_dte)))
        for dte in chosen_dtes:
            expiry = candidates[candidates["dte"].eq(dte)].copy()
            strikes = (
                expiry[["strike_price"]]
                .drop_duplicates()
                .assign(distance=lambda x: (x["strike_price"] - spot).abs())
                .nsmallest(2 * int(strikes_each_side) + 1, "distance")["strike_price"]
            )
            selected = expiry[expiry["strike_price"].isin(strikes)]
            symbols.update(selected["raw_symbol"].astype(str))
    return sorted(symbols)


def black76_price(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    rate: float,
    is_call: bool,
) -> float:
    if min(forward, strike, maturity, sigma) <= 0:
        return np.nan
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / (
        sigma * root_t
    )
    d2 = d1 - sigma * root_t
    cp = 1.0 if is_call else -1.0
    return math.exp(-rate * maturity) * cp * (
        forward * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2)
    )


def implied_volatility_black76(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    is_call: bool,
) -> float:
    if min(price, forward, strike, maturity) <= 0:
        return np.nan
    discount = math.exp(-rate * maturity)
    intrinsic = discount * max(
        (forward - strike) if is_call else (strike - forward), 0.0
    )
    upper = discount * (forward if is_call else strike)
    if price <= intrinsic + 1e-8 or price >= upper:
        return np.nan
    objective = lambda sigma: black76_price(
        forward, strike, maturity, sigma, rate, is_call
    ) - price
    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=100))
    except (ValueError, RuntimeError):
        return np.nan


def compute_slv_iv30(
    definitions: pd.DataFrame,
    option_prices: pd.DataFrame,
    underlying: pd.DataFrame,
    *,
    target_dte: int,
    minimum_dte: int,
    maximum_dte: int,
    rate: float,
) -> pd.DataFrame:
    defs = definitions.copy()
    defs["expiration_date"] = pd.to_datetime(defs["expiration"], utc=True).dt.tz_localize(
        None
    ).dt.normalize()
    prices = option_prices.merge(
        defs[["raw_symbol", "instrument_class", "strike_price", "expiration_date"]],
        on="raw_symbol",
        how="inner",
    )
    prices = prices.merge(
        underlying[["observation_date", "close", "available_at"]].rename(
            columns={"close": "spot", "available_at": "underlying_available_at"}
        ),
        on="observation_date",
        how="inner",
    )
    rows: list[dict] = []
    for date, daily in prices.groupby("observation_date"):
        date = pd.Timestamp(date).normalize()
        daily = daily.copy()
        daily["dte"] = (daily["expiration_date"] - date).dt.days
        daily = daily[daily["dte"].between(minimum_dte, maximum_dte)]
        expiry_rows: list[dict] = []
        for (expiration, dte), chain in daily.groupby(["expiration_date", "dte"]):
            pivot = chain.pivot_table(
                index="strike_price",
                columns="instrument_class",
                values="close",
                aggfunc="last",
            )
            if not {"C", "P"}.issubset(pivot.columns):
                continue
            pivot = pivot.dropna(subset=["C", "P"], how="any")
            if pivot.empty:
                continue
            maturity = float(dte) / 365.0
            forward_candidates = pivot.index.to_numpy(dtype=float) + math.exp(
                rate * maturity
            ) * (pivot["C"].to_numpy(dtype=float) - pivot["P"].to_numpy(dtype=float))
            spot = float(chain["spot"].iloc[0])
            plausible = forward_candidates[
                (forward_candidates > 0.7 * spot) & (forward_candidates < 1.3 * spot)
            ]
            if plausible.size == 0:
                continue
            forward = float(np.median(plausible))
            local = chain.assign(
                distance=(chain["strike_price"] - forward).abs()
            ).nsmallest(8, "distance")
            sigmas = []
            for option in local.itertuples(index=False):
                sigma = implied_volatility_black76(
                    float(option.close),
                    forward,
                    float(option.strike_price),
                    maturity,
                    rate,
                    str(option.instrument_class) == "C",
                )
                if np.isfinite(sigma) and 0.02 <= sigma <= 3.0:
                    sigmas.append(sigma)
            if len(sigmas) < 2:
                continue
            expiry_rows.append(
                {
                    "expiration_date": expiration,
                    "dte": int(dte),
                    "iv": float(np.median(sigmas)),
                    "option_count": len(sigmas),
                    "forward": forward,
                    "spot": spot,
                    "available_at": max(
                        pd.to_datetime(chain["available_at"], utc=True).max(),
                        pd.to_datetime(chain["underlying_available_at"], utc=True).max(),
                    ),
                }
            )
        if not expiry_rows:
            continue
        expiries = pd.DataFrame(expiry_rows).sort_values("dte")
        lower = expiries[expiries["dte"].le(target_dte)].tail(1)
        upper = expiries[expiries["dte"].ge(target_dte)].head(1)
        selected = pd.concat([lower, upper]).drop_duplicates("dte").sort_values("dte")
        if selected.empty:
            selected = expiries.iloc[[np.argmin(np.abs(expiries["dte"] - target_dte))]]
        selected = selected.copy()
        selected["total_variance"] = selected["iv"] ** 2 * selected["dte"] / 365.0
        target_t = target_dte / 365.0
        if len(selected) >= 2 and selected["dte"].min() < target_dte < selected["dte"].max():
            total_variance = float(
                np.interp(target_dte, selected["dte"], selected["total_variance"])
            )
            iv30 = math.sqrt(max(total_variance / target_t, 0.0))
            method = "total_variance_interpolation"
        else:
            nearest = selected.iloc[np.argmin(np.abs(selected["dte"] - target_dte))]
            iv30 = float(nearest["iv"])
            total_variance = iv30 * iv30 * target_t
            method = "nearest_expiry"
        rows.append(
            {
                "series_id": "SLV_IV30_OPRA",
                "observation_date": date,
                "value": iv30 * 100.0,
                "slv_iv30": iv30,
                "slv_iv30_pct": iv30 * 100.0,
                "slv_implied_var_daily": iv30 * iv30 / 252.0,
                "target_dte": int(target_dte),
                "method": method,
                "expiry_count": int(len(selected)),
                "option_count": int(selected["option_count"].sum()),
                "available_at": pd.to_datetime(selected["available_at"], utc=True).max(),
                "retrieved_at_utc": utc_now(),
            }
        )
    return pd.DataFrame(rows).sort_values("observation_date").reset_index(drop=True)


def _cost_request(request: dict) -> dict:
    return {key: value for key, value in request.items() if key != "stype_out"}


def _download_to_frame(client, request: dict) -> pd.DataFrame:
    return client.timeseries.get_range(**request).to_df()


def _monthly_snapshot_requests(base_request: dict, start_date, end_date) -> list[dict]:
    start = pd.Timestamp(start_date).normalize()
    final = pd.Timestamp(end_date).normalize()
    requests = []
    for month_start in pd.date_range(start.to_period("M").start_time, final, freq="MS"):
        candidates = pd.bdate_range(month_start, month_start + pd.offsets.MonthEnd(0))
        candidates = candidates[(candidates >= start) & (candidates < final)]
        if len(candidates) == 0:
            continue
        # A mid-month weekday is much less likely to be a US exchange holiday
        # than the first business day (for example New Year's Day or Labor
        # Day).  The downloader below still retries subsequent weekdays.
        cursor = pd.Timestamp(candidates[min(6, len(candidates) - 1)])
        boundary = min(cursor + pd.Timedelta(days=1), final)
        requests.append(
            {
                **base_request,
                "start": cursor.strftime("%Y-%m-%d"),
                "end": boundary.strftime("%Y-%m-%d"),
            }
        )
    return requests


def _definition_snapshot(
    client,
    request: dict,
    cache_directory: Path,
) -> pd.DataFrame:
    stamp = pd.Timestamp(request["start"]).strftime("%Y%m%d")
    path = cache_directory / f"slv_definitions_{stamp}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    last_error: Exception | None = None
    result: pd.DataFrame | None = None
    cursor = pd.Timestamp(request["start"])
    for offset in range(7):
        candidate = cursor + pd.Timedelta(days=offset)
        if candidate.dayofweek >= 5:
            continue
        candidate_request = {
            **request,
            "start": candidate.strftime("%Y-%m-%d"),
            "end": (candidate + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        try:
            raw = _download_to_frame(client, candidate_request)
            if raw.empty:
                continue
            candidate_result = normalise_slv_definitions(raw)
            if candidate_result.empty:
                continue
            result = candidate_result
            break
        except Exception as exc:  # network retries are deliberately bounded
            last_error = exc
    if result is None:
        if last_error is not None:
            raise RuntimeError(
                f"No usable SLV definition snapshot near {request['start']}."
            ) from last_error
        raise RuntimeError(f"No SLV definitions returned near {request['start']}.")
    write_parquet(result, path)
    return result


def _option_requests_by_expiry_window(
    definitions: pd.DataFrame,
    symbols: list[str],
    base_request: dict,
    global_start,
    global_end,
    *,
    minimum_dte: int,
    maximum_dte: int,
    batch_size: int = 400,
) -> list[dict]:
    selected = definitions[definitions["raw_symbol"].isin(symbols)].copy()
    selected["expiration_date"] = pd.to_datetime(
        selected["expiration"], utc=True
    ).dt.tz_localize(None).dt.normalize()
    selected["expiry_quarter"] = selected["expiration_date"].dt.to_period("Q")
    start_limit = pd.Timestamp(global_start).normalize()
    end_limit = pd.Timestamp(global_end).normalize()
    requests: list[dict] = []
    for _, group in selected.groupby("expiry_quarter", sort=True):
        group_symbols = sorted(group["raw_symbol"].astype(str).unique())
        request_start = max(
            start_limit,
            group["expiration_date"].min() - pd.Timedelta(days=int(maximum_dte)),
        )
        request_end = min(
            end_limit,
            group["expiration_date"].max()
            - pd.Timedelta(days=int(minimum_dte))
            + pd.Timedelta(days=1),
        )
        if request_start >= request_end:
            continue
        for index in range(0, len(group_symbols), int(batch_size)):
            requests.append(
                {
                    **base_request,
                    "symbols": group_symbols[index : index + int(batch_size)],
                    "start": request_start.strftime("%Y-%m-%d"),
                    "end": request_end.strftime("%Y-%m-%d"),
                }
            )
    return requests


def _option_price_batch(client, request: dict, cache_directory: Path) -> pd.DataFrame:
    symbol_key = "\n".join(request["symbols"]).encode("utf-8")
    digest = hashlib.sha256(symbol_key).hexdigest()[:12]
    path = cache_directory / (
        f"slv_options_{request['start']}_{request['end']}_{digest}.parquet"
    )
    if path.is_file():
        result = _label_daily_bar_dates(pd.read_parquet(path))
        write_parquet(result, path)
        return result
    raw = _download_to_frame(client, request)
    if raw.empty:
        result = pd.DataFrame(
            columns=[
                "raw_symbol",
                "observation_date",
                "ts_event",
                "available_at",
                "close",
                "retrieved_at_utc",
            ]
        )
    else:
        result = normalise_daily_ohlcv(raw, option_data=True)
    write_parquet(result, path)
    return result


def download_slv_iv_data(
    config: dict,
    start_date,
    end_date,
    *,
    force_execute: bool = False,
) -> dict[str, Path]:
    settings = config["data_sources"]["slv_options"]
    execute = bool(settings["execute_download"]) or bool(force_execute)
    if not execute:
        raise RuntimeError(
            "SLV OPRA download is disabled. Pass --download-slv-options or set "
            "data_sources.slv_options.execute_download=true."
        )
    client = _client()
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    definition_base = {
        "dataset": settings["options_dataset"],
        "schema": settings["definitions_schema"],
        "symbols": [settings["parent_symbol"]],
        "stype_in": "parent",
    }
    definition_requests = _monthly_snapshot_requests(
        definition_base, start, end
    )
    underlying_request = {
        "dataset": settings["underlying_dataset"],
        "schema": settings["underlying_schema"],
        "symbols": [settings["underlying_symbol"]],
        "stype_in": "raw_symbol",
        "start": start,
        "end": end,
    }
    # The cost of the complete parent range is a conservative upper bound for
    # the sparse monthly definition snapshots actually downloaded below.
    definition_cost_request = {
        **definition_base,
        "start": start,
        "end": end,
    }
    definition_cost = float(
        client.metadata.get_cost(**_cost_request(definition_cost_request))
    )
    underlying_cost = float(client.metadata.get_cost(**_cost_request(underlying_request)))
    maximum = float(settings["maximum_cost_usd"])
    if definition_cost + underlying_cost > maximum:
        raise RuntimeError(
            f"SLV definitions and underlying estimate ${definition_cost + underlying_cost:.4f} "
            f"exceeds configured cap ${maximum:.4f}."
        )
    cache_directory = resolve_path(
        config,
        settings.get(
            "definition_cache_dir", "data/raw/slv_definition_snapshots"
        ),
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        definition_pieces = list(
            executor.map(
                lambda request: _definition_snapshot(
                    client, request, cache_directory
                ),
                definition_requests,
            )
        )
    all_definitions = pd.concat(definition_pieces, ignore_index=True)
    # OPRA definition snapshots often omit the exchange activation timestamp.
    # The first monthly snapshot in which a raw symbol is observed is a causal,
    # conservative proxy. Keeping only the latest snapshot timestamp would
    # incorrectly make old contracts look unavailable until expiration.
    first_seen = all_definitions.groupby("raw_symbol")["ts_event"].min()
    definitions = (
        all_definitions.sort_values(["raw_symbol", "ts_event"])
        .drop_duplicates("raw_symbol", keep="last")
        .reset_index(drop=True)
    )
    definitions["activation"] = definitions["raw_symbol"].map(first_seen)
    underlying_path = resolve_path(config, settings["underlying_path"])
    if underlying_path.is_file():
        underlying = _label_daily_bar_dates(pd.read_parquet(underlying_path))
        write_parquet(underlying, underlying_path)
    else:
        underlying = normalise_daily_ohlcv(
            _download_to_frame(client, underlying_request), option_data=False
        )
        # Persist the inexpensive underlying immediately so a later OPRA
        # gateway timeout does not force the same request to be repeated.
        write_parquet(underlying, underlying_path)
    symbols = select_near_30d_symbols(
        definitions,
        underlying,
        target_dte=int(settings["target_dte_calendar_days"]),
        minimum_dte=int(settings["minimum_dte_calendar_days"]),
        maximum_dte=int(settings["maximum_dte_calendar_days"]),
        strikes_each_side=int(settings["strikes_each_side"]),
        strike_band_fraction=float(settings["strike_band_fraction"]),
        use_bracketing_expiries=bool(settings["use_bracketing_expiries"]),
    )
    if not symbols:
        raise RuntimeError("No near-30-day SLV option symbols were selected.")
    option_requests = _option_requests_by_expiry_window(
        definitions,
        symbols,
        {
            "dataset": settings["options_dataset"],
            "schema": settings["options_schema"],
            "stype_in": "raw_symbol",
        },
        start,
        end,
        minimum_dte=int(settings["minimum_dte_calendar_days"]),
        maximum_dte=int(settings["maximum_dte_calendar_days"]),
    )
    option_costs = [
        float(client.metadata.get_cost(**_cost_request(request)))
        for request in option_requests
    ]
    total_cost = definition_cost + underlying_cost + sum(option_costs)
    estimate = {
        "estimated_cost_usd": total_cost,
        "definitions_cost_usd": definition_cost,
        "definition_snapshot_count": len(definition_requests),
        "definitions_cost_is_conservative_parent_range_upper_bound": True,
        "underlying_cost_usd": underlying_cost,
        "option_prices_cost_usd": sum(option_costs),
        "selected_symbol_count": len(symbols),
        "batch_count": len(option_requests),
        "estimated_at_utc": utc_now().isoformat(),
    }
    root = Path(config["_project_root"])
    write_json(
        estimate,
        resolve_path(
            config,
            settings.get(
                "cost_estimate_path", "data/raw/databento_slv_cost_estimate.json"
            ),
        ),
    )
    if total_cost > maximum:
        raise RuntimeError(
            f"Estimated SLV OPRA cost ${total_cost:.4f} exceeds configured cap "
            f"${maximum:.4f}; no option-price download was made."
        )
    option_cache = resolve_path(
        config,
        settings.get("option_cache_dir", "data/raw/slv_option_batches"),
    )
    option_cache.mkdir(parents=True, exist_ok=True)
    pieces = [
        _option_price_batch(client, request, option_cache)
        for request in option_requests
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        raise RuntimeError("No SLV option daily prices were returned.")
    option_prices = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["observation_date", "raw_symbol"], keep="last"
    )
    iv = compute_slv_iv30(
        definitions,
        option_prices,
        underlying,
        target_dte=int(settings["target_dte_calendar_days"]),
        minimum_dte=int(settings["minimum_dte_calendar_days"]),
        maximum_dte=int(settings["maximum_dte_calendar_days"]),
        rate=float(settings["risk_free_rate"]),
    )
    paths = {
        "definitions": write_parquet(
            definitions, resolve_path(config, settings["definitions_path"])
        ),
        "underlying": write_parquet(
            underlying, resolve_path(config, settings["underlying_path"])
        ),
        "option_prices": write_parquet(
            option_prices, resolve_path(config, settings["option_prices_path"])
        ),
        "iv": write_parquet(iv, resolve_path(config, settings["iv_path"])),
    }
    return paths
