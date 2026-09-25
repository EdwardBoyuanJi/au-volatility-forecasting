from __future__ import annotations

from pathlib import Path

import pandas as pd

from au_rv.config import resolve_path
from au_rv.features.comex_nonoverlap import (
    aggregate_ohlcv_1m_to_5m,
    compute_comex_nonoverlap_daily,
)
from au_rv.features.events import build_event_counts
from au_rv.features.external_indicators import (
    add_gpr_features,
    add_slv_iv_features,
    align_external_indicator,
)
from au_rv.features.macro import align_fred_to_cutoffs
from au_rv.features.realized_variance import (
    add_har_features,
    compute_au_daily_measures,
    prepare_au_bars,
    select_daily_au_contracts,
)
from au_rv.features.targets import add_forward_targets
from au_rv.io import read_parquet, utc_now, write_csv, write_parquet

FORBIDDEN_COLUMNS = {
    "overnight_rv_share",
    "abs_overnight_gap",
    "event_score",
    "regime",
    "market_iv",
    "bid",
    "ask",
    "delta",
    "gamma",
    "theta",
}


def _completed_model_cutoffs(frame: pd.DataFrame, asof_utc) -> pd.DataFrame:
    """Exclude a future or still-open trade date before targets are formed."""

    result = frame.copy()
    cutoffs = pd.to_datetime(result["model_cutoff"], utc=True, errors="coerce")
    asof = pd.Timestamp(asof_utc)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    result = result[cutoffs.notna() & cutoffs.le(asof)].copy()
    result["build_asof_utc"] = asof
    return result.reset_index(drop=True)


def _quality_notes(row: pd.Series, required_features: list[str]) -> str:
    notes: list[str] = []
    if not bool(row.get("au_quality_flag", False)):
        notes.append("invalid_or_insufficient_au_bars")
    if not bool(row.get("comex_quality_flag", False)):
        notes.append("invalid_or_insufficient_comex_nonoverlap_bars")
    if bool(row.get("macro_missing_flag", True)):
        notes.append("missing_or_stale_macro")
    if bool(row.get("calendar_missing_flag", True)):
        notes.append("official_event_calendar_not_covered")
    missing = [name for name in required_features if pd.isna(row.get(name))]
    if missing:
        notes.append("missing_features:" + "|".join(missing))
    if not bool(row.get("availability_cutoff_pass", False)):
        notes.append("available_at_after_model_cutoff")
    return "ok" if not notes else ";".join(notes)


def build_feature_table_from_frames(
    config: dict,
    *,
    au_bars: pd.DataFrame,
    shfe_calendar: pd.DataFrame,
    gc_1m: pd.DataFrame,
    fred: pd.DataFrame,
    events: pd.DataFrame,
    gpr: pd.DataFrame | None = None,
    slv_iv: pd.DataFrame | None = None,
    asof_utc=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project = config["project"]
    market = config["market"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    epsilon = float(project["epsilon"])
    cutoff_time = project["model_cutoff_time"]

    prepared_au = prepare_au_bars(au_bars, shfe_calendar)
    selections = select_daily_au_contracts(prepared_au)
    au_daily, _ = compute_au_daily_measures(
        prepared_au,
        selections,
        cutoff_time=cutoff_time,
        bar_minutes=int(market["bar_minutes"]),
        minimum_bars=int(market["minimum_au_bars"]),
        jump_alpha=float(config["jump"]["alpha"]),
        finite_sample=bool(config["jump"]["use_finite_sample_correction"]),
    )
    au_daily = add_har_features(au_daily, epsilon)
    if asof_utc is not None:
        au_daily = _completed_model_cutoffs(au_daily, asof_utc)
    else:
        au_daily["build_asof_utc"] = pd.NaT

    gc_5m = aggregate_ohlcv_1m_to_5m(gc_1m)
    comex = compute_comex_nonoverlap_daily(
        gc_5m,
        prepared_au,
        au_daily["trade_date"],
        cutoff_time=cutoff_time,
        epsilon=epsilon,
        minimum_bars=int(market["minimum_comex_nonoverlap_bars"]),
    )

    macro = align_fred_to_cutoffs(
        fred,
        au_daily["trade_date"],
        series_map=config["data_sources"]["fred"]["series"],
        cutoff_time=cutoff_time,
        maximum_staleness_days=config["data_sources"]["fred"].get(
            "maximum_staleness_calendar_days_by_series",
            int(config["data_sources"]["fred"]["maximum_staleness_calendar_days"]),
        ),
    )
    gpr_aligned = align_external_indicator(
        gpr,
        au_daily["trade_date"],
        prefix="gpr",
        value_column="value",
        cutoff_time=cutoff_time,
        maximum_staleness_days=int(
            config["data_sources"]["gpr"]["maximum_staleness_calendar_days"]
        ),
    )
    slv_aligned = align_external_indicator(
        slv_iv,
        au_daily["trade_date"],
        prefix="slv_iv30_pct",
        value_column="slv_iv30_pct",
        cutoff_time=cutoff_time,
        maximum_staleness_days=int(
            config["data_sources"]["fred"]["maximum_staleness_calendar_days"]
        ),
    )
    open_dates = shfe_calendar.loc[
        shfe_calendar["is_open"].astype(int).eq(1), "cal_date"
    ]
    event_counts = build_event_counts(
        events,
        open_dates,
        au_daily["trade_date"],
        horizons=horizons,
        cutoff_time=cutoff_time,
    )

    frame = au_daily.merge(comex, on="trade_date", how="left")
    macro_drop = ["model_cutoff"]
    frame = frame.merge(macro.drop(columns=macro_drop), on="trade_date", how="left")
    frame = frame.merge(
        gpr_aligned.drop(columns=["model_cutoff"]), on="trade_date", how="left"
    )
    frame = frame.merge(
        slv_aligned.drop(columns=["model_cutoff"]), on="trade_date", how="left"
    )
    frame = add_gpr_features(frame)
    frame = add_slv_iv_features(frame)
    frame = frame.merge(event_counts, on="trade_date", how="left")
    frame = add_forward_targets(frame, horizons, epsilon)

    component_availability = [
        "au_available_at",
        "comex_available_at",
        "macro_available_at",
        "calendar_available_at",
    ]
    for column in component_availability + ["model_cutoff"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["feature_available_at_max"] = frame[component_availability].max(axis=1)
    frame["availability_cutoff_pass"] = (
        frame["feature_available_at_max"].notna()
        & frame["feature_available_at_max"].le(frame["model_cutoff"])
    )

    core = [
        "log_rv_1d",
        "log_rv_5d",
        "log_rv_22d",
        "realized_quarticity",
        "sqrt_realized_quarticity_1d",
        "harq_log_rv_rq_interaction_1d",
        "jump_var_1d",
        "jump_var_5d",
        "log_comex_nonoverlap_rv_1d",
        "abs_broad_dollar_ret_1d",
        "abs_us10y_real_chg_1d",
        "abs_usdcny_ret_1d",
    ]
    event_features = [
        f"{event_type}_count_{horizon}d"
        for horizon in horizons
        for event_type in ("cpi", "nfp", "fomc")
    ]
    required = core + event_features
    frame["feature_valid_flag"] = (
        frame[required].notna().all(axis=1)
        & frame["au_quality_flag"].fillna(False).astype(bool)
        & frame["comex_quality_flag"].fillna(False).astype(bool)
        & ~frame["macro_missing_flag"].fillna(True).astype(bool)
        & ~frame["calendar_missing_flag"].fillna(True).astype(bool)
        & frame["availability_cutoff_pass"]
    )
    frame["data_quality_notes"] = frame.apply(
        _quality_notes, axis=1, required_features=required
    )
    frame["data_quality_flag"] = frame["feature_valid_flag"].map(
        {True: "ok", False: "invalid_no_formal_prediction"}
    )
    experiment_required = required + [
        "daily_log_return",
        "log_gvz_implied_var_daily",
        "gvz_log_change_1d",
        "gvz_log_change_5d",
        "log_slv_implied_var_daily",
        "slv_iv_log_change_1d",
        "slv_iv_log_change_5d",
        "log_us_epu",
        "us_epu_log_change_5d",
        "log_gepu",
        "gepu_log_change_22d",
        "log_gpr",
        "gpr_log_change_5d",
        "gpr_log_mean_5d",
    ]
    external_availability = [
        "fred_indicator_available_at",
        "gpr_available_at",
        "slv_iv30_pct_available_at",
    ]
    for column in external_availability:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["experiment_feature_available_at_max"] = frame[
        component_availability + external_availability
    ].max(axis=1)
    frame["experiment_availability_cutoff_pass"] = (
        frame["experiment_feature_available_at_max"].notna()
        & frame["experiment_feature_available_at_max"].le(frame["model_cutoff"])
    )
    frame["experiment_feature_valid_flag"] = (
        frame[experiment_required].notna().all(axis=1)
        & frame["feature_valid_flag"].astype(bool)
        & frame["experiment_availability_cutoff_pass"]
    )
    forbidden_present = FORBIDDEN_COLUMNS.intersection(
        {column.lower() for column in frame.columns}
    )
    if forbidden_present:
        raise AssertionError(f"Forbidden model fields present: {sorted(forbidden_present)}")

    audit_columns = [
        "trade_date",
        "build_asof_utc",
        "model_cutoff",
        *component_availability,
        "feature_available_at_max",
        "availability_cutoff_pass",
        "feature_valid_flag",
        "experiment_feature_available_at_max",
        "experiment_availability_cutoff_pass",
        "experiment_feature_valid_flag",
        "data_quality_notes",
    ]
    audit = frame[audit_columns].copy()
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame, audit


def build_feature_table(config: dict) -> tuple[Path, Path]:
    root = Path(config["_project_root"])
    raw = root / "data/raw"
    sources = config.get("data_sources", {})
    gpr_path = sources.get("gpr", {}).get(
        "output_path", "data/raw/gpr_daily_recent.parquet"
    )
    slv_iv_path = sources.get("slv_options", {}).get(
        "iv_path", "data/raw/databento_slv_iv_30d.parquet"
    )
    required_inputs = {
        "TqSdk AU 5-minute bars": raw / "tqsdk_au_5m.parquet",
        "TqSdk SHFE calendar": raw / "tqsdk_shfe_trade_calendar.parquet",
        "Databento GC 1-minute bars": raw / "databento_gc_1m.parquet",
        "FRED initial-release daily data": raw / "fred_initial_release_daily.parquet",
        "complete official event calendar": raw / "official_macro_events.parquet",
        "GPR author daily workbook": resolve_path(config, gpr_path),
        "Databento OPRA SLV 30-day IV": resolve_path(config, slv_iv_path),
    }
    missing = [f"{label}: {path}" for label, path in required_inputs.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Cannot build formal features; required raw inputs are missing:\n- "
            + "\n- ".join(missing)
            + "\nRun scripts/01_download_data.py by source and resolve every failure first."
        )
    frame, audit = build_feature_table_from_frames(
        config,
        au_bars=read_parquet(required_inputs["TqSdk AU 5-minute bars"]),
        shfe_calendar=read_parquet(required_inputs["TqSdk SHFE calendar"]),
        gc_1m=read_parquet(required_inputs["Databento GC 1-minute bars"]),
        fred=read_parquet(required_inputs["FRED initial-release daily data"]),
        events=read_parquet(required_inputs["complete official event calendar"]),
        gpr=read_parquet(required_inputs["GPR author daily workbook"]),
        slv_iv=read_parquet(required_inputs["Databento OPRA SLV 30-day IV"]),
        asof_utc=utc_now(),
    )
    feature_path = resolve_path(config, config["outputs"]["features_path"])
    audit_path = resolve_path(config, config["outputs"]["leakage_audit_path"])
    write_parquet(frame, feature_path)
    write_csv(audit, audit_path)
    return feature_path, audit_path
