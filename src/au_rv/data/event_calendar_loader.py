from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from lxml import html as lxml_html

from au_rv.io import utc_now, write_parquet

ET = ZoneInfo("America/New_York")
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def _get_html(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AU-RV-Research/1.0)"},
    )
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _get_or_cache_html(url: str, cache_path: Path) -> str:
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")
    content = _get_html(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content, encoding="utf-8")
    return content


def _localized_event_row(
    event_type: str,
    event_name: str,
    event_local: pd.Timestamp,
    available_local: pd.Timestamp,
    source_url: str,
    availability_note: str,
) -> dict:
    event_local = pd.Timestamp(event_local)
    if event_local.tzinfo is None:
        event_local = event_local.tz_localize(ET, ambiguous="raise", nonexistent="shift_forward")
    available_local = pd.Timestamp(available_local)
    if available_local.tzinfo is None:
        available_local = available_local.tz_localize(
            ET, ambiguous="raise", nonexistent="shift_forward"
        )
    return {
        "event_type": event_type,
        "event_name": event_name,
        "timestamp_original": event_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "timezone_original": "America/New_York",
        "timestamp_utc": event_local.tz_convert("UTC"),
        "timestamp_shanghai": event_local.tz_convert("Asia/Shanghai"),
        "available_at": available_local.tz_convert("UTC"),
        "calendar_available_at_original": available_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_url": source_url,
        "availability_note": availability_note,
        "retrieved_at_utc": utc_now(),
    }


def parse_bls_year(html: str, source_url: str) -> pd.DataFrame:
    """Parse CPI and Employment Situation from an official annual BLS page.

    Each monthly table's official ``Last Modified Date`` is used as the
    schedule-availability date. Because the page supplies no time of day, the
    implementation uses 23:59:59 ET, a conservative choice.
    """

    tree = lxml_html.fromstring(html)
    rows: list[dict] = []
    for table in tree.xpath("//table"):
        try:
            table_html = lxml_html.tostring(table, encoding="unicode")
            parsed = pd.read_html(StringIO(table_html))[0]
        except (ValueError, IndexError):
            continue
        parsed.columns = [str(column).strip().lower() for column in parsed.columns]
        if not {"date", "time", "release"}.issubset(parsed.columns):
            continue
        note = ""
        for sibling in table.itersiblings():
            candidate = " ".join(sibling.text_content().split())
            if "Last Modified Date" in candidate:
                note = candidate
                break
            if sibling.tag == "table":
                break
        match = re.search(r"Last Modified Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", note)
        if match:
            modified = pd.to_datetime(match.group(1), errors="coerce")
            modified_label = match.group(1)
        else:
            modified = pd.NaT
            modified_label = ""
        for record in parsed.to_dict("records"):
            release = str(record["release"]).strip()
            if release.startswith("Consumer Price Index"):
                event_type = "cpi"
            elif release.startswith("Employment Situation"):
                event_type = "nfp"
            else:
                continue
            combined = f"{record['date']} {record['time']}"
            event_local = pd.to_datetime(combined, errors="coerce")
            if pd.isna(event_local):
                continue
            if pd.isna(modified):
                # Missing official revision metadata is not guessed. Marking it
                # available only at release time prevents anticipatory use.
                available = event_local
                availability_note = "BLS Last Modified missing; release-time fallback"
            else:
                available = modified + pd.Timedelta(hours=23, minutes=59, seconds=59)
                availability_note = (
                    f"BLS monthly table Last Modified Date: {modified_label}"
                )
            rows.append(
                _localized_event_row(
                    event_type,
                    release,
                    event_local,
                    available,
                    source_url,
                    availability_note,
                )
            )
    return pd.DataFrame(rows)


def _meeting_end(month_label: str, day_label: str, year: int) -> pd.Timestamp | None:
    if re.search(r"unscheduled|cancelled|notation", day_label, flags=re.I):
        return None
    months = [piece.strip() for piece in month_label.split("/") if piece.strip()]
    if not months or months[-1] not in MONTHS:
        return None
    numbers = [int(value) for value in re.findall(r"\d+", day_label)]
    if not numbers:
        return None
    end_day = numbers[-1]
    end_month = MONTHS[months[-1]] if len(months) > 1 else MONTHS[months[0]]
    return pd.Timestamp(year=year, month=end_month, day=end_day, hour=14)


def parse_fomc_historical_year(html: str, source_url: str, year: int) -> pd.DataFrame:
    tree = lxml_html.fromstring(html)
    rows: list[dict] = []
    for heading in tree.xpath("//h4 | //h5"):
        text = " ".join(heading.text_content().split())
        if not re.search(r"\bMeeting\b", text) or f"- {year}" not in text:
            continue
        if re.search(r"unscheduled|cancelled|notation vote", text, flags=re.I):
            continue
        match = re.match(
            r"(?P<month>[A-Za-z]+(?:/[A-Za-z]+)?)\s+"
            r"(?P<days>\d{1,2}(?:-\d{1,2})?)\s+Meeting",
            text,
        )
        if not match:
            continue
        event_local = _meeting_end(match.group("month"), match.group("days"), year)
        if event_local is None:
            continue
        known = pd.Timestamp(year=year, month=1, day=1, hour=23, minute=59, second=59)
        rows.append(
            _localized_event_row(
                "fomc",
                text,
                event_local,
                known,
                source_url,
                "Conservative FOMC schedule availability: start of event year",
            )
        )
    return pd.DataFrame(rows)


def parse_fomc_current(html: str, source_url: str) -> pd.DataFrame:
    tree = lxml_html.fromstring(html)
    rows: list[dict] = []
    meetings = tree.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' fomc-meeting ')]"
    )
    for meeting in meetings:
        month_nodes = meeting.xpath(
            ".//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' fomc-meeting__month ')]"
        )
        date_nodes = meeting.xpath(
            ".//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' fomc-meeting__date ')]"
        )
        if not month_nodes or not date_nodes:
            continue
        heading_nodes = meeting.xpath("preceding::h4[1] | preceding::h3[1]")
        if not heading_nodes:
            continue
        year_match = re.search(r"(20\d{2})", heading_nodes[-1].text_content())
        if year_match is None:
            continue
        year = int(year_match.group(1))
        month_label = " ".join(month_nodes[0].text_content().split())
        day_label = " ".join(date_nodes[0].text_content().split())
        event_local = _meeting_end(month_label, day_label, year)
        if event_local is None:
            continue
        known = pd.Timestamp(year=year, month=1, day=1, hour=23, minute=59, second=59)
        rows.append(
            _localized_event_row(
                "fomc",
                f"{month_label} {day_label} FOMC Meeting - {year}",
                event_local,
                known,
                source_url,
                "Conservative FOMC schedule availability: start of event year",
            )
        )
    return pd.DataFrame(rows)


def download_fred_release_events(
    release_ids: dict[str, int],
    start_date,
    end_date,
) -> pd.DataFrame:
    """Fallback CPI/NFP dates from FRED when BLS blocks automated access.

    Past schedules are marked available only at the release itself because the
    API does not expose point-in-time schedule-publication vintages. Future
    dates are marked available when this download retrieved them. This is
    deliberately conservative and cannot leak a historical future schedule.
    """

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the BLS calendar fallback.")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    retrieved = utc_now()
    retrieved_local = retrieved.tz_convert(ET)
    rows: list[dict] = []
    for event_type, release_id in release_ids.items():
        response = requests.get(
            FRED_RELEASE_DATES_URL,
            params={
                "release_id": int(release_id),
                "api_key": api_key,
                "file_type": "json",
                "include_release_dates_with_no_data": "true",
                "limit": 10000,
                "sort_order": "asc",
            },
            timeout=60,
        )
        if not response.ok:
            try:
                detail = response.json().get("error_message", "")
            except (ValueError, AttributeError):
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"FRED release {release_id} failed with HTTP {response.status_code}{suffix}"
            )
        for record in response.json().get("release_dates", []):
            event_date = pd.to_datetime(record.get("date"), errors="coerce")
            if pd.isna(event_date) or event_date < start or event_date > end:
                continue
            event_local = event_date + pd.Timedelta(hours=8, minutes=30)
            aware_event = event_local.tz_localize(
                ET, ambiguous="raise", nonexistent="shift_forward"
            )
            if aware_event <= retrieved_local:
                available_local = aware_event
                note = (
                    "FRED release-date fallback; historical schedule available "
                    "only at release time"
                )
            else:
                available_local = retrieved_local
                note = "FRED release-date fallback; future date known at retrieval"
            rows.append(
                _localized_event_row(
                    event_type,
                    f"FRED release {release_id}",
                    aware_event,
                    available_local,
                    f"https://fred.stlouisfed.org/release?rid={int(release_id)}",
                    note,
                )
            )
    return pd.DataFrame(rows)


def validate_event_calendar_coverage(
    events: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Reject partial official calendars instead of treating missing events as zero."""

    required_columns = {"event_type", "timestamp_utc", "available_at"}
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        raise ValueError(
            f"Official event calendar missing fields: {sorted(missing_columns)}"
        )
    result = events.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    result = result[
        result["timestamp_utc"].ge(start) & result["timestamp_utc"].lt(end)
    ].copy()
    if result.empty:
        raise ValueError("Official event calendar has no rows in the requested range.")

    result["event_year"] = result["timestamp_utc"].dt.tz_convert(ET).dt.year
    result["event_month"] = result["timestamp_utc"].dt.tz_convert(ET).dt.month
    problems: list[str] = []
    for event_type in ("cpi", "nfp", "fomc"):
        subset = result[result["event_type"].eq(event_type)]
        if subset.empty:
            problems.append(f"missing all {event_type} events")
            continue
        for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
            range_start = max(pd.Timestamp(start_date).normalize(), pd.Timestamp(year, 1, 1))
            range_end = min(pd.Timestamp(end_date).normalize(), pd.Timestamp(year, 12, 31))
            covered_months = len(pd.period_range(range_start, range_end, freq="M"))
            observed = int(subset["event_year"].eq(year).sum())
            if event_type in {"cpi", "nfp"}:
                minimum = max(1, covered_months - 2)
            else:
                minimum = max(1, int(covered_months * 0.5) - 1)
            if observed < minimum:
                problems.append(
                    f"{event_type} {year}: {observed} rows, expected at least {minimum}"
                )
    if result["available_at"].isna().any():
        problems.append("one or more events have missing available_at")
    if problems:
        raise ValueError("Incomplete official event calendar: " + "; ".join(problems))
    return result.drop(columns=["event_year", "event_month"])


def download_event_calendars(config: dict, start_date, end_date) -> Path:
    settings = config["data_sources"]["events"]
    root = Path(config["_project_root"])
    html_dir = root / settings["raw_html_dir"]
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    pieces: list[pd.DataFrame] = []
    failures: list[str] = []

    for year in range(start_year, end_year + 1):
        url = settings["bls_url_template"].format(year=year)
        try:
            html = _get_or_cache_html(url, html_dir / f"bls_{year}.html")
            pieces.append(parse_bls_year(html, url))
        except Exception as exc:  # keep other official sources downloadable
            failures.append(f"BLS {year}: {exc}")

    if settings.get("fred_release_fallback", False) and any(
        failure.startswith("BLS ") for failure in failures
    ):
        try:
            pieces.append(
                download_fred_release_events(
                    settings["fred_release_ids"], start_date, end_date
                )
            )
        except Exception as exc:
            failures.append(f"FRED release-calendar fallback: {exc}")

    current_url = settings["fomc_calendar_url"]
    try:
        current = parse_fomc_current(
            _get_or_cache_html(current_url, html_dir / "fomc_current.html"),
            current_url,
        )
        pieces.append(current)
        if current.empty or "timestamp_utc" not in current:
            current_years = set()
        else:
            current_years = set(pd.to_datetime(current["timestamp_utc"], utc=True).dt.year)
    except Exception as exc:
        failures.append(f"FOMC current: {exc}")
        current_years = set()

    for year in range(start_year, end_year + 1):
        if year in current_years:
            continue
        url = settings["fomc_historical_url_template"].format(year=year)
        try:
            html = _get_or_cache_html(url, html_dir / f"fomc_{year}.html")
            pieces.append(parse_fomc_historical_year(html, url, year))
        except Exception as exc:
            failures.append(f"FOMC {year}: {exc}")

    nonempty = [piece for piece in pieces if piece is not None and not piece.empty]
    if not nonempty:
        raise RuntimeError("No official event-calendar rows downloaded. " + "; ".join(failures))
    events = pd.concat(nonempty, ignore_index=True)
    # Official BLS rows are appended before the FRED fallback and therefore win
    # when both sources provide the same event.
    events = events.drop_duplicates(["event_type", "timestamp_utc"], keep="first")
    try:
        events = validate_event_calendar_coverage(events, start_date, end_date)
    except ValueError as exc:
        detail = "; ".join(failures)
        suffix = f" Download failures: {detail}" if detail else ""
        raise RuntimeError(f"{exc}.{suffix}") from exc
    events = events.sort_values("timestamp_utc").reset_index(drop=True)
    events["download_failures"] = "; ".join(failures)
    return write_parquet(events, root / "data/raw/official_macro_events.parquet")
