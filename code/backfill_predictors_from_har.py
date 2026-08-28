"""
backfill_predictors_from_har.py
--------------------------------
One-off, manually-run backfill for historical weather/rating/tide/swell/
wind/energy/consistency predictors, parsed from a HAR (HTTP Archive) file
instead of making live requests.

Why this exists: backfill_historical_predictors.py makes its own paced,
jittered requests against Surfline's historical API to stay bot-safe. An
alternative that's just as bot-safe (arguably more so, since it's a real
logged-in browser session driven by an actual person) is to click through
the site's own "Historical" view one day at a time, export the Network tab
as a HAR file, and hand it to this script — it extracts the forecast
endpoint responses already captured in the file and reuses
get_surf_predictors.py's merge/write logic. No requests are made by this
script at all.

Each request the site makes under the hood returns a multi-day window
(e.g. 16 days), not just the single day being viewed — but the later days
in that window are effectively a several-days-out forecast, not verified
day-of data, even though the whole window is nominally "historical". So
this script filters every response down to ONLY the record(s) matching
that request's own `start=` date — meaning you do need to click through
and capture each individual date you want data for, one HAR entry per
date (though one HAR file can contain many days' worth of clicks before
you export). If a date has no day-of data at all in any HAR (e.g. that
specific date wasn't directly clickable on the site), a target row for it
falls back to the 1-day-out record from the nearest earlier date's
request instead of being skipped — this only kicks in when day-of data is
genuinely unavailable, and every row written this way is reported by
name, both in --dry-run and the real write.

Known gap: the Historical view's network calls never include the weather
or consistency endpoints (confirmed via HAR capture) — temperature_f,
weather_condition, pressure_mb, and consistency_wave_count will stay
blank for every row backfilled this way, regardless of date. If needed
later, backfill those specific columns from another weather source.

Accepts multiple --har files in one run — all matching endpoint responses
across all files are merged into a single by_hour map before matching
against predictions.csv.

Usage:
    python code/backfill_predictors_from_har.py --har data/external/oct1-10.har
    python code/backfill_predictors_from_har.py --har data/external/a.har --har data/external/b.har
    python code/backfill_predictors_from_har.py --har data/external/a.har --dry-run
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_surf_predictors as sp  # noqa: E402

DEFAULT_OUT_CSV = sp.OUT_CSV

# The site's own "Historical" view does NOT call the same endpoint set as
# get_surf_predictors.py's live fetch. Confirmed via HAR capture
# (2026-08-26, 2025-10-20..10-30 date range): it calls rating, surf,
# swells, wind, energy, tides (plus sunlight/spectra/regions-conditions,
# which we don't use) — but never calls weather or consistency at all.
# `surf` + `swells` together carry the same fields the live `wave`
# endpoint carries in one response, just split across two endpoints, so
# they're combined below into a synthetic "wave" record list.
HISTORICAL_ONLY_PATHS = ["surf", "swells"]
UNAVAILABLE_VIA_HISTORICAL_VIEW = ["weather", "consistency"]

ENDPOINT_URL_RE = re.compile(
    r"/kbyg/spots/forecasts/(" + "|".join(sp.ENDPOINT_PATHS + HISTORICAL_ONLY_PATHS) + r")\b"
)


def combine_surf_and_swells(surf_records, swells_records):
    """surf: {timestamp, utcOffset, surf: {min,max,...}}
    swells: {timestamp, utcOffset, swells: [...]}
    -> synthetic wave-shaped records merge_into_by_hour expects: {timestamp, utcOffset, surf: {...}, swells: [...]}

    Unions timestamps from both lists rather than just iterating `surf` — the two
    endpoints don't always both appear for a given request (observed: some HAR
    requests captured swells but not surf), and dropping swells data just because
    surf is missing (or vice versa) would silently lose real data.
    """
    surf_by_ts = {s["timestamp"]: (s["utcOffset"], s["surf"]) for s in surf_records}
    swells_by_ts = {s["timestamp"]: s.get("swells") for s in swells_records}
    combined = []
    for ts in sorted(set(surf_by_ts) | set(swells_by_ts)):
        utc_offset, surf_val = surf_by_ts.get(ts, (None, {"min": 0, "max": 0}))
        if utc_offset is None:
            # fall back to a swells record's utcOffset if surf didn't have one for this ts
            utc_offset = next(s["utcOffset"] for s in swells_records if s["timestamp"] == ts)
        combined.append({
            "timestamp": ts,
            "utcOffset": utc_offset,
            "surf": surf_val,
            "swells": swells_by_ts.get(ts, []),
        })
    return combined


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--har", action="append", required=True, dest="har_paths",
                   help="Path to a HAR file (repeatable for multiple files)")
    p.add_argument("--out", default=str(DEFAULT_OUT_CSV), help=f"Output CSV path (default: {DEFAULT_OUT_CSV})")
    p.add_argument("--dry-run", action="store_true", help="Show what would be matched/written without writing")
    return p.parse_args()


def filter_to_offset(records, start_date, offset_days):
    """Keep only records whose local date is exactly `offset_days` after the request's
    own `start` param. offset_days=0 is the day actually being viewed (day-of, verified
    historical). offset_days=1 is one day out from that request — used only as an
    explicit fallback when day-of data for a date isn't available at all (e.g. the date
    itself couldn't be clicked directly on the site), never as the default."""
    kept = []
    for r in records:
        local_date = sp.local_hour_key(r["timestamp"], r["utcOffset"]).date()
        if (local_date - start_date).days == offset_days:
            kept.append(r)
    return kept


def extract_endpoint_responses(har_path):
    """Yields (endpoint_path, start_date, full_records_list) for each matching forecast
    request in the HAR — unfiltered by date; callers filter by offset as needed."""
    with open(har_path, encoding="utf-8") as f:
        har = json.load(f)

    entries = har.get("log", {}).get("entries", [])
    found = 0
    for entry in entries:
        url = entry.get("request", {}).get("url", "")
        m = ENDPOINT_URL_RE.search(url)
        if not m:
            continue
        endpoint = m.group(1)

        content = entry.get("response", {}).get("content", {})
        text = content.get("text")
        if not text:
            continue
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            print(f"  WARNING: {har_path}: non-JSON response body for {endpoint} request, skipping")
            continue

        try:
            records = body["data"][endpoint]
        except (KeyError, TypeError):
            print(f"  WARNING: {har_path}: unexpected response shape for {endpoint} ({url}), skipping")
            continue

        qs = parse_qs(urlparse(url).query)
        start_param = qs.get("start", [None])[0]
        if not start_param:
            print(f"  WARNING: {har_path}: {endpoint} request has no start= param, skipping (can't offset-filter)")
            continue
        start_date = datetime.strptime(start_param, "%Y-%m-%d").date()

        found += 1
        yield endpoint, start_date, records

    if found == 0:
        print(f"  WARNING: {har_path}: no matching forecast endpoint requests found "
              f"(expected URLs containing /kbyg/spots/forecasts/{{{','.join(sp.ENDPOINT_PATHS)}}})")


def main():
    args = parse_args()
    out_csv = Path(args.out)

    by_hour = {}
    fallback_by_hour = {}
    any_found = False
    for har_path in args.har_paths:
        har_path = Path(har_path)
        if not har_path.exists():
            print(f"ERROR: HAR file not found: {har_path}")
            return
        print(f"Parsing {har_path}...")

        raw_day0 = {}
        raw_day1 = {}
        for endpoint, start_date, records in extract_endpoint_responses(har_path):
            day0 = filter_to_offset(records, start_date, 0)
            day1 = filter_to_offset(records, start_date, 1)
            print(f"  [{endpoint}] start={start_date}: day-of={len(day0)}, "
                  f"1-day-out fallback={len(day1)} (of {len(records)} total in window)")
            raw_day0.setdefault(endpoint, []).extend(day0)
            raw_day1.setdefault(endpoint, []).extend(day1)
            any_found = True

        for raw in (raw_day0, raw_day1):
            if "surf" in raw or "swells" in raw:
                raw["wave"] = combine_surf_and_swells(raw.pop("surf", []), raw.pop("swells", []))

        sp.merge_into_by_hour(by_hour, **raw_day0)
        sp.merge_into_by_hour(fallback_by_hour, **raw_day1)

    if not any_found:
        print("\nNo predictor data found in any HAR file — nothing to do.")
        return

    print(f"\nNOTE: the site's Historical view never calls the weather or consistency "
          f"endpoints (confirmed empirically) — {', '.join(UNAVAILABLE_VIA_HISTORICAL_VIEW)} "
          f"will stay blank for all rows written from HAR files, regardless of what's in them.")

    covered_hours = sorted(by_hour.keys())
    print(f"\nMerged predictor data spans {covered_hours[0]} to {covered_hours[-1]} "
          f"({len(by_hour)} distinct hour(s) across all HAR files).")

    existing = sp.load_rows(out_csv)
    already_done = {r["filename"] for r in existing}

    preds_rows = sp.load_rows(sp.PREDS_CSV)
    targets = [
        (r["date"], r["time_local"], r["filename"])
        for r in preds_rows
        if r["filename"] not in already_done
    ]

    written = 0
    to_write = []
    fallback_dates_used = set()
    for date_str, time_local, filename in targets:
        target_dt = datetime.strptime(f"{date_str} {time_local}", "%Y-%m-%d %H:%M")
        hour_key = target_dt.replace(minute=0, second=0, microsecond=0)
        predictors = by_hour.get(hour_key)
        used_fallback = False
        if predictors is None:
            predictors = fallback_by_hour.get(hour_key)
            used_fallback = predictors is not None
        if predictors is None:
            continue
        if used_fallback:
            fallback_dates_used.add(date_str)
        to_write.append((date_str, time_local, filename, predictors, used_fallback))

    print(f"predictions.csv rows matched to fetched hours: {len(to_write)}")
    if fallback_dates_used:
        print(f"NOTE: {len(fallback_dates_used)} date(s) had no day-of data and used the "
              f"1-day-out fallback instead: {', '.join(sorted(fallback_dates_used))}")

    if args.dry_run:
        print("\n--dry-run: stopping before writing.")
        for date_str, time_local, filename, _, used_fallback in to_write[:20]:
            tag = " [1-day-out fallback]" if used_fallback else ""
            print(f"  would write: {filename} ({date_str} {time_local}){tag}")
        if len(to_write) > 20:
            print(f"  ... and {len(to_write) - 20} more")
        return

    for date_str, time_local, filename, predictors, used_fallback in to_write:
        sp.append_row(out_csv, sp.row_from_predictors(date_str, time_local, filename, predictors))
        written += 1

    print(f"\nDone. Wrote {written} row(s) to {out_csv}.")


if __name__ == "__main__":
    main()
