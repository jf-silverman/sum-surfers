"""
backfill_historical_predictors.py
----------------------------------
One-off, manually-run backfill for historical weather/rating/tide/swell/
wind/energy/consistency predictors, using Surfline's same
services.surfline.com/kbyg/spots/forecasts/* endpoints as
get_surf_predictors.py, but with a `start=YYYY-MM-DD` parameter and an
`x-auth-accesstoken` header pulled from a logged-in, premium Surfline
session (confirmed working 2026-08-25 via a HAR capture of the site's own
"Historical" view — see docs/PROJECT_HISTORY.md).

This is NOT part of the scheduled pipeline (like backfill_tide.py) and is
not meant to be automated:
  - The token is your personal account's session credential, not a public
    API key. Treat it as sensitive — never commit it, never put it
    somewhere that ends up in git history.
  - It may go stale (it's tied to your logged-in session, not a
    long-lived API key), so this is meant for occasional, deliberate,
    manual runs — grab a fresh token each time from your browser's
    DevTools Network tab (any services.surfline.com request → Headers →
    x-auth-accesstoken), not something to bake into cron.
  - Requests are paced with a random jittered pause between each one
    (default 3-35s) specifically to avoid looking like scripted/bot
    traffic against an authenticated session.

Usage:
    # token via env var (recommended — set it just for this shell session):
    export SURFLINE_HISTORICAL_TOKEN=<paste from DevTools>
    python code/backfill_historical_predictors.py --start 2025-10-15 --end 2025-10-20

    # or omit --token/env var and it'll prompt (hidden input) at runtime
    python code/backfill_historical_predictors.py --start 2025-10-15

    # preview what would be fetched without making any requests
    python code/backfill_historical_predictors.py --start 2025-10-15 --end 2025-11-01 --dry-run

    # smaller chunks = fewer days per request, more requests, gentler on the
    # authenticated session (3 days/request is a deliberately cautious setting)
    python code/backfill_historical_predictors.py --start 2026-08-22 --end 2026-09-07 \
        --chunk-days 3 --refetch-incomplete

`--refetch-incomplete` also repairs rows already in the output CSV that have
blank predictor columns — without it, any filename already present is skipped
no matter how empty it is. Rows are replaced in place (the file is rewritten
atomically), so re-running never leaves duplicate rows for one filename.
"""

import argparse
import csv
import getpass
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_surf_predictors as sp  # noqa: E402 — reuses ENDPOINT_PATHS, CSV_HEADER, merge_into_by_hour, local_hour_key

PROJECT_ROOT = sp.PROJECT_ROOT
PREDS_CSV = sp.PREDS_CSV
DEFAULT_OUT_CSV = sp.OUT_CSV

# Empirically observed to work in a single request (the site's own
# Historical view used days=16/17); kept a little under that as a margin
# of safety. Date ranges longer than this get split into multiple chunks,
# each making its own round of 7 requests (one per endpoint).
CHUNK_MAX_DAYS = 14

DEFAULT_MIN_PAUSE = 3.0
DEFAULT_MAX_PAUSE = 35.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="End date, YYYY-MM-DD (inclusive). Defaults to --start (single day).")
    p.add_argument("--token", help="Surfline x-auth-accesstoken. If omitted, reads SURFLINE_HISTORICAL_TOKEN env var, then prompts.")
    p.add_argument("--out", default=str(DEFAULT_OUT_CSV), help=f"Output CSV path (default: {DEFAULT_OUT_CSV})")
    p.add_argument("--min-pause", type=float, default=DEFAULT_MIN_PAUSE, help=f"Minimum seconds between requests (default {DEFAULT_MIN_PAUSE})")
    p.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE, help=f"Maximum seconds between requests (default {DEFAULT_MAX_PAUSE})")
    p.add_argument("--chunk-days", type=int, default=CHUNK_MAX_DAYS, help=f"Max days per API request (default {CHUNK_MAX_DAYS})")
    p.add_argument("--refetch-incomplete", action="store_true",
                   help="Also re-fetch rows already in the output CSV that have blank predictor "
                        "fields (e.g. the surf/swell columns left null while Surfline's retired "
                        "`wave` endpoint 404'd, or blank energy_* rows), replacing them in place "
                        "instead of skipping them. Without this, any filename already present is "
                        "skipped regardless of how complete it is.")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch and replace EVERY predictions.csv row in the date range, even "
                        "rows already fully populated. Use when the extraction logic itself "
                        "changed (e.g. primary_swell()'s selection rule) and existing rows are "
                        "complete but computed the old way. Implies --refetch-incomplete.")
    p.add_argument("--dry-run", action="store_true", help="Show what would be fetched/written without making any requests")
    return p.parse_args()


# Columns that come from the Surfline endpoints — a blank in any of these means
# the row didn't get everything it should have. date/time_local/filename are the
# row's identity, not fetched data, so they're excluded.
_FETCHED_COLUMNS = [c for c in sp.CSV_HEADER if c not in ("date", "time_local", "filename")]


def is_complete(row):
    """True if every Surfline-derived column in an existing output row has a value."""
    return all(str(row.get(c, "")).strip() != "" for c in _FETCHED_COLUMNS)


def write_rows(out_csv, rows):
    """Rewrite out_csv atomically (temp file + replace), so a crash mid-write
    can't leave a half-written predictors file behind."""
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sp.CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in sp.CSV_HEADER})
    tmp.replace(out_csv)


def resolve_token(cli_token):
    if cli_token:
        return cli_token
    env_token = os.environ.get("SURFLINE_HISTORICAL_TOKEN", "").strip()
    if env_token:
        return env_token
    print("No token provided via --token or SURFLINE_HISTORICAL_TOKEN.")
    print("Grab one from Chrome DevTools: Network tab -> any services.surfline.com")
    print("request -> Headers -> Request Headers -> x-auth-accesstoken.")
    return getpass.getpass("Paste x-auth-accesstoken (input hidden): ").strip()


def date_range_chunks(start_date, end_date, max_days):
    chunks = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=max_days - 1), end_date)
        span_days = (chunk_end - cur).days + 1
        chunks.append((cur, span_days))
        cur = chunk_end + timedelta(days=1)
    return chunks


def load_targets(start_date, end_date, already_done):
    """(date, time_local, filename) tuples from predictions.csv in range, not already backfilled."""
    rows = sp.load_rows(PREDS_CSV)
    targets = []
    for r in rows:
        row_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        if start_date <= row_date <= end_date and r["filename"] not in already_done:
            targets.append((r["date"], r["time_local"], r["filename"]))
    return targets


class RequestDenied(Exception):
    """Raised on a 400/429 response — signals the caller to stop the whole run
    rather than continue hammering an endpoint that's actively denying requests."""


def fetch_chunk(chunk_start, span_days, token, min_pause, max_pause):
    headers = {**sp.REQUEST_HEADERS, "x-auth-accesstoken": token}
    responses = {}
    for path in sp.ENDPOINT_PATHS:
        pause = random.uniform(min_pause, max_pause)
        print(f"    [{path}] waiting {pause:.1f}s before request...")
        time.sleep(pause)

        params = {"spotId": sp.SPOT_ID, "days": str(span_days), "start": chunk_start.isoformat(), "intervalHours": "1"}
        try:
            r = requests.get(
                f"https://services.surfline.com/kbyg/spots/forecasts/{path}",
                params=params, headers=headers, timeout=20,
            )
            if r.status_code == 400:
                raise RequestDenied(f"[{path}] 400: {r.text[:150]} (likely token expired/invalid, or not premium)")
            if r.status_code == 429:
                raise RequestDenied(f"[{path}] 429 rate-limited")
            r.raise_for_status()
            responses[path] = r.json()["data"][path]
            print(f"    [{path}] ok, {len(responses[path])} record(s)")
        except requests.exceptions.RequestException as e:
            print(f"    [{path}] ERROR: {e}")
            responses[path] = []
    return responses


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else start_date
    if end_date < start_date:
        print("ERROR: --end is before --start")
        return
    yesterday = (datetime.now() - timedelta(days=1)).date()
    if end_date > yesterday:
        print(f"ERROR: --end ({end_date}) is not in the past (today or future dates aren't 'historical' — "
              f"use get_surf_predictors.py for current/upcoming data instead)")
        return

    out_csv = Path(args.out)
    existing = sp.load_rows(out_csv)
    if args.force:
        # Nothing counts as done — every row in range is re-fetched and replaced.
        already_done = set()
        print("--force: re-fetching every row in range, including already-complete ones.")
    elif args.refetch_incomplete:
        # Only fully-populated rows count as done, so rows with blank predictor
        # columns become targets again and get replaced rather than skipped.
        already_done = {r["filename"] for r in existing if is_complete(r)}
        n_incomplete = len(existing) - len(already_done)
        print(f"--refetch-incomplete: {n_incomplete} existing row(s) have blank predictor "
              f"fields and are eligible for re-fetch.")
    else:
        already_done = {r["filename"] for r in existing}

    targets = load_targets(start_date, end_date, already_done)
    if not targets:
        print(f"No predictions.csv rows in {start_date}..{end_date} need backfilling (already covered, or no crops in that range).")
        return

    chunks = date_range_chunks(start_date, end_date, args.chunk_days)
    n_requests = len(chunks) * len(sp.ENDPOINT_PATHS)
    est_seconds = n_requests * (args.min_pause + args.max_pause) / 2

    print(f"Date range: {start_date} to {end_date} ({(end_date - start_date).days + 1} days)")
    print(f"Chunks: {len(chunks)} (max {args.chunk_days} days each) -> {n_requests} requests total")
    print(f"Estimated time: ~{est_seconds/60:.1f} min (pauses {args.min_pause}-{args.max_pause}s between requests)")
    print(f"predictions.csv rows to backfill: {len(targets)}")

    if args.dry_run:
        print("\n--dry-run: stopping before any requests are made.")
        for c_start, c_days in chunks:
            print(f"  would fetch: start={c_start} days={c_days}")
        return

    token = resolve_token(args.token)
    if not token:
        print("ERROR: no token provided, aborting.")
        return

    by_hour = {}
    denied = False
    for i, (chunk_start, chunk_days) in enumerate(chunks, start=1):
        print(f"\nChunk {i}/{len(chunks)}: start={chunk_start} days={chunk_days}")
        try:
            responses = fetch_chunk(chunk_start, chunk_days, token, args.min_pause, args.max_pause)
        except RequestDenied as e:
            print(f"\nSTOPPING: request denied — {e}")
            print(f"Stopped after {i - 1}/{len(chunks)} chunk(s) completed. Writing whatever was fetched before the denial.")
            denied = True
            break
        sp.merge_into_by_hour(by_hour, **responses)

    new_rows = {}
    unmatched = 0
    for date_str, time_local, filename in targets:
        target_dt = datetime.strptime(f"{date_str} {time_local}", "%Y-%m-%d %H:%M")
        hour_key = target_dt.replace(minute=0, second=0, microsecond=0)
        predictors = by_hour.get(hour_key)
        if predictors is None:
            unmatched += 1
            continue
        new_rows[filename] = sp.row_from_predictors(date_str, time_local, filename, predictors)

    # Re-read the file immediately before merging rather than reusing the
    # snapshot taken at startup. A full-history run takes ~2 hours, and the
    # scheduled pipeline appends to this same CSV partway through (Step 5,
    # get_surf_predictors.py) — rewriting from the stale startup snapshot
    # would silently delete every row the pipeline added while we were
    # fetching. Rows that appeared in the meantime are kept as-is unless this
    # run actually re-fetched that filename.
    current = sp.load_rows(out_csv)
    appeared = len(current) - len(existing)
    if appeared:
        print(f"note: {appeared} row(s) were added to {out_csv.name} by another process "
              f"during this run (likely the scheduled pipeline) — preserving them.")

    # Rewrite rather than append: with --refetch-incomplete/--force a target may
    # already have a row in the file, and appending would leave two rows for the
    # same filename. Existing rows keep their original position; anything
    # re-fetched is replaced in place, and genuinely new rows go on the end.
    replaced = sum(1 for r in current if r["filename"] in new_rows)
    merged = [new_rows.pop(r["filename"], r) for r in current]
    merged.extend(new_rows.values())
    added = len(merged) - len(current)
    write_rows(out_csv, merged)

    print(f"\nDone. {added} row(s) added, {replaced} replaced in {out_csv} "
          f"({unmatched} target row(s) had no matching hour in the fetched data).")
    if denied:
        print("Run was stopped early due to a denied request — re-run later (fresh token if needed) to pick up the rest.")


if __name__ == "__main__":
    main()
