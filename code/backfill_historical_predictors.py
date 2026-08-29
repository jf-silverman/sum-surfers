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
    p.add_argument("--dry-run", action="store_true", help="Show what would be fetched/written without making any requests")
    return p.parse_args()


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

    written = 0
    unmatched = 0
    for date_str, time_local, filename in targets:
        target_dt = datetime.strptime(f"{date_str} {time_local}", "%Y-%m-%d %H:%M")
        hour_key = target_dt.replace(minute=0, second=0, microsecond=0)
        predictors = by_hour.get(hour_key)
        if predictors is None:
            unmatched += 1
            continue
        sp.append_row(out_csv, sp.row_from_predictors(date_str, time_local, filename, predictors))
        written += 1

    print(f"\nDone. Wrote {written} row(s) to {out_csv} ({unmatched} target row(s) had no matching hour in the fetched data).")
    if denied:
        print("Run was stopped early due to a denied request — re-run later (fresh token if needed) to pick up the rest.")


if __name__ == "__main__":
    main()
