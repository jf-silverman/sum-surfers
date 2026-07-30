"""
get_surf_predictors.py
-----------------------
Pulls weather, condition rating, tide, and swell/wave predictors for Jack's
from Surfline's public forecast API and appends a row to
data/surfline_predictors.csv for each predictions.csv row that doesn't have
one yet (matched by filename, restricted to the same recent-days window
detect_surfers.py uses so a fresh machine doesn't try to backfill history).

Endpoints used (no accesstoken required — omitting it avoids a 403 on
`rating` that occurs when a token IS passed):
  https://services.surfline.com/kbyg/spots/forecasts/weather
  https://services.surfline.com/kbyg/spots/forecasts/rating
  https://services.surfline.com/kbyg/spots/forecasts/tides
  https://services.surfline.com/kbyg/spots/forecasts/wave

These only return forward-looking data (today onward) — there is no known
lightweight endpoint for historical dates. Surfline's site-side "Historical"
toggle renders past dates via a mechanism that isn't a visible fetch/XHR
call (confirmed via network + fetch/XHR patching), so backfilling old
predictions.csv rows with these predictors is not implemented here.

Usage:
    python code/get_surf_predictors.py
"""

import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = PROJECT_ROOT / "data" / "predictions.csv"
OUT_CSV = PROJECT_ROOT / "data" / "surfline_predictors.csv"

SPOT_ID = "5842041f4e65fad6a770880b"  # Jack's

REQUEST_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.surfline.com",
    "Referer": "https://www.surfline.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/26.5 Safari/605.1.15"
    ),
}

DAYS = 2  # today + 1, covers utcOffset edge cases around midnight

CSV_HEADER = [
    "date", "time_local", "filename",
    "temperature_f", "weather_condition", "pressure_mb",
    "rating_key", "rating_value",
    "tide_ft",
    "surf_min_ft", "surf_max_ft",
    "primary_swell_height_ft", "primary_swell_period_s", "primary_swell_direction_deg",
]

# Same "recent" scope logic as detect_surfers.py, so a fresh machine doesn't
# try to walk the entire crop history looking for matches.
DETECT_RECENT_DAYS = int(os.environ.get("DETECT_RECENT_DAYS", "7"))


def fetch(path, retries=3):
    params = {"spotId": SPOT_ID, "days": str(DAYS), "intervalHours": "1"}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(
                f"https://services.surfline.com/kbyg/spots/forecasts/{path}",
                params=params,
                headers=REQUEST_HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            return r.json()["data"][path]
        except requests.exceptions.HTTPError as e:
            last_exc = e
            time.sleep(2 * (attempt + 1))
    raise last_exc


def local_hour_key(timestamp, utc_offset):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=utc_offset)
    return dt.replace(minute=0, second=0, microsecond=0)


def build_predictor_map():
    weather = fetch("weather")
    rating = fetch("rating")
    tides = fetch("tides")
    wave = fetch("wave")

    by_hour = {}
    for w in weather:
        key = local_hour_key(w["timestamp"], w["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["temperature_f"] = round(w["temperature"], 1)
        by_hour[key]["weather_condition"] = w["condition"]
        by_hour[key]["pressure_mb"] = w["pressure"]

    for r in rating:
        key = local_hour_key(r["timestamp"], r["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["rating_key"] = r["rating"]["key"]
        by_hour[key]["rating_value"] = r["rating"]["value"]

    for w in wave:
        key = local_hour_key(w["timestamp"], w["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["surf_min_ft"] = w["surf"]["min"]
        by_hour[key]["surf_max_ft"] = w["surf"]["max"]
        swells = w.get("swells") or []
        primary = swells[0] if swells else {}
        by_hour[key]["primary_swell_height_ft"] = round(primary.get("height", 0), 2)
        by_hour[key]["primary_swell_period_s"] = primary.get("period", 0)
        by_hour[key]["primary_swell_direction_deg"] = round(primary.get("direction", 0), 1)

    # Tides are finer-grained than hourly; snap each to the nearest hour bucket
    # already created above (or start its own if no weather/wave/rating entry
    # exists for that hour, which shouldn't normally happen).
    for t in tides:
        key = local_hour_key(t["timestamp"], t["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key].setdefault("tide_ft", t["height"])

    return by_hour


def load_rows(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def in_recent_scope(date_str):
    row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    cutoff = (datetime.now() - timedelta(days=max(DETECT_RECENT_DAYS - 1, 0))).date()
    return row_date >= cutoff


def append_row(csv_path, row):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    preds = load_rows(PREDS_CSV)
    existing = load_rows(OUT_CSV)
    done_filenames = {r["filename"] for r in existing}

    candidates = [r for r in preds if in_recent_scope(r["date"]) and r["filename"] not in done_filenames]
    if not candidates:
        print("No new predictions.csv rows need predictors.")
        return

    print(f"Fetching Jack's forecast data (days={DAYS})...")
    by_hour = build_predictor_map()

    written = 0
    for r in candidates:
        target = datetime.strptime(f"{r['date']} {r['time_local']}", "%Y-%m-%d %H:%M")
        hour_key = target.replace(minute=0, second=0, microsecond=0)
        predictors = by_hour.get(hour_key)
        if predictors is None:
            continue  # outside the fetched forecast window

        append_row(OUT_CSV, {
            "date": r["date"],
            "time_local": r["time_local"],
            "filename": r["filename"],
            "temperature_f": predictors.get("temperature_f", ""),
            "weather_condition": predictors.get("weather_condition", ""),
            "pressure_mb": predictors.get("pressure_mb", ""),
            "rating_key": predictors.get("rating_key", ""),
            "rating_value": predictors.get("rating_value", ""),
            "tide_ft": predictors.get("tide_ft", ""),
            "surf_min_ft": predictors.get("surf_min_ft", ""),
            "surf_max_ft": predictors.get("surf_max_ft", ""),
            "primary_swell_height_ft": predictors.get("primary_swell_height_ft", ""),
            "primary_swell_period_s": predictors.get("primary_swell_period_s", ""),
            "primary_swell_direction_deg": predictors.get("primary_swell_direction_deg", ""),
        })
        written += 1

    print(f"Wrote {written} row(s) to {OUT_CSV} ({len(candidates) - written} had no matching forecast hour).")


if __name__ == "__main__":
    main()
