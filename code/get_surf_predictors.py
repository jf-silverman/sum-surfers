"""
get_surf_predictors.py
-----------------------
Pulls weather, condition rating, tide, swell/wave, wind, wave energy, and
consistency predictors for Jack's from Surfline's public forecast API and
appends a row to data/predictor_vars/surfline_predictors.csv for each predictions.csv row
that doesn't have one yet (matched by filename, restricted to the same
recent-days window detect_surfers.py uses so a fresh machine doesn't try to
backfill history).

Endpoints used (no accesstoken required for forward-looking data — omitting
it avoids a 403 on `rating` that occurs when a token IS passed):
  https://services.surfline.com/kbyg/spots/forecasts/weather
  https://services.surfline.com/kbyg/spots/forecasts/rating
  https://services.surfline.com/kbyg/spots/forecasts/tides
  https://services.surfline.com/kbyg/spots/forecasts/wave
  https://services.surfline.com/kbyg/spots/forecasts/wind
  https://services.surfline.com/kbyg/spots/forecasts/energy
  https://services.surfline.com/kbyg/spots/forecasts/consistency

These only return forward-looking data (today onward) for anonymous
requests. Historical dates ARE available from the same endpoints via a
`start=YYYY-MM-DD` parameter, but require an `x-auth-accesstoken` header
from a logged-in, premium Surfline session — see
code/backfill_historical_predictors.py, which shares the extraction logic
in this file (build_predictor_map / merge_into_by_hour / CSV_HEADER).

build_predictor_map() also merges in real_temperature_f/real_humidity_pct/
real_cloud_cover_pct/real_weather_code/real_pressure_mb from Open-Meteo's
live forecast API (api.open-meteo.com/v1/forecast — forward-looking,
distinct from the archive API backfill_openmeteo_weather.py uses for
historical rows) under the same field names the trained model expects.
Added 2026-08-28 to fix a real bug: predict_surf_count.py's live prediction
path only ever pulled Surfline data, so these model features (trained on,
and real_humidity_pct in particular a validated fog proxy — see
docs/PROJECT_HISTORY.md) were silently NaN on every live prediction. Non-fatal
if this fetch fails — predictions just fall back to the previous NaN
behavior for these fields, same as any other missing predictor.

Usage:
    python code/get_surf_predictors.py
"""

import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_clips import LOCATION as CAMERA_LOCATION  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT_CSV = PROJECT_ROOT / "data" / "predictor_vars" / "surfline_predictors.csv"

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

# Endpoints fetched, and the key each response's payload is nested under
# (data[<path>]) — same for every endpoint observed so far.
ENDPOINT_PATHS = ["weather", "rating", "tides", "wave", "wind", "energy", "consistency"]

# Same "recent" scope logic as detect_surfers.py, so a fresh machine doesn't
# try to walk the entire crop history looking for matches.
DETECT_RECENT_DAYS = int(os.environ.get("DETECT_RECENT_DAYS", "7"))

CSV_HEADER = [
    "date", "time_local", "filename",
    "temperature_f", "weather_condition", "pressure_mb",
    "rating_key", "rating_value",
    "tide_ft",
    "surf_min_ft", "surf_max_ft",
    "primary_swell_height_ft", "primary_swell_period_s", "primary_swell_direction_deg",
    "wind_speed_mph", "wind_direction_deg", "wind_direction_type", "wind_gust_mph",
    "energy_offshore_kj", "energy_nearshore_kj",
    "consistency_wave_count",
]


def fetch(path, params, headers=None, retries=3):
    """
    GET one forecast endpoint. `params` should already include spotId and
    days (and start/intervalHours/etc. as needed by the caller) — kept
    generic here so both the live (forward-looking) and historical
    (start=date) callers can share it.
    """
    headers = headers or REQUEST_HEADERS
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(
                f"https://services.surfline.com/kbyg/spots/forecasts/{path}",
                params=params,
                headers=headers,
                timeout=20,
            )
            r.raise_for_status()
            return r.json()["data"][path]
        except requests.exceptions.RequestException as e:
            # Broad exception type deliberately -- was `HTTPError` (only a bad
            # HTTP response, e.g. 404/500), which meant these retries never
            # actually fired for a real connectivity failure (DNS down, no
            # network, timeout): requests.get() itself raises ConnectionError
            # or Timeout, sibling exceptions HTTPError doesn't catch, so the
            # very first attempt would propagate immediately with 0 of the
            # intended `retries` actually attempted. See docs/PROJECT_HISTORY.md's
            # 2026-09-03 entry (same root-cause bug as build_predictor_map()'s
            # except clause, found the same day).
            last_exc = e
            time.sleep(2 * (attempt + 1))
    raise last_exc


def local_hour_key(timestamp, utc_offset):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=utc_offset)
    return dt.replace(minute=0, second=0, microsecond=0)


def merge_into_by_hour(by_hour, weather=None, rating=None, tides=None, wave=None,
                        wind=None, energy=None, consistency=None):
    """
    Pure merge step (no network calls) — folds raw per-endpoint response
    lists into a {local_hour_datetime: {field: value}} dict. Shared by the
    live (forward-looking) and historical backfill scripts so the field
    extraction logic only lives in one place.
    """
    for w in weather or []:
        key = local_hour_key(w["timestamp"], w["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["temperature_f"] = round(w["temperature"], 1)
        by_hour[key]["weather_condition"] = w["condition"]
        by_hour[key]["pressure_mb"] = w["pressure"]

    for r in rating or []:
        key = local_hour_key(r["timestamp"], r["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["rating_key"] = r["rating"]["key"]
        by_hour[key]["rating_value"] = r["rating"]["value"]

    for w in wave or []:
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
    for t in tides or []:
        key = local_hour_key(t["timestamp"], t["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key].setdefault("tide_ft", t["height"])

    for w in wind or []:
        key = local_hour_key(w["timestamp"], w["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["wind_speed_mph"] = round(w.get("speed", 0), 2)
        by_hour[key]["wind_direction_deg"] = round(w.get("direction", 0), 1)
        by_hour[key]["wind_direction_type"] = w.get("directionType", "")
        by_hour[key]["wind_gust_mph"] = round(w.get("gust", 0), 2)

    for e in energy or []:
        key = local_hour_key(e["timestamp"], e["utcOffset"])
        by_hour.setdefault(key, {})
        by_hour[key]["energy_offshore_kj"] = round(e.get("offshore", 0), 1)
        by_hour[key]["energy_nearshore_kj"] = round(e.get("nearshore", 0), 1)

    for c in consistency or []:
        try:
            key = local_hour_key(c["timestamp"], c["utcOffset"])
            by_hour.setdefault(key, {})
            by_hour[key]["consistency_wave_count"] = round(c["consistency"]["waveCount"], 2)
        except (KeyError, TypeError) as e:
            print(f"  WARNING: unexpected consistency record shape, skipping: {e}")

    return by_hour


def fetch_openmeteo_forecast(days=DAYS):
    """Live (forward-looking) Open-Meteo forecast fetch — same fields
    backfill_openmeteo_weather.py pulls from the archive API for historical
    rows, so the live predictor dict and the training data use identical
    field names/semantics. No auth needed. Raises on HTTP failure; caller
    decides how to handle that (see build_predictor_map)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CAMERA_LOCATION["latitude"], "longitude": CAMERA_LOCATION["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,weather_code,surface_pressure",
        "timezone": CAMERA_LOCATION["timezone"],
        "forecast_days": days,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()["hourly"]


def merge_openmeteo_forecast(by_hour, hourly):
    """Folds an Open-Meteo forecast response (as returned by
    fetch_openmeteo_forecast) into the same {local_hour_datetime: {field:
    value}} dict build_predictor_map() builds from Surfline data. Open-Meteo
    already returns local wall-clock time strings when `timezone` is passed,
    so no UTC-offset math is needed here (unlike local_hour_key(), which
    exists because Surfline's timestamps are raw UTC epoch seconds)."""
    for i, t in enumerate(hourly["time"]):
        key = datetime.fromisoformat(t)
        by_hour.setdefault(key, {})
        temp_c = hourly["temperature_2m"][i]
        by_hour[key]["real_temperature_f"] = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
        by_hour[key]["real_humidity_pct"] = hourly["relative_humidity_2m"][i]
        by_hour[key]["real_cloud_cover_pct"] = hourly["cloud_cover"][i]
        by_hour[key]["real_weather_code"] = hourly["weather_code"][i]
        by_hour[key]["real_pressure_mb"] = hourly["surface_pressure"][i]
    return by_hour


def build_predictor_map():
    """Live (forward-looking, today+DAYS) fetch — no auth token needed/used."""
    responses = {}
    for path in ENDPOINT_PATHS:
        params = {"spotId": SPOT_ID, "days": str(DAYS), "intervalHours": "1"}
        try:
            responses[path] = fetch(path, params)
        except requests.exceptions.RequestException as e:
            # Broad exception type deliberately -- catches HTTPError (bad response)
            # AND ConnectionError/Timeout/etc. (network down, DNS failure, ...).
            # A narrower except HTTPError here let a real DNS-resolution failure
            # (machine offline at cron time) propagate uncaught through main() and
            # crash the whole daily_chart.sh run before it ever reached its git
            # commit/push step -- no chart, no README update, no error surfaced
            # anywhere. See docs/PROJECT_HISTORY.md's 2026-09-03 entry.
            print(f"  WARNING: {path} fetch failed ({e}), continuing without it")
            responses[path] = []
    by_hour = merge_into_by_hour({}, **responses)

    try:
        openmeteo_hourly = fetch_openmeteo_forecast()
        by_hour = merge_openmeteo_forecast(by_hour, openmeteo_hourly)
    except requests.exceptions.RequestException as e:
        print(f"  WARNING: Open-Meteo forecast fetch failed ({e}), continuing without real_* fields")

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


def row_from_predictors(date_str, time_local, filename, predictors):
    return {
        "date": date_str, "time_local": time_local, "filename": filename,
        **{field: predictors.get(field, "") for field in CSV_HEADER if field not in ("date", "time_local", "filename")},
    }


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

        append_row(OUT_CSV, row_from_predictors(r["date"], r["time_local"], r["filename"], predictors))
        written += 1

    print(f"Wrote {written} row(s) to {OUT_CSV} ({len(candidates) - written} had no matching forecast hour).")


if __name__ == "__main__":
    main()
