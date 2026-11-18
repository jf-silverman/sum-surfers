"""
fetch_rain_forecast.py
----------------------
Records forecast rain for Jack's, hour by hour (pipeline step 6d, added
2026-11-18), so rain can be tested as a forecast input.

Why: the stormy, empty days 11-03, 11-12 and 11-13 were forecast busy. The
model's only rain input is Surfline's weather condition, which called rain in
just 12-33% of those days' hours. Forecast rain amount and chance of rain are
what a surfer reads the night before, and are known when the evening forecast
is made, unlike observed rain.

Sources (no account, nothing to do with Surfline):
  openmeteo   Open-Meteo forecast API, hourly precipitation (mm) and
              precipitation_probability (%), 8 days ahead.
  nws         NWS gridpoint MTR/93,66 (the grid cell holding Jack's):
              hourly probabilityOfPrecipitation, and quantitativePrecipitation
              issued in 6-hour blocks, spread evenly over its hours.
  openmeteo_archive
              `--backfill`: Open-Meteo's historical-forecast archive, the same
              model forecasts stitched from each run's first hours. Used for
              training rows; its lead time is shorter than the night-before
              forecast the pipeline makes, so it is a little better than what
              the live forecast will see. NWS keeps no such archive.

Output: data/predictor_vars/rain_forecast.csv, one row per source, fetch day
and valid hour (local):
    source, fetched_date, valid_local, precip_mm, pop_pct
Live rows keep every night's snapshot, so lead time can be studied later; a
re-run on the same night replaces that night's rows.

Usage:
    python code/fetch_rain_forecast.py
    python code/fetch_rain_forecast.py --backfill 2025-10-01
"""

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytz
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "predictor_vars" / "rain_forecast.csv"
FIELDS = ["source", "fetched_date", "valid_local", "precip_mm", "pop_pct"]
LAT, LON, TZ = 36.9577, -121.9688, "America/Los_Angeles"   # get_clips.py's camera point
NWS_GRID = "https://api.weather.gov/gridpoints/MTR/93,66"
NWS_HEADERS = {"User-Agent": "sum-surfers (github.com/jf-silverman/sum-surfers)",
               "Accept": "application/geo+json"}
TIMEOUT_SEC = 30
RETRY_WAITS_SEC = [10, 30]


def _get(url, params=None, headers=None):
    last = None
    for wait in RETRY_WAITS_SEC + [None]:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SEC)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            last = e
            if wait is not None:
                time.sleep(wait)
    raise last


def openmeteo_rows(hourly, source, fetched):
    return [{"source": source, "fetched_date": fetched, "valid_local": t,
             "precip_mm": p, "pop_pct": "" if q is None else q}
            for t, p, q in zip(hourly["time"], hourly["precipitation"],
                               hourly.get("precipitation_probability") or [None] * len(hourly["time"]))]


def fetch_openmeteo(fetched):
    j = _get("https://api.open-meteo.com/v1/forecast",
             {"latitude": LAT, "longitude": LON, "timezone": TZ, "forecast_days": 8,
              "hourly": "precipitation,precipitation_probability"})
    return openmeteo_rows(j["hourly"], "openmeteo", fetched)


def backfill_openmeteo(start, end):
    j = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": LAT, "longitude": LON, "timezone": TZ,
              "start_date": start, "end_date": end,
              "hourly": "precipitation,precipitation_probability"})
    return openmeteo_rows(j["hourly"], "openmeteo_archive", "")


def _interval(valid_time):
    """'2026-11-18T12:00:00+00:00/PT6H' -> (start UTC datetime, hours)."""
    start, dur = valid_time.split("/")
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?", dur)
    hours = int(m[1] or 0) * 24 + int(m[2] or 0)
    return datetime.fromisoformat(start), max(hours, 1)


def nws_hourly(props):
    """{local hour 'YYYY-MM-DDTHH:00': {'precip_mm', 'pop_pct'}} from a
    gridpoint's properties. A 6-hour precipitation amount is spread evenly
    over its hours; probability is issued hourly."""
    tz = pytz.timezone(TZ)
    out = {}
    for key, field, spread in (("quantitativePrecipitation", "precip_mm", True),
                               ("probabilityOfPrecipitation", "pop_pct", False)):
        for v in props.get(key, {}).get("values", []):
            start, hours = _interval(v["validTime"])
            val = v["value"]
            if val is None:
                continue
            for h in range(hours):
                t = (start + timedelta(hours=h)).astimezone(tz).strftime("%Y-%m-%dT%H:00")
                out.setdefault(t, {})[field] = round(val / hours, 3) if spread else val
    return out


def fetch_nws(fetched):
    props = _get(NWS_GRID, headers=NWS_HEADERS)["properties"]
    return [{"source": "nws", "fetched_date": fetched, "valid_local": t,
             "precip_mm": v.get("precip_mm", ""), "pop_pct": v.get("pop_pct", "")}
            for t, v in sorted(nws_hourly(props).items())]


def merge(new_rows, path=OUT_CSV):
    """Replace rows with the same (source, fetched_date, valid_local); keep the rest."""
    rows = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["source"], r["fetched_date"], r["valid_local"])] = r
    for r in new_rows:
        rows[(r["source"], r["fetched_date"], r["valid_local"])] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return len(rows)


def load_archive(path=OUT_CSV):
    """{local hour 'YYYY-MM-DDTHH:00': (precip_mm, pop_pct or None)} from the
    archive rows (training)."""
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "openmeteo_archive" and r["precip_mm"] != "":
                out[r["valid_local"]] = (float(r["precip_mm"]),
                                         float(r["pop_pct"]) if r["pop_pct"] != "" else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", metavar="START", help="archive forecasts from START to yesterday")
    args = ap.parse_args()
    if args.backfill:
        end = (date.today() - timedelta(days=1)).isoformat()
        n = merge(backfill_openmeteo(args.backfill, end))
        print(f"archive {args.backfill}..{end} merged; {n} rows in {OUT_CSV.name}")
        return
    fetched = date.today().isoformat()
    new, failed = [], []
    for name, fn in (("openmeteo", fetch_openmeteo), ("nws", fetch_nws)):
        try:
            rows = fn(fetched)
            new += rows
            print(f"{name}: {len(rows)} hours")
        except Exception as e:  # noqa: BLE001 -- one source down must not lose the other
            failed.append(name)
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
    # Keep the archive current too, so the training table has yesterday.
    try:
        y = (date.today() - timedelta(days=1)).isoformat()
        new += backfill_openmeteo((date.today() - timedelta(days=7)).isoformat(), y)
    except Exception as e:  # noqa: BLE001
        failed.append("openmeteo_archive")
        print(f"openmeteo_archive: FAILED ({type(e).__name__}: {e})")
    n = merge(new)
    print(f"{n} rows in {OUT_CSV.name}")
    sys.exit(1 if len(failed) == 3 else 0)


if __name__ == "__main__":
    main()
