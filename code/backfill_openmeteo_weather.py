"""
backfill_openmeteo_weather.py
-------------------------------
Adds REAL observed historical weather (Open-Meteo's archive API, ERA5-based
reanalysis — not a forecast) for every predictions.csv row, as a separate file
from surfline_predictors.csv (non-destructive — doesn't touch Surfline's
forecast-based weather_condition/temperature_f/pressure_mb).

Why this exists: Surfline's weather fields are either a live forecast (for
recent dates, pulled a couple times a week) or a historical-endpoint snapshot
(itself model-based, not necessarily true observations). Open-Meteo's archive
endpoint is real historical weather for the exact date/time, free, no auth,
no rate limits, and covers the whole project date range in a single request.

Also adds relative_humidity_2m and cloud_cover specifically because they're a
validated (if imperfect) proxy for the fog/blur conditions the image-quality
gate already flags — cross-checked against the 24 real `foggy_or_blurred`
rows in predictions.csv: mean humidity 87.5% on foggy rows vs 69.8% on clear
rows from the same dates (real, meaningful separation). Cloud_cover alone is
weaker (61.7 vs 55.6) and weather_code doesn't reliably flag fog at all — a
few known-foggy rows show clear/mostly-clear weather_code despite real fog,
consistent with the literature on coarse-grid reanalysis models struggling to
resolve localized coastal marine-layer fog. So humidity is the useful proxy
here, not a categorical fog flag.

Usage:
    python code/backfill_openmeteo_weather.py
"""

import csv
from datetime import datetime
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = _PROJECT_ROOT / "data" / "predictions.csv"
OUT_CSV = _PROJECT_ROOT / "data" / "openmeteo_weather.csv"

LATITUDE, LONGITUDE = 36.9577, -121.9688  # Pleasure Point, Santa Cruz (see get_clips.py)
TIMEZONE = "America/Los_Angeles"

CSV_HEADER = [
    "date", "time_local", "filename",
    "real_temperature_f", "real_humidity_pct", "real_cloud_cover_pct",
    "real_weather_code", "real_pressure_mb",
]


def fetch_archive(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,weather_code,surface_pressure",
        "timezone": TIMEZONE,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["hourly"]


def main():
    preds = list(csv.DictReader(open(PREDS_CSV)))
    if not preds:
        print("No predictions.csv rows found.")
        return

    dates = sorted(r["date"] for r in preds)
    start_date, end_date = dates[0], dates[-1]
    print(f"Fetching Open-Meteo archive for {start_date} to {end_date}...")
    h = fetch_archive(start_date, end_date)

    by_hour = {}
    for i, t in enumerate(h["time"]):
        by_hour[t] = {
            "temp_c": h["temperature_2m"][i],
            "humidity": h["relative_humidity_2m"][i],
            "cloud": h["cloud_cover"][i],
            "wcode": h["weather_code"][i],
            "pressure": h["surface_pressure"][i],
        }
    print(f"Fetched {len(by_hour)} hourly records.")

    written = 0
    unmatched = 0
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in preds:
            hour = r["time_local"].split(":")[0]
            key = f"{r['date']}T{hour}:00"
            v = by_hour.get(key)
            if v is None:
                unmatched += 1
                continue
            temp_f = v["temp_c"] * 9 / 5 + 32 if v["temp_c"] is not None else ""
            writer.writerow({
                "date": r["date"], "time_local": r["time_local"], "filename": r["filename"],
                "real_temperature_f": round(temp_f, 1) if temp_f != "" else "",
                "real_humidity_pct": v["humidity"],
                "real_cloud_cover_pct": v["cloud"],
                "real_weather_code": v["wcode"],
                "real_pressure_mb": v["pressure"],
            })
            written += 1

    print(f"\nDone. Wrote {written} row(s) to {OUT_CSV} ({unmatched} unmatched).")


if __name__ == "__main__":
    main()
