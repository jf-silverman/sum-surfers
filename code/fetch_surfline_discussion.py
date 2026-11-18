"""
fetch_surfline_discussion.py
----------------------------
Records Surfline's written forecast for the Santa Cruz region (pipeline step
5b, added 2026-11-18): each day's headline and discussion from the forecaster.

Why: Joel noticed that words like "jumbled swell mix" or "strong onshore wind"
mean a bad, empty day, and the numbers the model sees did not mark the stormy
days 11-03, 11-12 and 11-13 clearly. 2026-11-18's headline was "Jumbled swell
mix builds in. Strong onshore S wind to start then partial clean up late in
the day."

Source: services.surfline.com/kbyg/regions/forecasts/conditions for subregion
58581a836630e24c44879011 (Santa Cruz), with no token, the same way step 5
fetches the spot forecasts. It returns today plus the next days. There is no
history: `start=` is ignored (checked 2026-11-18), so the record starts now and
every night's snapshot is kept, since tomorrow's text is revised the next
morning.

Output: data/predictor_vars/surfline_discussion.csv
    fetched_date, forecast_day, human, day_to_watch, forecaster, headline, observation
(`observation` with its HTML tags removed). A re-run on the same night replaces
that night's rows.

Usage:
    python code/fetch_surfline_discussion.py
"""

import csv
import html
import re
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_surf_predictors as sp  # noqa: E402 -- REQUEST_HEADERS (Cloudflare)
import surfline_request_log as srl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "predictor_vars" / "surfline_discussion.csv"
URL = "https://services.surfline.com/kbyg/regions/forecasts/conditions"
SUBREGION_ID = "58581a836630e24c44879011"   # Santa Cruz, from Jack's spot report
DAYS = 5
FIELDS = ["fetched_date", "forecast_day", "human", "day_to_watch", "forecaster", "headline", "observation"]


def plain(text):
    """Surfline's discussion HTML -> one line of plain text."""
    t = re.sub(r"<br\s*/?>", " ", text or "")
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def rows_from(conditions, fetched):
    return [{"fetched_date": fetched, "forecast_day": c["forecastDay"],
             "human": c.get("human"), "day_to_watch": c.get("dayToWatch"),
             "forecaster": (c.get("forecaster") or {}).get("name", ""),
             "headline": plain(c.get("headline")), "observation": plain(c.get("observation"))}
            for c in conditions]


def merge(new_rows, path=OUT_CSV):
    rows = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["fetched_date"], r["forecast_day"])] = r
    for r in new_rows:
        rows[(r["fetched_date"], r["forecast_day"])] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return len(rows)


def main():
    r = requests.get(URL, params={"subregionId": SUBREGION_ID, "days": DAYS},
                     headers=sp.REQUEST_HEADERS, timeout=30)
    srl.record("discussion", URL, r, 1)
    r.raise_for_status()
    new = rows_from(r.json()["data"]["conditions"], date.today().isoformat())
    n = merge(new)
    for x in new:
        print(f"  {x['forecast_day']}  {x['headline'][:100]}")
    print(f"{len(new)} days saved; {n} rows in {OUT_CSV.name}")


if __name__ == "__main__":
    main()
