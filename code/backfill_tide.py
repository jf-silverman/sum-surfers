"""
backfill_tide.py
----------------
One-time backfill: estimate tide height (ft, MLLW) for every existing row in
predictions.csv, using NOAA CO-OPS high/low predictions for the Santa Cruz
station (9413745) and cosine interpolation between the surrounding high and
low.

Cosine interpolation assumes the tide curve between consecutive extrema is a
half-cosine (standard approximation absent full harmonic data): given a point
a fraction f of the way from one extreme (height h0) to the next (height h1),

    h(f) = h0 + (h1 - h0) * (1 - cos(pi * f)) / 2

This is a backfill-only script (Surfline will be the source for tide/swell
going forward) — not part of the scheduled pipeline.

Usage:
    python code/backfill_tide.py
Writes data/tide_backfill.csv with columns: date,time_local,filename,tide_ft_est
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT_CSV = PROJECT_ROOT / "data" / "tide_backfill.csv"

TIDE_STATION = "9413745"  # Santa Cruz, Monterey Bay


def fetch_hilo(begin_date: str, end_date: str):
    """Fetch high/low tide predictions. NOAA limits ranges to ~1 year; chunk by month to be safe."""
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?begin_date={begin_date}&end_date={end_date}&station={TIDE_STATION}"
        "&product=predictions&datum=MLLW&time_zone=lst_ldt&units=english"
        "&interval=hilo&format=json"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["predictions"]


def month_chunks(start: datetime, end: datetime):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + __import__("datetime").timedelta(days=4)).replace(day=1)
        chunk_end = min(nxt - __import__("datetime").timedelta(days=1), end)
        yield cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        cur = nxt


def estimate_tide(target: datetime, extrema: list[tuple[datetime, float]]) -> float | None:
    """Cosine-interpolate tide height at `target` from a sorted list of (time, height) extrema."""
    prev = None
    for t, h in extrema:
        if t <= target:
            prev = (t, h)
        else:
            if prev is None:
                return None  # target before first known extreme
            nxt = (t, h)
            span = (nxt[0] - prev[0]).total_seconds()
            if span <= 0:
                return prev[1]
            f = (target - prev[0]).total_seconds() / span
            return prev[1] + (nxt[1] - prev[1]) * (1 - math.cos(math.pi * f)) / 2
    return None  # target after last known extreme


def main():
    rows = list(csv.DictReader(open(PREDS_CSV, newline="")))
    dates = sorted({r["date"] for r in rows})
    start = datetime.strptime(dates[0], "%Y-%m-%d")
    end = datetime.strptime(dates[-1], "%Y-%m-%d")

    print(f"Fetching NOAA hi/lo tide predictions for {start.date()} to {end.date()}...")
    extrema = []
    for begin, chunk_end in month_chunks(start, end):
        preds = fetch_hilo(begin, chunk_end)
        for p in preds:
            t = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
            extrema.append((t, float(p["v"])))
    extrema.sort(key=lambda x: x[0])
    print(f"Got {len(extrema)} high/low points.")

    out_rows = []
    missing = 0
    for r in rows:
        target = datetime.strptime(f"{r['date']} {r['time_local']}", "%Y-%m-%d %H:%M")
        est = estimate_tide(target, extrema)
        if est is None:
            missing += 1
        out_rows.append({
            "date": r["date"],
            "time_local": r["time_local"],
            "filename": r["filename"],
            "tide_ft_est": round(est, 2) if est is not None else "",
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "time_local", "filename", "tide_ft_est"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT_CSV} ({missing} without an estimate).")


if __name__ == "__main__":
    main()
