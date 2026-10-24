"""
Pull every event listed by Visit Santa Cruz County (santacruz.org, the county's
visitor bureau) from 2025-07-01 to 2026-12-31 through the site's public events
API (The Events Calendar plugin, /wp-json/tribe/events/v1/events), and write
one row per event-day to visit_sc_events.csv.

Why this source: it is local, current, machine-readable and goes back past the
start of our count record (2025-10-11), so a feature built from it can be
backtested. Newspapers (Lookout's BOLO calendar, Good Times, the Sentinel) were
also checked; they publish listings as articles, not a feed.

The site refuses Python's default user agent, so requests sends a curl one.
Multi-day events are expanded to one row per day, capped at 7 days so that
season-long listings (exhibitions, steam trains) do not flood every date.

Usage:
    python analysis/packed_hours/fetch_visit_sc_events.py
"""
from __future__ import annotations

import html
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
URL = "https://www.santacruz.org/wp-json/tribe/events/v1/events"
HEAD = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}
OUT = HERE / "visit_sc_events.csv"


def fetch(start="2025-07-01", end="2026-12-31"):
    rows, page = [], 1
    while True:
        r = requests.get(URL, headers=HEAD, timeout=60, params={
            "start_date": start, "end_date": end, "per_page": 50, "page": page})
        r.raise_for_status()
        d = r.json()
        for e in d.get("events", []):
            v = e.get("venue") or {}
            v = v if isinstance(v, dict) else {}
            rows.append({
                "id": e["id"], "title": html.unescape(e["title"]),
                "start": e["start_date"], "end": e["end_date"],
                "all_day": e.get("all_day", False),
                "venue": html.unescape(v.get("venue", "") or ""),
                "city": v.get("city", "") or "",
                "categories": "|".join(html.unescape(c["name"]) for c in e.get("categories", [])),
                "url": e.get("url", "")})
        if page >= d.get("total_pages", 1):
            return pd.DataFrame(rows), d.get("total")
        page += 1
        time.sleep(1)


def main():
    ev, total = fetch()
    days = []
    for _, e in ev.iterrows():
        s, t = pd.Timestamp(e.start).normalize(), pd.Timestamp(e.end).normalize()
        span = (t - s).days + 1
        for d in pd.date_range(s, s + pd.Timedelta(days=min(span, 7) - 1)):
            days.append({**e.to_dict(), "date": d.strftime("%Y-%m-%d"), "span_days": span})
    out = pd.DataFrame(days).drop_duplicates(["id", "date"])
    out.to_csv(OUT, index=False)
    print(f"{len(ev)} events (API total {total}), {len(out)} event-days -> {OUT.name}")


if __name__ == "__main__":
    main()
