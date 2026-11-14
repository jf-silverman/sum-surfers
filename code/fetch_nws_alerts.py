"""
fetch_nws_alerts.py
-------------------
Records National Weather Service alerts for Jack's (pipeline step 6c, added
2026-11-14), so official "stay out of the water" days can be flagged now and
tested as a forecast input once enough of them are counted.

Why: 2026-11-03 and 11-12 were empty, stormy days the forecast called busy,
and the model ignores rating, wind and rain (too few examples). Joel noticed
NOAA and state agencies telling people to stay away from the water on such
days. NWS alerts are issued ahead of time, so they are available when the
evening forecast is made.

Zones: CAZ529 (Northern Monterey Bay, the land zone containing Jack's) and
PZZ535 (Monterey Bay, marine). Source: the public NWS API
(api.weather.gov/alerts/active), no account, nothing to do with Surfline.
"Active" includes alerts issued for a later onset, which is what makes
tomorrow's warning visible tonight.

Output: data/predictor_vars/nws_alerts.csv, one row per alert, updated in place
when the NWS revises it:
    alert_id, event, severity, sent, onset, ends, expires, zones, headline,
    first_seen, last_seen, source

`--backfill` adds the Iowa Environmental Mesonet archive of every alert that
covered Jack's point since 2025-09 (source "iem"; issue / expire only), for
testing against the counted record.

Usage:
    python code/fetch_nws_alerts.py
    python code/fetch_nws_alerts.py --backfill 2025-09-01
"""

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = _ROOT / "data" / "predictor_vars" / "nws_alerts.csv"
ZONES = ["CAZ529", "PZZ535"]
LAT, LON = 36.9577, -121.9688          # the camera (get_clips.LOCATION)
HEADERS = {"User-Agent": "sum-surfers (github.com/jf-silverman/sum-surfers)",
           "Accept": "application/geo+json"}
FIELDS = ["alert_id", "event", "severity", "sent", "onset", "ends", "expires", "zones",
          "headline", "first_seen", "last_seen", "source"]

# Alerts that say "dangerous to be in or near the water / out in the storm".
# Left out on the record so far: Beach Hazards Statements go out on big-swell
# days that draw surfers (counted hours under one averaged 18.2 against 15.0
# otherwise, forecast on target); Coastal Flood Advisories are very high tides,
# which the model already sees; Small Craft Advisories are routine wind on the
# bay. Checked 2026-11-13 against the IEM archive.
FLAG_EVENTS = {
    "High Surf Warning", "High Surf Advisory", "High Wind Warning", "High Wind Watch",
    "Wind Advisory", "Flood Advisory", "Flood Warning", "Flood Watch", "Flash Flood Warning",
    "Gale Warning", "Storm Warning", "Hazardous Seas Warning", "Hurricane Force Wind Warning",
    "Tsunami Warning", "Tsunami Advisory",
}


def load():
    if not OUT_CSV.exists():
        return {}
    with open(OUT_CSV, newline="") as f:
        return {r["alert_id"]: r for r in csv.DictReader(f)}


def save(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (r.get("onset") or r.get("sent") or "")):
            w.writerow({k: r.get(k, "") for k in FIELDS})


def fetch_active():
    r = requests.get("https://api.weather.gov/alerts/active", params={"zone": ",".join(ZONES)},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        p = f["properties"]
        out.append({"alert_id": p.get("id") or f.get("id"), "event": p.get("event", ""),
                    "severity": p.get("severity", ""), "sent": p.get("sent", ""),
                    "onset": p.get("onset") or p.get("effective") or "", "ends": p.get("ends") or "",
                    "expires": p.get("expires") or "",
                    "zones": " ".join(z.rsplit("/", 1)[-1] for z in p.get("affectedZones", [])),
                    "headline": (p.get("headline") or "").replace("\n", " "), "source": "nws"})
    return out


def fetch_archive(start):
    r = requests.get("https://mesonet.agron.iastate.edu/json/vtec_events_bypoint.py",
                     params={"lat": LAT, "lon": LON, "sdate": start,
                             "edate": (date.today() + timedelta(days=1)).isoformat()},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        out.append({"alert_id": f"iem:{e['wfo']}.{e['phenomena']}.{e['significance']}.{e['eventid']}."
                                f"{e['issue'][:4]}.{e['ugc']}",
                    "event": e["name"], "onset": e["issue"], "ends": e["expire"],
                    "expires": e["expire"], "zones": e["ugc"], "source": "iem"})
    return out


def flagged_alerts(day, path=None, first_hour=6, last_hour=18):
    """Flag-worthy alerts (FLAG_EVENTS) whose onset-to-end overlaps `day`'s
    daylight hours, local time: [(event, onset, ends), ...], soonest first."""
    p = Path(path) if path else OUT_CSV
    if not p.exists():
        return []
    lo = datetime(day.year, day.month, day.day, first_hour)
    hi = datetime(day.year, day.month, day.day, last_hour)
    out = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r["event"] not in FLAG_EVENTS:
                continue
            try:
                on = local_naive(r["onset"] or r["sent"])
                end = local_naive(r["ends"] or r["expires"])
            except ValueError:
                continue
            if on < hi and end > lo:
                key = (r["event"], end)
                out[key] = (r["event"], on, end)
    return sorted(out.values(), key=lambda a: a[1])


def flag_text(day, path=None):
    """One short line naming the day's flag-worthy NWS alerts, or "".

    e.g. "NWS High Surf Warning (until Sun 4 AM): expect fewer surfers than
    forecast". The forecast model does not see these (2026-11-14); this only
    warns the reader.
    """
    flags = flagged_alerts(day, path)
    if not flags:
        return ""
    parts = []
    for event, _on, end in flags:
        until = end.strftime("%a %-I %p") if end.date() != day else end.strftime("%-I %p")
        parts.append(f"{event} (until {until})")
    return "NWS " + "; ".join(parts) + ": expect fewer surfers than forecast"


def local_naive(ts):
    """ISO time with an offset or a Z -> naive Pacific local time."""
    import pytz
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(pytz.timezone("America/Los_Angeles")).replace(tzinfo=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", help="also add the IEM archive from this date")
    args = ap.parse_args()
    rows = load()
    now = datetime.now().isoformat(timespec="seconds")
    new = fetch_active()
    if args.backfill:
        new += fetch_archive(args.backfill)
    added = 0
    for a in new:
        old = rows.get(a["alert_id"])
        a["first_seen"] = old["first_seen"] if old else now
        a["last_seen"] = now
        added += old is None
        rows[a["alert_id"]] = {**(old or {}), **a}
    save(rows)
    tomorrow = date.today() + timedelta(days=1)
    flags = flagged_alerts(tomorrow)
    print(f"{len(new)} alert(s) fetched, {added} new; {len(rows)} on file -> {OUT_CSV}")
    print(f"flag-worthy for {tomorrow}: " + (", ".join(f"{e} until {end:%a %-I %p}" for e, _o, end in flags)
                                              or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
