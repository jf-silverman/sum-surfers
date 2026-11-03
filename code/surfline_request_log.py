"""
surfline_request_log.py
-----------------------
One CSV row per request to Surfline, so Cloudflare blocks can be traced to a
cause rather than guessed at (added 2026-11-02).

Until now only failures reached the logs, as "403 (bot check)" lines with no
record of what else was sent that night. Three heavy nights in a row
(2026-10-30 to 11-01, 6 to 28 refusals each, after one or two a month) came
right after the between-hours clips (2026-10-26) roughly doubled the clip
requests, and overlapped an expired-token night, and nothing could tell those
apart. With every request logged, a block can be lined up against the request
rate, the endpoint, the step and the time.

Written: `data/logs/surfline_requests.csv` (gitignored), columns

    time_local     when the response came back, local clock, to the second
    script         the running script (get_clips.py, get_subhourly_frames.py, ...)
    kind           forecast / clip_json / clip_mp4
    endpoint       the URL path only. Query strings are never written: the clip
                   API carries the account token in its query.
    status         HTTP status, or "error" for no response (network, timeout)
    cf_mitigated   Cloudflare's cf-mitigated header ("challenge" on a bot check)
    cf_ray         Cloudflare's request id, for matching against anything they report
    elapsed_ms     request time
    attempt        retry number within the caller (1 = first try)

Logging must never break a request: every failure here is swallowed.
"""

import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

LOG_CSV = Path(__file__).resolve().parent.parent / "data" / "logs" / "surfline_requests.csv"
HEADER = ["time_local", "script", "kind", "endpoint", "status", "cf_mitigated",
          "cf_ray", "elapsed_ms", "attempt"]


def endpoint_of(url):
    """The host and path of a URL, never its query string."""
    parts = urlsplit(url or "")
    return f"{parts.netloc}{parts.path}"


def record(kind, url, response=None, attempt=1, path=None):
    """Append one row. `response` is a requests.Response, or None if none came back."""
    try:
        target = Path(path) if path else LOG_CSV
        target.parent.mkdir(parents=True, exist_ok=True)
        new = not target.exists()
        headers = getattr(response, "headers", {}) or {}
        elapsed = getattr(response, "elapsed", None)
        row = {
            "time_local": datetime.now().isoformat(timespec="seconds"),
            "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "",
            "kind": kind,
            "endpoint": endpoint_of(getattr(response, "url", None) or url),
            "status": getattr(response, "status_code", "error") if response is not None else "error",
            "cf_mitigated": headers.get("cf-mitigated", ""),
            "cf_ray": headers.get("cf-ray", ""),
            "elapsed_ms": int(elapsed.total_seconds() * 1000) if elapsed is not None else "",
            "attempt": attempt,
        }
        with open(target, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def blocks_on(day=None, path=None):
    """Summary of one day's refusals, for the evening email.

    Returns None when there were none (or no log), else a dict:
    {"refused": n, "requests": n, "first": "HH:MM:SS", "last": "HH:MM:SS",
     "by_kind": {kind: n}, "by_script": {script: n}}
    """
    target = Path(path) if path else LOG_CSV
    day = (day or date.today()).isoformat()
    if not target.exists():
        return None
    try:
        with open(target, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["time_local"].startswith(day)]
    except Exception:
        return None
    refused = [r for r in rows if r["status"] == "403"]
    if not refused:
        return None
    by_kind, by_script = {}, {}
    for r in refused:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        by_script[r["script"]] = by_script.get(r["script"], 0) + 1
    return {"refused": len(refused), "requests": len(rows),
            "first": refused[0]["time_local"][11:], "last": refused[-1]["time_local"][11:],
            "by_kind": by_kind, "by_script": by_script}
