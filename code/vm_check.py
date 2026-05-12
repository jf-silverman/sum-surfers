#!/usr/bin/env python3
"""
vm_check.py — VM-side pre-detection helper.

Prints two space-separated integers to stdout:
  <new_crop_count> <days_since_last_local_run>

  new_crop_count         : crops in CROPS_DIR not yet in predictions.csv
  days_since_last_local_run : calendar days since .last_local_run was written
                             (-1 if the file does not exist)

Exit codes:
  0 — new crops found; caller should run detection
  1 — no new crops; caller should consider sending a reminder email
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = _PROJECT_ROOT / "data" / "j_shore_cam" / "surf_crops"
PREDS_CSV = _PROJECT_ROOT / "data" / "predictions.csv"
LAST_LOCAL_RUN = _PROJECT_ROOT / "data" / ".last_local_run"


def count_new_crops() -> int:
    processed: set[str] = set()
    if PREDS_CSV.exists():
        with open(PREDS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename", "").strip()
                if fn:
                    processed.add(fn)

    if not CROPS_DIR.exists():
        return 0

    new = [p for p in CROPS_DIR.rglob("*.jpg") if p.name not in processed]
    return len(new)


def days_since_last_local_run() -> int:
    if not LAST_LOCAL_RUN.exists():
        return -1
    raw = LAST_LOCAL_RUN.read_text().strip().rstrip("Z")
    try:
        last = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - last).days)
    except ValueError:
        return -1


if __name__ == "__main__":
    new = count_new_crops()
    days = days_since_last_local_run()
    print(f"{new} {days}")
    sys.exit(0 if new > 0 else 1)
