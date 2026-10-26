"""
One-time migration to the median/max hybrid count (2026-10-26).

Adds the five-frame columns to data/predictions/predictions.csv
(frame_count_t05/t15/t35/t45, frame_count_hybrid), fills them for past hours
from analysis/packed_hours/extra_frame_counts.csv (the four one-second frames
counted with the production detector; the 2.5 s frame is frame_count_2), and
sets surfer_count to the hybrid on those rows so history and new rows use one
scale. frame_count_mean is untouched, so the old count stays recoverable as
round(frame_count_mean). Rows without all five frames keep their count.

detect_surfers.py refuses to append to a predictions.csv without the new
columns, so this has to run once before the first nightly run on the new code.

Dry run by default: prints what would change. --apply writes, after copying
predictions.csv to predictions.csv.bak-<date>, and removes the chart's model
cache so the next run refits on the new counts.

Usage:
    python code/migrate_hybrid_counts.py            # dry run
    python code/migrate_hybrid_counts.py --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date
from pathlib import Path

import detect_surfers as ds

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "data/predictions/predictions.csv"
EXTRA = ROOT / "analysis/packed_hours/extra_frame_counts.csv"
CACHE = ROOT / "data/model_cache/chart_models.joblib"
SRC = {"t05": "f05", "t15": "f15", "t35": "f35", "t45": "f45"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--apply", action="store_true")
    args = a.parse_args()

    with open(PREDS, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(EXTRA, newline="") as f:
        extra = {r["filename"]: r for r in csv.DictReader(f)}

    changed, same, filled = 0, 0, 0
    for r in rows:
        for s in ds.SECOND_FRAME_SUFFIXES:
            r.setdefault(f"frame_count_{s}", "")
        r.setdefault("frame_count_hybrid", "")
        e = extra.get(r["filename"])
        if (e is None or (r.get("frame_count_2") or "") == ""
                or any(e[v] == "" for v in SRC.values()) or r["frame_count_hybrid"] != ""):
            continue
        sec = [int(float(e[SRC[s]])) for s in ds.SECOND_FRAME_SUFFIXES]
        for s, c in zip(ds.SECOND_FRAME_SUFFIXES, sec):
            r[f"frame_count_{s}"] = c
        h = ds.hybrid_count(sec[:2] + [int(float(r["frame_count_2"]))] + sec[2:])
        r["frame_count_hybrid"] = h
        filled += 1
        if str(r["surfer_count"]).strip() != "" and int(float(r["surfer_count"])) == h:
            same += 1
        else:
            changed += 1
        r["surfer_count"] = h

    print(f"{len(rows)} rows; hybrid filled for {filled} ({changed} counts change, {same} unchanged)")
    if not args.apply:
        print("Dry run; nothing written. Re-run with --apply.")
        return
    bak = PREDS.with_name(f"{PREDS.name}.bak-{date.today().isoformat()}")
    shutil.copy2(PREDS, bak)
    with open(PREDS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ds.CSV_HEADER, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {PREDS} (backup at {bak.name})")
    if CACHE.exists():
        CACHE.unlink()
        print(f"Removed {CACHE.name}; the next chart run refits on the new counts")


if __name__ == "__main__":
    main()
