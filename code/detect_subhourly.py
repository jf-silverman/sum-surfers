"""
Counts the sub-hourly frame sets (pipeline step 4b, added 2026-10-27).

get_subhourly_frames.py (step 2b) saves two extra sets per hour, ten crops
each, one every 6 s across a one-minute clip taken 18 and 36 minutes after the
main clip. This runs the production detector over them and writes one row per
set to data/predictions/subhourly_counts.csv:

  date, time_local      the set's clip start
  quality_ok, quality_reason
                        detect_surfers.compute_image_quality on the middle
                        frame (f05); a set that fails is recorded, not counted
  n_frames              frames counted (a set is only counted when all 10 exist)
  count_f00 ... count_f09
  count_median, count_max
  set_count             the hourly count's rule (detect_surfers.hybrid_count):
                        the median of the ten below 6, their max from 6 up

set_count is what the evening email plots between the hourly counts. It was
the median of the ten until 2026-10-28, when Joel switched it to the hourly
rule so both kinds of point are read the same way: on a good day surfers ride
out of frame behind the cliff during sets, so a single look runs low, and the
max catches the lineup between sets. On the first five days (2026-10-22 to
10-26, 124 sets) the max of the ten ran 8% above the interpolated hourly
counts (15+ surfers) and the median 1.5% below; a max over ten frames across a
minute picks up more than a max over five frames in four seconds. Neither is
checked against hand counts yet. count_median and count_max stay in the file.

Nothing in training or forecasting reads this file.

Scope is the same as detect_surfers.py (DETECT_MODE / DETECT_RECENT_DAYS /
DETECT_START_DATE). Sets already in the CSV are skipped.

Usage:
    python code/detect_subhourly.py
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "j_shore_cam" / "subhourly_crops"
OUT_CSV = ROOT / "data" / "predictions" / "subhourly_counts.csv"
N_FRAMES = 10
HEADER = (["date", "time_local", "quality_ok", "quality_reason", "n_frames"]
          + [f"count_f{i:02d}" for i in range(N_FRAMES)]
          + ["count_median", "count_max", "set_count"])
SET_RE = re.compile(r"^sub(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})_f(\d{2})$")


def in_scope(day):
    """Same window as detect_surfers.in_detection_scope, by date."""
    if ds.DETECT_MODE == "all":
        return True
    if ds.DETECT_START_DATE:
        return day >= datetime.strptime(ds.DETECT_START_DATE, "%Y-%m-%d").date()
    return (datetime.now().date() - day).days <= ds.DETECT_RECENT_DAYS


def find_sets():
    """{(date, 'HH:MM'): [10 paths in frame order]} for complete sets in scope."""
    sets = {}
    for p in CROPS.glob("*/sub*_f*.jpg"):
        m = SET_RE.match(p.stem)
        if not m:
            continue
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if in_scope(day):
            sets.setdefault((m.group(1), f"{m.group(2)}:{m.group(3)}"), {})[int(m.group(4))] = p
    return {k: [v[i] for i in range(N_FRAMES)] for k, v in sets.items()
            if all(i in v for i in range(N_FRAMES))}


def done_keys():
    if not OUT_CSV.exists():
        return set()
    with open(OUT_CSV, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != HEADER:
            sys.exit(f"{OUT_CSV} header does not match; refusing to append.")
        return {(row["date"], row["time_local"]) for row in r}


def main():
    todo = sorted(find_sets().items())
    done = done_keys()
    todo = [(k, v) for k, v in todo if k not in done]
    if not todo:
        print("No new sub-hourly sets to count.")
        return
    print(f"Counting {len(todo)} sub-hourly set(s)...")
    model = ds.load_model()
    new_file = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        for (day, hhmm), paths in todo:
            ok, reason, _b, _l = ds.compute_image_quality(paths[N_FRAMES // 2])
            if not ok:
                w.writerow([day, hhmm, False, reason, 0] + [""] * (N_FRAMES + 3))
                print(f"  {day} {hhmm}  ->  skipped ({reason})")
                continue
            counts = [len(ds.run_inference_with_boxes(model, p)) for p in paths]
            med = statistics.median(counts)
            set_count = ds.hybrid_count(counts)
            w.writerow([day, hhmm, True, "ok", N_FRAMES] + counts
                       + [med, max(counts), set_count])
            f.flush()
            print(f"  {day} {hhmm}  ->  {set_count}  (frames {counts})")
    print(f"Done. Results appended to {OUT_CSV}")


if __name__ == "__main__":
    main()
