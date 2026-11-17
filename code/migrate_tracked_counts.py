"""
Adds the `frame_count_tracked` column to predictions.csv (2026-11-16).

detect_surfers.py now writes, beside the median/max hybrid, the number of
surfers present at 2.5 s across a clip's 7 frames (track_frames.py). Existing
rows get it from analysis/occlusion_tracking/tracked_hourly.csv, computed with
the same code and the current detector on every hourly clip whose 7 crops were
on disk; rows without all 7 crops stay blank. surfer_count is not touched.

Writes a dated backup first. Dry run by default.

Usage:
    python code/migrate_tracked_counts.py            # report only
    python code/migrate_tracked_counts.py --apply
"""

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACKED = ROOT / "analysis" / "occlusion_tracking" / "tracked_hourly.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    p = pd.read_csv(ds.PREDS_CSV, dtype=str, keep_default_na=False)
    if "frame_count_tracked" in p.columns:
        sys.exit("predictions.csv already has frame_count_tracked.")
    t = pd.read_csv(TRACKED, dtype=str)[["filename", "present25"]]
    p = p.merge(t.rename(columns={"present25": "frame_count_tracked"}), on="filename", how="left")
    p["frame_count_tracked"] = p["frame_count_tracked"].fillna("")
    p = p[ds.CSV_HEADER]
    filled = (p["frame_count_tracked"] != "").sum()
    print(f"{len(p)} rows; frame_count_tracked filled for {filled}, blank for {len(p) - filled}")
    if not args.apply:
        print("dry run; pass --apply to write")
        return
    backup = ds.PREDS_CSV.with_name(f"predictions.csv.bak-{date.today().isoformat()}")
    shutil.copy(ds.PREDS_CSV, backup)
    p.to_csv(ds.PREDS_CSV, index=False)
    print(f"written; backup at {backup.name}")


if __name__ == "__main__":
    main()
