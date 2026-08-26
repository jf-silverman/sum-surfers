"""
backfill_multiframe_counts.py
------------------------------
One-off backfill (not part of the scheduled pipeline): for predictions.csv
rows whose raw clip.mp4 is still on disk, extracts the two side frames
(same convention as get_cropped_frame.py — FRAME_TIME_SEC +/- SIDE_FRAME_
OFFSET_SEC) and runs multi-frame detection, populating frame_count_1/2/3,
frame_count_mean, frame_count_stdev.

Non-destructive by design: the existing surfer_count/confidence_avg values
are left untouched (they've already been referenced by manual review work
elsewhere in this project) — this only fills in the new frame_count_*
columns as additional data.

Only processes rows where:
  - quality_ok is True (no point multi-frame-detecting a frame already
    flagged as too dark/foggy to trust)
  - frame_count_mean is not already populated (safe to re-run / resume)
  - the corresponding clip.mp4 still exists on disk

This is a local, compute-bound job (no network calls, no rate-limit
concerns) — runs as fast as the CPU allows. For a large date range this
can take a while (~3x normal detection time per image, since it runs
inference on 3 frames instead of 1); consider --start/--end to work
through it in chunks, same spirit as backfill_historical_predictors.py.

Usage:
    python code/backfill_multiframe_counts.py --start 2025-10-11 --end 2025-10-20
    python code/backfill_multiframe_counts.py --start 2025-10-11 --dry-run
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402
import get_cropped_frame as gcf  # noqa: E402

CLIPS_DIR = gcf.INPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="End date, YYYY-MM-DD (inclusive). Defaults to --start.")
    p.add_argument("--dry-run", action="store_true", help="Show what would be processed without running detection")
    return p.parse_args()


def clip_path_for(date_str, time_local):
    """clips are stored as data/.../surf_clips/YYYY-MM-DD/HH_MM/clip.mp4"""
    hh_mm = time_local.replace(":", "_")
    return CLIPS_DIR / date_str / hh_mm / "clip.mp4"


def load_rows():
    with open(ds.PREDS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(ds.PREDS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ds.CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ds.CSV_HEADER})


def main():
    args = parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else start_date
    if end_date < start_date:
        print("ERROR: --end is before --start")
        return

    rows = load_rows()

    targets = []
    for i, r in enumerate(rows):
        row_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        if not (start_date <= row_date <= end_date):
            continue
        if r.get("quality_ok") != "True":
            continue
        if r.get("frame_count_mean", "").strip() != "":
            continue
        clip_path = clip_path_for(r["date"], r["time_local"])
        if not clip_path.exists():
            continue
        targets.append((i, r, clip_path))

    print(f"Date range: {start_date} to {end_date}")
    print(f"Rows needing multi-frame backfill (quality_ok, no existing frame_count_mean, clip on disk): {len(targets)}")

    if not targets:
        return
    if args.dry_run:
        print("--dry-run: stopping before any processing.")
        for i, r, clip_path in targets[:20]:
            print(f"  {r['filename']}  <- {clip_path}")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more")
        return

    model = ds.load_model()

    updated = 0
    for i, r, clip_path in targets:
        primary_path = ds.CROPS_DIR / r["filename"]
        side1_path, side2_path = ds.side_frame_paths(primary_path)

        try:
            if not side1_path.exists():
                gcf.extract_frame_at(clip_path, gcf.FRAME_TIME_SEC - gcf.SIDE_FRAME_OFFSET_SEC, side1_path)
            if not side2_path.exists():
                gcf.extract_frame_at(clip_path, gcf.FRAME_TIME_SEC + gcf.SIDE_FRAME_OFFSET_SEC, side2_path)

            result = ds.run_inference_multi(model, primary_path)
        except Exception as e:
            print(f"  ERROR {r['filename']}: {e}")
            continue

        rows[i]["frame_count_1"] = result["frame_count_1"]
        rows[i]["frame_count_2"] = result["frame_count_2"]
        rows[i]["frame_count_3"] = result["frame_count_3"]
        rows[i]["frame_count_mean"] = result["frame_count_mean"]
        rows[i]["frame_count_stdev"] = result["frame_count_stdev"]
        updated += 1

        print(f"  [{updated}/{len(targets)}] {r['filename']}  "
              f"orig_surfer_count={r['surfer_count']}  "
              f"frames={result['frame_count_1']},{result['frame_count_2']},{result['frame_count_3']}  "
              f"mean={result['frame_count_mean']} stdev={result['frame_count_stdev']}")

        # Write progress incrementally so a long run can be interrupted
        # without losing completed work.
        if updated % 25 == 0:
            write_rows(rows)

    write_rows(rows)
    print(f"\nDone. Backfilled frame_count_* for {updated} row(s). surfer_count/confidence_avg left untouched.")


if __name__ == "__main__":
    main()
