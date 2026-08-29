"""
analyze_hourly_variability.py
------------------------------
One-off analysis script (not part of the scheduled pipeline). Uses the 12
back-to-back 5-minute clips from data/not_needed_in_repo/hourly_variability/
(downloaded via pull_hourly_variability_clips.py) to see how much the surfer
count varies across a single hour.

For each 5-minute clip: extracts 5 ROI-cropped frames spread evenly across
the clip's ~300s span (seconds 0, 75, 150, 225, 299), runs the production
image-quality gate and the production tiling/NMS/false-positive-filtered
detector on each frame that passes, and averages the surviving frames to get
one count estimate per 5-minute window. This spread (5 frames across the
full clip) follows the finding in analyze_frame_timing.py/docs/PROJECT_HISTORY.md
that spacing frames apart reduces detector noise far more than bunching them
close together.

Reuses ROI_X/Y/W/H from get_cropped_frame.py and load_model()/
compute_image_quality()/run_inference_with_boxes() from detect_surfers.py --
does not reimplement any of that logic.

Usage:
    python analysis/hourly_variability_8to9am/analyze_hourly_variability.py [--date YYYY-MM-DD]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code"))
import detect_surfers as ds  # noqa: E402
from get_cropped_frame import ROI_X, ROI_Y, ROI_W, ROI_H  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
BASE_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "hourly_variability"
OUT_FRAMES_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "hourly_variability_frames"
OUT_CSV = HERE / "hourly_variability_analysis.csv"

SAMPLE_SECONDS = [0, 75, 150, 225, 299]  # 5 frames spread across a ~300s clip

CSV_HEADER = ["date", "clip_time", "second", "quality_ok", "quality_reason",
              "brightness", "lap_var", "count"]


def extract_at_seconds(video_path, out_dir, target_seconds):
    """Open the clip once, read sequentially, and save one ROI-cropped frame
    at each requested integer second. Sequential-read only -- cap.set()
    seeking is unreliable on these clips (see get_cropped_frame.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot get FPS for {video_path}")

    wanted = sorted(target_seconds)
    saved = []
    idx_wanted = 0
    frame_idx = 0
    while idx_wanted < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t >= wanted[idx_wanted]:
            crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            out_path = out_dir / f"sec_{wanted[idx_wanted]:03d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved.append((wanted[idx_wanted], out_path))
            idx_wanted += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-08-28")
    args = parser.parse_args()

    day_dir = BASE_DIR / args.date
    clip_times = sorted(p.name for p in day_dir.iterdir() if p.is_dir())
    if not clip_times:
        raise SystemExit(f"No clip folders found under {day_dir}")

    model = ds.load_model()
    rows = []

    for clip_time in clip_times:
        clip_path = day_dir / clip_time / "clip.mp4"
        if not clip_path.exists():
            print(f"WARNING: missing clip {clip_path}, skipping")
            continue

        out_dir = OUT_FRAMES_DIR / args.date / clip_time
        frames = extract_at_seconds(clip_path, out_dir, SAMPLE_SECONDS)
        print(f"{clip_time}: extracted {len(frames)} frames")

        for second, img_path in frames:
            try:
                quality_ok, reason, brightness, lap_var = ds.compute_image_quality(img_path)
            except Exception as e:
                print(f"  ERROR (quality) {clip_time} sec={second}: {e}")
                continue

            if not quality_ok:
                rows.append({
                    "date": args.date, "clip_time": clip_time, "second": second,
                    "quality_ok": False, "quality_reason": reason,
                    "brightness": round(brightness, 1), "lap_var": round(lap_var, 1),
                    "count": "",
                })
                continue

            try:
                boxes = ds.run_inference_with_boxes(model, img_path)
            except Exception as e:
                print(f"  ERROR (inference) {clip_time} sec={second}: {e}")
                continue

            rows.append({
                "date": args.date, "clip_time": clip_time, "second": second,
                "quality_ok": True, "quality_reason": "ok",
                "brightness": round(brightness, 1), "lap_var": round(lap_var, 1),
                "count": len(boxes),
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # Per-clip summary
    from collections import defaultdict
    by_clip = defaultdict(list)
    for r in rows:
        if r["quality_ok"] and r["count"] != "":
            by_clip[r["clip_time"]].append(r["count"])

    print("\nPer-5-minute-window mean count (n frames used):")
    for ct in clip_times:
        vals = by_clip.get(ct, [])
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {ct}: mean={mean:.1f}  n={len(vals)}  raw={vals}")
        else:
            print(f"  {ct}: no quality_ok frames")


if __name__ == "__main__":
    main()
