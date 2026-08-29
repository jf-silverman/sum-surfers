"""
analyze_hourly_variability_full.py
------------------------------------
One-off analysis script (not part of the scheduled pipeline). Full-density
version of analyze_hourly_variability.py: instead of sampling 5 frames per
5-minute clip, extracts 1 ROI-cropped frame per second across all 12
back-to-back clips in data/not_needed_in_repo/hourly_variability/<date>/
(downloaded via pull_hourly_variability_clips.py) — 12 clips x ~300s each,
about 3600 frames total ("60 x 60": 60 minutes x 60 frames/minute).

Runs the production image-quality gate and the production tiling/NMS/
false-positive-filtered detector on every frame, so poor-quality seconds
are flagged/skipped exactly like the real pipeline would. Reuses ROI_X/Y/W/H
from get_cropped_frame.py and load_model()/compute_image_quality()/
run_inference_with_boxes() from detect_surfers.py.

Usage:
    python analysis/hourly_variability_8to9am/analyze_hourly_variability_full.py [--date YYYY-MM-DD]
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
OUT_FRAMES_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "hourly_variability_frames_full"
OUT_CSV = HERE / "hourly_variability_full.csv"

CSV_HEADER = ["date", "clip_time", "second", "quality_ok", "quality_reason",
              "brightness", "lap_var", "count"]


def extract_all_seconds(video_path, out_dir):
    """Open the clip once, read sequentially, and save one ROI-cropped frame
    per integer second across the whole clip. Sequential-read only --
    cap.set() seeking is unreliable on these clips (see get_cropped_frame.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot get FPS for {video_path}")

    saved = []
    frame_idx = 0
    next_second = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t >= next_second:
            crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            out_path = out_dir / f"sec_{next_second:03d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved.append((next_second, out_path))
            next_second += 1
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

    for ci, clip_time in enumerate(clip_times, 1):
        clip_path = day_dir / clip_time / "clip.mp4"
        if not clip_path.exists():
            print(f"WARNING: missing clip {clip_path}, skipping")
            continue

        out_dir = OUT_FRAMES_DIR / args.date / clip_time
        frames = extract_all_seconds(clip_path, out_dir)
        print(f"[{ci}/{len(clip_times)}] {clip_time}: extracted {len(frames)} frames, running detection...")

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

        # Flush progress after every clip so a long run can be checked on partway through.
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

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
            print(f"  {ct}: mean={mean:.1f}  n={len(vals)}  min={min(vals)}  max={max(vals)}")
        else:
            print(f"  {ct}: no quality_ok frames")


if __name__ == "__main__":
    main()
