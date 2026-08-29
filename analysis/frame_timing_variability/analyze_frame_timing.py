"""
analyze_frame_timing.py
------------------------
One-off analysis script (not part of the scheduled pipeline). Studies how
much the detector's per-frame surfer count varies over time within a single
clip, to inform how many frames to average and how far apart to space them
(production currently uses 3 frames spaced ~1.5-3s apart in
get_cropped_frame.py/detect_surfers.py's run_inference_multi).

Uses the 14 real ~60-second clips collected on 2026-08-27 via a one-off
CLIP_DURATION_OVERRIDE_FILE run (get_clips.py's usual default is 5s), one per
daylight hour. For each clip: extracts one ROI-cropped frame per integer
second, runs the production image-quality gate and the production tiling/
NMS/false-positive-filtered detector on every frame that passes, and writes
one row per (clip, second) to frame_variability_analysis.csv (same folder).

Reuses ROI_X/Y/W/H from get_cropped_frame.py and load_model()/
compute_image_quality()/run_inference_with_boxes() from detect_surfers.py
directly -- does not reimplement any of that logic.

Usage:
    python analysis/frame_timing_variability/analyze_frame_timing.py
"""

import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code"))
import detect_surfers as ds  # noqa: E402
from get_cropped_frame import ROI_X, ROI_Y, ROI_W, ROI_H  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
CLIPS_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "surf_clips" / "2026-08-27"
OUT_FRAMES_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "frame_variability_analysis"
OUT_CSV = HERE / "frame_variability_analysis.csv"

CLIP_TIMES = [
    "05_50", "06_44", "07_38", "08_32", "09_26", "10_20", "11_14",
    "12_08", "13_02", "13_56", "14_50", "15_44", "16_38", "18_26",
]

CSV_HEADER = ["date", "clip_time", "second", "quality_ok", "quality_reason",
              "brightness", "lap_var", "count"]


def extract_seconds(video_path, out_dir):
    """Open the clip once, read sequentially, and save one ROI-cropped frame
    per integer second (0..floor(duration)-1). Returns list of (second, path).
    Sequential-read only -- cap.set(CAP_PROP_POS_FRAMES) is unreliable on
    these clips (see get_cropped_frame.py's extract_frame_at comment)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot get FPS for {video_path}")

    saved = []
    next_second = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t >= next_second:
            crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            out_path = out_dir / f"sec_{next_second:02d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved.append((next_second, out_path))
            next_second += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    model = ds.load_model()
    rows = []
    n_ok = 0
    n_skip = 0
    skip_reasons = {}

    for clip_time in CLIP_TIMES:
        clip_path = CLIPS_DIR / clip_time / "clip.mp4"
        if not clip_path.exists():
            print(f"WARNING: missing clip {clip_path}, skipping")
            continue

        out_dir = OUT_FRAMES_DIR / clip_time
        frames = extract_seconds(clip_path, out_dir)
        print(f"{clip_time}: extracted {len(frames)} frames")

        for second, img_path in frames:
            try:
                quality_ok, reason, brightness, lap_var = ds.compute_image_quality(img_path)
            except Exception as e:
                print(f"  ERROR (quality) {clip_time} sec={second}: {e}")
                continue

            if not quality_ok:
                n_skip += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                rows.append({
                    "date": "2026-08-27", "clip_time": clip_time, "second": second,
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

            n_ok += 1
            rows.append({
                "date": "2026-08-27", "clip_time": clip_time, "second": second,
                "quality_ok": True, "quality_reason": "ok",
                "brightness": round(brightness, 1), "lap_var": round(lap_var, 1),
                "count": len(boxes),
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal rows: {len(rows)}  quality_ok: {n_ok}  skipped: {n_skip}")
    if skip_reasons:
        print(f"Skip reasons: {skip_reasons}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
