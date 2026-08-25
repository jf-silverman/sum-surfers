"""
detect_surfers.py
-----------------
Runs tiled YOLOv8 inference on cropped surf frames and appends results to
predictions.csv. Designed to be called from local_pipeline.sh on a schedule.

Tiling mirrors the training setup:
  - Input crop: 1280 x 180 px (full ROI width)
  - 4 horizontal tiles of 376 x 180 px with 20 % overlap (75 px)
  - Detections are deduplicated across tile boundaries via NMS on merged boxes
"""

import os
import csv
import torch
from pathlib import Path
from datetime import datetime, timedelta

from ultralytics import YOLO

# ---------- CONFIG ----------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model path: override with MODEL_PATH env var, otherwise use the trained best.pt
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(_PROJECT_ROOT / "data" / "model_out" / "20251013" / "train"
            / "runs" / "detect" / "train13" / "weights" / "best.pt"),
    )
)

CROPS_DIR    = _PROJECT_ROOT / "data" / "j_shore_cam" / "surf_crops"
PREDS_CSV    = _PROJECT_ROOT / "data" / "predictions.csv"

# Inference settings
CONF_THRESH  = 0.195   # match validation threshold used in notebook
IOU_NMS      = 0.45    # NMS IoU threshold for cross-tile deduplication

# Tiling parameters (must match training)
TILE_W       = 376
TILE_H       = 180
OVERLAP_PX   = int(TILE_W * 0.20)   # 75 px
STEP_X       = TILE_W - OVERLAP_PX  # 301 px
NUM_TILES    = 4

# Static false-positive exclusion zones, in full-crop ROI pixel coords
# (1280x180). Derived from a 2026-08-30 manual review (see
# data/model_review_50/review_counts.csv) cross-checked against the
# original CVAT test set to confirm neither zone clips real detections.
#   - Tree bough (right edge): consistently isolated, low/mid confidence,
#     always touching the tiling boundary at x=1279 — safe to always drop.
#   - Flag/wind sock (bottom-middle): occupies the same spot real surfers
#     sometimes sit in, so it's only dropped below CONF_THRESH_FLAG_MASK —
#     shrinking the zone alone wasn't enough to separate the two.
TREE_MASK = (1268, 58, 1280, 87)      # x1, y1, x2, y2
FLAG_MASK = (720, 140, 750, 165)
CONF_THRESH_FLAG_MASK = 0.5

# Image-quality gate: skips detection entirely on frames too dark (night/
# dawn/dusk, incl. streetlight-lit fog) or too low-detail (fog, lens
# condensation, or naturally smooth/low-texture water) to reliably count.
# Thresholds fit on 169 hand-labeled images across 4 review batches
# (2026-08 model review — see PROJECT_HISTORY.md); a two-branch rule
# (brightness OR lap_var) outperformed a single linear combination of
# features because the two failure modes pull brightness in opposite
# directions (night = dark, fog = artificially bright). 5-fold CV
# accuracy ~86% on the labeled set — treat as a first-pass estimate, not
# a solved problem, and re-tune as more labeled data comes in.
QUALITY_BRIGHTNESS_THRESH = 75.4   # mean grayscale pixel value; below -> too dark
QUALITY_LAPVAR_THRESH = 12.7       # Laplacian variance (blur/detail); below -> too foggy/blurred
# ----------------------------

CSV_HEADER = [
    "date", "time_local", "filename", "surfer_count", "confidence_avg",
    "quality_ok", "quality_reason", "brightness", "lap_var",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection scope controls:
#   DETECT_MODE=recent (default) -> only process recent crops
#   DETECT_MODE=all             -> process all unprocessed crops
#   DETECT_RECENT_DAYS=7        -> window for recent mode
#   DETECT_START_DATE=YYYY-MM-DD -> optional manual backfill lower bound
DETECT_MODE = os.environ.get("DETECT_MODE", "recent").strip().lower()
DETECT_RECENT_DAYS = int(os.environ.get("DETECT_RECENT_DAYS", "7"))
DETECT_START_DATE = os.environ.get("DETECT_START_DATE", "").strip()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH}  (device={DEVICE})")
    return YOLO(str(MODEL_PATH))


def tile_image_paths(img_path):
    """
    Return a list of (tile_index, x_offset, tile_array) tuples for the
    4-tile horizontal split, written to a temp directory.
    We pass paths to YOLO so we use tmp files.
    """
    import cv2, tempfile, numpy as np

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot open image: {img_path}")

    h, w = img.shape[:2]
    tiles = []
    tmp_dir = Path(tempfile.mkdtemp())
    for i in range(NUM_TILES):
        x0 = i * STEP_X
        x1 = min(x0 + TILE_W, w)
        x0 = max(x1 - TILE_W, 0)   # right-align last tile if short
        tile = img[0:TILE_H, x0:x1]
        tile_path = tmp_dir / f"tile{i}.jpg"
        cv2.imwrite(str(tile_path), tile)
        tiles.append((i, x0, tile_path))
    return tiles, tmp_dir


def nms_across_tiles(all_boxes_global, iou_thresh=IOU_NMS):
    """
    all_boxes_global: list of [x1, y1, x2, y2, conf]  in global (ROI) coords
    Returns list of surviving boxes after NMS.
    """
    if not all_boxes_global:
        return []
    import torch

    boxes  = torch.tensor([b[:4] for b in all_boxes_global], dtype=torch.float32)
    scores = torch.tensor([b[4]  for b in all_boxes_global], dtype=torch.float32)
    from torchvision.ops import nms
    keep = nms(boxes, scores, iou_thresh)
    return [all_boxes_global[i] for i in keep.tolist()]


def _in_zone(box, zone):
    x1, y1, x2, y2, _conf = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def filter_false_positive_zones(boxes):
    """Drop boxes landing in the static tree-bough/flag false-positive zones."""
    kept = []
    for b in boxes:
        if _in_zone(b, TREE_MASK):
            continue
        if _in_zone(b, FLAG_MASK) and b[4] < CONF_THRESH_FLAG_MASK:
            continue
        kept.append(b)
    return kept


def run_inference(model, img_path):
    """
    Tile the image, run inference on each tile, merge boxes into global
    ROI coordinates, apply NMS, and return (surfer_count, confidence_avg).
    """
    import shutil
    tiles, tmp_dir = tile_image_paths(img_path)

    all_boxes = []
    for tile_idx, x_offset, tile_path in tiles:
        results = model.predict(
            source=str(tile_path),
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False,
        )[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            conf = float(box.conf[0].cpu())
            # translate tile-local x coords back to global ROI coords
            all_boxes.append([x1 + x_offset, y1, x2 + x_offset, y2, conf])

    shutil.rmtree(tmp_dir, ignore_errors=True)

    kept = nms_across_tiles(all_boxes)
    kept = filter_false_positive_zones(kept)
    count = len(kept)
    avg_conf = (sum(b[4] for b in kept) / count) if count > 0 else 0.0
    return count, round(avg_conf, 4)


def compute_image_quality(img_path):
    """
    Returns (quality_ok, reason, brightness, lap_var). Cheap (no model
    load), so it's meant to run before detection and skip inference on
    frames that fail — see QUALITY_BRIGHTNESS_THRESH / QUALITY_LAPVAR_THRESH.
    """
    import cv2

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot open image: {img_path}")

    brightness = float(img.mean())
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    if brightness < QUALITY_BRIGHTNESS_THRESH:
        return False, "dark_or_night", brightness, lap_var
    if lap_var < QUALITY_LAPVAR_THRESH:
        return False, "foggy_or_blurred", brightness, lap_var
    return True, "ok", brightness, lap_var


def already_processed(existing_rows, filename):
    return any(r["filename"] == filename for r in existing_rows)


def load_existing(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(csv_path, row):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_crop_datetime(img_path):
    try:
        return datetime.strptime(img_path.stem, "crop%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def in_detection_scope(img_path):
    if DETECT_MODE == "all":
        return True

    dt = parse_crop_datetime(img_path)
    if dt is None:
        return False

    if DETECT_START_DATE:
        try:
            start_date = datetime.strptime(DETECT_START_DATE, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("DETECT_START_DATE must be YYYY-MM-DD")
        return dt.date() >= start_date

    cutoff_date = (datetime.now() - timedelta(days=max(DETECT_RECENT_DAYS - 1, 0))).date()
    return dt.date() >= cutoff_date


def main():
    model = load_model()
    existing = load_existing(PREDS_CSV)
    processed_files = {r["filename"] for r in existing}

    crop_files = sorted(CROPS_DIR.glob("crop*.jpg"))
    scoped_files = [p for p in crop_files if in_detection_scope(p)]
    new_files = [p for p in scoped_files if p.name not in processed_files]

    if not new_files:
        print("No new crop images to process.")
        return

    if DETECT_MODE == "all":
        print(f"Detection scope: all crops. Candidates={len(scoped_files)}")
    elif DETECT_START_DATE:
        print(f"Detection scope: crops from {DETECT_START_DATE} onward. Candidates={len(scoped_files)}")
    else:
        print(f"Detection scope: last {DETECT_RECENT_DAYS} day(s). Candidates={len(scoped_files)}")

    print(f"Processing {len(new_files)} new image(s)...")

    for img_path in new_files:
        # filename format: cropYYYY-MM-DD_HH-MM-SS.jpg
        try:
            dt = parse_crop_datetime(img_path)
            if dt is None:
                raise ValueError("Cannot parse crop timestamp")
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")
        except ValueError:
            date_str = "unknown"
            time_str = "unknown"

        try:
            quality_ok, reason, brightness, lap_var = compute_image_quality(img_path)
        except Exception as e:
            print(f"  ERROR (quality check) {img_path.name}: {e}")
            continue

        if not quality_ok:
            print(f"  {img_path.name}  →  skipped ({reason}, brightness={brightness:.1f}, lap_var={lap_var:.1f})")
            append_row(PREDS_CSV, {
                "date": date_str, "time_local": time_str, "filename": img_path.name,
                "surfer_count": "", "confidence_avg": "",
                "quality_ok": False, "quality_reason": reason,
                "brightness": round(brightness, 1), "lap_var": round(lap_var, 1),
            })
            continue

        try:
            count, avg_conf = run_inference(model, img_path)
            print(f"  {img_path.name}  →  {count} surfer(s)  (conf avg={avg_conf})")
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            count, avg_conf = -1, 0.0

        append_row(PREDS_CSV, {
            "date":          date_str,
            "time_local":    time_str,
            "filename":      img_path.name,
            "surfer_count":  count,
            "confidence_avg": avg_conf,
            "quality_ok": True, "quality_reason": "ok",
            "brightness": round(brightness, 1), "lap_var": round(lap_var, 1),
        })

    print(f"\nDone. Results appended to {PREDS_CSV}")


if __name__ == "__main__":
    main()
