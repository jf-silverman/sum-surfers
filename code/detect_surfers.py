"""
detect_surfers.py
-----------------
Runs tiled YOLOv8 inference on cropped surf frames and appends results to
predictions.csv.  Designed to be called from run_pipeline.sh daily.

Tiling mirrors the training setup:
  - Input crop: 1280 x 180 px (full ROI width)
  - 4 horizontal tiles of 376 x 180 px with 20 % overlap (75 px)
  - Detections are deduplicated across tile boundaries via NMS on merged boxes
"""

import os
import csv
import torch
from pathlib import Path
from datetime import datetime

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
# ----------------------------

CSV_HEADER = ["date", "time_local", "filename", "surfer_count", "confidence_avg"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    count = len(kept)
    avg_conf = (sum(b[4] for b in kept) / count) if count > 0 else 0.0
    return count, round(avg_conf, 4)


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


def main():
    model = load_model()
    existing = load_existing(PREDS_CSV)
    processed_files = {r["filename"] for r in existing}

    crop_files = sorted(CROPS_DIR.glob("crop*.jpg"))
    new_files  = [p for p in crop_files if p.name not in processed_files]

    if not new_files:
        print("No new crop images to process.")
        return

    print(f"Processing {len(new_files)} new image(s)...")

    for img_path in new_files:
        # filename format: cropYYYY-MM-DD_HH-MM-SS.jpg
        try:
            dt = datetime.strptime(img_path.stem, "crop%Y-%m-%d_%H-%M-%S")
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")
        except ValueError:
            date_str = "unknown"
            time_str = "unknown"

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
        })

    print(f"\nDone. Results appended to {PREDS_CSV}")


if __name__ == "__main__":
    main()
