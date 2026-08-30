"""
train_model.py
-----------------
Rebuilds the surfer-detection YOLOv8 model from a CVAT COCO annotation
export: tiles images/annotations to match production inference tiling,
converts to YOLO format, fine-tunes from a COCO-pretrained checkpoint, and
validates the result on the held-out val split.

Replaces the old exploratory detect_surfers_v2.ipynb notebook workflow
(early data/model exploration) with a script meant to be rerun whenever
there's a new or larger CVAT export to train on -- the notebook is kept
for its historical training log/plots, not as the live retraining path.

Tile dimensions (TILE_W/TILE_H/OVERLAP_PX/STEP_X/NUM_TILES) are imported
directly from detect_surfers.py rather than duplicated, so training tiling
can never silently drift out of sync with production inference tiling.

Each run writes to FRESH, dated output folders rather than overwriting the
current production model/training data -- nothing production-facing
changes until you manually point detect_surfers.py's MODEL_PATH (or the
MODEL_PATH env var) at the new weights, after reviewing the printed
validation metrics:
    data/cvat_out_coco/splits_tiled_<run-name>/
    data/cvat_out_yolo_rebuilt_<run-name>/
    data/model_out/<run-name>/train/weights/best.pt

Usage:
    # Retrain from the current CVAT export (data/cvat_out_coco/splits/):
    python code/train_model.py

    # Retrain from a new/larger CVAT export, with a custom run name:
    python code/train_model.py --cvat-coco-dir data/cvat_out_coco/splits_2026_11 --run-name 2026_11_retrain --epochs 80
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_surfers import TILE_W, TILE_H, STEP_X, NUM_TILES  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"

# The checkpoint the current production model (data/model_out/20251013/)
# was fine-tuned from -- see docs/PROJECT_FILES.md's model_out/20251013/
# entry (confirmed present, 22.5MB, 2026-08-29). If this file is ever
# missing, ultralytics auto-downloads a fresh COCO-pretrained yolov8s.pt.
DEFAULT_BASE_CHECKPOINT = DATA_DIR / "model_out" / "20251013" / "train" / "yolov8s.pt"

SPLITS = ["train", "val", "test"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cvat-coco-dir", type=Path, default=DATA_DIR / "cvat_out_coco" / "splits",
                    help="CVAT COCO export root: expects instances_{train,val,test}.json "
                         "+ {train,val,test}/ image folders (default: the current export)")
    p.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d"),
                    help="Label for this run's output folders (default: today's date, YYYYMMDD)")
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT,
                    help="COCO-pretrained YOLOv8 checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps", help="'mps' (Apple Silicon), '0' (CUDA GPU), or 'cpu'")
    p.add_argument("--degrees", type=float, default=10)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--force", action="store_true",
                    help="Overwrite this run's output folders if they already exist "
                         "(default: refuse, to avoid silently clobbering a previous run)")
    return p.parse_args()


def tile_coco_split(cvat_coco_dir, out_dir):
    """Tiles each image into NUM_TILES horizontal TILE_W x TILE_H tiles
    (matching production inference exactly, via detect_surfers.py's own
    constants) and re-projects COCO annotations onto each tile, cropping/
    dropping boxes that fall outside it. Same logic as
    detect_surfers_v2.ipynb's tiling cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split in SPLITS:
        in_json = cvat_coco_dir / f"instances_{split}.json"
        if not in_json.exists():
            raise FileNotFoundError(
                f"Missing {in_json} -- --cvat-coco-dir must point at a CVAT COCO export root "
                f"(expects instances_{{train,val,test}}.json + {{train,val,test}}/ image folders)"
            )
        with open(in_json) as f:
            coco = json.load(f)

        img_dir = cvat_coco_dir / split
        out_img_dir = out_dir / split
        out_img_dir.mkdir(parents=True, exist_ok=True)

        new_coco = {k: coco[k] for k in ["info", "licenses", "categories"] if k in coco}
        new_coco["images"], new_coco["annotations"] = [], []
        ann_id = img_id = 1

        id_to_anns = {}
        for ann in coco["annotations"]:
            id_to_anns.setdefault(ann["image_id"], []).append(ann)

        for img_info in coco["images"]:
            img_path = img_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: skipping missing image {img_path}")
                continue

            for i in range(NUM_TILES):
                x0 = i * STEP_X
                x1 = x0 + TILE_W
                if x1 > img.shape[1]:
                    x1 = img.shape[1]
                    x0 = x1 - TILE_W

                tile = img[0:TILE_H, x0:x1]
                tile_name = f"{Path(img_info['file_name']).stem}_tile{i}.jpg"
                cv2.imwrite(str(out_img_dir / tile_name), tile)

                new_anns = []
                for ann in id_to_anns.get(img_info["id"], []):
                    x, y, w, h = ann["bbox"]
                    x2 = x + w
                    if x2 < x0 or x > x0 + TILE_W:
                        continue
                    new_x1 = max(x - x0, 0)
                    new_x2 = min(x2 - x0, TILE_W)
                    new_w = new_x2 - new_x1
                    if new_w <= 1:
                        continue
                    ann_new = ann.copy()
                    ann_new["bbox"] = [new_x1, y, new_w, h]
                    ann_new["id"] = ann_id
                    ann_new["image_id"] = img_id
                    new_anns.append(ann_new)
                    ann_id += 1

                new_coco["images"].append({"id": img_id, "width": TILE_W, "height": TILE_H, "file_name": tile_name})
                new_coco["annotations"].extend(new_anns)
                img_id += 1

        out_json = out_dir / f"instances_{split}.json"
        with open(out_json, "w") as f:
            json.dump(new_coco, f, indent=2)
        counts[split] = (len(new_coco["images"]), len(new_coco["annotations"]))
        print(f"  {split}: {counts[split][0]} tiled images, {counts[split][1]} annotations -> {out_json}")

    return counts


def coco_to_yolo(coco_tiled_dir, yolo_dir):
    """Converts the tiled COCO export to YOLO format (images/, labels/,
    data.yaml). Same logic as detect_surfers_v2.ipynb's conversion cell."""
    for split in SPLITS:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        json_path = coco_tiled_dir / f"instances_{split}.json"
        with open(json_path) as f:
            coco_data = json.load(f)

        id_to_info = {img["id"]: img for img in coco_data["images"]}
        img_to_anns = {img_id: [] for img_id in id_to_info}
        for ann in coco_data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        copied = 0
        for img_id, info in id_to_info.items():
            file_name = info["file_name"]
            src_img = coco_tiled_dir / split / file_name
            if not src_img.exists():
                print(f"  WARNING: missing tiled image {src_img}")
                continue
            shutil.copy2(src_img, yolo_dir / "images" / split / file_name)

            dst_lbl = yolo_dir / "labels" / split / (Path(file_name).stem + ".txt")
            with open(dst_lbl, "w") as f_out:
                for ann in img_to_anns.get(img_id, []):
                    cls = ann["category_id"] - 1  # YOLO classes are 0-indexed
                    x, y, w, h = ann["bbox"]
                    x_c = (x + w / 2) / info["width"]
                    y_c = (y + h / 2) / info["height"]
                    w_n, h_n = w / info["width"], h / info["height"]
                    f_out.write(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            copied += 1
        counts[split] = copied
        print(f"  {split}: {copied} images + labels -> {yolo_dir}")

    data_yaml = f"""train: {yolo_dir}/images/train
val: {yolo_dir}/images/val
test: {yolo_dir}/images/test

nc: 1
names: ['surfer']
"""
    (yolo_dir / "data.yaml").write_text(data_yaml)
    print(f"  wrote {yolo_dir / 'data.yaml'}")
    return counts


def main():
    args = parse_args()

    if not args.cvat_coco_dir.exists():
        raise SystemExit(f"--cvat-coco-dir not found: {args.cvat_coco_dir}")

    tiled_dir = DATA_DIR / "cvat_out_coco" / f"splits_tiled_{args.run_name}"
    yolo_dir = DATA_DIR / f"cvat_out_yolo_rebuilt_{args.run_name}"
    model_out_dir = DATA_DIR / "model_out" / args.run_name

    for d, label in [(tiled_dir, "tiled COCO"), (yolo_dir, "YOLO dataset"), (model_out_dir, "model_out")]:
        if d.exists() and not args.force:
            raise SystemExit(
                f"{label} output dir already exists: {d}\n"
                f"Pick a different --run-name, or pass --force to overwrite."
            )

    print(f"=== Tiling {args.cvat_coco_dir} -> {tiled_dir} ===")
    tile_coco_split(args.cvat_coco_dir, tiled_dir)

    print(f"\n=== Converting COCO -> YOLO: {tiled_dir} -> {yolo_dir} ===")
    coco_to_yolo(tiled_dir, yolo_dir)

    base_ckpt = args.base_checkpoint
    if not base_ckpt.exists():
        print(f"\nWARNING: base checkpoint {base_ckpt} not found -- "
              f"ultralytics will auto-download a fresh yolov8s.pt instead.")
        base_ckpt = Path("yolov8s.pt")  # bare name -- ultralytics resolves via its own download cache

    print(f"\n=== Training from {base_ckpt} ({args.epochs} epochs, device={args.device}) ===")
    model = YOLO(str(base_ckpt))
    model.train(
        data=str(yolo_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        device=args.device,
        project=str(model_out_dir),
        name="train",
    )

    print("\n=== Validating on held-out val split ===")
    metrics = model.val()
    results = metrics.results_dict
    precision = results.get("metrics/precision(B)")
    recall = results.get("metrics/recall(B)")
    map50 = results.get("metrics/mAP50(B)")
    map50_95 = results.get("metrics/mAP50-95(B)")
    print(f"  precision={precision:.5f}  recall={recall:.5f}  mAP50={map50:.5f}  mAP50-95={map50_95:.5f}")

    weights_path = model_out_dir / "train" / "weights" / "best.pt"
    print(f"""
=== Done ===
New weights:        {weights_path}
Training log/plots: {model_out_dir / "train"}
Validation:          precision={precision:.5f}  recall={recall:.5f}  mAP50={map50:.5f}  mAP50-95={map50_95:.5f}

Compare against the current production model before switching:
  DETECTOR_PRECISION/DETECTOR_RECALL in code/plot_daily_prediction.py
  (currently 0.87843 / 0.80618, from data/model_out/20251013/).

To put these weights into production, set in .env:
  MODEL_PATH={weights_path}
and update the hardcoded numbers above once you've confirmed the new
model is actually better -- don't just trust a higher mAP without also
spot-checking real detections.
""")


if __name__ == "__main__":
    main()
