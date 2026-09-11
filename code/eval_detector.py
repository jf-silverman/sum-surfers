"""
eval_detector.py
----------------
Evaluates the committed detector weights against the held-out test split,
so the detection numbers quoted in the README can be re-derived rather than
taken on faith.

Everything this needs is already in the repo — the labeled splits, the tiled
YOLO dataset, and the trained weights — so it runs on a fresh clone with no
account, no token, and no access to the unpublished prediction corpus.

It reports two different things, because they answer different questions:

1. **Tile-level detection metrics** (precision / recall / mAP), via
   Ultralytics' own validator on the tiled dataset. This is the same
   measurement the training log produced, so the `val` split here should
   reproduce the 87.8% / 80.6% quoted in the README, and the `test` split
   shows what those look like on images never used for training or model
   selection.

2. **Count-level accuracy on whole frames**, which is what actually feeds
   the forecast model. Tile metrics score individual boxes; the pipeline
   only ever uses the *number* of surfers in a full 1280x180 crop, after
   tiling, cross-tile NMS, and false-positive-zone filtering. A detector can
   look respectable per-box and still undercount crowded frames. This runs
   the production inference path (`detect_surfers.run_inference`, at the
   production confidence threshold) on each full test image and compares the
   count to the number of ground-truth boxes.

Usage:
    python code/eval_detector.py                 # test split (default)
    python code/eval_detector.py --split val     # reproduce the README numbers
    python code/eval_detector.py --split val --split test
"""

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

YOLO_DIR = _PROJECT_ROOT / "data" / "cvat_out_yolo_rebuilt"       # tiled images + labels
COCO_SPLITS_DIR = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits"  # whole frames + boxes
EVAL_OUT_DIR = _PROJECT_ROOT / "data" / "eval_out"

# Published in plot_daily_prediction.py and the README as the detector's
# accuracy. Both are final-epoch *val* numbers from the training log, so they
# are the reconciliation target for --split val, not for test.
PUBLISHED_VAL_PRECISION = 0.87843
PUBLISHED_VAL_RECALL = 0.80618


def portable_data_yaml(tmp_dir):
    """Write a data.yaml with paths resolved from this file's location.

    The committed data.yaml has absolute paths from the machine that built it
    (/Users/.../sum-surfers/...), so it only resolves on that machine. Writing
    a fresh one is what lets this script work on anyone else's clone.
    """
    names = ["surfer"]
    committed = YOLO_DIR / "data.yaml"
    if committed.exists():
        # Keep the class names from the real export rather than assuming;
        # a future multi-class export should not need this script edited.
        for line in committed.read_text().splitlines():
            if line.startswith("names:"):
                names = eval(line.split(":", 1)[1].strip())  # noqa: S307 - repo-local file
    path = Path(tmp_dir) / "data.yaml"
    path.write_text(
        f"train: {YOLO_DIR}/images/train\n"
        f"val: {YOLO_DIR}/images/val\n"
        f"test: {YOLO_DIR}/images/test\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n"
    )
    return path


def tile_level_metrics(model, split):
    """Ultralytics validation on the tiled dataset — the training log's measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        metrics = model.val(
            data=str(portable_data_yaml(tmp)),
            split=split,
            project=str(EVAL_OUT_DIR),
            name=f"val_{split}",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def ground_truth_counts(split):
    """{image filename: number of labeled boxes} for the whole (untiled) frames."""
    coco = json.loads((COCO_SPLITS_DIR / f"instances_{split}.json").read_text())
    per_image = {img["id"]: 0 for img in coco["images"]}
    for ann in coco["annotations"]:
        per_image[ann["image_id"]] += 1
    return {img["file_name"]: per_image[img["id"]] for img in coco["images"]}


def count_level_metrics(model, split):
    """Production inference path vs. ground-truth box counts, per whole frame."""
    truth = ground_truth_counts(split)
    rows = []
    for file_name, actual in sorted(truth.items()):
        img_path = COCO_SPLITS_DIR / split / file_name
        if not img_path.exists():
            print(f"  WARNING: {img_path} missing, skipping")
            continue
        predicted, avg_conf = ds.run_inference(model, img_path)
        rows.append({
            "image": file_name,
            "actual": actual,
            "predicted": predicted,
            "error": predicted - actual,
            "avg_conf": avg_conf,
        })
    return rows


def print_count_table(rows):
    print(f"  {'image':<34} {'actual':>7} {'pred':>6} {'error':>7} {'avg conf':>9}")
    print(f"  {'-' * 34} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 9}")
    for r in rows:
        print(f"  {r['image']:<34} {r['actual']:>7} {r['predicted']:>6} "
              f"{r['error']:>+7} {r['avg_conf']:>9.4f}")

    errors = [r["error"] for r in rows]
    actuals = [r["actual"] for r in rows]
    mae = statistics.mean(abs(e) for e in errors)
    bias = statistics.mean(errors)
    total_actual, total_pred = sum(actuals), sum(r["predicted"] for r in rows)
    print(f"\n  n frames        : {len(rows)}")
    print(f"  total actual    : {total_actual}")
    print(f"  total predicted : {total_pred}")
    print(f"  MAE             : {mae:.2f} surfers")
    # Sign matters more than magnitude here: a consistently negative bias is
    # the undercount the training-data expansion work is meant to address,
    # and it is invisible in MAE alone.
    print(f"  mean bias       : {bias:+.2f} surfers ({'under' if bias < 0 else 'over'}counting)")
    if total_actual:
        print(f"  total-count err : {(total_pred - total_actual) / total_actual:+.1%}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", action="append", choices=["train", "val", "test"],
                   help="Split to evaluate; repeatable (default: test)")
    p.add_argument("--skip-tile-metrics", action="store_true",
                   help="Only run the whole-frame count evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    splits = args.split or ["test"]

    if not YOLO_DIR.exists():
        sys.exit(f"Tiled YOLO dataset not found at {YOLO_DIR}")
    if not ds.MODEL_PATH.exists():
        sys.exit(f"Model weights not found at {ds.MODEL_PATH}")

    print(f"Weights : {ds.MODEL_PATH}")
    print(f"Device  : {ds.DEVICE}")
    print(f"Conf    : {ds.CONF_THRESH} (production threshold, used for the count evaluation)")
    model = ds.load_model()

    for split in splits:
        print(f"\n{'=' * 72}\n{split.upper()} SPLIT\n{'=' * 72}")

        if not args.skip_tile_metrics:
            print(f"\n-- Tile-level detection metrics ({split}) --")
            m = tile_level_metrics(model, split)
            print(f"  precision   : {m['precision']:.5f}")
            print(f"  recall      : {m['recall']:.5f}")
            print(f"  mAP@0.5     : {m['map50']:.5f}")
            print(f"  mAP@0.5:0.95: {m['map50_95']:.5f}")
            if split == "val":
                print(f"\n  Published in README/plot_daily_prediction.py: "
                      f"precision {PUBLISHED_VAL_PRECISION:.5f}, recall {PUBLISHED_VAL_RECALL:.5f}")
                print(f"  Difference: precision {m['precision'] - PUBLISHED_VAL_PRECISION:+.5f}, "
                      f"recall {m['recall'] - PUBLISHED_VAL_RECALL:+.5f}")

        print(f"\n-- Whole-frame count accuracy ({split}, production inference path) --")
        rows = count_level_metrics(model, split)
        print_count_table(rows)


if __name__ == "__main__":
    main()
