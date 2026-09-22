"""
Side-by-side old-vs-new detector panels for the 60-second clear-day review sets.

Rebuilds `data/compare_detector_20260921/clearday_scenes/`. The first version of
those images was produced inline on 2026-09-22 at 08:49, half an hour before
nested-box suppression was added to `detect_surfers.py`, so the new-model panels
showed one surfer boxed twice in several scenes. Keeping this as a script means
the comparison can be regenerated whenever post-processing changes, instead of
quietly ageing out of date.

Both models run through the SAME current post-processing (cross-tile NMS,
nested-box suppression, false-positive zones), so the panels isolate the weights
rather than mixing in pipeline changes.

Human counts come from `data/reviews/count_60sec_var/review_counts.csv`; rows
whose count is not a number (e.g. "lens condensation") are skipped.

Usage:
    python code/compare_detector_scenes.py
    python code/compare_detector_scenes.py --second 21 --old-model <path>
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = _PROJECT_ROOT / "data" / "reviews" / "count_60sec_var"
OUT_DIR = _PROJECT_ROOT / "data" / "compare_detector_20260921" / "clearday_scenes"
OLD_MODEL = (_PROJECT_ROOT / "data" / "model_out" / "20251013" / "train" / "runs"
             / "detect" / "train13" / "weights" / "best.pt")

BAR_H = 26
OLD_COLOR = (255, 255, 255)
NEW_COLOR = (0, 255, 0)


def human_counts():
    out = {}
    with open(REVIEW_DIR / "review_counts.csv") as f:
        for row in csv.DictReader(f):
            try:
                out[row["filename"]] = int(row["human_count"])
            except (ValueError, TypeError):
                continue
    return out


def draw(img, boxes, color, label):
    canvas = img.copy()
    for x1, y1, x2, y2, _conf in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    bar = cv2.copyMakeBorder(canvas, 0, BAR_H, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(bar, label, (8, canvas.shape[0] + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def load(path):
    from ultralytics import YOLO
    print(f"Loading {path}")
    return YOLO(str(path))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--second", type=int, default=None,
                   help="only this second of each clip (default: whatever the "
                        "existing panel filenames use, else 21)")
    p.add_argument("--old-model", type=Path, default=OLD_MODEL)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def existing_seconds():
    """Which second each set's current panel was built from, so a rebuild matches."""
    out = {}
    for f in OUT_DIR.glob("set*_sec_*.png"):
        stem = f.stem                       # set5_09_26_sec_21
        set_id = stem.split("_")[0]
        out[set_id] = int(stem.rsplit("_", 1)[1])
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    humans = human_counts()
    per_set = existing_seconds()

    old_model = load(args.old_model)
    new_model = load(ds.MODEL_PATH)

    print(f"\nPost-processing: containment {ds.CONTAINMENT_THRESH}, "
          f"conf {ds.CONF_THRESH}, NMS IoU {ds.IOU_NMS}\n")
    print(f"  {'scene':<22} {'old':>5} {'new':>5} {'human':>6}")

    for set_dir in sorted(REVIEW_DIR.glob("set*")):
        if not set_dir.is_dir():
            continue
        set_id = set_dir.name.split("_")[0]
        sec = args.second if args.second is not None else per_set.get(set_id, 21)
        img_path = set_dir / f"sec_{sec:02d}.jpg"
        if not img_path.exists():
            print(f"  {set_dir.name}: no sec_{sec:02d}.jpg, skipping")
            continue
        rel = f"{set_dir.name}/{img_path.name}"
        human = humans.get(rel)
        if human is None:
            print(f"  {set_dir.name}: no numeric human count, skipping")
            continue

        img = cv2.imread(str(img_path))
        old_boxes = ds.run_inference_with_boxes(old_model, img_path)
        new_boxes = ds.run_inference_with_boxes(new_model, img_path)

        label_set = set_dir.name.replace("_", " ", 1).replace("_", ":")
        top = draw(img, old_boxes, OLD_COLOR,
                   f"{label_set} sec_{sec:02d}.jpg | OLD {len(old_boxes)}  (human {human})")
        bottom = draw(img, new_boxes, NEW_COLOR,
                      f"{label_set} sec_{sec:02d}.jpg | NEW {len(new_boxes)}  (human {human})")
        stacked = cv2.vconcat([top, bottom])

        out_path = args.out_dir / f"{set_dir.name}_sec_{sec:02d}.png"
        cv2.imwrite(str(out_path), stacked)
        print(f"  {set_dir.name:<22} {len(old_boxes):>5} {len(new_boxes):>5} {human:>6}")

    print(f"\nWrote panels to {args.out_dir}")


if __name__ == "__main__":
    main()
