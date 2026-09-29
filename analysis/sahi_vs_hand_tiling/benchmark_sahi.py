"""
Does SAHI beat this project's hand-rolled 4-tile split?

SAHI (Slicing Aided Hyper Inference) is a maintained library that does what
detect_surfers.py does by hand: cut a large image into overlapping slices, run a
detector on each, and merge the boxes back. The question is whether the library
is better than the ~60 lines here, on this project's own held-out frames.

Both paths use the SAME weights and the SAME confidence threshold, so the only
variable is the slicing and merging. Scored against the 32 whole frames in the
production detector's test split -- frames it never trained on -- with the
ground-truth box count from the CVAT labels.

The count is what matters here, not box placement: this project answers "how
crowded is it", so a box that lands on the right surfer counts that surfer
whether or not it hugs the outline.

Usage:
    python analysis/sahi_vs_hand_tiling/benchmark_sahi.py
    python analysis/sahi_vs_hand_tiling/benchmark_sahi.py --limit 5   # quick pass
"""
import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

V2_DIR = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits_v2"
TILED_DIR = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits_tiled_20260921_fog"


def held_out_frames():
    """[(image_path, true_box_count), ...] for frames in the production test split.

    The tiled split is what the model was actually trained on, and its tile names
    carry the source frame stem, so the frame-level split is recoverable from
    them. Ground-truth counts come from the whole-frame labels in splits_v2 --
    summing boxes across tiles would double-count anything in an overlap.
    """
    test_stems = set()
    o = json.loads((TILED_DIR / "instances_test.json").read_text())
    for im in o["images"]:
        test_stems.add(re.sub(r"_tile\d+\.jpg$", "", im["file_name"]))

    out = []
    for split in ("train", "val", "test"):
        path = V2_DIR / f"instances_{split}.json"
        if not path.exists():
            continue
        o = json.loads(path.read_text())
        per = collections.Counter()
        for a in o["annotations"]:
            per[a["image_id"]] += 1
        for im in o["images"]:
            stem = im["file_name"].rsplit(".", 1)[0]
            if stem not in test_stems:
                continue
            img = V2_DIR / split / im["file_name"]
            if img.exists():
                out.append((img, per.get(im["id"], 0)))
    return sorted(out)


def sahi_boxes(model_path, img_path, slice_w, slice_h, overlap):
    """SAHI's merged boxes as [x1, y1, x2, y2, conf], this project's format."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    global _SAHI_MODEL
    if "_SAHI_MODEL" not in globals() or _SAHI_MODEL is None:
        _SAHI_MODEL = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(model_path),
            confidence_threshold=ds.CONF_THRESH, device="cpu")
    res = get_sliced_prediction(
        str(img_path), _SAHI_MODEL,
        slice_width=slice_w, slice_height=slice_h,
        overlap_width_ratio=overlap, overlap_height_ratio=0.0,
        verbose=0)
    out = []
    for o in res.object_prediction_list:
        b = o.bbox
        out.append([b.minx, b.miny, b.maxx, b.maxy, float(o.score.value)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="only the first N frames")
    args = p.parse_args()

    frames = held_out_frames()
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print("No held-out frames found.")
        return 1
    print(f"{len(frames)} held-out frames (production test split), "
          f"{sum(n for _f, n in frames)} labelled surfers\n")

    model = ds.load_model()
    rows = []
    t_hand = t_sahi = 0.0
    for img, truth in frames:
        t0 = time.perf_counter()
        hand = len(ds.run_inference_with_boxes(model, img))
        t_hand += time.perf_counter() - t0

        t0 = time.perf_counter()
        sb = sahi_boxes(ds.MODEL_PATH, img, ds.TILE_W, ds.TILE_H, 0.20)
        t_sahi += time.perf_counter() - t0
        sahi_n = len(sb)
        # Third arm: SAHI's slicing and merging, then THIS project's
        # post-processing. Without it the comparison conflates two different
        # things -- whether the library slices better, and whether the extra
        # filtering this project added is doing real work.
        sp = ds.filter_false_positive_zones(ds.suppress_contained(sb))
        rows.append((img.name, truth, hand, sahi_n, len(sp)))
        print(f"  {img.name:<34} true {truth:>3}   hand {hand:>3}   "
              f"sahi {sahi_n:>3}   sahi+post {len(sp):>3}")

    truth = np.array([r[1] for r in rows], float)
    hand = np.array([r[2] for r in rows], float)
    sahi = np.array([r[3] for r in rows], float)
    sahi_post = np.array([r[4] for r in rows], float)

    def stats(pred, label, secs):
        err = pred - truth
        print(f"  {label:<22} MAE {np.abs(err).mean():5.2f}   bias {err.mean():+5.2f}   "
              f"total {pred.sum():>4.0f} vs {truth.sum():.0f} "
              f"({pred.sum() / truth.sum():.1%})   {secs / len(rows):.2f}s/frame")

    print(f"\n{'':<22} {'against the labels':}")
    stats(hand, "hand-rolled 4 tiles", t_hand)
    stats(sahi, "SAHI sliced", t_sahi)
    stats(sahi_post, "SAHI + this post-proc", t_sahi)

    d = np.abs(sahi - truth) - np.abs(hand - truth)
    print(f"\n  SAHI minus hand-rolled, per-frame absolute error: {d.mean():+.2f}")
    print(f"  SAHI closer on {(d < 0).sum()} frames, worse on {(d > 0).sum()}, "
          f"tied on {(d == 0).sum()}")
    from scipy import stats as st
    if len(rows) > 2:
        t = st.ttest_rel(np.abs(sahi - truth), np.abs(hand - truth))
        print(f"  paired t-test p = {t.pvalue:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
