"""
Sweep the detection confidence threshold on held-out frames (B05).

CONF_THRESH (0.195) was tuned on the October 2025 weights and never revisited
after the September 2026 fog retrain, which changed what the model is confident
about. This script re-measures precision/recall/F1 and whole-frame count error
across a range of thresholds, and breaks the result down by image row band and
by hour of day — the two axes the conditional-threshold ideas depend on
(B03: whitewater band recall; the dawn/dusk and row-gradient ideas).

Method: inference runs ONCE per frame at a low floor (--floor, default 0.02) and
every candidate threshold is then applied to the cached boxes. That is equivalent
to re-running detection at each threshold because every post-processing step is
score-ordered or per-box — NMS and containment suppression both keep the
highest-scoring box first, so extra low-scoring boxes can never displace a
high-scoring one, and the false-positive zone filter treats boxes independently.
The equivalence is asserted against the real production path at CONF_THRESH.

Raw boxes are cached to --cache so repeat sweeps are instant.

Usage:
    python code/tune_confidence.py --coco-dir data/cvat_out_coco/splits_v2
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

IOU_MATCH = 0.5          # standard detection match criterion
ROW_BANDS = [(0, 60, "top (0-60)"), (60, 120, "middle (60-120)"), (120, 10_000, "bottom (120+)")]


def raw_boxes_for(model, img_path, floor):
    """Every box the model emits at `floor` confidence, in global ROI coords."""
    tiles, tmp_dir = ds.tile_image_paths(img_path)
    boxes = []
    for _tile_idx, x_offset, tile_path in tiles:
        results = model.predict(source=str(tile_path), conf=floor,
                                device=ds.DEVICE, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            boxes.append([x1 + x_offset, y1, x2 + x_offset, y2, float(box.conf[0].cpu())])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return boxes


def post_process(boxes, thresh):
    """The production post-processing chain, applied at an arbitrary threshold."""
    kept = [b for b in boxes if b[4] >= thresh]
    kept = ds.suppress_contained(ds.nms_across_tiles(kept))
    return ds.filter_false_positive_zones(kept)


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match(preds, truths):
    """Greedy highest-confidence-first matching. Returns (tp_pairs, fps, fns)."""
    unmatched = list(range(len(truths)))
    tp_pairs, fps = [], []
    for p in sorted(preds, key=lambda b: -b[4]):
        best, best_iou = None, IOU_MATCH
        for ti in unmatched:
            v = iou(p, truths[ti])
            if v >= best_iou:
                best, best_iou = ti, v
        if best is None:
            fps.append(p)
        else:
            unmatched.remove(best)
            tp_pairs.append((p, truths[best]))
    return tp_pairs, fps, [truths[i] for i in unmatched]


def band_of(box):
    cy = (box[1] + box[3]) / 2
    for lo, hi, name in ROW_BANDS:
        if lo <= cy < hi:
            return name
    return ROW_BANDS[-1][2]


def hour_of(file_name):
    try:
        return datetime.strptime(Path(file_name).stem.split("_side")[0],
                                 "crop%Y-%m-%d_%H-%M-%S").hour
    except ValueError:
        return None


def load_truth(coco_dir, split):
    coco = json.loads((coco_dir / f"instances_{split}.json").read_text())
    by_id = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        by_id[ann["image_id"]].append([x, y, x + w, y + h])
    return {img["file_name"]: by_id[img["id"]] for img in coco["images"]}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def collect(model, coco_dir, splits, floor, cache_path, excluded):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("floor") != floor or cache.get("model") != str(ds.MODEL_PATH):
            cache = {}
    frames = cache.get("frames", {})

    data = []
    for split in splits:
        truth = load_truth(coco_dir, split)
        for file_name, boxes in sorted(truth.items()):
            if file_name in excluded:
                continue
            img_path = coco_dir / split / file_name
            if not img_path.exists():
                print(f"  WARNING: {img_path} missing, skipping")
                continue
            key = f"{split}/{file_name}"
            if key not in frames:
                frames[key] = raw_boxes_for(model, img_path, floor)
                if len(frames) % 10 == 0:
                    print(f"  ... {len(frames)} frames inferred")
            data.append((file_name, frames[key], boxes))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"floor": floor, "model": str(ds.MODEL_PATH), "frames": frames}))
    return data


def sweep(data, thresholds):
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        abs_err = signed = 0.0
        for _file_name, raw, truths in data:
            preds = post_process(raw, t)
            tps, fps, fns = match(preds, truths)
            tp += len(tps); fp += len(fps); fn += len(fns)
            err = len(preds) - len(truths)
            abs_err += abs(err); signed += err
        p, r, f = prf(tp, fp, fn)
        n = len(data)
        rows.append({"thresh": t, "precision": p, "recall": r, "f1": f,
                     "tp": tp, "fp": fp, "fn": fn,
                     "count_mae": abs_err / n, "count_bias": signed / n})
    return rows


def breakdown(data, thresholds, key_fn, label):
    """precision/recall by group, per threshold. key_fn(box, file_name) -> group."""
    print(f"\n{label}")
    groups = sorted({key_fn(b, fnm) for fnm, _raw, truths in data for b in truths
                     for _ in [0]} | {key_fn(b, fnm) for fnm, raw, _t in data
                                      for b in post_process(raw, min(thresholds))},
                    key=str)
    header = "  {:<16}".format("group") + "".join(f"{t:>14.3f}" for t in thresholds)
    print(header)
    for g in groups:
        cells = []
        for t in thresholds:
            tp = fp = fn = 0
            for file_name, raw, truths in data:
                preds = [b for b in post_process(raw, t) if key_fn(b, file_name) == g]
                gts = [b for b in truths if key_fn(b, file_name) == g]
                tps, fps, fns = match(preds, gts)
                tp += len(tps); fp += len(fps); fn += len(fns)
            p, r, _f = prf(tp, fp, fn)
            cells.append(f"   P{p*100:4.0f} R{r*100:4.0f}")
        print(f"  {str(g):<16}" + "".join(cells))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "cvat_out_coco" / "splits_v2")
    p.add_argument("--split", action="append", choices=["train", "val", "test"],
                   help="repeatable; default val+test")
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--cache", type=Path,
                   default=_PROJECT_ROOT / "data" / "conf_sweep_cache.json")
    p.add_argument("--min", type=float, default=0.05)
    p.add_argument("--max", type=float, default=0.60)
    p.add_argument("--step", type=float, default=0.025)
    return p.parse_args()


def main():
    args = parse_args()
    splits = args.split or ["val", "test"]

    excluded = set()
    for parent in ds.MODEL_PATH.parents:
        f = parent / "train_filenames.txt"
        if f.exists():
            excluded = {ln.strip() for ln in f.read_text().splitlines()
                        if ln.strip() and not ln.startswith("#")}
            break

    print(f"Model   : {ds.MODEL_PATH}")
    print(f"Splits  : {', '.join(splits)} from {args.coco_dir}")
    print(f"Excluded: {len(excluded)} training frames")
    print(f"Floor   : {args.floor} (one inference pass per frame)")

    model = ds.load_model()
    data = collect(model, args.coco_dir, splits, args.floor, args.cache, excluded)
    total_truth = sum(len(t) for _f, _r, t in data)
    print(f"Frames  : {len(data)} held out, {total_truth} labeled surfers\n")

    # Equivalence check: cached boxes re-thresholded must match the production path.
    sample = data[0]
    img = next((args.coco_dir / s / sample[0] for s in splits
                if (args.coco_dir / s / sample[0]).exists()))
    prod_count, _conf = ds.run_inference(model, img)
    cached_count = len(post_process(sample[1], ds.CONF_THRESH))
    status = "OK" if prod_count == cached_count else "MISMATCH"
    print(f"Equivalence check on {sample[0]}: production {prod_count} vs "
          f"re-thresholded {cached_count}  [{status}]")
    if status == "MISMATCH":
        print("  Post-processing is not score-order-invariant after all — "
              "treat the sweep below as approximate.")

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    rows = sweep(data, thresholds)
    print(f"\n{'conf':>6} {'prec':>7} {'recall':>7} {'F1':>7} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'cnt MAE':>8} {'cnt bias':>9}")
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_mae = min(rows, key=lambda r: r["count_mae"])
    for r in rows:
        marks = []
        if r is best_f1:
            marks.append("best F1")
        if r is best_mae:
            marks.append("best MAE")
        if abs(r["thresh"] - ds.CONF_THRESH) < 1e-9:
            marks.append("production")
        print(f"{r['thresh']:>6.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
              f"{r['count_mae']:>8.2f} {r['count_bias']:>+9.2f}"
              + ("   <- " + ", ".join(marks) if marks else ""))

    print(f"\nBest F1   : {best_f1['thresh']:.3f} "
          f"(P {best_f1['precision']:.3f}, R {best_f1['recall']:.3f}, F1 {best_f1['f1']:.3f})")
    print(f"Best count: {best_mae['thresh']:.3f} "
          f"(MAE {best_mae['count_mae']:.2f}, bias {best_mae['count_bias']:+.2f})")

    focus = [t for t in (0.10, 0.15, ds.CONF_THRESH, 0.25, 0.35, 0.45) if t in thresholds] \
        or thresholds[::4]
    breakdown(data, focus, lambda b, _f: band_of(b), "Precision/recall by row band:")
    breakdown(data, focus, lambda _b, f: hour_of(f), "Precision/recall by hour of day:")


if __name__ == "__main__":
    main()
