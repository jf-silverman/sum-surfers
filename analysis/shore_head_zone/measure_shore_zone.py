"""Can a bottom-centre zone with a raised threshold remove the on-shore heads?

Joel's proposal (2026-10-17): the people standing on the cliff at the bottom
centre of the frame are in the FOREGROUND, so their head-and-shoulders should
be bigger than a surfer out in the water at the same image height -- which
would make a FLAG_MASK-style zone (drop below a raised confidence inside a
rectangle) safe if it also required a minimum box size.

This measures that instead of arguing it. Every labelled frame is run through
the production pipeline, each surviving box is matched to ground truth at
IoU 0.5, and a candidate rule is scored by what it would actually cost:

    surfers lost   true positives the rule drops
    FPs removed    false positives the rule drops

A rule is only worth adopting if it removes several FPs for zero or near-zero
lost surfers, and if the FPs it removes are the shore heads rather than any
old false positive that happens to sit low in the frame.

Usage:
    python analysis/shore_head_zone/measure_shore_zone.py
    python analysis/shore_head_zone/measure_shore_zone.py --rebuild   # re-detect
"""

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

SPLITS = _ROOT / "data" / "cvat_out_coco" / "splits_tiled_20260921_fog"
CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
PREDS_CSV = _ROOT / "data" / "predictions" / "predictions.csv"
OUT_DIR = Path(__file__).resolve().parent
BOXES_CSV = OUT_DIR / "production_boxes_on_labeled.csv"
SAMPLE_CSV = OUT_DIR / "sample_boxes_unlabeled.csv"

# The candidate: boxes clipped by the bottom edge of the ROI, in the x range
# where the bluff path rises high enough for a standing head to reach it.
ZONE = (400, 164, 680, 180)
EDGE_Y = 179.0

TILE_OFFSETS = []
for _i in range(ds.NUM_TILES):
    _x0 = _i * ds.STEP_X
    _x1 = min(_x0 + ds.TILE_W, 1280)
    TILE_OFFSETS.append(max(_x1 - ds.TILE_W, 0))


def ground_truth():
    """Labelled boxes per frame in ROI coords, reassembled from the tiled split.

    The tiled export is the one the production model was trained on, so it is
    the complete pool; tiles overlap by 75px, so the same surfer appears twice
    and has to be deduplicated on rounded global coords.
    """
    gt = collections.defaultdict(set)
    for sp in ("train", "val", "test"):
        j = json.load(open(SPLITS / f"instances_{sp}.json"))
        names = {i["id"]: i["file_name"] for i in j["images"]}
        for a in j["annotations"]:
            stem, tile = names[a["image_id"]].rsplit("_tile", 1)
            off = TILE_OFFSETS[int(tile.split(".")[0])]
            x, y, w, h = a["bbox"]
            gt[stem].add((round(x + off, 1), round(y, 1),
                          round(x + off + w, 1), round(y + h, 1)))
    return {k: [list(b) for b in v] for k, v in gt.items()}


def untiled_images():
    idx = {}
    for p in (glob.glob(str(_ROOT / "data/cvat_out_coco/*/images/*.jpg"))
              + glob.glob(str(_ROOT / "data/cvat_out_coco/splits*/*/*.jpg"))):
        b = os.path.basename(p)
        if "_tile" in b:
            continue
        idx.setdefault(b[:-4], p)
    return idx


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build():
    gt = ground_truth()
    imgs = untiled_images()
    model = ds.load_model()
    rows = []
    for n, (stem, truth) in enumerate(sorted(gt.items()), 1):
        path = imgs.get(stem)
        if path is None:
            continue
        kept = ds.run_inference_with_boxes(model, Path(path))
        # greedy highest-confidence-first match, one GT per detection
        taken = set()
        for b in sorted(kept, key=lambda z: -z[4]):
            best, best_i = 0.0, -1
            for i, g in enumerate(truth):
                if i in taken:
                    continue
                v = iou(b, g)
                if v > best:
                    best, best_i = v, i
            tp = best >= 0.5
            if tp:
                taken.add(best_i)
            rows.append(dict(frame=stem, x1=round(b[0], 1), y1=round(b[1], 1),
                             x2=round(b[2], 1), y2=round(b[3], 1),
                             conf=round(b[4], 4), iou=round(best, 3),
                             tp=int(tp)))
        if n % 25 == 0:
            print(f"  {n}/{len(gt)}", flush=True)
    pd.DataFrame(rows).to_csv(BOXES_CSV, index=False)
    print(f"{len(rows)} production boxes over {len(gt)} labelled frames -> {BOXES_CSV}")


def score(d, zone, conf_max, min_w=0.0, max_w=np.inf, edge_only=False):
    """Boxes a rule would drop: inside the zone, below conf_max, within the
    width window, and (optionally) clipped by the bottom edge of the ROI.
    Returns (surfers_lost, fps_removed)."""
    zx1, zy1, zx2, zy2 = zone
    hit = ((d.cx >= zx1) & (d.cx <= zx2) & (d.cy >= zy1) & (d.cy <= zy2)
           & (d.conf < conf_max) & (d.w >= min_w) & (d.w <= max_w))
    if edge_only:
        hit &= d.y2 >= EDGE_Y
    return int(d.loc[hit, "tp"].sum()), int((~d.loc[hit, "tp"].astype(bool)).sum())


def sample_rate(n):
    """How often the zone fires across the production record, which the 142
    labelled frames cannot say -- they are not a random sample of it."""
    import random
    p = pd.read_csv(PREDS_CSV)
    p = p[p["quality_ok"].astype(str) == "True"]
    files = [f for f in p["filename"] if "_side" not in str(f) and (CROPS / str(f)).exists()]
    random.seed(11)
    samp = random.sample(files, min(n, len(files)))
    model = ds.load_model()
    rows = []
    for i, f in enumerate(samp, 1):
        for b in ds.run_inference_with_boxes(model, CROPS / f):
            rows.append(dict(filename=f, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=b[4]))
        if i % 50 == 0:
            print(f"  {i}/{len(samp)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(SAMPLE_CSV, index=False)
    d["cx"] = (d.x1 + d.x2) / 2
    hit = d[(d.y2 >= EDGE_Y) & (d.cx >= ZONE[0]) & (d.cx <= ZONE[2]) & (d.conf < 0.50)]
    print(f"\n{len(samp)} random quality-passed frames, {len(d)} boxes "
          f"({len(d) / len(samp):.2f} per frame)")
    print(f"  the rule fires on {len(hit)} box(es) in {hit.filename.nunique()} frame(s) "
          f"-- {len(hit) / len(samp):.3f} per frame, "
          f"{100 * len(hit) / len(d):.1f}% of all detections")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rate", type=int, default=0,
                    help="also sample N unlabelled production frames to measure "
                         "how often the rule would fire")
    args = ap.parse_args()
    if args.rebuild or not BOXES_CSV.exists():
        build()

    d = pd.read_csv(BOXES_CSV)
    d["w"], d["h"] = d.x2 - d.x1, d.y2 - d.y1
    d["cx"], d["cy"] = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
    d["area"] = d.w * d.h
    print(f"\n{len(d)} production boxes, {d.tp.sum()} true positives, "
          f"{len(d) - d.tp.sum()} false positives, {d.frame.nunique()} frames")

    print("\n--- what is down there at all (no rule applied) ---")
    for y0 in (155, 160, 164, 168):
        s = d[d.cy >= y0]
        print(f"  cy >= {y0}: {len(s):4d} boxes, {s.tp.sum():3d} TP, "
              f"{len(s) - s.tp.sum():3d} FP")
    print("\n--- bottom-centre strip x 500-650 ---")
    for y0 in (160, 164, 168):
        s = d[(d.cx >= 500) & (d.cx <= 650) & (d.cy >= y0)]
        print(f"  cy >= {y0}: {len(s):4d} boxes, {s.tp.sum():3d} TP, "
              f"{len(s) - s.tp.sum():3d} FP")

    print("\n--- size, inside the strip (x 500-650, cy >= 164) ---")
    s = d[(d.cx >= 500) & (d.cx <= 650) & (d.cy >= 164)]
    if len(s):
        print(s.groupby("tp").agg(n=("w", "size"), w_med=("w", "median"),
                                  w_min=("w", "min"), w_max=("w", "max"),
                                  area_med=("area", "median"),
                                  conf_med=("conf", "median")).round(1).to_string())

    print("\n--- boxes clipped by the bottom edge of the ROI (y2 >= 179) ---")
    # A bluff-top head cannot be anywhere else: the cliff path sits just BELOW
    # the crop (ROI is rows 420-600 of a 720-row frame, and the fence, bench and
    # standing people are at roughly y 590-660), so only the top of a head
    # reaches the image and its box is always cut off by the edge.
    e = d[d.y2 >= EDGE_Y]
    print(e.groupby("tp").agg(n=("w", "size"), w_med=("w", "median"),
                              w_min=("w", "min"), w_max=("w", "max"),
                              h_med=("h", "median"), conf_med=("conf", "median"),
                              conf_max=("conf", "max"), cx_min=("cx", "min"),
                              cx_max=("cx", "max")).round(2).to_string())

    zone = ZONE
    z = e[(e.cx >= zone[0]) & (e.cx <= zone[2])]
    print(f"\n--- edge-clipped AND inside x {zone[0]}-{zone[2]} "
          f"(where the bluff top rises into the crop) ---")
    print(f"  {int(z.tp.sum())} true positives, {int((~z.tp.astype(bool)).sum())} false positives")
    print("  false positives, by confidence:",
          ", ".join(f"{c:.2f}" for c in sorted(z.loc[z.tp == 0, "conf"])))
    print("  true  positives, by confidence:",
          ", ".join(f"{c:.2f}" for c in sorted(z.loc[z.tp == 1, "conf"])))

    print("\n--- rules: raise the threshold for edge-clipped boxes in the zone ---")
    for c in (0.25, 0.30, 0.40, 0.45, 0.50, 0.52, 0.55, 0.60):
        lost, removed = score(d, zone, c, edge_only=True)
        print(f"      conf < {c:.2f}  ->  {lost} surfer(s) lost, {removed} FP(s) removed")

    print("\n--- the same rules without the edge condition (zone alone) ---")
    for c in (0.30, 0.40, 0.50, 0.60):
        lost, removed = score(d, zone, c, edge_only=False)
        print(f"      conf < {c:.2f}  ->  {lost} surfer(s) lost, {removed} FP(s) removed")

    if args.rate:
        sample_rate(args.rate)

    print("\n--- size, inside the zone (the foreground hypothesis) ---")
    print("  a rule keyed on BIG boxes instead of confidence:")
    for mw in (12.0, 15.0, 18.0, 20.0):
        lost, removed = score(d, zone, 1.01, min_w=mw, edge_only=True)
        print(f"      w >= {mw:4.0f}, any conf  ->  {lost} surfer(s) lost, "
              f"{removed} FP(s) removed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
