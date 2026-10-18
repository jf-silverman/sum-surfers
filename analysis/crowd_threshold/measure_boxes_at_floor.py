"""Every box the detector proposes on the labelled frames, down to a 0.02 floor.

Why a floor and not `CONF_THRESH`. Two questions need answering here and both
need boxes the production threshold throws away:

  1. Is a crowd-dependent confidence threshold worth building? The 263-frame
     hand-count re-check (analysis/conf_thresh_recheck/) showed lowering the
     threshold helps crowded frames and hurts empty ones, so the instrument has
     to be indexed on crowd level -- which means scoring many thresholds.
  2. Why do several boxes pile onto one surfer when surfers cluster? That is a
     suppression question, so the raw proposals have to survive to be re-suppressed.

So this caches the proposals ONCE, pre-NMS and pre-suppression, and leaves every
threshold sweep to `score_crowd_rules.py`, which replays the production path
offline. One inference pass, any number of rules.

Ground truth comes from the tiled COCO export the production model was trained
on, reassembled into ROI coords. The split each frame came from is recorded with
it, which matters: 96 of the 172 labelled frames are TRAIN frames, so a rule
scored on the pooled set is scored partly on the model's own training data. Every
number `score_crowd_rules.py` reports is therefore broken out train vs held-out.

Usage:
    python analysis/crowd_threshold/measure_boxes_at_floor.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

SPLITS = _ROOT / "data" / "cvat_out_coco" / "splits_tiled_20260921_fog"
OUT = Path(__file__).resolve().parent
FLOOR = 0.02

TILE_OFFSETS = []
for _i in range(ds.NUM_TILES):
    _x0 = _i * ds.STEP_X
    _x1 = min(_x0 + ds.TILE_W, 1280)
    TILE_OFFSETS.append(max(_x1 - ds.TILE_W, 0))


def ground_truth_with_split():
    """{stem: (split, [boxes])} in ROI coords, deduplicated across tile overlap.

    Tiles overlap by 75px so a surfer in the seam is annotated twice; rounding
    the global coords and using a set collapses the pair.
    """
    gt = collections.defaultdict(set)
    split_of = {}
    for sp in ("train", "val", "test"):
        j = json.load(open(SPLITS / f"instances_{sp}.json"))
        names = {i["id"]: i["file_name"] for i in j["images"]}
        for a in j["annotations"]:
            stem, tile = names[a["image_id"]].rsplit("_tile", 1)
            off = TILE_OFFSETS[int(tile.split(".")[0])]
            x, y, w, h = a["bbox"]
            gt[stem].add((round(x + off, 1), round(y, 1),
                          round(x + off + w, 1), round(y + h, 1)))
            split_of[stem] = sp
    return {k: (split_of[k], [list(b) for b in v]) for k, v in gt.items()}


def untiled_images():
    idx = {}
    for p in (glob.glob(str(_ROOT / "data/cvat_out_coco/*/images/*.jpg"))
              + glob.glob(str(_ROOT / "data/cvat_out_coco/splits*/*/*.jpg"))):
        b = os.path.basename(p)
        if "_tile" in b:
            continue
        idx.setdefault(b[:-4], p)
    return idx


def proposals(model, img_path):
    """Per-tile boxes at the floor, in global ROI coords. No NMS, no suppression."""
    import shutil
    tiles, tmp = ds.tile_image_paths(img_path)
    out = []
    for idx, xoff, tp in tiles:
        r = model.predict(source=str(tp), conf=FLOOR, device=ds.DEVICE,
                          verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
            out.append((idx, x1 + xoff, y1, x2 + xoff, y2, float(b.conf[0].cpu())))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    gt = ground_truth_with_split()
    imgs = untiled_images()
    model = ds.load_model()
    brows, grows = [], []
    for n, (stem, (split, truth)) in enumerate(sorted(gt.items()), 1):
        path = imgs.get(stem)
        if path is None:
            continue
        for g in truth:
            grows.append(dict(frame=stem, split=split, x1=g[0], y1=g[1],
                              x2=g[2], y2=g[3]))
        for tile, x1, y1, x2, y2, conf in proposals(model, Path(path)):
            brows.append(dict(frame=stem, split=split, tile=tile,
                              x1=round(x1, 1), y1=round(y1, 1),
                              x2=round(x2, 1), y2=round(y2, 1),
                              conf=round(conf, 4)))
        if n % 25 == 0:
            print(f"  {n}/{len(gt)}", flush=True)

    b = pd.DataFrame(brows)
    g = pd.DataFrame(grows)
    b.to_csv(OUT / "raw_boxes_at_floor.csv", index=False)
    g.to_csv(OUT / "ground_truth_boxes.csv", index=False)
    nf = b.frame.nunique()
    print(f"\n{len(b)} proposals over {nf} frames at floor {FLOOR} "
          f"-> {OUT / 'raw_boxes_at_floor.csv'}")
    print(f"{len(g)} ground-truth boxes -> {OUT / 'ground_truth_boxes.csv'}")
    print(b.groupby("split").frame.nunique().to_string())
    print(f"proposals/frame: median {b.groupby('frame').size().median():.0f}, "
          f"max {b.groupby('frame').size().max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
