"""
Production boxes on the labelled frames with the false-positive zones OFF.

`analysis/shore_head_zone/production_boxes_on_labeled.csv` was written through
`ds.run_inference_with_boxes()`, which applies `TREE_MASK`, `FLAG_MASK` and now
`BLUFF_MASK`. Anything those zones already drop is therefore absent from it by
construction -- which makes it useless for judging a zone. Scoring `TREE_MASK`
against it returns "0 false positives caught, 0 true positives clipped", which
is an artifact of the file, not a property of the zone.

This rebuilds the same table with everything up to the zones (NMS across tile
seams, containment suppression) and nothing after, so a zone can be scored on
what the detector actually proposes. Ground truth and IoU-0.5 greedy matching
are reused unchanged from `measure_shore_zone.py`.

Usage:
    python analysis/tree_bough_zone/measure_unfiltered.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
sys.path.insert(0, str(_ROOT / "analysis" / "shore_head_zone"))
import detect_surfers as ds  # noqa: E402
from measure_shore_zone import ground_truth, untiled_images, iou  # noqa: E402

OUT = Path(__file__).resolve().parent / "unfiltered_boxes_on_labeled.csv"


def unfiltered(model, img_path):
    """Everything run_inference_with_boxes does EXCEPT the zone filter."""
    import shutil
    tiles, tmp = ds.tile_image_paths(img_path)
    boxes = []
    for _, xoff, tp in tiles:
        r = model.predict(source=str(tp), conf=ds.CONF_THRESH,
                          device=ds.DEVICE, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
            boxes.append([x1 + xoff, y1, x2 + xoff, y2, float(b.conf[0].cpu())])
    shutil.rmtree(tmp, ignore_errors=True)
    return ds.suppress_contained(ds.nms_across_tiles(boxes))


def main():
    gt = ground_truth()
    imgs = untiled_images()
    model = ds.load_model()
    rows = []
    for n, (stem, truth) in enumerate(sorted(gt.items()), 1):
        path = imgs.get(stem)
        if path is None:
            continue
        taken = set()
        for b in sorted(unfiltered(model, Path(path)), key=lambda z: -z[4]):
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
                             conf=round(b[4], 4), iou=round(best, 3), tp=int(tp)))
        if n % 25 == 0:
            print(f"  {n}/{len(gt)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f"{len(d)} unfiltered boxes over {len(gt)} labelled frames "
          f"({int(d.tp.sum())} TP, {int((d.tp == 0).sum())} FP) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
