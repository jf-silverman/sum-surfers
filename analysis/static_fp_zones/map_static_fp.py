"""
Where do STATIC false positives actually fire, and is the mask the right shape?

The detector has two hardcoded false-positive zones (detect_surfers.TREE_MASK and
FLAG_MASK). Both were drawn by hand. This script tests a candidate zone against
the record instead of eyeballing it.

THE SIGNATURE. A static object detects at the SAME pixel location across many
different days; a surfer detects once, somewhere, at high confidence. So binning
box centres and counting DISTINCT DAYS per bin separates them:

    many days, middling confidence  -> a fixed object
    one or two days, high confidence -> real surfers

Distinct days rather than frames, because a dozen frames from one morning is one
observation of one lighting condition. Sampling spans all hours for the same
reason -- an earlier pass that took only the first frame of each day could not
have seen an object that is lit at midday and invisible at dawn.

RESULTS, 2026-10-15.

Tree boughs (x >= 1150, 160 frames over 93 days, all hours). Joel described the
conifer as several horizontal boughs at different heights, not connected
vertically, the lower ones reaching further into the frame -- which matters,
because it rules out a single tall rectangle. Only ONE cluster has the static
signature:

      bin        days  mean_conf
    1276, 72      18     0.348
    1276, 76      16     0.231
    1276, 68      11     0.389
    1276, 80       3     0.357
    1268, 128      2     0.234
    everything else 1-3   0.60-0.77

That cluster is already inside TREE_MASK (1268, 58, 1280, 87), and all 14 boxes
Joel flagged as tree boughs in the confidence-band review sit in it. The lower
boughs are clearly visible in the image -- one measured reaching to about x 1242
at y 114-128 -- but they do not generate false positives, presumably because
they are larger and sit against the dark shoreline rather than being thin
silhouettes against water.

VERDICT: no change. A rectangle spanning y 58 to y 158 would blanket the water
between boughs, and that water is productive -- bins at y 76-92, x 1150-1200
carry real detections at confidence 0.69-0.77. Masking it would delete real
surfers to remove false positives that are not occurring. The one watch item is
(1268, 128): 2 days at mean confidence 0.234, which matches the lower bough and
clears the 0.195 production threshold, so if it recurs it is the first candidate
for a second thin rectangle.

Shore heads (x 500-660, y >= 150, same sample). Joel flagged 8 boxes as "head of
person on shore, center bottom edge. We may need to mask out this area", and
unlike the boughs all 8 DO survive into production. But the day-count test comes
back empty -- no bin reaches 5 distinct days, the best being 3. The reason is
that these are not a fixed object: they are people, who stand in different
places, so a fixed object's signature cannot appear however many days are
sampled. The unit has to be the band, not the bin.

By band, the separation is clean:

    y 164-180 (the shore edge)     11 bins, 18 boxes, mean conf 0.359
    y < 164   (water, same x)       9 bins, 10 boxes, mean conf 0.741

which points at a confidence-conditional zone, the idiom FLAG_MASK already uses.
Against the reviewed boxes, a candidate zone (500, 164, 625, 180) contains 8 of
them: 7 not-surfers at confidence 0.108-0.181, and ONE confirmed surfer at
(596, 175), confidence 0.181 -- the top of the false-positive range, so no
confidence cut separates them. That one is also the row Joel annotated "edge
truncated; could be on shore person's head" at js_confidence 0.6, his least
certain yes.

Looking at the strip settles it: x 480-650, y 150-180 is WATER, not sand, with
head-sized dark blobs in it. It is the inside/shore break where people really do
stand and sometimes surf.

VERDICT: no mask. 7-for-1 looks like a good trade until you notice the 1 is
inside water at a confidence the 7 reach, and the sample is 8 boxes. Masking
would be fitting a rule to a handful of cases in a region where real surfers
appear. Revisit when more reviewed rows land in the band; until then this is a
known, measured, accepted false-positive source rather than an open question.

BOTH verdicts are negative, which is the point: the two causes Joel flagged for
masking in the review notes both turn out not to warrant one, for opposite
reasons -- the boughs because the mask already catches the only one that fires,
the shore heads because no mask can separate them from real surfers.

Usage:
    python analysis/static_fp_zones/map_static_fp.py --xmin 1150
    python analysis/static_fp_zones/map_static_fp.py --xmin 500 --xmax 660 --ymin 150
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

PREDS_CSV = _ROOT / "data" / "predictions" / "predictions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, default=0)
    ap.add_argument("--xmax", type=float, default=1e9)
    ap.add_argument("--ymin", type=float, default=0)
    ap.add_argument("--ymax", type=float, default=1e9)
    ap.add_argument("--floor", type=float, default=0.10,
                    help="detection floor; 0.10 is the confidence-band review's")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    preds = pd.read_csv(PREDS_CSV, float_precision="round_trip")
    preds = preds[preds["quality_ok"].astype(str) == "True"]
    # Across ALL hours, not one frame per day -- a fixed object can be lit at
    # one time of day and invisible at another.
    sample = preds.sample(n=min(args.frames, len(preds)), random_state=7)
    print(f"{len(sample)} frames, {sample['date'].nunique()} distinct days, "
          f"floor {args.floor}")
    print(f"region x {args.xmin}-{args.xmax}, y {args.ymin}-{args.ymax}\n")

    model = ds.load_model()
    days, confs = defaultdict(set), defaultdict(list)
    for r in sample.itertuples():
        p = ds.CROPS_DIR / r.filename
        if not p.exists():
            continue
        tiles, tmp = ds.tile_image_paths(p)
        boxes = []
        for _, xoff, tp in tiles:
            res = model.predict(source=str(tp), conf=args.floor,
                                device=ds.DEVICE, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
                boxes.append([x1 + xoff, y1, x2 + xoff, y2, float(b.conf[0].cpu())])
        shutil.rmtree(tmp, ignore_errors=True)
        # Production geometry, so the bins describe boxes that can really be
        # emitted -- but WITHOUT filter_false_positive_zones, since the point is
        # to see what the mask would have to catch.
        for b in ds.suppress_contained(ds.nms_across_tiles(boxes)):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (args.xmin <= cx <= args.xmax and args.ymin <= cy <= args.ymax):
                continue
            key = (int(cx // args.bin) * args.bin, int(cy // args.bin) * args.bin)
            days[key].add(r.date)
            confs[key].append(b[4])

    if not days:
        print("No detections in this region.")
        return 0
    d = pd.DataFrame([dict(x=k[0], y=k[1], days=len(v), mean_conf=np.mean(confs[k]),
                           boxes=len(confs[k]))
                      for k, v in days.items()]).sort_values("days", ascending=False)
    print(f"bins by distinct days (static = many days; surfers = 1-2 at high conf):")
    print(d.head(args.top).round(3).to_string(index=False))

    static = d[d["days"] >= 5]
    if len(static):
        print(f"\ncandidate static cluster(s): x {static.x.min()}-{static.x.max() + args.bin}, "
              f"y {static.y.min()}-{static.y.max() + args.bin}, "
              f"{int(static.days.sum())} detection-days, "
              f"mean conf {static.mean_conf.mean():.3f}")
        print(f"  suggested zone, centre-based as _in_zone uses: "
              f"({int(static.x.min()) - 2}, {int(static.y.min()) - 2}, "
              f"{int(static.x.max()) + args.bin + 2}, {int(static.y.max()) + args.bin + 2})")
        rest = d[d["days"] < 5]
        print(f"  the other {len(rest)} bin(s) average {rest.mean_conf.mean():.3f} "
              f"confidence on {rest.days.mean():.1f} days -- the surfer signature")
    else:
        print("\nNo bin reaches 5 distinct days: nothing here looks static at this "
              "sample size, and a mask would be fitted to noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
