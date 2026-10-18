"""Are the overlapping boxes on clustered surfers duplicates, or real neighbours?

Joel's report: when several surfers are close together the detector puts too many
overlapping boxes on them. Two readings of that, with opposite fixes:

  DUPLICATES  -- one surfer wearing several boxes. Fix: suppress harder.
  REAL NEIGHBOURS -- adjacent surfers that look like a pile. Suppressing harder
                  deletes people. This is what Adaptive NMS (Liu et al., CVPR
                  2019) warns about: in a crowd, high overlap is evidence of two
                  pedestrians rather than one detected twice, so the suppression
                  threshold should RISE with local density, not fall.

Box-level ground truth settles it. For every box production keeps, this records
whether it matched a real surfer at IoU 0.5, how much it overlaps the nearest
other kept box, and how many kept boxes sit within NEIGHBOUR_PX of its centre.
A false positive stacked on a counted surfer is a duplicate; one sitting alone
is something else, and the two call for different fixes.

Then it scores density-adaptive containment both ways -- looser in crowds, as
Adaptive NMS argues, and tighter, as the duplicate reading implies -- against the
flat `CONTAINMENT_THRESH`.

Reported train / held-out separately: 81 of the 142 labelled frames trained the
production model, so only the 61 held-out frames are evidence.

Usage:
    python analysis/crowd_threshold/score_duplicate_rules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402
from score_crowd_rules import iou, load, match, summarise, PROD_THRESH  # noqa: E402

HERE = Path(__file__).resolve().parent
NEIGHBOUR_PX = 40          # centre-to-centre radius counted as "close together"
STACK_IOU = 0.3            # overlap at which a box is "stacked on" another


def density(boxes, radius=NEIGHBOUR_PX):
    """Kept boxes within `radius` of each box's centre, excluding itself."""
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
    if len(c) < 2:
        return [0] * len(c)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    return (d <= radius).sum(axis=1) - 1


def suppress_adaptive(boxes, base, slope, radius=NEIGHBOUR_PX):
    """Containment suppression whose threshold moves with local density.

    slope > 0 is the Adaptive NMS direction: a denser neighbourhood raises the
    bar for calling an overlap a duplicate, so genuinely adjacent surfers
    survive. slope < 0 suppresses harder in crowds instead.
    """
    if len(boxes) < 2:
        return list(boxes)
    dens = density(boxes, radius)
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][4])
    kept = []
    for i in order:
        b = boxes[i]
        t = min(max(base + slope * dens[i], 0.1), 1.0)
        area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
        dup = False
        for k in kept:
            iw = min(b[2], k[2]) - max(b[0], k[0])
            ih = min(b[3], k[3]) - max(b[1], k[1])
            if iw <= 0 or ih <= 0:
                continue
            smaller = min(area_b, (k[2] - k[0]) * (k[3] - k[1]))
            if smaller > 0 and (iw * ih) / smaller > t:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def production(boxes, thresh=PROD_THRESH):
    kept = [b for b in boxes if b[4] >= thresh]
    return ds.filter_false_positive_zones(
        ds.suppress_contained(ds.nms_across_tiles(kept)))


def main():
    props, truth, split = load()

    # --- 1. anatomy of what production keeps ---
    rows = []
    for frame, boxes in props.items():
        kept = production(boxes)
        dens = density(kept)
        taken, flag = set(), {}
        for j, b in enumerate(sorted(range(len(kept)), key=lambda i: -kept[i][4])):
            bb = kept[b]
            best, bi = 0.0, -1
            for i, g in enumerate(truth[frame]):
                if i in taken:
                    continue
                v = iou(bb, g)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                taken.add(bi)
            flag[b] = best >= 0.5
        for i, b in enumerate(kept):
            stack = max([iou(b, k) for j, k in enumerate(kept) if j != i], default=0.0)
            partner_tp = None
            if stack >= STACK_IOU:
                j = int(np.argmax([iou(b, k) if j2 != i else -1
                                   for j2, k in enumerate(kept)]))
                partner_tp = flag[j]
            rows.append(dict(frame=frame, split=split[frame], tp=int(flag[i]),
                             conf=b[4], neighbours=int(dens[i]),
                             max_iou=round(stack, 3),
                             stacked=int(stack >= STACK_IOU),
                             partner_tp=partner_tp))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "kept_box_anatomy.csv", index=False)

    for s in ("held-out", "train"):
        sub = d[d.split != "train"] if s == "held-out" else d[d.split == "train"]
        fp = sub[sub.tp == 0]
        print(f"\n=== {s}: {len(sub)} kept boxes, {len(fp)} false positives ===")
        print(f"  stacked on another kept box (IoU >= {STACK_IOU}): "
              f"{int(sub.stacked.sum())} boxes ({sub.stacked.mean():.1%})")
        print(f"  of the {len(fp)} FPs: {int(fp.stacked.sum())} stacked "
              f"({fp.stacked.mean():.1%}), {int((fp.stacked == 0).sum())} isolated")
        st = fp[fp.stacked == 1]
        if len(st):
            print(f"    stacked FPs whose partner is a real surfer (duplicate): "
                  f"{int(st.partner_tp.fillna(False).sum())}")
        print("  FP rate by local neighbour count:")
        g = sub.assign(nb=pd.cut(sub.neighbours, [-1, 0, 1, 2, 4, 99],
                                 labels=["0", "1", "2", "3-4", "5+"])) \
               .groupby("nb", observed=True).agg(boxes=("tp", "size"),
                                                 fp_rate=("tp", lambda x: 1 - x.mean()))
        print(g.round(3).to_string())

    # --- 2. density-adaptive containment ---
    print(f"\n=== Density-adaptive containment (radius {NEIGHBOUR_PX}px) ===")
    variants = [("flat 0.7 (production)", 0.7, 0.0)]
    for slope in (0.05, 0.10, 0.15):
        variants.append((f"looser in crowds  0.7 +{slope:.2f}/neighbour", 0.7, slope))
    for slope in (-0.05, -0.10):
        variants.append((f"tighter in crowds 0.7 {slope:.2f}/neighbour", 0.7, slope))
    variants.append(("flat 0.9", 0.9, 0.0))
    variants.append(("off", 1.01, 0.0))

    out = []
    for name, base, slope in variants:
        rr = []
        for frame, boxes in props.items():
            kept = [b for b in boxes if b[4] >= PROD_THRESH]
            kept = ds.nms_across_tiles(kept)
            kept = (ds.suppress_contained(kept, base) if slope == 0
                    else suppress_adaptive(kept, base, slope))
            kept = ds.filter_false_positive_zones(kept)
            tp, fp = match(kept, truth[frame])
            n = len(truth[frame])
            rr.append(dict(frame=frame, split=split[frame], truth=n, tp=tp, fp=fp,
                           fn=n - tp, err=len(kept) - n))
        out += summarise(pd.DataFrame(rr), name)
    t = pd.DataFrame(out)
    for s in ("held-out", "train"):
        print(f"\n-- {s} --")
        print(t[t.set == s].set_index("rule")[["precision", "recall", "mae", "signed"]]
              .round(3).to_string())
    t.to_csv(HERE / "containment_scores.csv", index=False)
    print(f"\n-> {HERE / 'containment_scores.csv'}")
    print(f"-> {HERE / 'kept_box_anatomy.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
