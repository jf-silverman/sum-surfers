"""Score crowd-dependent thresholds and crowd-aware duplicate suppression.

Reads the cached floored proposals from `measure_boxes_at_floor.py` and replays
the production path -- NMS across tile seams, containment suppression, the static
false-positive zones -- at any setting, scoring each result against box-level
ground truth with IoU-0.5 greedy matching.

Two rules are on trial, and they pull in opposite directions in exactly the
frames that matter:

  CROWD-DEPENDENT THRESHOLD. A fixed `CONF_THRESH` trades empty-frame precision
  against crowded-frame recall: the 263-frame hand-count re-check measured
  +0.270 count error on the 37 frames that already overcount against -0.453 on
  the 106 that undercount. Indexing the threshold on the frame's own crowd level
  is the standard answer -- Chattopadhyay et al. 2016 set per-image detector
  thresholds from a predicted count, and NumBench (2026) steps the threshold
  down from a count of 3 toward 0.3 at a count of 10. Crowd level is not known
  before thresholding, so this is a two-pass rule: count at the base threshold,
  look the threshold up from that count, re-threshold.

  CROWD-AWARE CONTAINMENT. `CONTAINMENT_THRESH` is one global constant too.
  Adaptive NMS (Liu et al., CVPR 2019) argues the suppression threshold should
  RISE with local density, because in a crowd high overlap is evidence of two
  real people rather than one person detected twice. Joel's report of boxes
  piling onto clustered surfers argues it should FALL. Both are testable: this
  scores containment as a function of each box's local neighbour count.

TRAIN CONTAMINATION. 96 of the labelled frames trained the production model.
Every table below is therefore reported train / held-out (val+test) separately,
and only the held-out column is evidence. The pooled number is printed for
continuity with earlier zone work, which did not make this split.

Usage:
    python analysis/crowd_threshold/score_crowd_rules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

HERE = Path(__file__).resolve().parent
PROD_THRESH = 0.195
# Candidate schedules: count at PROD_THRESH -> threshold to re-apply.
# Each is (name, [(min_count, thresh), ...]) read as the last matching step.
SCHEDULES = [
    ("flat 0.195 (production)", [(0, 0.195)]),
    ("flat 0.150",              [(0, 0.150)]),
    ("step  >=10 -> 0.15",      [(0, 0.195), (10, 0.150)]),
    ("step  >=20 -> 0.15",      [(0, 0.195), (20, 0.150)]),
    ("step  >=20 -> 0.14",      [(0, 0.195), (20, 0.140)]),
    ("step  >=10 ->.17 >=25 ->.14",
     [(0, 0.195), (10, 0.170), (25, 0.140)]),
    ("ramp  5->.19 15->.17 25->.15 35->.13",
     [(0, 0.210), (5, 0.190), (15, 0.170), (25, 0.150), (35, 0.130)]),
    ("tighten empties  <5 -> 0.25",
     [(0, 0.250), (5, 0.195)]),
    ("tighten+loosen  <5 ->.25 >=20 ->.15",
     [(0, 0.250), (5, 0.195), (20, 0.150)]),
]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load():
    b = pd.read_csv(HERE / "raw_boxes_at_floor.csv")
    g = pd.read_csv(HERE / "ground_truth_boxes.csv")
    props, truth, split = {}, {}, {}
    for frame, d in b.groupby("frame"):
        props[frame] = d[["x1", "y1", "x2", "y2", "conf"]].values.tolist()
        split[frame] = d.split.iloc[0]
    for frame, d in g.groupby("frame"):
        truth[frame] = d[["x1", "y1", "x2", "y2"]].values.tolist()
    return props, truth, split


def survive(boxes, thresh, containment=None, nms_iou=None):
    """The production path from a confidence threshold downward."""
    kept = [b for b in boxes if b[4] >= thresh]
    kept = ds.nms_across_tiles(kept, ds.IOU_NMS if nms_iou is None else nms_iou)
    kept = ds.suppress_contained(kept, containment)
    return ds.filter_false_positive_zones(kept)


def match(kept, truth):
    """IoU-0.5 greedy TP/FP counts, highest confidence first."""
    taken, tp = set(), 0
    for b in sorted(kept, key=lambda z: -z[4]):
        best, bi = 0.0, -1
        for i, g in enumerate(truth):
            if i in taken:
                continue
            v = iou(b, g)
            if v > best:
                best, bi = v, i
        if best >= 0.5:
            taken.add(bi)
            tp += 1
    return tp, len(kept) - tp


def sched_thresh(schedule, count):
    t = schedule[0][1]
    for lo, v in schedule:
        if count >= lo:
            t = v
    return t


def score(props, truth, split, schedule, containment=None):
    """Two-pass: count at PROD_THRESH, look up the threshold, re-score."""
    rows = []
    for frame, boxes in props.items():
        base = len(survive(boxes, PROD_THRESH, containment))
        t = sched_thresh(schedule, base)
        kept = survive(boxes, t, containment)
        tp, fp = match(kept, truth[frame])
        n = len(truth[frame])
        rows.append(dict(frame=frame, split=split[frame], truth=n, thresh=t,
                         count=len(kept), tp=tp, fp=fp, fn=n - tp,
                         err=len(kept) - n))
    return pd.DataFrame(rows)


def summarise(d, label):
    out = []
    for name, sub in (("train", d[d.split == "train"]),
                      ("held-out", d[d.split != "train"]),
                      ("pooled", d)):
        tp, fp, fn = sub.tp.sum(), sub.fp.sum(), sub.fn.sum()
        out.append(dict(rule=label, set=name, frames=len(sub),
                        precision=tp / (tp + fp) if tp + fp else np.nan,
                        recall=tp / (tp + fn) if tp + fn else np.nan,
                        mae=sub.err.abs().mean(), signed=sub.err.mean()))
    return out


def main():
    props, truth, split = load()
    print(f"{len(props)} frames, {sum(len(v) for v in props.values())} proposals, "
          f"{sum(len(v) for v in truth.values())} ground-truth boxes")
    print(pd.Series({k: sum(1 for s in split.values() if s == k)
                     for k in ('train', 'val', 'test')}).to_string())

    print("\n=== 1. Crowd-dependent confidence threshold ===")
    rows, base = [], None
    for name, sch in SCHEDULES:
        d = score(props, truth, split, sch)
        if base is None:
            base = d
        rows += summarise(d, name)
    t = pd.DataFrame(rows)
    for s in ("held-out", "train"):
        print(f"\n-- {s} --")
        v = t[t.set == s].set_index("rule")[["frames", "precision", "recall", "mae", "signed"]]
        print(v.round(3).to_string())

    print("\n=== 2. Where the fixed threshold's error sits, by crowd level ===")
    base["q"] = pd.qcut(base.truth, 4, labels=["Q1 emptiest", "Q2", "Q3", "Q4 busiest"],
                        duplicates="drop")
    for s in ("held-out", "train"):
        sub = base[base.split != "train"] if s == "held-out" else base[base.split == "train"]
        g = sub.groupby("q", observed=True).agg(
            frames=("err", "size"), truth=("truth", "mean"),
            signed=("err", "mean"), mae=("err", lambda x: x.abs().mean()),
            fp=("fp", "sum"), fn=("fn", "sum"))
        print(f"\n-- {s} --")
        print(g.round(2).to_string())

    print("\n=== 3. Containment suppression, globally ===")
    rows = []
    # 1.01 disables it: overlap/smaller-area can never exceed 1, so nothing is
    # dropped. Passing None would NOT disable it -- suppress_contained() reads
    # None as "use CONTAINMENT_THRESH", i.e. production's 0.7.
    for c in (1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        d = score(props, truth, split, [(0, PROD_THRESH)], containment=c)
        label = "containment off" if c > 1 else f"containment {c}"
        rows += summarise(d, label + (" (production)" if c == 0.7 else ""))
    t = pd.DataFrame(rows)
    for s in ("held-out", "train"):
        print(f"\n-- {s} --")
        print(t[t.set == s].set_index("rule")[["precision", "recall", "mae", "signed"]]
              .round(3).to_string())
    t.to_csv(HERE / "rule_scores.csv", index=False)
    print(f"\n-> {HERE / 'rule_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
