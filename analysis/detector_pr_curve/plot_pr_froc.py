"""
Precision-recall and FROC curves for the surfer detector, across every
confidence threshold (2026-10-26).

Why not ROC. An ROC curve needs a false positive RATE, i.e. a count of true
negatives, and a detector has none to count: every patch of empty water is a
candidate "correct rejection". The detection-world equivalents are:

  precision-recall   precision against recall as the threshold moves; its area
                     is average precision (AP), the standard single number
  FROC               free-response ROC (medical imaging): recall against false
                     boxes per frame, i.e. ROC with the FP rate replaced by a
                     rate per image

Boxes come from analysis/crowd_threshold/raw_boxes_at_floor.csv: every
proposal on the 172 hand-labelled frames down to confidence 0.02, before
cross-tile NMS. Each threshold replays the production path offline (threshold,
NMS, contained-box suppression, the fixed false-positive zones), so the curve
is the pipeline's, not the raw network's. Only held-out frames are scored (val
+ test splits of the production model, 61 frames, 1,509 labelled surfers).

A box matches a labelled surfer at IoU >= 0.5 (the usual detection standard)
or, as a looser second curve, IoU >= 0.3: surfers here are 10-20 px boxes,
where a few pixels of offset cost a lot of IoU without the box being on the
wrong surfer.

Writes pr_froc.png and pr_froc_thresholds.csv next to this script.

Usage:
    python analysis/detector_pr_curve/plot_pr_froc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "analysis" / "crowd_threshold"))
import detect_surfers as ds  # noqa: E402
from score_crowd_rules import iou, load  # noqa: E402

BG, INK, INK_DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA, LIME = "#3ab4c9", "#9de35a"
THRESHOLDS = np.round(np.concatenate([np.arange(0.02, 0.10, 0.01), np.arange(0.10, 0.951, 0.025)]), 3)
PROD = ds.CONF_THRESH


def survive(boxes, t):
    kept = ds.nms_across_tiles([b for b in boxes if b[4] >= t], ds.IOU_NMS)
    return ds.filter_false_positive_zones(ds.suppress_contained(kept))


def tp_count(kept, truth, thr):
    taken, tp = set(), 0
    for b in sorted(kept, key=lambda z: -z[4]):
        best, bi = 0.0, -1
        for i, g in enumerate(truth):
            if i not in taken:
                v = iou(b, g)
                if v > best:
                    best, bi = v, i
        if best >= thr:
            taken.add(bi)
            tp += 1
    return tp


def sweep(props, truth, frames, thr):
    rows = []
    n_true = sum(len(truth.get(f, [])) for f in frames)
    for t in THRESHOLDS:
        tp = fp = cnt_err = 0
        for f in frames:
            kept = survive(props.get(f, []), t)
            k = tp_count(kept, truth.get(f, []), thr)
            tp += k
            fp += len(kept) - k
            cnt_err += abs(len(kept) - len(truth.get(f, [])))
        rows.append({"iou": thr, "threshold": t, "tp": tp, "fp": fp, "fn": n_true - tp,
                     "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / n_true,
                     "fp_per_frame": fp / len(frames), "count_mae": cnt_err / len(frames)})
    return pd.DataFrame(rows)


def ap(d):
    """Area under the PR curve, all-point interpolation (precision made
    monotone from the right, then integrated over recall)."""
    d = d.sort_values("recall")
    r = np.concatenate([[0], d.recall.to_numpy(), [d.recall.max()]])
    p = np.concatenate([[d.precision.iloc[0]], d.precision.to_numpy(), [0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum(np.diff(r) * p[1:]))


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9.5)
    ax.grid(color=GRID, linewidth=0.8)


def main():
    props, truth, split = load()
    frames = sorted(f for f in set(truth) | set(props) if split.get(f) in ("val", "test")
                    or (f not in split and f in truth))
    gt = pd.read_csv(ROOT / "analysis/crowd_threshold/ground_truth_boxes.csv")
    held = sorted(gt[gt.split.isin(["val", "test"])].frame.unique())
    frames = held
    print(f"{len(frames)} held-out frames, {sum(len(truth.get(f, [])) for f in frames)} labelled surfers")
    res = pd.concat([sweep(props, truth, frames, thr) for thr in (0.5, 0.3)])
    res.round(4).to_csv(HERE / "pr_froc_thresholds.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=BG)
    for ax in (a1, a2):
        style(ax)
    for thr, color, ls in [(0.5, AQUA, "-"), (0.3, LIME, (0, (6, 4)))]:
        d = res[res.iou == thr]
        a = ap(d)
        lab = f"match at IoU ≥ {thr}  (AP = {a:.3f})"
        a1.plot(d.recall, d.precision, color=color, linewidth=2.0, linestyle=ls, label=lab, zorder=3)
        a2.plot(d.fp_per_frame, d.recall, color=color, linewidth=2.0, linestyle=ls, label=lab, zorder=3)
        pr = d.iloc[(d.threshold - PROD).abs().argmin()]
        for ax, x, y in [(a1, pr.recall, pr.precision), (a2, pr.fp_per_frame, pr.recall)]:
            ax.scatter([x], [y], s=70, color=color, edgecolors=BG, linewidths=2, zorder=4)
        print(f"IoU {thr}: AP {a:.3f}; at production threshold {PROD}: precision {pr.precision:.3f}, "
              f"recall {pr.recall:.3f}, {pr.fp_per_frame:.2f} false boxes per frame, count MAE {pr.count_mae:.2f}")
    a1.set_xlim(0, 1.0); a1.set_ylim(0, 1.02)
    a1.set_xlabel("recall: share of labelled surfers found", color=INK, fontsize=10.5)
    a1.set_ylabel("precision: share of boxes on a surfer", color=INK, fontsize=10.5)
    a1.set_title("Precision vs recall, every threshold", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left")
    a2.set_xlim(0, res.fp_per_frame.max() * 1.04)
    a2.set_ylim(0, 1.02)
    a2.set_xlabel("false boxes per frame", color=INK, fontsize=10.5)
    a2.set_ylabel("recall: share of labelled surfers found", color=INK, fontsize=10.5)
    a2.set_title("FROC: surfers found vs false boxes", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left")
    for ax, loc in [(a1, "lower left"), (a2, "lower right")]:
        h, l = ax.get_legend_handles_labels()
        h.append(plt.Line2D([], [], marker="o", color=INK_DIM, linestyle="", markersize=8))
        l.append(f"production threshold ({PROD})")
        ax.legend(h, l, loc=loc, fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=INK)
    fig.suptitle(f"Surfer detector across confidence thresholds 0.02-0.95  ·  "
                 f"{len(frames)} held-out frames, {sum(len(truth.get(f, [])) for f in frames):,} labelled surfers",
                 color=INK_DIM, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "pr_froc.png", facecolor=BG, dpi=150)
    print(f"wrote {HERE / 'pr_froc.png'}")


if __name__ == "__main__":
    main()
