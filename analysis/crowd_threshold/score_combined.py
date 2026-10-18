"""The crowd-dependent threshold and adaptive containment together, on all three
truth protocols at once.

The point of this file is that there are THREE truth sources, not two, and they
do not line up the way the earlier re-check assumed:

  box-level CVAT annotation   61 held-out labelled frames, boxes drawn by hand
  `reviewdays`                220 frames, truth = true_positives + missed from a
                              box-by-box review of the 0.195 boxes
  `spotcheck`                 43 frames, truth = a direct `my_count` of the scene

`analysis/conf_thresh_recheck/` treated `spotcheck` as the set a change must
clear, because it is independent of the review the 0.150 floor came from. That
reasoning missed that box-level annotation is independent of BOTH -- and at
matched crowd levels it agrees with `reviewdays` almost band for band while
`spotcheck` sits apart. All three ratios below are plain count/truth, so no IoU
matching enters the comparison.

Both rules are scored on all three. A change that moves two independent
protocols the same way is a different proposition from one that moves a single
set.

Usage:
    python analysis/crowd_threshold/score_combined.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "code"))
sys.path.insert(0, str(HERE))
import detect_surfers as ds  # noqa: E402
from score_crowd_rules import load, match, sched_thresh  # noqa: E402
from score_duplicate_rules import suppress_adaptive  # noqa: E402

PROD_THRESH = 0.195
N_BOOT = 5000
SEED = 23

# (name, schedule, containment base, containment slope)
RULES = [
    ("production (0.195, containment flat 0.7)", [(0, 0.195)], 0.7, 0.0),
    ("threshold only  ramp 10/20/30", [(0, 0.195), (10, 0.17), (20, 0.15), (30, 0.12)], 0.7, 0.0),
    ("containment only  0.7 +0.10/nb", [(0, 0.195)], 0.7, 0.10),
    ("both  ramp + 0.7 +0.10/nb",
     [(0, 0.195), (10, 0.17), (20, 0.15), (30, 0.12)], 0.7, 0.10),
    ("both, gentler  ramp 15/25/40 + 0.7 +0.05/nb",
     [(0, 0.195), (15, 0.17), (25, 0.14), (40, 0.10)], 0.7, 0.05),
]


def pipeline(boxes, thresh, base, slope):
    kept = [b for b in boxes if b[4] >= thresh]
    kept = ds.nms_across_tiles(kept)
    kept = (ds.suppress_contained(kept, base) if slope == 0
            else suppress_adaptive(kept, base, slope))
    return ds.filter_false_positive_zones(kept)


def apply_rule(boxes, schedule, base, slope):
    """Two-pass: count at production settings, look up the threshold, re-run."""
    n0 = len(pipeline(boxes, PROD_THRESH, base, slope))
    return pipeline(boxes, sched_thresh(schedule, n0), base, slope)


def main():
    props, gt, split = load()
    with (HERE / "handcount_proposals.pkl").open("rb") as fh:
        c = pickle.load(fh)
    hc, store = c["frames"], c["boxes"]

    rng = np.random.default_rng(SEED)
    out = []

    # --- box-level truth, held-out frames only ---
    ho = [f for f in props if split[f] != "train"]
    print(f"=== box-level CVAT truth: {len(ho)} held-out labelled frames, "
          f"mean truth {np.mean([len(gt[f]) for f in ho]):.1f} ===")
    print("  rule                                          signed   |err|  ratio"
          "   prec  recall   d|err|      95% CI            p")
    base_err = None
    for name, sch, b0, sl in RULES:
        rows = []
        for f in ho:
            kept = apply_rule(props[f], sch, b0, sl)
            tp, fp = match(kept, gt[f])
            rows.append((len(kept) - len(gt[f]), len(gt[f]), len(kept), tp, fp))
        e = np.array([r[0] for r in rows], float)
        tr = sum(r[1] for r in rows); cn = sum(r[2] for r in rows)
        tp = sum(r[3] for r in rows); fp = sum(r[4] for r in rows)
        fn = tr - tp
        if base_err is None:
            base_err = e
        dl = np.abs(e) - np.abs(base_err)
        bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(N_BOOT)])
        p = (np.abs(bs - dl.mean()) >= abs(dl.mean())).mean() if dl.mean() else 1.0
        print(f"  {name:<44} {e.mean():+6.2f}  {np.abs(e).mean():5.2f}  {cn/tr:.3f}  "
              f"{tp/(tp+fp):.3f}  {tp/(tp+fn):.3f}  {dl.mean():+7.3f}  "
              f"{np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f}  {p:.3f}")
        out.append(dict(truth="box-level (held-out)", rule=name, signed=e.mean(),
                        mae=np.abs(e).mean(), ratio=cn/tr, precision=tp/(tp+fp),
                        recall=tp/(tp+fn), d_mae=dl.mean(), p=p))

    # --- the two count-level sets ---
    counts = {name: [len(apply_rule(store[f], sch, b0, sl)) for f in hc.filename]
              for name, sch, b0, sl in RULES}
    for name in counts:
        hc[name] = counts[name]
    for sname, g in hc.groupby("set"):
        prod = RULES[0][0]
        base_err = (g[prod] - g.truth).to_numpy(float)
        print(f"\n=== {sname}: {len(g)} frames, mean hand count {g.truth.mean():.1f} ===")
        print("  rule                                          signed   |err|  ratio"
              "   d|err|      95% CI            p")
        for name, _, _, _ in RULES:
            e = (g[name] - g.truth).to_numpy(float)
            dl = np.abs(e) - np.abs(base_err)
            bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(N_BOOT)])
            p = (np.abs(bs - dl.mean()) >= abs(dl.mean())).mean() if dl.mean() else 1.0
            print(f"  {name:<44} {e.mean():+6.2f}  {np.abs(e).mean():5.2f}  "
                  f"{g[name].sum()/g.truth.sum():.3f}  {dl.mean():+7.3f}  "
                  f"{np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f}  {p:.3f}")
            out.append(dict(truth=sname, rule=name, signed=e.mean(),
                            mae=np.abs(e).mean(), ratio=g[name].sum()/g.truth.sum(),
                            precision=np.nan, recall=np.nan, d_mae=dl.mean(), p=p))

    print("\n=== Sparse-end check: mean |count error| on the emptiest frames ===")
    for lo, hi in ((0, 2), (3, 5)):
        s = hc[(hc.truth >= lo) & (hc.truth <= hi)]
        print(f"  hand count {lo}-{hi} (n={len(s)}): " + "  ".join(
            f"{n.split()[0]}={((s[n]-s.truth).abs().mean()):.3f}" for n, _, _, _ in RULES))

    pd.DataFrame(out).to_csv(HERE / "combined_scores.csv", index=False)
    print(f"\n-> {HERE / 'combined_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
