"""Does a crowd-dependent confidence threshold beat the fixed 0.195?

The box-level labelled set cannot answer this. Its emptiest quartile still holds
a mean of 8.3 surfers -- the frames people bothered to annotate are the busy ones
-- so a rule indexed on crowd level has no sparse end to act on there, and the
"tighten the empties" schedules come out bit-identical to production.

The 263 hand-counted frames do have the sparse end (emptiest quartile: mean 1.1)
and `analysis/conf_thresh_recheck/counts_by_threshold.csv` already holds each
frame's count at each candidate threshold. A count-indexed schedule is therefore
pure replay: read the frame's count at 0.195, look the threshold up from it, read
the frame's count at that threshold. No inference, no new truth.

The precedent for indexing on a predicted count is Chattopadhyay et al. 2016
(per-image detector thresholds from a predicted count) and NumBench 2026 (step
the threshold down above a count of 3, reaching 0.3 at 10). What those cannot
supply is the schedule for this camera, which is what this measures.

Scored the same way as the re-check it extends: per-frame signed and absolute
count error, a paired bootstrap on the change in mean |error| against flat
0.195, and the two hand-count sets kept apart -- `spotcheck` is the only one
independent of the review the 0.150 floor came from, so it is the one a change
has to clear.

Usage:
    python analysis/crowd_threshold/score_count_schedule.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "conf_thresh_recheck" / "counts_by_threshold.csv"
BASE = "0.195"
AVAIL = ["0.195", "0.18", "0.17", "0.16", "0.15", "0.14", "0.12", "0.1"]
N_BOOT = 5000
SEED = 13

# (name, [(min_count_at_0.195, threshold), ...]); the last matching step wins.
# Thresholds must exist in AVAIL -- this is replay, not interpolation.
SCHEDULES = [
    ("flat 0.195 (production)",        [(0, "0.195")]),
    ("flat 0.150",                     [(0, "0.15")]),
    ("step >=10 -> 0.15",              [(0, "0.195"), (10, "0.15")]),
    ("step >=20 -> 0.15",              [(0, "0.195"), (20, "0.15")]),
    ("step >=30 -> 0.15",              [(0, "0.195"), (30, "0.15")]),
    ("step >=20 -> 0.12",              [(0, "0.195"), (20, "0.12")]),
    ("step >=30 -> 0.10",              [(0, "0.195"), (30, "0.1")]),
    ("ramp >=10 ->.17 >=20 ->.15 >=30 ->.12",
     [(0, "0.195"), (10, "0.17"), (20, "0.15"), (30, "0.12")]),
    ("ramp >=15 ->.17 >=25 ->.14 >=40 ->.10",
     [(0, "0.195"), (15, "0.17"), (25, "0.14"), (40, "0.1")]),
]


def sched_thresh(schedule, count):
    t = schedule[0][1]
    for lo, v in schedule:
        if count >= lo:
            t = v
    return t


def apply_schedule(g, schedule):
    """Per-frame count under the schedule, indexed on the frame's 0.195 count."""
    base = g[f"n_{BASE}"]
    return pd.Series([g[f"n_{sched_thresh(schedule, c)}"].iloc[i]
                      for i, c in enumerate(base)], index=g.index)


def main():
    d = pd.read_csv(SRC)
    print(f"{len(d)} hand-counted frames from {SRC.name}")
    print(d.groupby("set").truth.agg(["size", "mean"]).round(2).to_string())

    rng = np.random.default_rng(SEED)
    out = []
    for name, g in d.groupby("set"):
        base_err = (g[f"n_{BASE}"] - g.truth)
        print(f"\n=== {name}: {len(g)} frames, mean hand count {g.truth.mean():.1f} ===")
        print("  schedule                                 signed   |err|   ratio"
              "   d|err|      95% CI            p")
        for sname, sch in SCHEDULES:
            c = apply_schedule(g, sch)
            e = c - g.truth
            dl = (e.abs() - base_err.abs()).to_numpy(float)
            bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean()
                           for _ in range(N_BOOT)])
            p = (np.abs(bs - dl.mean()) >= abs(dl.mean())).mean() if dl.mean() else 1.0
            print(f"  {sname:<40} {e.mean():+6.2f}  {e.abs().mean():5.2f}  "
                  f"{c.sum() / g.truth.sum():.3f}  {dl.mean():+7.3f}  "
                  f"{np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}  {p:.3f}")
            out.append(dict(set=name, schedule=sname, signed=e.mean(),
                            mae=e.abs().mean(), ratio=c.sum() / g.truth.sum(),
                            d_mae=dl.mean(), p=p))

    # The sparse end is the whole question, so show it directly.
    print("\n=== By crowd quartile, pooled, for the best-looking schedules ===")
    d["q"] = pd.qcut(d.truth, 4, labels=["Q1 emptiest", "Q2", "Q3", "Q4 busiest"],
                     duplicates="drop")
    for sname, sch in SCHEDULES:
        c = apply_schedule(d, sch)
        d[f"e::{sname}"] = (c - d.truth)
    agg = {"frames": ("truth", "size"), "truth": ("truth", "mean")}
    rows = []
    for q, sub in d.groupby("q", observed=True):
        r = dict(q=q, frames=len(sub), truth=round(sub.truth.mean(), 1))
        for sname, _ in SCHEDULES:
            r[sname] = round(sub[f"e::{sname}"].abs().mean(), 3)
        rows.append(r)
    print(pd.DataFrame(rows).set_index("q").T.to_string())

    pd.DataFrame(out).to_csv(HERE / "count_schedule_scores.csv", index=False)
    print(f"\n-> {HERE / 'count_schedule_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
