"""
What is the measurement noise in the forecast TARGET, on a like-for-like basis?

MODEL_CARD publishes a "target noise floor is 0.68 surfers", defined as the
spread between one extracted frame and the 3-frame mean of its clip, over 1,750
clips, and tells the reader to judge the headline MAE against it. Two problems,
both found 2026-10-13:

  1. The frame being scored is INSIDE the mean it is scored against, which
     biases the figure down mechanically.
  2. Those 3 frames sit seconds apart, and counts decorrelate only over about
     60 seconds (r = 0.83 at 1s lag, 0.58 at 10s, 0.03 at 60s). So a 3-frame
     mean of adjacent frames is barely more informative than one frame -- it is
     not a stand-in for the clip's true level.

Together they measure short-timescale detector jitter, not the error in the
target. The target in predictions.csv is ONE frame standing for an hour.

This script computes three definitions on IDENTICAL draws from the dense
per-second hours, so the gap is attributable to the definition alone and not to
different data or crowd levels:

  self_inclusive  frame vs the 3-frame mean containing it   (the published one)
  leave_one_out   frame vs the mean of the OTHER 2 frames   (unbiased, but
                  still only a 2-frame reference seconds away)
  vs_clip_truth   frame vs the clip's full per-second mean  (what we want)

It also reports each against crowd level, because noise does scale with level
and a project-wide floor has to say at what level it applies.

Result, 2026-10-13, on 2026-08-28 08:00-09:00 (3,614 frames, mean 24.65):

    frame vs 3-frame mean containing it   0.70   <- reproduces the published 0.68
    frame vs mean of the other 2 frames   1.05
    frame vs the clip true 300s mean      2.08

The published figure reproduces almost exactly on an hour whose mean count is
24.7, well above the 11.5-15.9 the project usually sees. So 0.68 is NOT a
low-crowd artifact -- it is what that definition yields, and the like-for-like
number at this level is about 3x larger.

Consequence: 0.68 does not bound forecast accuracy and should not be quoted as
the floor the MAE is read against. Pending more dense hours at other crowd
levels, the defensible statement is that a single frame carries roughly 2
surfers of error against its clip's true level at a mean count near 25.

Usage:
    python analysis/count_variability/noise_floor/derive_noise_floor.py
    python analysis/count_variability/noise_floor/derive_noise_floor.py --draws 8000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DENSE_DIR = HERE.parent / "hour_dense"
TRIO_GAP = 1          # seconds between the 3 frames, matching clip extraction


def dense_sources():
    """Every per-second dense-hour CSV, so new hours are picked up automatically."""
    out = []
    for p in sorted(DENSE_DIR.glob("hourly_variability_full*.csv")):
        d = pd.read_csv(p)
        if {"clip_time", "second", "count"} <= set(d.columns):
            out.append((p.name, d))
    return out


def analyse(d, draws, rng):
    rows = []
    for clip, g in d.groupby("clip_time"):
        v = g.sort_values("second")["count"].to_numpy()
        if len(v) < 3 * TRIO_GAP + 3:
            continue
        truth = v.mean()
        span = len(v) - 2 * TRIO_GAP - 1
        idx = rng.integers(0, span, size=draws)
        trio = np.stack([v[idx], v[idx + TRIO_GAP], v[idx + 2 * TRIO_GAP]])
        f = trio[0]
        rows.append(dict(
            clip_time=clip, mean_count=truth,
            self_inclusive=np.abs(f - trio.mean(axis=0)).mean(),
            leave_one_out=np.abs(f - trio[1:].mean(axis=0)).mean(),
            vs_clip_truth=np.abs(f - truth).mean(),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    srcs = dense_sources()
    if not srcs:
        raise SystemExit(f"no dense-hour CSVs found in {DENSE_DIR}")

    all_rows = []
    for name, d in srcs:
        res = analyse(d, args.draws, rng)
        res["source"] = name
        all_rows.append(res)
        n_frames = len(d)
        print(f"\n=== {name}: {n_frames} frames, {res.clip_time.nunique()} clips, "
              f"mean count {d['count'].mean():.2f} ===")
        print(f"  {'frame vs 3-frame mean containing it':38s} "
              f"{res.self_inclusive.mean():.2f}   <- the published 0.68 definition")
        print(f"  {'frame vs mean of the other 2 frames':38s} {res.leave_one_out.mean():.2f}")
        print(f"  {'frame vs the clip true per-second mean':38s} {res.vs_clip_truth.mean():.2f}")

    res = pd.concat(all_rows, ignore_index=True)

    print("\n=== by crowd level (per clip, pooled over sources) ===")
    bands = pd.cut(res.mean_count, [0, 8, 15, 22, 100],
                   labels=["<8", "8-15", "15-22", "22+"])
    by = res.groupby(bands, observed=True).agg(
        clips=("clip_time", "size"), mean_count=("mean_count", "mean"),
        self_inclusive=("self_inclusive", "mean"),
        leave_one_out=("leave_one_out", "mean"),
        vs_clip_truth=("vs_clip_truth", "mean"))
    print(by.round(2).to_string())
    if len(by) > 1:
        print("\nIf vs_clip_truth tracks mean_count proportionally, target noise is a")
        print("RATE and the forecast's level-proportional bands are right in shape.")
        print("If it is flat, the noise is additive and the band shape is wrong.")
    else:
        print("\nOnly one crowd band present -- collect dense hours at other levels")
        print("before stating a project-wide floor.")


if __name__ == "__main__":
    main()
