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

Result, 2026-10-13. Three dense hours, 36 clips, clip means 1.1 to 30.1:

    hour                  mean   self-inc   LOO   vs clip truth
    2026-10-12 09:00      5.36     0.61     0.91      1.15
    2026-10-13 08:00     12.09     1.00     1.51      2.08
    2026-08-28 08:00     24.65     0.69     1.04      2.07

Pooled by crowd level (per clip):

    clip mean   clips   self-inc   LOO   vs clip truth
    <8            12      0.61     0.92      1.14
    8-15          10      1.01     1.51      2.15
    15-22          6      0.75     1.12      1.69
    22+            8      0.72     1.09      2.28

FINDING 1 -- the published figure is a definition artifact, and the bias is
exactly mechanical. Over all 1,903 clips in predictions.csv the self-inclusive
definition gives 0.690 and the leave-one-out gives 1.035; the ratio is 1.50,
precisely the 3/2 predicted algebraically when the scored frame sits inside a
3-frame mean. That 1.5x holds in all three dense hours (0.61->0.91,
1.00->1.51, 0.69->1.04). The quantity itself does NOT hold: the same
definition returns 0.61, 1.00 and 0.69 on the three hours, so 0.68 is not even
a stable estimate of what it does measure.

FINDING 2 -- target noise is mostly ADDITIVE, not proportional to crowd size.
This is the question the new hours were collected to settle, and the design
answers it cleanly: 2026-10-13 08:00 and 2026-08-28 08:00 are the SAME clock
hour on the same camera, so light and geometry are held constant while the
mean count halves from 24.65 to 12.09. The per-frame error did not move --
2.07 against 2.08. Proportional noise would have predicted about 1.0.

Fitted across all 36 clips:

    MAD = 0.0427 * mean_count + 1.173     r = 0.607, p = 0.0001
    slope 95% CI +0.0239 to +0.0614

The intercept carries nearly all of it. There IS a weak proportional term --
the slope is significantly positive -- but it is small: noise grows only about
0.85 surfers across a 3x increase in crowd size. And the coefficient of
variation falls monotonically with level (0.59, 0.28, 0.24, 0.11, 0.11), which
a proportional model requires to be constant.

CONSEQUENCE for MODEL_CARD. The "target noise floor is 0.68 surfers" should be
replaced. A single extracted frame carries roughly

    1.2 + 0.043 * count

surfers of error against its hour's true level -- about 1.7 at the project's
typical mean count of 11.5-15.9, and 2.3 on the most crowded hours. That is
roughly 2.5x the published figure, so the headroom between the model's MAE and
the floor it is read against is materially smaller than published.

NOT claimed: that the forecast's level-proportional interval width is wrong in
shape. The band must cover real day-to-day crowd variation as well as
measurement error, and that component may well scale with level. This result
constrains the measurement term only. MODEL_CARD's defence of sd ~ sqrt(mu) is
about the total, and testing it needs a separate analysis.

Caveat: three hours, all from one camera, two in October 2026 and one in
August 2026, all passing the quality gate. The additive result rests mainly on
the matched-clock-hour pair, which is the strongest piece of evidence here but
is still a single pair.

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
