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

estimator_contest() answers a separate question: would a median beat the mean,
the median being the standard defence against outliers? It does not, at any
configuration tested, and the tail statistics say why.

Result, 2026-10-14. Three dense hours, 36 clips, clip means 1.1 to 30.1:

    hour                mean   self-inc   LOO   1 frame   PRODUCTION TARGET
    2026-10-12 09:00    5.36     0.61     0.91    1.15          1.00
    2026-10-13 08:00   12.09     1.00     1.51    2.08          1.82
    2026-08-28 08:00   24.65     0.69     1.04    2.07          1.96

Pooled by crowd level (per clip):

    clip mean   clips   1 frame   PRODUCTION TARGET
    <8            12      1.14          0.98
    8-15          10      2.15          1.91
    15-22          6      1.69          1.51
    22+            8      2.28          2.17

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
mean count halves from 24.65 to 12.09. Single-frame error did not move --
2.07 against 2.08. Proportional noise would have predicted about 1.0.

Fitted across all 36 clips, on the production target:

    MAE = 0.0454 * mean_count + 0.954     r = 0.643, p < 0.0001
    slope 95% CI +0.0272 to +0.0636

The intercept carries most of it. A weak proportional term exists -- the slope
is significantly positive -- but noise grows only about 1.2 surfers across the
full 1-to-30 range of clip means, and the coefficient of variation falls
monotonically with level (0.59, 0.28, 0.24, 0.11, 0.11) where a proportional
model requires it constant.

FINDING 3 -- multi-frame averaging buys much less than sqrt(3). Production
extracts 3 frames at 1.0/2.5/4.0s and trains on round(frame_count_mean), so
the target is a rounded 3-frame mean over a 3-second span. Against the clip's
true level that scores 1.59 pooled, versus 2.00 for a single frame -- a 20%
reduction where independent samples would give 42%. The reason is in the
autocorrelation: counts 1-3s apart correlate at r = 0.83 to 0.77, so the three
frames are nearly the same observation. Spreading them across the full clip
instead would cut single-frame error from 2.07 to 1.17 on the 2026-08-28 hour.
That is a real and cheap improvement, logged in the ideas doc, not adopted
here -- it needs a re-extraction and a re-detection pass over the archive.

CONSEQUENCE for MODEL_CARD. The "target noise floor is 0.68 surfers" is
replaced by about 1.5 at the project's typical mean count of 13, from the
production-target fit above: roughly 1.0 on a quiet hour and 2.2 on a crowded
one. That is about 2.3x the published figure, so the headroom between the
model's MAE and the floor it is read against is materially smaller than
published.

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
        # The quantity that actually matters: production extracts 3 frames at
        # 1.0/2.5/4.0s (get_cropped_frame.py FRAME_TIME_SEC 2.5,
        # SIDE_FRAME_OFFSET_SEC 1.5) and build_training_features.py trains on
        # round(frame_count_mean). So the forecast target is a ROUNDED 3-FRAME
        # MEAN over a 3-second span, not one frame. At 1 fps the two integer
        # windows spanning 3s bracket production's middle frame; they agree, so
        # the exact placement does not matter.
        prod = []
        for off in ((0, 1, 3), (0, 2, 3)):
            lim = len(v) - max(off)
            m = np.stack([v[np.arange(lim) + o] for o in off]).mean(axis=0)
            prod.append(np.abs(np.round(m) - truth).mean())

        rows.append(dict(
            clip_time=clip, mean_count=truth,
            self_inclusive=np.abs(f - trio.mean(axis=0)).mean(),
            leave_one_out=np.abs(f - trio[1:].mean(axis=0)).mean(),
            vs_clip_truth=np.abs(f - truth).mean(),
            production_target=float(np.mean(prod)),
        ))
    return pd.DataFrame(rows)


def estimator_contest(srcs, draws, rng):
    """Mean vs median, and whether the within-clip tails are heavy enough to
    justify a robust estimator.

    Asked because the median is the standard defence against outliers. Three
    configurations: the production trio (3 frames over a 3s span), and k frames
    spread across the whole clip (one drawn from each of k equal segments), which
    is the alternative logged in the ideas doc. Scored against each clip's own
    true per-second mean, so this measures precision, not robustness to a
    detector failure that corrupts a whole clip -- the quality gate handles that
    case and every frame here passed it.
    """
    per_clip, resid = [], []
    for name, d in srcs:
        for clip, g in d.groupby("clip_time"):
            v = g.sort_values("second")["count"].to_numpy().astype(float)
            if len(v) < 12:
                continue
            truth = v.mean()
            row = {"source": name, "clip_time": clip, "mean_count": truth}

            # production trio: both integer windows spanning 3s, averaged
            for tag, fn in (("mean", np.mean), ("med", np.median)):
                errs = []
                for off in ((0, 1, 3), (0, 2, 3)):
                    lim = len(v) - max(off)
                    stack = np.stack([v[np.arange(lim) + o] for o in off])
                    errs.append(np.abs(np.round(fn(stack, axis=0)) - truth).mean())
                row[f"{tag}3"] = float(np.mean(errs))

            # k frames spread across the clip, one per equal segment
            for k in (3, 5):
                edges = np.linspace(0, len(v), k + 1).astype(int)
                pick = np.stack([rng.integers(edges[i], edges[i + 1], size=draws)
                                 for i in range(k)])
                stack = v[pick]
                row[f"mean{k}_spread"] = float(
                    np.abs(np.round(stack.mean(axis=0)) - truth).mean())
                row[f"med{k}_spread"] = float(
                    np.abs(np.round(np.median(stack, axis=0)) - truth).mean())

            per_clip.append(row)
            sd = v.std(ddof=1)
            if sd > 0:
                resid.append((v - truth) / sd)

    e = pd.DataFrame(per_clip)
    from scipy import stats  # noqa: PLC0415

    print(f"\n=== mean vs median, {len(e)} clips, MAE against each clip's true level ===")
    for a, b in (("mean3", "med3"), ("mean3_spread", "med3_spread"),
                 ("mean5_spread", "med5_spread")):
        t = stats.ttest_rel(e[b], e[a])
        wins = int((e[b] < e[a]).sum())
        print(f"  {a:14s} {e[a].mean():.3f}    {b:14s} {e[b].mean():.3f}"
              f"    median-mean {e[b].mean() - e[a].mean():+.3f}"
              f"  paired p={t.pvalue:.2e}  median better in {wins}/{len(e)}")

    z = np.concatenate(resid)
    print(f"\n=== within-clip tail shape, {len(z)} frames pooled ===")
    print(f"  excess kurtosis {stats.kurtosis(z):+.3f}   (0 = normal tails)")
    print(f"  skew            {stats.skew(z):+.3f}")
    for c, expect in ((3, 0.270), (4, 0.0063)):
        print(f"  |z|>{c} rate      {100 * (np.abs(z) > c).mean():.4f}%"
              f"   (normal expects {expect}%)")
    print("  Normal-ish tails leave nothing for the median to protect against,")
    print("  while the median of 3 discards two thirds of the information.")
    return e


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
        print(f"  {'PRODUCTION TARGET vs that true mean':38s} {res.production_target.mean():.2f}"
              f"   <- round(3-frame mean), what the forecast trains on")

    res = pd.concat(all_rows, ignore_index=True)

    print("\n=== by crowd level (per clip, pooled over sources) ===")
    bands = pd.cut(res.mean_count, [0, 8, 15, 22, 100],
                   labels=["<8", "8-15", "15-22", "22+"])
    by = res.groupby(bands, observed=True).agg(
        clips=("clip_time", "size"), mean_count=("mean_count", "mean"),
        self_inclusive=("self_inclusive", "mean"),
        leave_one_out=("leave_one_out", "mean"),
        vs_clip_truth=("vs_clip_truth", "mean"),
        production_target=("production_target", "mean"))
    print(by.round(2).to_string())

    from scipy import stats  # noqa: PLC0415
    for col in ("vs_clip_truth", "production_target"):
        r = stats.linregress(res.mean_count, res[col])
        print(f"\n{col}: MAE = {r.slope:.4f} * mean_count + {r.intercept:.3f}"
              f"   r={r.rvalue:.3f} p={r.pvalue:.4f}")
        print(f"  slope 95% CI {r.slope - 1.96*r.stderr:+.4f} to {r.slope + 1.96*r.stderr:+.4f}"
              f"; at the project's typical mean count of 13 -> {r.intercept + r.slope*13:.2f}")
    estimator_contest(srcs, args.draws, rng)

    if len(by) > 1:
        print("\nIf vs_clip_truth tracks mean_count proportionally, target noise is a")
        print("RATE and the forecast's level-proportional bands are right in shape.")
        print("If it is flat, the noise is additive and the band shape is wrong.")
    else:
        print("\nOnly one crowd band present -- collect dense hours at other levels")
        print("before stating a project-wide floor.")


if __name__ == "__main__":
    main()
