"""
Does a weekend x hour interaction help the forecast?

Background, 2026-10-17. Splitting the per-hour box plot by weekend showed the
weekend effect is not a lift on the whole day. Below 3 ft of tide, the medians:

    06:00   8 vs  8      (+0)        14:00  21 vs 35   (+14)
    07:00  15 vs 19.5  (+4.5)        15:00  23 vs 43   (+20)
    08:00  19 vs 34     (+15)        17:00  22 vs 38   (+16)
    10:00  24.5 vs 39.5 (+15)        18:00  24 vs 26    (+2)
    11:00  25 vs 34      (+9)        19:00  12 vs 12.5  (+0.5)

Dawn patrol and the last session are the same seven days a week; the whole gap
sits in the hours a weekday has to be at work. The live model carries a plain
binary `is_weekend` alongside `hour_sin`/`hour_cos`, which cannot express that
shape as a linear term -- so the question is whether handing the interaction
over explicitly buys anything.

The prior is that it does not. The model is a gradient-boosted tree
ensemble, and trees represent interactions natively by splitting on one feature
inside a branch of another. An explicit product column is a hint, not new
information. This is why the test is worth running rather than assuming either
way: the shape is clearly real in the data, and the question is purely whether
the learner needed help finding it.

ARMS (all share one target, so every arm is scored against identical truth;
only the feature matrix changes):

    base        production's 33 features                        <- baseline
    wx_trig     + is_weekend * hour_sin, is_weekend * hour_cos  (the builder in
                forecast_features.py, build_dow_features)
    wx_raw      + is_weekend * hour_local
    wx_work     + is_weekend * 1[08:00 <= hour <= 17:00]        <- the shape the
                chart actually shows
    dow_onehot  + seven day-of-week indicators + dow_idx * hour (the rest of the
                same builder, already on the MODEL_CARD's rejected list; rerun
                here so the two sit on one set of origins)

CONTROLS. Two, because they answer different questions.

    shuffleN    adds a column with the arm's exact marginal distribution
                attached to the wrong rows. The null for "add a plausible
                column, any column" -- the placebo required by the procedure
                adopted 2026-10-14 for any arm claiming under +/-0.5 MAE.

    fakewkndN   permutes the weekend LABEL across dates, then builds wx_work
                from the fake label. Stronger: it keeps the interaction's exact
                structure -- a binary flag crossed with the working-day window,
                constant within a date -- and breaks only which days are
                actually weekends. If a fake weekend does as well as the real
                one, the arm is picking up the shape of the encoding rather than
                the fact.

08:00-17:00 is the ordinary working day, fixed before looking at any MAE rather
than scanned for the best window. It is not tuned, and no other window was
tried.

RESULT, 2026-10-17, 1,943 rows / 146 days. VERDICT: NO -- and the run order is
the finding.

                16 origins        30 origins       101 origins (all)
    base          6.887             6.693              7.044
    wx_trig      -0.063            +0.075             +0.010   p=0.78  56/101
    wx_raw       -0.080            +0.046             +0.012   p=0.73  46/101
    wx_work      -0.084            +0.101             +0.029   p=0.27  43/101
    dow_onehot   +0.012            +0.006             +0.032   p=0.31  41/101

At 16 origins every interaction arm improved and cleared both controls at 0/8.
At 30 -- the same cut list, more of it -- every arm was worse, at 12/12. At all
101 they land on nothing, and 6/6 placebo seeds match `wx_work`. No arm beats
the add-a-column null; nothing is adopted.

WHY THE SIGN FLIPPED, and the number to carry forward. The per-origin delta has
a standard deviation of 0.27-0.36 MAE. The standard error of the mean is that
over sqrt(n):

    16 origins   se ~ 0.09      30 origins   se ~ 0.065
    101 origins  se ~ 0.030

So the -0.084 and the +0.101 are both within one standard error of zero, and of
each other. Sixteen origins cannot resolve an effect of 0.1 MAE -- it can only
resolve about 0.25. Resolving 0.1 needs roughly 80 origins.

This project has used 16 origins as the default (tide residual, target
rounding). Those runs concluded NO, and a null that survives a noisy test is
still a null, so no published conclusion changes. But the MAGNITUDES reported
at 16 origins carry an uncertainty of about +/-0.09, and 16 should not be used
again for anything claiming less than about 0.25 MAE.

The interaction is real in the counts -- the box plot is not in doubt, and the
weekday/weekend medians printed at the top of this run show it at every hour
from 08:00 to 17:00. The tree ensemble simply already has it: trees represent
interactions by splitting on one feature inside a branch of another, so the
product column is a hint the learner did not need.

Usage:
    python analysis/forecast_weekend_hour/test_weekend_hour.py
    python analysis/forecast_weekend_hour/test_weekend_hour.py --origins 101 --seeds 6
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
from fit_surfer_count_model import load_and_prepare  # noqa: E402

TEST_BLOCK = 5
MIN_TRAIN_DAYS = 40
WORK_LO, WORK_HI = 8, 17
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
GBT = dict(loss="poisson", max_iter=300, learning_rate=0.05,
           max_depth=4, l2_regularization=1.0, random_state=42)


def fit_mae(X_tr, y_tr, X_te, y_te):
    if len(X_tr) < 50 or y_tr.sum() == 0:
        return np.nan
    m = HistGradientBoostingRegressor(**GBT).fit(X_tr, y_tr)
    return mean_absolute_error(y_te, m.predict(X_te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    X, y, df, _ = load_and_prepare()
    X = X.drop(columns=["const"])

    hour = pd.to_numeric(df["hour_local"], errors="coerce").to_numpy(float)
    wknd = pd.to_numeric(df["is_weekend"], errors="coerce").to_numpy(float)
    dow = df["day_of_week"].astype(str).to_numpy()
    work = ((hour >= WORK_LO) & (hour <= WORK_HI)).astype(float)
    date = pd.to_datetime(df["date"]).dt.normalize()

    day = date
    days = np.array(sorted(day.unique()))
    lo, hi = MIN_TRAIN_DAYS, len(days) - TEST_BLOCK
    cuts = days[lo:hi]
    if len(cuts) > args.origins:
        cuts = cuts[np.linspace(0, len(cuts) - 1, args.origins).round().astype(int)]

    print(f"{len(df)} rows / {len(days)} days, {len(cuts)} rolling origins, "
          f"{TEST_BLOCK}-day test blocks")
    print(f"weekend rows: {int(wknd.sum())} of {len(wknd)} "
          f"({100 * wknd.mean():.1f}%); working-window rows: "
          f"{int(work.sum())} ({100 * work.mean():.1f}%)")
    gap = []
    for h in range(6, 20):
        m = hour == h
        a, b = y[m & (wknd == 0)], y[m & (wknd == 1)]
        if len(a) >= 10 and len(b) >= 10:
            gap.append((h, np.median(a), np.median(b)))
    print("  median by hour (weekday vs weekend, all tides): "
          + "  ".join(f"{h:02d}:{a:.0f}/{b:.0f}" for h, a, b in gap) + "\n")

    def fake_weekend(rng):
        """Weekend label permuted across DATES, so the fake flag is still
        constant within a day and still marks 2 days in 7."""
        uniq = np.array(sorted(day.unique()))
        flag = pd.Series(wknd, index=day.values).groupby(level=0).max()
        shuffled = rng.permutation(flag.reindex(uniq).to_numpy())
        lookup = dict(zip(uniq, shuffled))
        return np.array([lookup[d] for d in day.values], float)

    def matrix(arm, rng=None):
        Z = X.copy()
        if arm == "wx_trig":
            Z["weekend_x_hour_sin"] = wknd * np.sin(2 * np.pi * hour / 24)
            Z["weekend_x_hour_cos"] = wknd * np.cos(2 * np.pi * hour / 24)
        elif arm == "wx_raw":
            Z["weekend_x_hour"] = wknd * hour
        elif arm == "wx_work":
            Z["weekend_x_work"] = wknd * work
        elif arm == "dow_onehot":
            idx = pd.Series(dow).map({n: i for i, n in enumerate(DAYS)}).to_numpy(float)
            for n in DAYS:
                Z[f"dow_{n}"] = (dow == n).astype(float)
            Z["dow_x_hour"] = idx * hour
        elif arm.startswith("shuffle"):
            Z["weekend_x_work"] = rng.permutation(wknd * work)
        elif arm.startswith("fakewknd"):
            Z["weekend_x_work"] = fake_weekend(rng) * work
        return Z

    real = ["wx_trig", "wx_raw", "wx_work", "dow_onehot"]
    arms = (["base"] + real
            + [f"shuffle{s}" for s in range(args.seeds)]
            + [f"fakewknd{s}" for s in range(args.seeds)])
    scores = {a: [] for a in arms}

    for cut in cuts:
        te_days = set(days[days >= cut][:TEST_BLOCK])
        te = np.array([d in te_days for d in day])
        tr = (day < cut).values
        if tr.sum() < 50 or te.sum() == 0:
            continue
        for arm in arms:
            rng = None
            if arm.startswith("shuffle"):
                rng = np.random.default_rng(6000 + int(arm[7:]))
            elif arm.startswith("fakewknd"):
                rng = np.random.default_rng(7000 + int(arm[8:]))
            Z = matrix(arm, rng)
            scores[arm].append(fit_mae(Z[tr], y[tr], Z[te], y[te]))

    b = np.array(scores["base"], float)
    print(f"  {'arm':14s} {'MAE':>7s} {'vs base':>8s} {'p':>7s} {'better':>8s}   spread")
    print(f"  {'base':14s} {np.nanmean(b):7.3f} {'--':>8s} {'--':>7s} {'--':>8s}")
    deltas = {}
    for arm in arms[1:]:
        a = np.array(scores[arm], float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        deltas[arm] = np.nanmean(a) - np.nanmean(b)
        if arm in real:
            t = stats.ttest_rel(a[ok], b[ok])
            per = a[ok] - b[ok]
            print(f"  {arm:14s} {np.nanmean(a):7.3f} {deltas[arm]:+8.3f} "
                  f"{t.pvalue:7.3f} {int((a[ok] < b[ok]).sum()):4d}/{int(ok.sum()):<3d}"
                  f"  per-origin sd {per.std(ddof=1):.3f}, "
                  f"se of mean {per.std(ddof=1) / np.sqrt(len(per)):.3f}")

    sh = np.array([deltas[a] for a in arms if a.startswith("shuffle")])
    print(f"\n  add-a-column placebo (permuted interaction), {len(sh)} seeds: "
          f"mean {sh.mean():+.3f}, sd {sh.std(ddof=1):.3f}, "
          f"range {sh.min():+.3f} to {sh.max():+.3f}")
    for arm in real:
        print(f"    {arm:11s} {deltas[arm]:+.3f}  ->  "
              f"{int((sh <= deltas[arm]).sum())}/{len(sh)} placebo seeds did as well or better")

    fk = np.array([deltas[a] for a in arms if a.startswith("fakewknd")])
    print(f"\n  FAKE-WEEKEND control (label permuted across dates, same encoding),\n"
          f"  {len(fk)} seeds: mean {fk.mean():+.3f}, sd {fk.std(ddof=1):.3f}, "
          f"range {fk.min():+.3f} to {fk.max():+.3f}")
    print(f"    wx_work     {deltas['wx_work']:+.3f}  ->  "
          f"{int((fk <= deltas['wx_work']).sum())}/{len(fk)} fake-weekend seeds "
          f"did as well or better")
    print("\n  Read in this order: an arm must first beat the add-a-column null,")
    print("  then beat a fake weekend. Clearing the first alone only says the")
    print("  column is not inert; clearing the second says the WEEKEND is what")
    print("  the column is carrying.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
