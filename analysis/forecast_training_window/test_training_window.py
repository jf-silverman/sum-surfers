"""
Does restricting or reweighting the TRAINING SET by time help the forecast?

Two questions Joel asked on 2026-10-13, neither previously tested:

  1. Train on only the last month or two, discarding older days outright.
  2. Weight the fall season higher than other seasons.

These are not the same as the recency weighting already rejected (L02, 16
rolling origins, 2026-09-23: exponential half-life made MAE worse at every
setting -- +0.06 at 90 days, +0.34 at 45). Exponential recency decays by
ELAPSED TIME, so it downweights autumn 2025 exactly when autumn 2025 is the
only season-matched data available. Season matching is the opposite
instruction: pull October 2025 UP, not down. A hard window is a third thing
again -- more aggressive than any half-life tried, with zero weight outside it.

Arms, all fit with the production model and feature matrix:

  all         every prior day, equal weight (the baseline, = production)
  win30/60/90 only prior days within N days of the cut
  season_hi_K rows whose calendar month is within +/-1 circular month of the
              test block's month get weight K, all others weight 1
  season_only rows outside that +/-1 month band are dropped entirely

Scored on rolling origins: at each cut day, fit on days strictly before it and
score the next TEST_BLOCK days. Paired t-test on per-origin MAE against the
baseline, because a single split of this size cannot support an MAE claim
(that is the lesson of eval_forecast_features.py).

Result, 2026-10-13, 1,904 rows / 143 days, 16 rolling origins:

    arm              MAE   vs all       p   better
    all            7.106       --      --       --
    win30          8.469   +1.704   0.049    3/14
    win60          8.114   +1.067   0.048    5/15
    win90          7.852   +0.804   0.036    6/15
    season_only    7.933   +1.168   0.064    4/14
    season_hi_1.5  7.175   +0.069   0.447    8/16
    season_hi_2    7.085   -0.022   0.794    6/16
    season_hi_2.5  7.109   +0.002   0.983    7/16
    season_hi_3    7.201   +0.095   0.419    7/16
    season_hi_5    7.260   +0.153   0.277    5/16
    anti_season_hi_2  7.207  +0.100  0.211    8/16

    placebo null (10 random reweightings): mean +0.039, sd 0.055,
                                           range -0.087 to +0.088

Verdict on question 1 (short windows): NO, and firmly. Every window is worse,
all three significantly, and they lose two origins in three. The mechanism is
legible per origin: a 30-day window at the 2026-07-07 cut scores 14.97 against
7.23, because 30 days of July is ~200 rows covering one weather regime, and a
tree cannot predict a condition it has never seen. At ~1,900 rows total,
discarding data costs far more than staleness does. These effects are +0.8 to
+1.7 MAE -- an order of magnitude above refit noise -- so they are robust.
The win30 and season_only NaNs at the March and May cuts are the same problem
in extreme form: the collection record has gaps, so those windows hold almost
nothing and no model is fitted.

Verdict on question 2 (weight fall higher): NO. Nothing here.

This arm was briefly reported as the best in-model result on this project
(-0.143 MAE, 11/16 origins, p=0.053, measured hours earlier the same day at
1,890 rows). It was wrong, and HOW it was wrong is the useful part:

  * It failed the placebo. Upweighting a RANDOM subset of the same size gives
    a null distribution of mean +0.039, sd 0.055. The seasonal arm's -0.022 is
    inside it, and 1 of 10 random reweightings did as well or better.
    Reweighting perturbs the fit; a perturbation sometimes helps by accident.
  * It failed to survive 14 new rows. The nightly pipeline added one day
    (2026-10-13) between the two runs -- 1,890 to 1,904, a 0.7% increase --
    and the gain fell from -0.143 to -0.022, from 11/16 origins to 6/16.
  * Its shape was already the tell. A sharp peak at 2x with decay on both
    sides is what noise looks like at this sample size.

It also contradicted nothing: the project had ALREADY measured that one extra
day of data moves predictions 0.78 surfers on refit noise alone. A -0.143
claim is well below that floor and should never have been ranked.

PROCEDURE, the durable lesson: any forecast arm claiming less than about
+/-0.5 MAE needs the placebo arms run alongside it BEFORE the result is
reported, and the training row count recorded next to the number so a shifted
baseline is visible rather than silent. Both are now built into this script.

Season and year also remain perfectly confounded here (October-December exist
only for 2025, March-September only for 2026), so even a surviving seasonal
effect would have been partly indistinguishable from a 2025 effect.

Usage:
    python analysis/forecast_training_window/test_training_window.py
    python analysis/forecast_training_window/test_training_window.py --origins 20
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

TEST_BLOCK = 5          # days scored after each cut
MIN_TRAIN_DAYS = 40     # a cut needs this much history to be worth scoring
GBT = dict(loss="poisson", max_iter=300, learning_rate=0.05,
           max_depth=4, l2_regularization=1.0, random_state=42)


def month_distance(m_row, m_target):
    """Circular distance in calendar months (Dec and Jan are 1 apart)."""
    d = np.abs(m_row - m_target)
    return np.minimum(d, 12 - d)


def fit_mae(X_tr, y_tr, X_te, y_te, w=None):
    if len(X_tr) < 50 or y_tr.sum() == 0:
        return np.nan
    m = HistGradientBoostingRegressor(**GBT).fit(X_tr, y_tr, sample_weight=w)
    return mean_absolute_error(y_te, m.predict(X_te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=10,
                    help="random-reweighting placebo arms (the null distribution)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    X, y, df, _ = load_and_prepare()
    X = X.drop(columns=["const"])          # tree model; no intercept column
    day = pd.to_datetime(df["date"]).dt.normalize()
    month = pd.to_datetime(df["date"]).dt.month.values

    days = np.array(sorted(day.unique()))
    # Candidate cuts: every day that leaves MIN_TRAIN_DAYS behind it and a full
    # test block ahead. Take the most recent `origins` of them, evenly spaced.
    lo, hi = MIN_TRAIN_DAYS, len(days) - TEST_BLOCK
    cuts = days[lo:hi]
    if len(cuts) > args.origins:
        cuts = cuts[np.linspace(0, len(cuts) - 1, args.origins).round().astype(int)]
    print(f"{len(df)} rows / {len(days)} days; {len(cuts)} rolling origins, "
          f"{TEST_BLOCK}-day test blocks\n")

    arms = (["all", "win30", "win60", "win90", "season_only"]
            + [f"season_hi_{k}" for k in (1.5, 2, 2.5, 3, 5)]
            + ["anti_season_hi_2"]
            + [f"placebo{s}" for s in range(args.seeds)])
    scores = {a: [] for a in arms}
    rows = []

    for cut in cuts:
        te_days = set(days[(days >= cut)][:TEST_BLOCK])
        te = np.array([d in te_days for d in day])
        tr_all = (day < cut).values
        if tr_all.sum() < 50 or te.sum() == 0:
            continue

        X_te, y_te = X[te], y[te]
        # Season band is defined by the month of the test block, which is what a
        # real forecast knows: the target date is always known in advance.
        m_target = pd.Series(month[te]).mode().iat[0]
        near = month_distance(month, m_target) <= 1

        per_arm = {}
        for arm in arms:
            w = None
            tr = tr_all.copy()
            if arm.startswith("win"):
                n = int(arm[3:])
                tr &= (day > cut - pd.Timedelta(days=n)).values
            elif arm == "season_only":
                tr &= near
            elif arm == "anti_season_hi_2":
                # Control: upweight the months FURTHEST from the target month.
                # If this helps too, the mechanism is not season.
                far = month_distance(month, m_target) >= 5
                w = np.where(far[tr], 2.0, 1.0)
            elif arm.startswith("placebo"):
                # Control: upweight a RANDOM subset of the same size as the
                # season band. Gives a null distribution for "reweight
                # something, anything" -- reweighting perturbs the fit, and a
                # perturbation sometimes helps by accident.
                rng = np.random.default_rng(1000 + int(arm[7:]))
                idx = np.where(tr)[0]
                w = np.ones(len(idx))
                w[rng.choice(len(idx), size=min(int(near[tr].sum()), len(idx)),
                             replace=False)] = 2.0
            elif arm.startswith("season_hi_"):
                k = float(arm.rsplit("_", 1)[1])
                w = np.where(near[tr], k, 1.0)
            mae = fit_mae(X[tr], y[tr], X_te, y_te, w)
            per_arm[arm] = mae
            scores[arm].append(mae)

        rows.append(dict(cut=pd.Timestamp(cut).date(), train_days=int(tr_all.sum()),
                         month=m_target, n_test=int(te.sum()), **per_arm))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))

    base = np.array(scores["all"], float)
    print(f"\n{'arm':14s} {'MAE':>6s} {'vs all':>8s} {'p':>7s} {'better':>8s}")
    print(f"{'all':14s} {np.nanmean(base):6.3f} {'--':>8s} {'--':>7s} {'--':>8s}")
    for arm in arms[1:]:
        a = np.array(scores[arm], float)
        ok = ~(np.isnan(a) | np.isnan(base))
        d = a[ok] - base[ok]
        t = stats.ttest_rel(a[ok], base[ok])
        print(f"{arm:14s} {np.nanmean(a):6.3f} {d.mean():+8.3f} {t.pvalue:7.3f} "
              f"{(d < 0).sum():4d}/{ok.sum():<3d}")
    print("\nNegative 'vs all' = better than the production baseline.")

    # The placebo arms are a null distribution, not individual results: compare
    # any small seasonal gain against their SPREAD, not against zero.
    pl = np.array([np.nanmean(np.array(scores[f"placebo{s}"], float) - base)
                   for s in range(args.seeds)])
    if len(pl):
        sd = pl.std(ddof=1) if len(pl) > 1 else float("nan")
        print(f"\nplacebo null ({args.seeds} random reweightings): "
              f"mean {pl.mean():+.3f}, sd {sd:.3f}, "
              f"range {pl.min():+.3f} to {pl.max():+.3f}")
        sh = np.nanmean(np.array(scores["season_hi_2"], float) - base)
        print(f"season_hi_2 delta {sh:+.3f} -> {int((pl <= sh).sum())}/{args.seeds} "
              f"random reweightings did as well or better")
        print("\nAny arm inside that spread has not been shown to do anything.")


if __name__ == "__main__":
    main()
