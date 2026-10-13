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

Result, 2026-10-13, 1,890 rows / 142 days, 16 rolling origins:

    arm              MAE   vs all       p   better
    all            7.112       --      --       --
    win30          7.928   +1.157   0.129    6/14
    win60          8.136   +1.083   0.042    5/15
    win90          7.948   +0.895   0.033    4/15
    season_only    7.887   +1.116   0.079    4/14
    season_hi_1.5  7.031   -0.080   0.254   10/16
    season_hi_2    6.969   -0.143   0.053   11/16
    season_hi_2.5  7.026   -0.086   0.319   10/16
    season_hi_3    7.076   -0.036   0.721    8/16
    season_hi_5    7.122   +0.010   0.939    7/16

Verdict on question 1: NO. Every hard window is worse, win60 and win90
significantly so, and they lose in two origins out of three. The direction is
the same as the rejected exponential recency weighting but larger, which is
what a harsher version of the same instruction should produce. The reason is
visible in the table: a 30-day window at the 2026-07-07 cut scores 14.97
against 7.23, because 30 days of July is ~200 rows covering one weather regime.
Throwing away data costs more than staleness does at this sample size. The
win30/season_only NaNs at the March and May cuts are the same problem in its
extreme form -- the collection record has gaps, so those windows hold almost
nothing and no model is fitted.

Verdict on question 2: WORTH WATCHING, NOT WORTH SHIPPING. Doubling the weight
on rows within +/-1 calendar month of the target month is the best in-model
result this project has measured (previous best: day-level aggregates, -0.09)
and wins 11 of 16 origins, but -0.143 MAE at p=0.053 is a tenth of the 0.5 MAE
adoption bar. It peaks at 2x and fades at 1.5x and 2.5x; at this sample size a
sharp peak flanked by decay is as consistent with noise as with a real optimum.
Re-run it when autumn 2026 is complete -- the hypothesis predicts the gain
should GROW as the season-matched band fills, since the +/-1 month band for an
October target currently draws mostly on October-November 2025.

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
            + [f"season_hi_{k}" for k in (1.5, 2, 2.5, 3, 5)])
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


if __name__ == "__main__":
    main()
