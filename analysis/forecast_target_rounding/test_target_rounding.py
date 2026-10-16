"""
Does training the forecast on the UNROUNDED 3-frame mean help?

Joel asked on 2026-10-15, after the noise-floor work established what the
target actually is. Production trains on `round(frame_count_mean)`
(build_training_features.py's resolve_target_count), and the rounding was never
a measured choice -- it is there because detect_surfers.py publishes an integer
count and the features table copied it.

The rounding destroys real information. A 3-frame mean of integers lands on a
third, so over the 1,924 rows carrying frame_count_mean:

    42.9%  mean is already an integer      (rounding is a no-op)
    57.0%  mean sits on a third            (rounding moves it 0.33)
     0.2%  exactly 0.5                     (moves it 0.5)
    mean |round(m) - m| = 0.189

So 57% of rows have a sub-integer level that the target throws away. Whether
recovering it helps is a different question: the information is small against
the ~1.5-surfer measurement floor in the target itself, and a Poisson objective
is built for counts.

DESIGN -- the trap this script exists to avoid. An arm trained on a different
target cannot be scored against its own target: "MAE against the unrounded
mean" and "MAE against the rounded count" are different quantities and the
smoother one is mechanically easier to hit. Every arm here is therefore scored
against the SAME held-out truth, and both choices of truth are reported:

    vs exact   the unrounded frame_count_mean -- the best available estimate of
               the hour's level, and the thing a forecast should want to be near
    vs round   round(frame_count_mean) -- what production publishes

Only the TRAINING target varies between arms.

Arms:
    round_poisson    y = round(mean), poisson loss     <- production baseline
    exact_poisson    y = mean,        poisson loss
    round_squared    y = round(mean), squared error
    exact_squared    y = mean,        squared error
    shuffleN         y = round(mean) - permuted rounding deltas, poisson

The shuffle arms are the placebo required by the procedure adopted on
2026-10-14 (see analysis/forecast_training_window/): any arm claiming less than
about +/-0.5 MAE runs placebo arms alongside it before the result is reported.
Write m for the exact mean, r = round(m), d = r - m. The exact target is r - d.
A shuffle arm trains on r - d_permuted: the same marginal distribution of
perturbation off the rounded value, assigned to the wrong rows. If unrounding
wins by the same margin the shuffles do, the mechanism is "perturb the target",
not "recover the sub-integer level".

Scored on rolling origins, paired t-test per origin against the baseline, and
the training row count is printed, both for the same reason as the
training-window script: a single split of this size cannot support an MAE
claim, and one extra day of data moves predictions 0.78 surfers on refit noise
alone.

RESULT, 2026-10-15, 1,917 rows / 144 days, 16 rolling origins:

    arm              MAE*   vs base       p   better
    round_poisson   6.927       --      --       --     <- production
    exact_poisson   6.992   +0.066   0.253    7/16
    round_squared   7.355   +0.428   0.046    3/16
    exact_squared   7.379   +0.453   0.032    5/16

    shuffled-delta placebo, 8 seeds: mean +0.078, sd 0.061,
                                     range -0.013 to +0.138
    exact_poisson +0.066 -> 3/8 placebo seeds did as well or better

    * scored against the exact target; the rounded-target table is within
      0.01 of this one throughout, so the choice of truth changes nothing.

VERDICT: NO. Unrounding does not help -- it is slightly worse (+0.066), not
significant (p = 0.25), and wins only 7 of 16 origins. The placebo settles it:
randomly permuting the same rounding deltas among the training rows gives a
null of mean +0.078, sd 0.061, and the exact arm sits inside that distribution
with 3 of 8 random seeds doing as well or better. Whatever unrounding does to
the fit is indistinguishable from perturbing the target by the same amount in
the wrong places, so there is no evidence the sub-integer level is being used.

Why this is the expected answer in hindsight: the information rounding destroys
is 0.188 surfers of mean displacement, against a target whose own measurement
noise is about 1.5 surfers at typical crowd levels
(analysis/count_variability/noise_floor/). Recovering a 0.19 detail of a
quantity that is itself +/-1.5 is below the noise, and far below the 0.78-surfer
refit noise the project already measured. Production keeps round().

A real finding came out of the control arms instead: SQUARED ERROR IS CLEARLY
WORSE THAN POISSON, +0.43 MAE on both targets, p < 0.05, winning 3 of 16
origins. That is the only significant effect in this table and it is an order
of magnitude above the rounding question.

analysis/surf_count_model_selection/ had already ranked the losses in September
(Poisson 6.238, squared_error 6.461, absolute_error 6.585), so the direction was
known. What it could not say is whether the ranking held on a correct split: it
used the random-hour split later found to leak, and reported no significance
test. Both are supplied here, and the gap is wider on a forward split. Gamma
and Tweedie were not tested: HistGradientBoostingRegressor's gamma loss needs a
strictly positive target and about 15% of hours are empty.

Usage:
    python analysis/forecast_target_rounding/test_target_rounding.py
    python analysis/forecast_target_rounding/test_target_rounding.py --origins 20 --seeds 10
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

PREDICTIONS_CSV = _ROOT / "data" / "predictions" / "predictions.csv"
TEST_BLOCK = 5          # days scored after each cut
MIN_TRAIN_DAYS = 40     # a cut needs this much history to be worth scoring
GBT = dict(max_iter=300, learning_rate=0.05, max_depth=4,
           l2_regularization=1.0, random_state=42)


def exact_means(df):
    """The unrounded frame_count_mean for each feature row, joined on filename.

    The features table stores only the rounded value, so the exact one has to
    come back from predictions.csv. Returns a float Series aligned to df, NaN
    where the row predates multi-frame extraction.
    """
    p = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    m = (p.assign(fcm=pd.to_numeric(p["frame_count_mean"], errors="coerce"))
          .dropna(subset=["fcm"])
          .drop_duplicates(subset="filename", keep="last")
          .set_index("filename")["fcm"])
    return df["filename"].map(m)


def fit_predict(X_tr, y_tr, X_te, loss):
    if len(X_tr) < 50 or np.nansum(y_tr) == 0:
        return None
    m = HistGradientBoostingRegressor(loss=loss, **GBT).fit(X_tr, y_tr)
    return m.predict(X_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8,
                    help="shuffled-delta placebo arms (the null distribution)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    X, y_round, df, _ = load_and_prepare()
    X = X.drop(columns=["const"])          # tree model; no intercept column

    exact = exact_means(df)
    keep = exact.notna().values
    # Rows without a 3-frame mean cannot answer this question at all -- their
    # target is the legacy single-frame count, where there is nothing to round.
    # Dropping them keeps every arm on identical data.
    X, df = X[keep].reset_index(drop=True), df[keep].reset_index(drop=True)
    y_round, exact = y_round[keep].reset_index(drop=True), exact[keep].reset_index(drop=True)

    m = exact.to_numpy(float)
    r = np.round(m)
    d = r - m                              # the information rounding destroys

    day = pd.to_datetime(df["date"]).dt.normalize()
    days = np.array(sorted(day.unique()))
    lo, hi = MIN_TRAIN_DAYS, len(days) - TEST_BLOCK
    cuts = days[lo:hi]
    if len(cuts) > args.origins:
        cuts = cuts[np.linspace(0, len(cuts) - 1, args.origins).round().astype(int)]

    print(f"{len(df)} rows / {len(days)} days with a 3-frame mean; "
          f"{len(cuts)} rolling origins, {TEST_BLOCK}-day test blocks")
    print(f"rounding moves the target on {100 * (d != 0).mean():.1f}% of rows, "
          f"mean |delta| {np.abs(d).mean():.3f}\n")

    arms = ["round_poisson", "exact_poisson", "round_squared", "exact_squared"]
    arms += [f"shuffle{s}" for s in range(args.seeds)]
    # Two scoring regimes, both against truth held fixed across arms.
    scores = {a: {"exact": [], "round": []} for a in arms}

    for cut in cuts:
        te_days = set(days[days >= cut][:TEST_BLOCK])
        te = np.array([dd in te_days for dd in day])
        tr = (day < cut).values
        if tr.sum() < 50 or te.sum() == 0:
            continue
        X_tr, X_te = X[tr], X[te]

        for arm in arms:
            if arm.startswith("shuffle"):
                rng = np.random.default_rng(2000 + int(arm[7:]))
                # Permute the deltas among the TRAINING rows only; a test-set
                # permutation would change the truth the arm is scored against.
                dp = d.copy()
                idx = np.where(tr)[0]
                dp[idx] = rng.permutation(d[idx])
                y_tr, loss = np.clip(r[tr] - dp[tr], 0, None), "poisson"
            else:
                tgt, loss_name = arm.split("_")
                y_tr = r[tr] if tgt == "round" else m[tr]
                loss = "poisson" if loss_name == "poisson" else "squared_error"

            pred = fit_predict(X_tr, y_tr, X_te, loss)
            if pred is None:
                scores[arm]["exact"].append(np.nan)
                scores[arm]["round"].append(np.nan)
                continue
            scores[arm]["exact"].append(mean_absolute_error(m[te], pred))
            scores[arm]["round"].append(mean_absolute_error(r[te], pred))

    base = "round_poisson"
    for truth in ("exact", "round"):
        b = np.array(scores[base][truth], float)
        print(f"=== scored against the {truth.upper()} target "
              f"({'unrounded frame_count_mean' if truth == 'exact' else 'what production publishes'}) ===")
        print(f"  {'arm':16s} {'MAE':>7s} {'vs base':>8s} {'p':>7s} {'better':>8s}")
        print(f"  {base:16s} {np.nanmean(b):7.3f} {'--':>8s} {'--':>7s} {'--':>8s}")
        deltas = {}
        for arm in arms:
            if arm == base:
                continue
            a = np.array(scores[arm][truth], float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            t = stats.ttest_rel(a[ok], b[ok])
            dlt = np.nanmean(a) - np.nanmean(b)
            deltas[arm] = dlt
            if not arm.startswith("shuffle"):
                print(f"  {arm:16s} {np.nanmean(a):7.3f} {dlt:+8.3f} {t.pvalue:7.3f} "
                      f"{int((a[ok] < b[ok]).sum()):4d}/{int(ok.sum()):<3d}")
        sh = np.array([deltas[a] for a in arms if a.startswith("shuffle")])
        print(f"  shuffled-delta placebo, {len(sh)} seeds: mean {sh.mean():+.3f}, "
              f"sd {sh.std(ddof=1):.3f}, range {sh.min():+.3f} to {sh.max():+.3f}")
        ex = deltas["exact_poisson"]
        print(f"  exact_poisson delta {ex:+.3f}  ->  "
              f"{int((sh <= ex).sum())}/{len(sh)} placebo seeds did as well or better\n")


if __name__ == "__main__":
    main()
