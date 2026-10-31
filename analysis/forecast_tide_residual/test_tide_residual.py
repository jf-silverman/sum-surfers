"""
Does correcting tide_ft with the non-tidal residual help the forecast?

Background. `tide_ft` is Surfline's HARMONIC prediction -- astronomy, with no
non-tidal signal in it by construction. The actual water level differs by the
residual, which has run 0.24-0.99 ft over this record and reached 1.83 ft hourly
during the October 2026 Kelvin wave (see code/backfill_tide_residual.py).

`tide_ft` is also the model's most-leaned-on feature: permutation importance on
a forward split ranks it 1 of 33 at 6.81, against 0.92 for the next feature.
(That figure is inflated -- tide correlates with hour and date, and permutation
breaks those joint structures -- so read it as heavy reliance, not as 6.8
surfers of causal signal.) So the dominant feature is systematically wrong about
the physical quantity, which is why this is worth a test.

THE COMPETING HYPOTHESES, and why this test can tell them apart. Joel's, on
2026-10-15: surfers are not tracking the Kelvin wave. They read the Surfline
tide chart and paddle out when it says the tide is low. If that is the whole
story, the harmonic prediction is the BEHAVIOURALLY correct input, the residual
is irrelevant to the decision, and correcting tide_ft should make the forecast
WORSE -- it would be replacing the number people act on with one they never see.

The competing channel is physical: actual water depth over the reef sets how the
wave breaks, so a foot of extra water changes the surf regardless of what anyone
read. That acts on people once they arrive and look at it, not on the decision
to drive over.

These predict opposite signs, which is what makes this worth running rather than
assuming. A null result is also informative: it says the residual is too small
against the ~1.5-surfer target noise floor for either channel to show up.

Arms (all share one target, so every arm is scored against identical truth;
only the FEATURE MATRIX changes):

    base          tide_ft as production has it                  <- baseline
    corrected     tide_ft replaced by tide_ft + residual_ft     (physical level)
    both          tide_ft kept, residual_ft added as a column   (model chooses)
    dev_only      tide_ft kept, residual_dev_ft added           (detrended, so
                  not confounded with the record's rising trend)
    shuffleN      tide_ft kept, a PERMUTED residual_ft added    <- placebo

The shuffle arms are the placebo required by the procedure adopted 2026-10-14:
any arm claiming less than about +/-0.5 MAE runs placebos alongside it before
the result is reported. They add a column with the residual's exact marginal
distribution attached to the wrong rows, which gives the null for "add a
plausible-looking column, any column".

Note what this design avoids. The arrival-timing question -- does the Kelvin
wave change WHEN people show up -- is not testable yet: only 190 rows sit in the
elevated period and the raw residual is near-collinear with time across the
record. `corrected` dodges that entirely, because it adds no column. It changes
an existing one, and the residual has been nonzero all along (0.24-0.6 ft
through 2025-26), so the correction has 143 days of leverage rather than 14.

RESULT, 2026-10-15, 1,927 rows / 145 days, 16 rolling origins. Residual over
these rows: mean +0.447 ft, sd 0.354, range -0.49 to +1.38.

    arm            MAE   vs base       p   better
    base         6.753       --      --       --
    corrected    7.033   +0.280   0.102    5/16
    both         6.786   +0.033   0.507    7/16
    dev_only     6.842   +0.089   0.165    6/16

    add-a-column placebo   (8 seeds): mean +0.017, sd 0.030
    SCRAMBLED correction   (8 seeds): mean +0.631, sd 0.103  -> 0/8 beat `corrected`
    DATE-SHIFTED control   (8 seeds): mean +0.545, sd 0.198  -> 1/8 beat `corrected`

VERDICT: NO. Nothing helps. Every way of using the residual is worse than
leaving tide_ft alone, and no arm is significant. `both` -- simply adding the
residual as a column -- lands at +0.033 inside the add-a-column null, which is
this project's standard result for a new feature.

The controls are where the interest is, and they have to be read in order.

`corrected` costs +0.280, but SCRAMBLING the same residual costs +0.631, so the
real correction is less than half as damaging as a fake one. That looks like
evidence the correction carries signal, and on its own it is not: a permuted
residual is white noise hour to hour while the real one is smooth, so the
comparison partly measures "smooth beats jagged" rather than "right beats
wrong". The DATE-SHIFTED control exists to remove that confound -- rolling the
residual by whole days preserves its autocorrelation exactly and breaks only
the alignment with the hours it belongs to. `corrected` beats it in 7 of 8
seeds, about 1.3 sd, with one seed at +0.216 doing better.

So the residual does appear to carry some information about these specific
hours, weakly. It is still not worth using: beating a WRONG correction while
remaining worse than NO correction means the signal is real but too small to
pay for disturbing the model's most-relied-upon feature.

WHAT THIS DOES AND DOES NOT SETTLE. Joel's hypothesis -- that surfers read the
Surfline tide chart and go when it says the tide is low, so the harmonic value
is the behaviourally correct input -- is consistent with every number here. But
the test cannot separate it from the simpler explanation that the physical
effect exists and is below the noise floor, because both predict exactly this
pattern. What it does rule out is that correcting tide_ft is free: it is a
measurable loss, not a wash.

Worth revisiting if the winter brings a larger residual. These rows top out at
+1.38 ft and the October event alone reached 1.83 ft hourly; a 2 ft residual
would give the physical channel a materially stronger test than anything in
this record.

Usage:
    python analysis/forecast_tide_residual/test_tide_residual.py
    python analysis/forecast_tide_residual/test_tide_residual.py --origins 20
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
import backfill_tide_residual as btr  # noqa: E402

RESID_CSV = _ROOT / "data" / "predictor_vars" / "tide_residual.csv"
TEST_BLOCK = 5
MIN_TRAIN_DAYS = 40
GBT = dict(loss="poisson", max_iter=300, learning_rate=0.05,
           max_depth=4, l2_regularization=1.0, random_state=42)


def join_residual(df):
    """Residual columns aligned to the feature rows, matched on date + hour.

    The residual file is hourly on the clock; feature rows carry an hour_local
    integer. Matching on the hour (not the clip's exact minute) is the same
    convention the forecast table and the evening report already use.
    """
    r = pd.read_csv(RESID_CSV)
    r["hour"] = pd.to_datetime(r["hour_local"], format="%H:%M").dt.hour
    r = r.drop_duplicates(subset=["date", "hour"], keep="last").set_index(["date", "hour"])
    key = pd.MultiIndex.from_arrays([df["date"].astype(str), df["hour_local"].astype(int)])
    return (r["residual_ft"].reindex(key).to_numpy(float),
            r["residual_dev_ft"].reindex(key).to_numpy(float))


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

    resid, dev = join_residual(df)
    # The servable form (2026-10-30): the previous day's mean detrended residual,
    # by the same rule the nightly forecast uses (btr.prior_dev_mean).
    means = btr.daily_dev_means()
    prior = np.array([btr.prior_dev_mean(d, means) for d in pd.to_datetime(df["date"]).dt.date], dtype=object)
    prior = np.array([np.nan if v == "" else v for v in prior], float)
    keep = ~np.isnan(resid) & ~np.isnan(dev)
    # Rows the gauge cannot cover (outages, dates outside the pull) are dropped
    # from EVERY arm, so the baseline is scored on identical data.
    X, df = X[keep].reset_index(drop=True), df[keep].reset_index(drop=True)
    y = y[keep].reset_index(drop=True)
    resid, dev = resid[keep], dev[keep]

    day = pd.to_datetime(df["date"]).dt.normalize()
    days = np.array(sorted(day.unique()))
    lo, hi = MIN_TRAIN_DAYS, len(days) - TEST_BLOCK
    cuts = days[lo:hi]
    if len(cuts) > args.origins:
        cuts = cuts[np.linspace(0, len(cuts) - 1, args.origins).round().astype(int)]

    print(f"{len(df)} rows / {len(days)} days with both features and a residual; "
          f"{len(cuts)} rolling origins, {TEST_BLOCK}-day test blocks")
    print(f"residual over these rows: mean {resid.mean():+.3f} ft, "
          f"sd {resid.std():.3f}, range {resid.min():+.2f} to {resid.max():+.2f}\n")

    def matrix(arm, rng=None):
        Z = X.copy()
        if arm == "corrected":
            Z["tide_ft"] = Z["tide_ft"].to_numpy() + resid
        elif arm == "both":
            Z["tide_residual_ft"] = resid
        elif arm == "dev_only":
            Z["tide_residual_dev_ft"] = dev
        elif arm == "dev_prior":
            Z["tide_resid_dev_prior_ft"] = prior
        elif arm.startswith("shuffle"):
            Z["tide_residual_ft"] = rng.permutation(resid)
        elif arm.startswith("shifted"):
            # The control that `corrupt` is NOT. A permuted residual is white
            # noise hour to hour, while the real one is smooth and slowly
            # varying, so "real beats permuted" can be no more than "smooth
            # beats jagged". A CIRCULAR SHIFT by whole days keeps the
            # autocorrelation structure exactly and breaks only the alignment
            # with the dates it belongs to. This is the control that isolates
            # whether the residual is right ABOUT THESE HOURS.
            shift = int(rng.integers(20, len(resid) - 20))
            Z["tide_ft"] = Z["tide_ft"].to_numpy() + np.roll(resid, shift)
        elif arm.startswith("corrupt"):
            # The control that `shuffle` does NOT provide. `corrected` perturbs
            # the model's most-relied-upon feature, and ANY distortion of it
            # would hurt, so a worse score is not by itself evidence that the
            # harmonic value is the behaviourally right one. This applies the
            # same marginal perturbation to tide_ft with the rows scrambled: if
            # it is as bad as `corrected`, the mechanism is "tide_ft was
            # disturbed", not "the correction was wrong".
            Z["tide_ft"] = Z["tide_ft"].to_numpy() + rng.permutation(resid)
        return Z

    arms = (["base", "corrected", "both", "dev_only", "dev_prior"]
            + [f"shuffle{s}" for s in range(args.seeds)]
            + [f"corrupt{s}" for s in range(args.seeds)]
            + [f"shifted{s}" for s in range(args.seeds)])
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
                rng = np.random.default_rng(3000 + int(arm[7:]))
            elif arm.startswith("corrupt"):
                rng = np.random.default_rng(4000 + int(arm[7:]))
            elif arm.startswith("shifted"):
                rng = np.random.default_rng(5000 + int(arm[7:]))
            Z = matrix(arm, rng)
            scores[arm].append(fit_mae(Z[tr], y[tr], Z[te], y[te]))

    b = np.array(scores["base"], float)
    print(f"  {'arm':14s} {'MAE':>7s} {'vs base':>8s} {'p':>7s} {'better':>8s}")
    print(f"  {'base':14s} {np.nanmean(b):7.3f} {'--':>8s} {'--':>7s} {'--':>8s}")
    deltas = {}
    for arm in arms[1:]:
        a = np.array(scores[arm], float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        deltas[arm] = np.nanmean(a) - np.nanmean(b)
        if not arm.startswith(("shuffle", "corrupt", "shifted")):
            t = stats.ttest_rel(a[ok], b[ok])
            print(f"  {arm:14s} {np.nanmean(a):7.3f} {deltas[arm]:+8.3f} {t.pvalue:7.3f} "
                  f"{int((a[ok] < b[ok]).sum()):4d}/{int(ok.sum()):<3d}")
    sh = np.array([deltas[a] for a in arms if a.startswith("shuffle")])
    print(f"\n  shuffled-residual placebo, {len(sh)} seeds: mean {sh.mean():+.3f}, "
          f"sd {sh.std(ddof=1):.3f}, range {sh.min():+.3f} to {sh.max():+.3f}")
    for arm in ("both", "dev_only", "dev_prior"):
        print(f"  {arm:10s} {deltas[arm]:+.3f}  ->  "
              f"{int((sh <= deltas[arm]).sum())}/{len(sh)} placebo seeds did as well or better")

    cr = np.array([deltas[a] for a in arms if a.startswith("corrupt")])
    print(f"\n  SCRAMBLED-correction control (tide_ft + permuted residual), "
          f"{len(cr)} seeds:\n    mean {cr.mean():+.3f}, sd {cr.std(ddof=1):.3f}, "
          f"range {cr.min():+.3f} to {cr.max():+.3f}")
    print(f"  corrected  {deltas['corrected']:+.3f}  ->  "
          f"{int((cr <= deltas['corrected']).sum())}/{len(cr)} scrambled seeds did as well or better")
    sf = np.array([deltas[a] for a in arms if a.startswith("shifted")])
    print(f"\n  DATE-SHIFTED control (tide_ft + residual rolled by whole days, so the\n"
          f"  smoothness is identical and only the alignment is wrong), {len(sf)} seeds:\n"
          f"    mean {sf.mean():+.3f}, sd {sf.std(ddof=1):.3f}, "
          f"range {sf.min():+.3f} to {sf.max():+.3f}")
    print(f"  corrected  {deltas['corrected']:+.3f}  ->  "
          f"{int((sf <= deltas['corrected']).sum())}/{len(sf)} shifted seeds did as well or better")
    print("\n  THIS is the comparison that matters. If `corrected` beats the shifted")
    print("  control, the residual carries information about these specific hours.")
    print("  If it does not, the residual adds nothing beyond being a smooth")
    print("  perturbation, and the harmonic value is what behaviour follows.")

    print("\n  If `corrected` is no worse than the scrambled control, the damage is from")
    print("  disturbing tide_ft at all, and this says nothing about WHICH tide")
    print("  value is right. If it is clearly worse, the real correction is")
    print("  worse than a fake one, which is evidence the harmonic value is what")
    print("  behaviour follows.")


if __name__ == "__main__":
    main()
