"""
Out-of-sample test of post-hoc corrections to the forecast point estimate.

Context: every IN-MODEL feature idea tried on this project has come back flat or
worse -- lags, lunar tide, weekday one-hots, raw hour, deeper trees,
weekend x hour, within-day ranks, tide bands, window structure, recency
weighting. The best was day-level aggregates at -0.09 MAE, not significant.

These are a different category: corrections applied OUTSIDE the model, fitted on
the scored record. A tree cannot extrapolate past its training range, which is
why recency weighting failed when the crowd level drifted below it; an additive
offset or an affine map has no such limit.

Every correction here is fitted ONLY on rows scored for EARLIER target dates and
then applied to the current one. Nothing is fitted on the rows it is scored
against.

Result, 2026-10-03, 625 rows over 10 target dates:

    global affine  (a + b*pred)     MAE 4.747 -> 4.471  (-0.276, p<0.001)
    per-lead affine                            -> 4.547  (-0.199, p=0.001)
    global EWMA offset, 3d half-life           -> 4.567  (-0.179, p=0.001)
    per-lead EWMA offset, 7d                   -> 4.646  (-0.100, p=0.067)

All four help; the simplest helps most. Per-lead variants split 10 target dates
too thin to estimate seven separate corrections, which is the same small-sample
problem the band-inflation factors have.

Bias falls from +2.04 to about +1.5 but does not vanish, because each correction
is fitted on history while the level keeps moving -- the correction is always
chasing. That is expected and is the argument for refitting it often rather than
for a bigger correction.

NOT wired into production. This measures the idea; adopting it changes every
forecast the project publishes.

Usage:
    python analysis/forecast_recalibration/test_recalibration.py
"""

import numpy as np, pandas as pd
from scipy import stats

f = pd.read_csv("data/forecasts/forecast_log.csv")
s = f[f.actual.notna() & f.lead_days.notna()].copy()
s["lead_days"] = s.lead_days.astype(int)
s["d"] = pd.to_datetime(s.date)
s = s.sort_values(["d", "hour"]).reset_index(drop=True)
dates = sorted(s.d.unique())
print(f"{len(s)} scored rows over {len(dates)} target dates\n")

MIN_PRIOR = 40          # need some history before any correction is attempted

def run(fn, label):
    base_e, corr_e = [], []
    for D in dates:
        prior = s[s.d < D]
        cur = s[s.d == D]
        if len(prior) < MIN_PRIOR or cur.empty:
            continue
        adj = fn(prior, cur)
        base_e.extend((cur.predicted - cur.actual).tolist())
        corr_e.extend((adj - cur.actual).tolist())
    base_e, corr_e = np.array(base_e), np.array(corr_e)
    if not len(base_e):
        return
    t = stats.ttest_rel(np.abs(corr_e), np.abs(base_e))
    print(f"  {label:<40} MAE {np.abs(base_e).mean():.3f} -> {np.abs(corr_e).mean():.3f} "
          f"({np.abs(corr_e).mean()-np.abs(base_e).mean():+.3f}, p={t.pvalue:.3f})  "
          f"bias {base_e.mean():+.2f} -> {corr_e.mean():+.2f}   n={len(base_e)}")

print("A. GLOBAL EWMA BIAS OFFSET  (pred + ewma of prior actual-minus-pred)")
for hl in (3, 7, 14, 30, 1e9):
    def fn(prior, cur, hl=hl):
        err = (prior.actual - prior.predicted)
        w = 0.5 ** (((prior.d.max() - prior.d).dt.days) / hl)
        off = float((err * w).sum() / w.sum())
        return cur.predicted + off
    run(fn, f"half-life {('inf' if hl>1e8 else int(hl)):>3} days")

print("\nB. PER-LEAD EWMA OFFSET")
for hl in (7, 14, 1e9):
    def fn(prior, cur, hl=hl):
        out = cur.predicted.copy()
        for L in cur.lead_days.unique():
            p = prior[prior.lead_days == L]
            if len(p) < 15:
                continue
            w = 0.5 ** (((p.d.max() - p.d).dt.days) / hl)
            off = float(((p.actual - p.predicted) * w).sum() / w.sum())
            out.loc[cur.lead_days == L] = cur.predicted[cur.lead_days == L] + off
        return out
    run(fn, f"half-life {('inf' if hl>1e8 else int(hl)):>3} days")

print("\nC. PER-LEAD AFFINE  (a_L + b_L * pred, least squares on prior rows)")
def affine(prior, cur):
    out = cur.predicted.copy()
    for L in cur.lead_days.unique():
        p = prior[prior.lead_days == L]
        if len(p) < 25:
            continue
        sl, ic, *_ = stats.linregress(p.predicted, p.actual)
        out.loc[cur.lead_days == L] = ic + sl * cur.predicted[cur.lead_days == L]
    return out.clip(lower=0)
run(affine, "fitted per lead")

print("\nD. GLOBAL AFFINE  (one a,b for all leads)")
def affine_g(prior, cur):
    sl, ic, *_ = stats.linregress(prior.predicted, prior.actual)
    return (ic + sl * cur.predicted).clip(lower=0)
run(affine_g, "fitted on all prior rows")
