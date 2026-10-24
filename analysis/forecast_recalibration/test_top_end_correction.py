"""
Can a correction applied after the model fix the forecast's top end?

When the model forecasts 31-40 surfers the hour averages about 40 (-8.6, -21%;
see analysis/forecast_miss_patterns/). The production correction is one
straight line (fit_point_recalibration in plot_daily_prediction.py), which
cannot pull the low end down and push the top end up by different amounts.
This tests shapes that can, against that line, under the same rules:

  - fitted only on rows scored for EARLIER target dates, applied to the next
  - the production guards (400 rows and 7 dates before anything is applied)
  - the result clamped into the hour's band and at zero, as production does

Arms
  raw              the model's own median (predicted_raw)
  affine           production today: a + b*pred
  hinge k          a + b*pred + c*max(0, pred-k): a line with a second slope
                   above k, so the top can steepen without moving the bottom
  isotonic         any monotone step shape (sklearn IsotonicRegression);
                   flat beyond the largest forecast it has seen
  hinge20 + hour   the k=20 hinge plus a separate offset for each hour of the
                   day, since the morning runs high and the afternoon low
  quantile map     sends the forecast's percentile in the prior forecasts to
                   the same percentile of the prior actuals. Matches the
                   spread rather than the conditional mean, so it should
                   fix the top-end bias and cost MAE elsewhere.

Scored on every lead, and on lead 1 (what the evening email reads). Per-date
comparison against the affine arm is a Wilcoxon test over target dates, since
hours within a date are not independent.

Result, 2026-10-24, 2,293 rows over 23 target dates (Oct 1-23), all leads:

    arm                MAE    bias when forecast 25+   bias when actual 30+
    affine (prod)      5.605          -4.35                  -15.21
    hinge k=20         5.359          -0.91                  -12.61
    isotonic           5.410          -0.87                  -12.48
    quantile map       5.332          +3.71                   -8.77

The hinge at 20 is the best calibrated: bias within 1 surfer in every
forecast bin (+0.9, +0.9, +0.4, -0.9). Quantile mapping does most for the
packed hours and has the lowest MAE, but over-forecasts whenever it says 25+.
Neither beats the affine on enough dates to be significant yet (15/23,
p=0.27; 13/23, p=0.75). Hour-of-day offsets add nothing over the hinge. The
band clamp is not what holds the affine's top down: it only ever lifts values
up to the band's lower edge (12% of rows), never caps them.

Usage:
    python analysis/forecast_recalibration/test_top_end_correction.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
FLOG = ROOT / "data" / "forecasts" / "forecast_log.csv"
MIN_ROWS, MIN_DATES = 400, 7          # RECAL_MIN_ROWS / RECAL_MIN_DATES in production


def affine(px, py, x):
    b, a, *_ = stats.linregress(px, py)
    return a + b * x if 0.5 <= b <= 1.5 else x      # production's slope guard


def hinge(k):
    def fn(px, py, x):
        A = np.c_[np.ones_like(px), px, np.maximum(0, px - k)]
        coef, *_ = np.linalg.lstsq(A, py, rcond=None)
        return coef[0] + coef[1] * x + coef[2] * np.maximum(0, x - k)
    return fn


def hinge_hour(px, py, x, ph, h, k=20):
    hours = np.unique(ph)
    def design(xx, hh):
        return np.c_[xx, np.maximum(0, xx - k), (hh[:, None] == hours[None, :]).astype(float)]
    coef, *_ = np.linalg.lstsq(design(px, ph), py, rcond=None)
    known = np.isin(h, hours)
    out = hinge(k)(px, py, x)                      # an hour never seen before falls back
    out[known] = design(x[known], h[known]) @ coef
    return out


def isotonic(px, py, x):
    return IsotonicRegression(out_of_bounds="clip").fit(px, py).predict(x)


def quantile_map(px, py, x):
    q = np.searchsorted(np.sort(px), x, side="right") / len(px)
    return np.quantile(py, np.clip(q, 0, 1))


ARMS = {"raw": None, "affine (production)": affine, "hinge k=15": hinge(15),
        "hinge k=20": hinge(20), "hinge k=25": hinge(25), "hinge20 + hour": hinge_hour,
        "isotonic": isotonic,
        "quantile map": quantile_map}


def main():
    f = pd.read_csv(FLOG)
    s = f[f.actual.notna()].copy()
    s["raw"] = s.predicted_raw.fillna(s.predicted)
    s = s.sort_values(["date", "hour", "lead_days"]).reset_index(drop=True)
    dates = sorted(s.date.unique())
    out = {k: np.full(len(s), np.nan) for k in ARMS}
    clamped = {k: 0 for k in ARMS}
    for D in dates:
        prior, cur = s[s.date < D], s.date == D
        if len(prior) < MIN_ROWS or prior.date.nunique() < MIN_DATES:
            continue
        x = s.loc[cur, "raw"].to_numpy()
        lo, hi = s.loc[cur, "lower_q10"].to_numpy(), s.loc[cur, "upper_q90"].to_numpy()
        for k, fn in ARMS.items():
            if fn is None:
                v = x
            elif fn is hinge_hour:
                v = fn(prior.raw.to_numpy(), prior.actual.to_numpy(), x,
                       prior.hour.to_numpy(), s.loc[cur, "hour"].to_numpy())
            else:
                v = fn(prior.raw.to_numpy(), prior.actual.to_numpy(), x)
            clamped[k] += int(((v > np.maximum(hi, lo)) | (v < np.maximum(lo, 0))).sum())
            out[k][cur.to_numpy()] = np.minimum(np.maximum(v, np.maximum(lo, 0)), np.maximum(hi, lo))
    keep = ~np.isnan(out["raw"])
    s = s[keep].reset_index(drop=True)
    out = {k: v[keep] for k, v in out.items()}
    print(f"Scored {len(s)} rows over {s.date.nunique()} target dates "
          f"({s.date.min()} to {s.date.max()}); corrections fitted forward in time.\n")

    for label, m in [("ALL LEADS", np.ones(len(s), bool)), ("LEAD 1", (s.lead_days == 1).to_numpy())]:
        a, y = s[m], s.actual.to_numpy()[m]
        rows = []
        for k, v in out.items():
            v = v[m]
            e = v - y
            pb = pd.cut(a.raw, [-1, 10, 20, 25, 100], labels=["<=10", "10-20", "20-25", "25+"])
            byp = pd.Series(e).groupby(pb.to_numpy(), observed=False).mean()
            per_date = pd.DataFrame({"d": a.date.to_numpy(), "e": np.abs(e),
                                     "base": np.abs(out["affine (production)"][m] - y)}).groupby("d").mean()
            better = int((per_date.e < per_date.base).sum()); n = len(per_date)
            p = stats.wilcoxon(per_date.e - per_date.base).pvalue if k != "affine (production)" else np.nan
            rows.append({"arm": k, "MAE": np.abs(e).mean(), "bias": e.mean(),
                         "bias fcst<=10": byp["<=10"], "bias fcst 10-20": byp["10-20"],
                         "bias fcst 20-25": byp["20-25"], "bias fcst 25+": byp["25+"],
                         "bias actual 30+": e[y >= 30].mean(), "MAE actual 30+": np.abs(e[y >= 30]).mean(),
                         "dates better than affine": f"{better}/{n}" if k != "affine (production)" else "",
                         "p (Wilcoxon, dates)": p, "clamped by band %": 100 * clamped[k] / len(s)})
        r = pd.DataFrame(rows).set_index("arm")
        n25 = int((a.raw > 25).sum())
        print(f"{label}  (n={m.sum()}, {n25} rows forecast above 25, {(y >= 30).sum()} actual 30+)")
        print(r.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
