"""
band_calibration.py
-------------------
Nightly calibration of the forecast band's edges from the scored record
(added 2026-11-23).

Why: the band's edges are fitted quantiles, and nothing checked them against
what happened. Over the 30 days to 11-21, 31% of hours forecast at 5+ surfers
came in below the band against a 15% target, while their mean forecast matched
the actual (20.1 vs 19.0): the lower edge sat too close to the median. Tested
out of sample (analysis/forecast_band/test_lower_edge_calibration.py), fitting
the stretch below on earlier days only brought that to 15%.

Method (the same form as calibrate_lead_bands.py): on hours forecast at
BUSY_MIN+ surfers, the distance from the point forecast to each edge is
multiplied by its own factor,

    lower = point - k_lower * (point - lower_raw)
    upper = point + k_upper * (upper_raw - point)

with each k the smallest value, at least 1, that leaves no more than that
edge's target share (the band's own quantile: 15% below, 10% above) of the last
WINDOW_DAYS days' busy hours outside it. Never narrows. Quiet hours (forecast
under BUSY_MIN) are left alone: their misses were on target (11% below).

The fit always reads the raw, uncalibrated edges (`lower_raw` / `upper_raw`,
written beside the published ones), so it never fits on its own output. Rows
from before 2026-11-23 have no raw columns; their published edges were raw.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

BUSY_MIN = 5.0          # matches email_daily_report.BUSY_FORECAST_MIN
WINDOW_DAYS = 30
MIN_HOURS = 60          # busy hours needed in the window before any stretch is applied
MAX_K = 3.0             # sanity ceiling; reaching it means something else is wrong


def raw_edges(df):
    """(lower, upper) before any calibration: the raw columns when present,
    else the published ones with any week-chart lead widening undone."""
    lower = df["lower_q10"].astype(float).copy()
    upper = df["upper_q90"].astype(float).copy()
    if "band_factor" in df:
        f = pd.to_numeric(df["band_factor"], errors="coerce").fillna(1.0).clip(lower=1.0)
        p = df["predicted"].astype(float)
        lower = (p - (p - lower) / f).where(f > 1.0, lower)
        upper = (p + (upper - p) / f).where(f > 1.0, upper)
    for col, s in (("lower_raw", lower), ("upper_raw", upper)):
        if col in df:
            r = pd.to_numeric(df[col], errors="coerce")
            s.loc[r.notna()] = r[r.notna()]
    return lower, upper


def required_k(beyond, gap, target):
    """Smallest k >= 1 such that at most `target` of hours have beyond > k * gap.
    `beyond` is how far past the point the actual fell (positive = toward the
    edge), `gap` the point-to-edge distance."""
    beyond, gap = np.asarray(beyond, float), np.asarray(gap, float)
    need = np.where(gap > 0, beyond / np.where(gap > 0, gap, 1.0), np.where(beyond > 0, np.inf, 0.0))
    if len(need) == 0:
        return 1.0
    k = float(np.quantile(need, 1 - target, method="higher"))
    return float(min(max(k, 1.0), MAX_K))


def fit(log, today, lower_q, upper_q, window_days=WINDOW_DAYS, min_hours=MIN_HOURS):
    """{'k_lower', 'k_upper', 'n_hours', 'below_raw', 'above_raw', 'applied',
    'reason'} from the day-ahead rows of forecast_log scored in the window
    before `today` and made with the same band quantiles."""
    out = dict(k_lower=1.0, k_upper=1.0, n_hours=0, below_raw=np.nan, above_raw=np.nan,
               applied=False, reason="")
    if log is None or log.empty:
        out["reason"] = "no forecast log"
        return out
    s = log[(log.get("forecast_kind", "day_ahead") == "day_ahead") & log["actual"].notna()].copy()
    if "band_lower_q" in s:
        s = s[np.isclose(s["band_lower_q"].astype(float), lower_q)
              & np.isclose(s["band_upper_q"].astype(float), upper_q)]
    d = pd.to_datetime(s["date"]).dt.date
    s = s[(d >= today - timedelta(days=window_days)) & (d < today)]
    s = s[s["predicted"].astype(float) >= BUSY_MIN]
    out["n_hours"] = len(s)
    if len(s) < min_hours:
        out["reason"] = f"only {len(s)} busy hours scored in the last {window_days} days (need {min_hours})"
        return out
    p, a = s["predicted"].astype(float), s["actual"].astype(float)
    lo, hi = raw_edges(s)
    out["below_raw"] = float((a < lo).mean())
    out["above_raw"] = float((a > hi).mean())
    out["k_lower"] = required_k(p - a, (p - lo).clip(lower=0), lower_q)
    out["k_upper"] = required_k(a - p, (hi - p).clip(lower=0), 1 - upper_q)
    out["applied"] = out["k_lower"] > 1.0 or out["k_upper"] > 1.0
    out["reason"] = (f"{len(s)} busy hours, {out['below_raw']:.0%} below / "
                     f"{out['above_raw']:.0%} above the raw band")
    return out


def apply(point, lower, upper, cal):
    """Calibrated (lower, upper) for one hour; unchanged below BUSY_MIN."""
    if point < BUSY_MIN:
        return lower, upper
    return (max(point - cal["k_lower"] * (point - lower), 0.0),
            point + cal["k_upper"] * (upper - point))


def load_and_fit(log_path, today, lower_q, upper_q):
    try:
        log = pd.read_csv(log_path)
    except (OSError, ValueError) as e:
        out = fit(None, today, lower_q, upper_q)
        out["reason"] = f"could not read the forecast log ({type(e).__name__})"
        return out
    return fit(log, today, lower_q, upper_q)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plot_daily_prediction as pdp
    print(load_and_fit(pdp.FORECAST_LOG_PATH, date.today(), pdp.FAN_LEVELS[0], pdp.FAN_LEVELS[-1]))
