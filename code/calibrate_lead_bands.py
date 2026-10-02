"""
calibrate_lead_bands.py
-----------------------
Measures how much the 80% prediction interval needs widening at each forecast
lead time, and writes the factors to data/forecasts/lead_band_calibration.json.

Why this is needed: the interval width is almost entirely a function of the
PREDICTED LEVEL (r = +0.900 against predicted count, measured 2026-10-01), which
is right for count data -- variance scales with the mean -- but leaves no
channel for "how reliable are my inputs". At longer lead the forecast regresses
toward the middle, so the band narrows along with it, exactly where the inputs
are least reliable. Measured coverage falls from ~85% at a day out to ~57% at
seven, against an 80% target.

Why the model cannot simply learn this: `training_features.csv` pairs every hour
with predictors fetched at essentially lead 0-1. There is no lead variation in
the training data at all, so adding `lead_days` as a feature would add a column
that is constant during training and the model would learn nothing from it. The
correction has to come from the scored record instead, which is what this does.

Method: for each lead, find the multiplier k such that scaling the distance from
the point prediction to each band edge by k would have achieved 80% coverage
historically. Those raw k values are noisy at these sample sizes (lead 6 wants
1.02 while lead 4 wants 1.29), so a line is fitted across leads and forced
monotone -- a longer lead can never get a narrower correction than a shorter
one. The fitted slope is +0.031/day and NOT significant (p = 0.19), so this is a
deliberately gentle correction, not a confident one.

Run it after enough new days have been scored; it is not part of the nightly
pipeline, so the factors change only when someone decides they should.

Usage:
    python code/calibrate_lead_bands.py
    python code/calibrate_lead_bands.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = _PROJECT_ROOT / "data" / "forecasts" / "forecast_log.csv"
OUT = _PROJECT_ROOT / "data" / "forecasts" / "lead_band_calibration.json"

TARGET_COVERAGE = 0.80
MIN_HOURS_PER_LEAD = 20      # below this a lead's own coverage is not worth reading
MAX_FACTOR = 2.0             # a sanity ceiling; hitting it means something else is wrong


def coverage(g, k):
    lo = g.predicted - k * (g.predicted - g.lower_q10)
    hi = g.predicted + k * (g.upper_q90 - g.predicted)
    return float(((g.actual >= lo) & (g.actual <= hi)).mean())


def required_k(g):
    """Smallest k in [0.6, MAX_FACTOR] reaching TARGET_COVERAGE, else MAX_FACTOR."""
    for k in np.arange(0.6, MAX_FACTOR + 0.001, 0.01):
        if coverage(g, k) >= TARGET_COVERAGE:
            return float(round(k, 2))
    return MAX_FACTOR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = p.parse_args()

    if not LOG.exists():
        print(f"  calibrate_lead_bands: {LOG} not found — nothing to calibrate")
        return 1
    s = pd.read_csv(LOG)
    s = s[s.actual.notna() & s.lead_days.notna()].copy()
    s["lead_days"] = s.lead_days.astype(int)
    if s.empty:
        print("  calibrate_lead_bands: no scored rows yet")
        return 1

    raw, usable = {}, {}
    print(f"{'lead':>5}{'hours':>7}{'coverage':>10}{'k needed':>10}")
    for lead, g in s.groupby("lead_days"):
        k = required_k(g)
        raw[lead] = k
        if len(g) >= MIN_HOURS_PER_LEAD:
            usable[lead] = k
        flag = "" if len(g) >= MIN_HOURS_PER_LEAD else "  (too few, excluded from the fit)"
        print(f"{lead:>5}{len(g):>7}{coverage(g, 1.0):>9.0%}{k:>10.2f}{flag}")

    if len(usable) < 3:
        print("\n  Not enough leads with data to fit — writing no calibration.")
        return 1

    leads = np.array(sorted(usable))
    ks = np.array([usable[L] for L in leads])
    slope, icept, r, pval, _ = stats.linregress(leads, ks)
    fitted = {}
    running = 1.0
    for L in range(1, 8):
        k = max(1.0, icept + slope * L)
        running = max(running, k)          # monotone: never narrower further out
        fitted[L] = float(round(min(running, MAX_FACTOR), 3))

    print(f"\nfit across leads: slope {slope:+.3f}/day, p={pval:.3f}, r={r:.2f}")
    print("fitted monotone factors:", {k: v for k, v in fitted.items()})

    blob = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_scored_hours": int(len(s)),
        "target_coverage": TARGET_COVERAGE,
        "method": ("multiplier on the distance from the point prediction to each band edge, "
                   "fitted linearly across lead times and forced monotone"),
        "fit": {"slope_per_day": round(float(slope), 4), "intercept": round(float(icept), 4),
                "p_value": round(float(pval), 4), "r": round(float(r), 4)},
        "raw_required_k": {str(k): v for k, v in raw.items()},
        "factors": {str(k): v for k, v in fitted.items()},
    }
    if args.dry_run:
        print("\n--dry-run, not written:")
        print(json.dumps(blob, indent=2))
        return 0
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
