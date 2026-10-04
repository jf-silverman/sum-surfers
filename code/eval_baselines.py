"""
eval_baselines.py
-----------------
Scores the forecast model against naive reference forecasts, and against the
noise floor of its own target.

Why this exists: until 2026-10-03 this project reported MAE with nothing to
compare it to. A point-accuracy figure without a reference forecast is not a
claim about quality -- M4 normalises against Naive2, MASE exists so MAE can be
read against a naive denominator, and meteorology scores skill against
climatology or persistence, taking whichever naive reference is STRONGEST. The
context here is unflattering enough to make the omission matter: mean hourly
count is about 15, so a backtest MAE of 7.6 is roughly half the mean.

The four references, all computable from data already on hand:

  climatology      mean count by (hour of day x is_weekend), fitted on train only
  persistence      the same hour yesterday
  seasonal naive   the same hour 7 days ago (preserves day of week)
  trailing level   the mean of the last 7 days, which is the reference that
                   specifically exposes the level-lag problem

A naive forecast is only scored where its own input exists (persistence needs
yesterday to have been counted), so each is reported with its own n. Scoring
them only on the intersection of all four would discard most of the test set and
flatter whichever reference needs the least history.

The floor comes from the detector: three frames are extracted 1.5s apart per
clip, so the spread between one frame and their mean is a free estimate of how
much noise is in the target itself. No forecast can beat that.

Usage:
    python code/eval_baselines.py
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
from fit_surfer_count_model import (  # noqa: E402
    load_and_prepare, standardize, split_by_day, fit_quantile_model_robust,
)
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PREDS = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
POINT_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=4,
                    l2_regularization=1.0, random_state=42)


def target_noise_floor():
    """MAE of a single extracted frame against the 3-frame mean for that clip.

    Not a forecast error at all -- it is how much the measured target moves
    between frames 1.5s apart. A forecast cannot be more accurate than the thing
    it is predicting is stable.
    """
    df = pd.read_csv(PREDS, float_precision="round_trip")
    cols = ["frame_count_1", "frame_count_2", "frame_count_3", "frame_count_mean"]
    if not all(c in df.columns for c in cols):
        return None
    d = df.dropna(subset=cols)
    if d.empty:
        return None
    errs = np.concatenate([(d[c] - d.frame_count_mean).abs().to_numpy()
                           for c in cols[:3]])
    return float(errs.mean()), len(d)


def main():
    X, y, df, numeric_cols = load_and_prepare()
    tr, te = split_by_day(df, test_size=0.2, scheme="forward")

    d = df.copy()
    d["count"] = y.values
    d["dt"] = pd.to_datetime(d["date"])
    d["hour"] = d["hour_local"].astype(int) if d["hour_local"].dtype != object \
        else d["hour_local"].astype(str).str.slice(0, 2).astype(int)
    d["is_weekend"] = d["is_weekend"].astype(int)

    train, test = d.iloc[tr], d.iloc[te]
    print(f"Forward split: {len(train)} train rows ({train.dt.min().date()}–{train.dt.max().date()}), "
          f"{len(test)} test rows ({test.dt.min().date()}–{test.dt.max().date()})")
    print(f"Mean actual count in the test set: {test['count'].mean():.2f}\n")

    # lookup of (date, hour) -> count, built from TRAIN ONLY where a naive
    # forecast would have had to use history; test-set history is allowed for
    # persistence because by then that day has happened.
    hist = {(r.dt.date(), r.hour): r.count for r in d.itertuples()}

    results = {}

    # --- climatology: fitted on train only ---------------------------------
    clim = train.groupby(["hour", "is_weekend"])["count"].mean()
    overall = train["count"].mean()
    pred = [clim.get((r.hour, r.is_weekend), overall) for r in test.itertuples()]
    results["climatology (hour x weekend)"] = (np.array(pred), test["count"].to_numpy())

    # --- persistence and seasonal naive ------------------------------------
    for label, days_back in (("persistence (yesterday, same hour)", 1),
                             ("seasonal naive (7 days ago)", 7)):
        p, a = [], []
        for r in test.itertuples():
            prev = hist.get(((r.dt - timedelta(days=days_back)).date(), r.hour))
            if prev is not None:
                p.append(prev); a.append(r.count)
        results[label] = (np.array(p), np.array(a))

    # --- trailing 7-day level ----------------------------------------------
    p, a = [], []
    daily = d.groupby(d.dt.dt.date)["count"].mean()
    for r in test.itertuples():
        window = [daily[k] for k in daily.index
                  if r.dt.date() - timedelta(days=7) <= k < r.dt.date()]
        if window:
            p.append(float(np.mean(window))); a.append(r.count)
    results["trailing 7-day mean level"] = (np.array(p), np.array(a))

    # --- the model itself, same split --------------------------------------
    Xtr, Xte = standardize(X.iloc[tr], X.iloc[te], numeric_cols)
    m = HistGradientBoostingRegressor(loss="poisson", **POINT_KWARGS).fit(Xtr, y.iloc[tr].values)
    results["THE MODEL"] = (np.clip(m.predict(Xte), 0, None), y.iloc[te].to_numpy())

    # --- report -------------------------------------------------------------
    naive_maes = {k: np.abs(p - a).mean() for k, (p, a) in results.items() if k != "THE MODEL"}
    best_naive = min(naive_maes, key=naive_maes.get)
    denom = naive_maes[best_naive]

    print(f"{'reference':<34}{'n':>6}{'MAE':>8}{'bias':>8}{'MASE*':>8}{'skill':>8}")
    for k, (p, a) in results.items():
        mae = np.abs(p - a).mean()
        print(f"{k:<34}{len(p):>6}{mae:>8.2f}{(p - a).mean():>+8.2f}"
              f"{mae / denom:>8.2f}{(1 - mae / denom) * 100:>7.0f}%")
    print(f"\n*MASE and skill are against the STRONGEST naive reference "
          f"({best_naive}, MAE {denom:.2f}), which is the demanding convention.")

    floor = target_noise_floor()
    if floor:
        print(f"\nTarget noise floor: {floor[0]:.2f} surfers "
              f"(single frame vs the 3-frame mean, {floor[1]} clips).")
        print("No forecast can beat that; it is measurement noise in the target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
