"""
eval_crowd_rating.py
---------------------
Derives and validates the 1-5 crowd rating used by the forecast charts.

Two questions, both answered from the data rather than asserted:

1. **Where do the band edges go?** The bands are the quintiles of the actual
   hourly counts the model is fit on (`data/training_features.csv`, after
   `load_and_prepare()`'s dropna), so each level means "quieter than / busier
   than this share of recorded hours" rather than an invented cut-off.

2. **Which statistic should the rating be read off?** The forecast regresses
   to the mean (docs/known_bugs.md L02: it over-predicts quiet hours and
   under-predicts crowded ones), so a rating taken from the median point
   estimate cannot reach level 5 as often as reality does. This script
   measures that on a held-out split instead of assuming it, and compares
   the median against two upper-tail readings of the SAME three fitted
   quantile models the chart already fits (0.10 / 0.50 / 0.90) — no extra
   model fits, since q0.75 is interpolated between q0.50 and q0.90.

Usage:
    python code/eval_crowd_rating.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crowd_rating import (  # noqa: E402
    CROWD_LEVELS, RATING_QUANTILE, crowd_level, quantile_at, quintile_edges,
)
from fit_surfer_count_model import load_and_prepare, fit_quantile_model_robust  # noqa: E402

FAN_LEVELS = [0.10, 0.50, 0.90]


def main():
    X, y, df, numeric_cols = load_and_prepare()

    print("\n=== Count distribution the bands come from ===")
    print(f"{len(y):,} hourly rows, {df['date'].min()} to {df['date'].max()}")
    print(f"mean {y.mean():.2f}, median {y.median():.0f}, max {y.max():.0f}, "
          f"{(y == 0).mean():.1%} empty hours")
    edges = quintile_edges(y)
    print(f"Measured quintile edges (20/40/60/80th pct): {edges}")
    print(f"Band edges in use (CROWD_LEVELS): {[b['max'] for b in CROWD_LEVELS[:-1]]}")
    for band in CROWD_LEVELS:
        lo, hi = band["min"], band["max"]
        in_band = (y >= lo) & (y <= (hi if hi is not None else np.inf))
        hi_s = f"{hi:.0f}" if hi is not None else "+"
        print(f"  L{band['level']} {band['name']:<9} {lo:.0f}-{hi_s:<3} "
              f"{in_band.sum():4d} hours ({in_band.mean():5.1%})")

    train_mean = X[numeric_cols].mean()
    train_std = X[numeric_cols].std().replace(0, 1)
    X_std = X.copy()
    X_std[numeric_cols] = (X_std[numeric_cols] - train_mean) / train_std
    X_tr, X_te, y_tr, y_te = train_test_split(X_std, y, test_size=0.2, random_state=42)

    print(f"\nFitting {len(FAN_LEVELS)} quantile models on {len(X_tr)} rows "
          f"(held out {len(X_te)})...")
    models = {lv: fit_quantile_model_robust(X_tr, y_tr, lv, allow_degenerate=True)[0]
              for lv in FAN_LEVELS}
    preds = {lv: np.maximum(0.0, models[lv].predict(X_te)) for lv in FAN_LEVELS}
    # Monotone, same as the chart does.
    preds[0.50] = np.maximum(preds[0.10], preds[0.50])
    preds[0.90] = np.maximum(preds[0.50], preds[0.90])

    q = {lv: preds[lv] for lv in FAN_LEVELS}
    def at(p):
        return np.array([quantile_at(p, {0.10: a, 0.50: b, 0.90: c})
                         for a, b, c in zip(q[0.10], q[0.50], q[0.90])])

    candidates = {"median (q0.50)": preds[0.50]}
    for p in (0.55, 0.60, 0.65, 0.70, 0.75):
        candidates[f"q{p:.2f} (interpolated)"] = at(p)
    candidates["q0.90"] = preds[0.90]

    actual_level = np.array([crowd_level(v) for v in y_te])
    n5 = int((actual_level == 5).sum())
    print(f"\nHeld-out hours: {len(y_te)}; {n5} of them are truly level 5 "
          f"({n5 / len(y_te):.1%}), MAE of the median model "
          f"{np.abs(preds[0.50] - y_te.values).mean():.2f} surfers")

    print("\n=== Rating source comparison (held-out) ===")
    print(f"{'source':<22} {'exact':>7} {'within1':>8} {'meanErr':>8} "
          f"{'L5 recall':>10} {'L5 pred':>8} {'L1 recall':>10}")
    for name, values in candidates.items():
        lvl = np.array([crowd_level(v) for v in values])
        exact = (lvl == actual_level).mean()
        within1 = (np.abs(lvl - actual_level) <= 1).mean()
        mean_err = (lvl - actual_level).mean()
        l5_recall = (lvl[actual_level == 5] == 5).mean() if n5 else float("nan")
        l5_pred = (lvl == 5).mean()
        l1_mask = actual_level == 1
        l1_recall = (lvl[l1_mask] == 1).mean() if l1_mask.any() else float("nan")
        star = "  <- in use" if abs(RATING_QUANTILE - _quantile_of(name)) < 1e-9 else ""
        print(f"{name:<22} {exact:>6.1%} {within1:>8.1%} {mean_err:>+8.2f} "
              f"{l5_recall:>9.1%} {l5_pred:>7.1%} {l1_recall:>9.1%}{star}")

    print("\nColumns: exact = rating equals the rating of the real count; within1 = "
          "off by at most one level; meanErr = mean signed level error (negative = "
          "the rating reads quieter than reality); L5 recall = share of genuinely "
          "level-5 hours the rating calls level 5; L5 pred = share of all hours the "
          "rating calls level 5 (reality's share is printed above).")

    print("\n=== Confusion, rating in use (rows = actual level, cols = predicted) ===")
    in_use = np.array([crowd_level(v) for v in
                       [quantile_at(RATING_QUANTILE, {0.10: a, 0.50: b, 0.90: c})
                        for a, b, c in zip(q[0.10], q[0.50], q[0.90])]])
    cm = pd.crosstab(pd.Series(actual_level, name="actual"),
                     pd.Series(in_use, name="rated"), dropna=False)
    print(cm.to_string())


def _quantile_of(name):
    if name.startswith("median"):
        return 0.50
    return float(name.split("q")[1][:4])


if __name__ == "__main__":
    main()
