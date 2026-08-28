"""
demo_predictions.py
--------------------
Small demo: picks 5 random held-out rows, shows the point prediction + 80%
interval from fit_surfer_count_model.py's GBT models alongside the actual
count and the main predictor conditions for that day/hour, so predictions
can be eyeballed against real inputs rather than just aggregate metrics.

Usage:
    python code/demo_predictions.py
    python code/demo_predictions.py --seed 7 --n 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_surfer_count_model import load_and_prepare, standardize  # noqa: E402

POINT_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
# Quantile models need l2=0.0 / shallower trees — l2=1.0 collapses the lower-quantile
# model to a constant 0 prediction for every row (see fit_surfer_count_model.py's
# fit_quantile_intervals docstring for the full diagnosis).
QUANTILE_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=0.0, random_state=42)

DISPLAY_PREDICTORS = [
    "rating_value", "tide_ft", "surf_min_ft", "surf_max_ft",
    "primary_swell_height_ft", "primary_swell_period_s",
    "wind_speed_mph", "weather_condition", "is_weekend",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42, help="Random seed for row selection (default 42)")
    p.add_argument("--n", type=int, default=5, help="Number of rows to show (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    X, y, df, numeric_cols = load_and_prepare()
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )
    X_train, X_test = standardize(X_train, X_test, numeric_cols)

    point_model = HistGradientBoostingRegressor(loss="poisson", **POINT_KWARGS).fit(X_train, y_train)
    lower_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **QUANTILE_KWARGS).fit(X_train, y_train)
    upper_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **QUANTILE_KWARGS).fit(X_train, y_train)

    pred_point = np.clip(point_model.predict(X_test), 0, None)
    pred_lower = np.clip(lower_model.predict(X_test), 0, None)
    pred_upper = np.clip(upper_model.predict(X_test), 0, None)
    pred_upper = np.maximum(pred_upper, pred_lower)

    rng = np.random.RandomState(args.seed)
    sample_idx = rng.choice(len(y_test), size=min(args.n, len(y_test)), replace=False)

    rows = []
    for i in sample_idx:
        d = df_test.iloc[i]
        row = {
            "date": d["date"],
            "time": d["time_local"],
            "actual": int(d["surfer_count"]),
            "predicted": round(float(pred_point[i]), 1),
            "range_80pct": f"[{pred_lower[i]:.0f}, {pred_upper[i]:.0f}]",
        }
        for col in DISPLAY_PREDICTORS:
            row[col] = d[col]
        rows.append(row)

    table = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
