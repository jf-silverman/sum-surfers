"""
export_model_release.py
-----------------------
Packages the surfer-count forecast model so its accuracy can be checked from
a clone of this repo, without publishing the training corpus.

What it writes to data/model_release/:

    surf_count_model.joblib   fitted point + lower/upper quantile models,
                              the train-set standardization statistics, and
                              the exact feature column order
    holdout_test_set.csv      the 292 held-out rows the models never saw —
                              raw (un-standardized, human-readable) predictor
                              values, the true surfer count, and this run's
                              predictions
    release_metadata.json     library versions, split parameters, row counts

Why the fitted model and not just a scored CSV: a table of predictions next
to actuals is self-reported — someone can recompute the error from it but
cannot check that the model was not fit on those same rows. Shipping the
fitted model with a held-out set lets a reader run it themselves on rows it
never saw. It also means the training rows stay unpublished: the release
contains the 20% test split only.

The split is the same one used everywhere else in this project, so these are
literally the rows the reported metrics were computed on, not a fresh resample.
Since 2026-09-23 that means a FORWARD-IN-TIME split on whole days: the model is
fitted on the earliest days and held out on the most recent. It used to be a
random 80/20, which leaked — with ~13 rows per day, random assignment put hours
from the same day on both sides, so the model learned a day's crowd level from
its own hours and was scored on the rest of that day. That flattered the
published MAE by about 2 surfers (B21). The number here is now the one a
day-ahead forecast actually faces.

Usage:
    python code/export_model_release.py
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_surfer_count_model import (  # noqa: E402
    load_and_prepare, standardize, fit_quantile_model_robust, split_by_day,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = _PROJECT_ROOT / "data" / "model_release"

TEST_SIZE = 0.2
RANDOM_STATE = 42
LOWER_Q, UPPER_Q = 0.10, 0.90

# Same hyperparameters as fit_surfer_count_model.py and demo_predictions.py.
POINT_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=4,
                    l2_regularization=1.0, random_state=42)

# Context columns carried into the released CSV so a row is readable on its
# own (which day and hour it is) rather than being 31 anonymous numbers.
# is_night / is_weekend are deliberately absent: they are already feature
# columns, and repeating them here would put duplicate names in the CSV, which
# read_csv silently renames (is_night.1) and makes column selection ambiguous.
CONTEXT_COLS = ["filename", "date", "time_local", "hour_local", "day_of_week",
                "month", "weather_simple"]


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    X, y, df, numeric_cols = load_and_prepare()
    train_idx, test_idx = split_by_day(df, test_size=TEST_SIZE, scheme="forward")
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    X_train, X_test = standardize(X_train_raw, X_test_raw, numeric_cols)

    print(f"\nTrain rows: {len(y_train)} (not published)  |  held-out test rows: {len(y_test)}")

    point_model = HistGradientBoostingRegressor(loss="poisson", **POINT_KWARGS).fit(X_train, y_train)
    lower_model, lower_leaf = fit_quantile_model_robust(X_train, y_train, LOWER_Q)
    upper_model, upper_leaf = fit_quantile_model_robust(X_train, y_train, UPPER_Q)

    pred = np.clip(point_model.predict(X_test), 0, None)
    lower = np.clip(lower_model.predict(X_test), 0, None)
    upper = np.maximum(np.clip(upper_model.predict(X_test), 0, None), lower)

    # The standardization statistics are part of the model: without them, raw
    # predictor values in the released CSV cannot be fed to these models.
    # Recomputing them from the test rows would use different (and leaky)
    # numbers, so they are shipped rather than re-derived.
    bundle = {
        "point_model": point_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "lower_quantile": LOWER_Q,
        "upper_quantile": UPPER_Q,
        "feature_columns": list(X.columns),
        "numeric_cols": list(numeric_cols),
        "standardize_mean": X_train_raw[numeric_cols].mean(),
        "standardize_std": X_train_raw[numeric_cols].std().replace(0, 1),
    }
    model_path = RELEASE_DIR / "surf_count_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"Wrote {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")

    # Raw, un-standardized features: readable as real quantities (tide in feet,
    # energy in kJ) instead of z-scores, and the eval script applies the shipped
    # standardization itself.
    out = pd.concat(
        [df_test[CONTEXT_COLS].reset_index(drop=True),
         X_test_raw.reset_index(drop=True)],
        axis=1,
    )
    out["actual_surfer_count"] = y_test.to_numpy()
    out["predicted_surfer_count"] = pred.round(4)
    out[f"pred_q{int(LOWER_Q * 100):02d}"] = lower.round(4)
    out[f"pred_q{int(UPPER_Q * 100):02d}"] = upper.round(4)
    csv_path = RELEASE_DIR / "holdout_test_set.csv"
    # Default float formatting already writes round-trippable repr; the matching
    # float_precision="round_trip" is needed on the READ side (see
    # eval_surf_count_model.py) because pandas' default C parser is not
    # correctly rounded.
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(out)} rows, {len(out.columns)} columns, "
          f"{csv_path.stat().st_size / 1024:.0f} KB)")

    covered = ((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)).mean()
    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": {"scheme": "forward_by_day",
                  "description": ("The most recent 20% of DAYS are held out and the model is "
                                  "fitted on the earlier ones. No day appears on both sides. "
                                  "This replaced a random 80/20 split on 2026-09-23, which "
                                  "leaked: at ~13 rows per day it put hours from the same day "
                                  "in train and test, flattering MAE by about 2 surfers."),
                  "test_size": TEST_SIZE,
                  "n_train": int(len(y_train)), "n_test": int(len(y_test))},
        "test_date_range": [str(df_test["date"].min()), str(df_test["date"].max())],
        "quantiles": {"lower": LOWER_Q, "upper": UPPER_Q,
                      "lower_min_samples_leaf": lower_leaf, "upper_min_samples_leaf": upper_leaf},
        "point_model_kwargs": POINT_KWARGS,
        "n_features": int(X.shape[1]),
        "holdout_metrics": {
            "mae": float(np.mean(np.abs(pred - y_test.to_numpy()))),
            "rmse": float(np.sqrt(np.mean((pred - y_test.to_numpy()) ** 2))),
            "mean_bias": float(np.mean(pred - y_test.to_numpy())),
            "interval_coverage_80pct": float(covered),
        },
        # The README's calibration section reports 71.6% for the same interval
        # on the same rows. Both are right: the calibration chart fits a
        # 9-level quantile ladder (0.10..0.90) and enforces monotonicity across
        # all of it, which nudges the q0.90 bound up and catches two more rows.
        # This release ships only the q0.10/q0.90 pair, so it enforces
        # monotonicity between those two alone. The gap is 2 rows out of 292.
        # No specific README figure is quoted here: an earlier version hardcoded
        # one (71.6%), and it went stale the moment the underlying counts changed.
        "coverage_note": (
            "Coverage here is computed from the shipped q0.10/q0.90 pair alone. The README's "
            "calibration chart fits a 9-level quantile ladder and enforces monotonicity "
            "across all levels, so its 80% figure can differ from this one by a row or two."
        ),
        # Version-sensitive: a joblib pickle is not guaranteed to load under a
        # different scikit-learn. Recorded so a load failure is diagnosable
        # rather than mysterious.
        "versions": {"python": platform.python_version(),
                     "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "joblib": joblib.__version__},
    }
    meta_path = RELEASE_DIR / "release_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}")

    print("\nHeld-out metrics recorded in the release:")
    for k, v in meta["holdout_metrics"].items():
        print(f"  {k:<24} {v:.4f}")


if __name__ == "__main__":
    main()
