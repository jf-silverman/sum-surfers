"""
eval_surf_count_model.py
------------------------
Re-derives the forecast model's accuracy from the released artifacts, on rows
the model never trained on.

This is the check a reader runs. It needs only what is committed to the repo —
data/model_release/ — and deliberately does not import the training code or
touch the (unpublished) training corpus, so it works on a fresh clone.

What it does:

1. Loads the fitted models and re-predicts the held-out rows from their raw
   predictor values, applying the standardization statistics shipped with the
   models. It then checks those predictions against the ones recorded in the
   CSV. That check is the point: it shows the numbers below come out of the
   released model rather than being copied from a file.
2. Reports point accuracy (MAE, RMSE, bias) and prediction-interval coverage,
   with the two tail-miss rates reported separately — an interval can hit its
   overall coverage target while being badly wrong on one side, and averaging
   the two would hide it.

Usage:
    python code/eval_surf_count_model.py
    python code/eval_surf_count_model.py --show 15
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = _PROJECT_ROOT / "data" / "model_release"

COUNT_BUCKETS = [(0, 5), (5, 15), (15, 30), (30, 10_000)]


def load_release():
    model_path = RELEASE_DIR / "surf_count_model.joblib"
    csv_path = RELEASE_DIR / "holdout_test_set.csv"
    meta_path = RELEASE_DIR / "release_metadata.json"
    for p in (model_path, csv_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run `python code/export_model_release.py` first.")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        versions = meta.get("versions", {})
        sys.exit(
            f"Could not load {model_path.name} ({type(e).__name__}: {e}).\n"
            f"It was written with scikit-learn {versions.get('scikit_learn', '?')} / "
            f"Python {versions.get('python', '?')}; a pickled model is not guaranteed to "
            f"load under a different scikit-learn. Install the pinned versions from "
            f"docs/requirements.txt and retry."
        )
    # float_precision="round_trip" is required, not a nicety. pandas' default C
    # parser is not correctly rounded, so features come back up to ~1e-16 off.
    # That is normally irrelevant, but this model bins its features: a value
    # landing on the far side of a bin edge changes which leaf a row falls in,
    # and a 1e-16 input difference showed up as a 0.89-surfer prediction
    # difference on a real row here.
    return bundle, pd.read_csv(csv_path, float_precision="round_trip"), meta


def predict(bundle, df):
    """Rebuild the model's feature matrix from raw values and predict."""
    X = df[bundle["feature_columns"]].copy()
    numeric = bundle["numeric_cols"]
    X[numeric] = (X[numeric] - bundle["standardize_mean"]) / bundle["standardize_std"]

    point = np.clip(bundle["point_model"].predict(X), 0, None)
    lower = np.clip(bundle["lower_model"].predict(X), 0, None)
    upper = np.maximum(np.clip(bundle["upper_model"].predict(X), 0, None), lower)
    return point, lower, upper


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=10, help="Example rows to print (default 10)")
    args = ap.parse_args()

    bundle, df, meta = load_release()
    lo_q, hi_q = bundle["lower_quantile"], bundle["upper_quantile"]
    nominal = hi_q - lo_q

    print(f"Held-out rows : {len(df)}")
    if meta:
        s = meta["split"]
        print(f"Split         : test_size={s['test_size']}, random_state={s['random_state']} "
              f"({s['n_train']} train rows, not published)")
        print(f"Date range    : {meta['test_date_range'][0]} -> {meta['test_date_range'][1]}")
        print(f"Built with    : scikit-learn {meta['versions']['scikit_learn']}, "
              f"Python {meta['versions']['python']}")

    point, lower, upper = predict(bundle, df)

    # Reproducibility check: does the released model reproduce the released numbers?
    recorded = df["predicted_surfer_count"].to_numpy()
    max_drift = float(np.max(np.abs(point - recorded)))
    status = "OK" if max_drift < 1e-3 else "MISMATCH"
    print(f"\nReproduced predictions from the released model: {status} "
          f"(max difference vs. the CSV: {max_drift:.2e})")
    if status == "MISMATCH":
        print("  The shipped model does not reproduce the shipped predictions. Treat the\n"
              "  metrics below as coming from the model, not from the CSV, and re-export.")

    actual = df["actual_surfer_count"].to_numpy(dtype=float)
    err = point - actual
    print(f"\n{'=' * 60}\nPOINT ACCURACY\n{'=' * 60}")
    print(f"  MAE            : {np.mean(np.abs(err)):.3f} surfers")
    print(f"  RMSE           : {np.sqrt(np.mean(err ** 2)):.3f} surfers")
    print(f"  mean bias      : {np.mean(err):+.3f} surfers")
    print(f"  median |error| : {np.median(np.abs(err)):.3f} surfers")
    print(f"  mean actual    : {actual.mean():.2f} surfers")

    print(f"\n  By actual count:")
    print(f"    {'bucket':<12} {'n':>5} {'MAE':>8} {'bias':>9}")
    for lo, hi in COUNT_BUCKETS:
        m = (actual >= lo) & (actual < hi)
        if not m.any():
            continue
        label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        print(f"    {label:<12} {m.sum():>5} {np.mean(np.abs(err[m])):>8.2f} {np.mean(err[m]):>+9.2f}")

    below = float((actual < lower).mean())
    above = float((actual > upper).mean())
    covered = 1.0 - below - above
    print(f"\n{'=' * 60}\nPREDICTION INTERVAL (nominal {nominal:.0%}, q{lo_q:.2f}-q{hi_q:.2f})\n{'=' * 60}")
    print(f"  actual coverage : {covered:.1%}  (target {nominal:.0%})")
    print(f"  missed low      : {below:.1%}  (actual below the lower bound)")
    print(f"  missed high     : {above:.1%}  (actual above the upper bound)")
    print(f"  mean width      : {np.mean(upper - lower):.2f} surfers")
    if meta.get("coverage_note"):
        print(f"\n  Note: {meta['coverage_note']}")
    if covered < nominal:
        print(f"\n  Under nominal: the interval is narrower than it claims, so treat it as\n"
              f"  roughly a {covered:.0%} interval rather than a {nominal:.0%} one.")

    print(f"\n{'=' * 60}\nEXAMPLE ROWS\n{'=' * 60}")
    show = df.head(args.show)
    print(f"  {'date':<12} {'time':<9} {'actual':>7} {'pred':>7} {'interval':>14} {'hit':>4}")
    for i, row in show.iterrows():
        hit = "yes" if lower[i] <= row["actual_surfer_count"] <= upper[i] else "NO"
        print(f"  {row['date']:<12} {str(row['time_local']):<9} "
              f"{int(row['actual_surfer_count']):>7} {point[i]:>7.1f} "
              f"{f'[{lower[i]:.0f}, {upper[i]:.0f}]':>14} {hit:>4}")


if __name__ == "__main__":
    main()
