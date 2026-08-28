"""
predict_surf_count.py
----------------------
Phase 3 of the surfer-count modeling plan: pulls live forward-looking
predictors from Surfline (today + tomorrow, via get_surf_predictors.py's
build_predictor_map() — no auth token needed) and outputs a surfer-count
prediction with an 80% range, for one or more times.

Trains fresh, "production" versions of the GBT point + quantile models on
ALL of data/training_features.csv (not an 80/20 split — the held-out split
in fit_surfer_count_model.py already validated generalization; here we want
every available row) using the same feature engineering and hyperparameters
validated there.

Usage:
    python code/predict_surf_count.py                          # tomorrow, several default daylight hours
    python code/predict_surf_count.py --date 2026-08-28 --hours 07:10,12:00
    python code/predict_surf_count.py --hours 07:10
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_surf_predictors as sp  # noqa: E402
from fit_surfer_count_model import load_and_prepare, fit_quantile_model_robust  # noqa: E402
from build_training_features import simplify_weather_condition  # noqa: E402

DEFAULT_HOURS = ["07:00", "10:00", "13:00", "16:00"]

MEAN_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Target date YYYY-MM-DD (default: tomorrow, local)")
    p.add_argument("--hours", help=f"Comma-separated HH:MM times (default: {','.join(DEFAULT_HOURS)})")
    return p.parse_args()


def train_production_models(X, y, weather_categories):
    """Fit 10th/50th/90th quantile GBT models plus a separate mean-based (Poisson
    loss) model, on ALL rows. The MEDIAN model is the range-consistent point
    estimate shown first — it's guaranteed to fall inside its own 80% range, which
    a mean-based estimate isn't for a skewed target like this one (right-skewed
    count data: the mean sits above the median, pulled up by the long right tail of
    crowded days). The MEAN model is kept and reported alongside it (not dropped)
    specifically so mean-vs-median divergence stays visible as a diagnostic — worth
    tracking as the dataset grows and models keep improving; a growing gap would
    signal increasing skew, a shrinking gap would signal it's easing. Quantile
    models use fit_quantile_model_robust() — see its docstring for why a fixed
    hyperparameter combo isn't safe here (silent collapse to a near-constant
    prediction, observed twice on this dataset already)."""
    lower_model, _ = fit_quantile_model_robust(X, y, 0.1)
    median_model, _ = fit_quantile_model_robust(X, y, 0.5)
    upper_model, _ = fit_quantile_model_robust(X, y, 0.9)
    mean_model = HistGradientBoostingRegressor(loss="poisson", **MEAN_KWARGS).fit(X, y)
    return median_model, lower_model, upper_model, mean_model


def build_feature_row(target_dt, predictors, numeric_cols, weather_categories, train_mean, train_std, template_columns):
    """Build one row of features matching the training design matrix exactly —
    same cyclical encoding, same weather-category collapsing, same standardization
    (using the production model's own train-set mean/std), reindexed to the exact
    training column set (missing dummy columns filled 0)."""
    row = {}
    for c in numeric_cols:
        val = predictors.get(c)
        row[c] = float(val) if val not in (None, "") else np.nan

    row["hour_sin"] = np.sin(2 * np.pi * target_dt.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * target_dt.hour / 24)
    row["month_sin"] = np.sin(2 * np.pi * target_dt.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * target_dt.month / 12)
    wind_dir = predictors.get("wind_direction_deg", 0) or 0
    swell_dir = predictors.get("primary_swell_direction_deg", 0) or 0
    row["wind_dir_sin"] = np.sin(2 * np.pi * wind_dir / 360)
    row["wind_dir_cos"] = np.cos(2 * np.pi * wind_dir / 360)
    row["swell_dir_sin"] = np.sin(2 * np.pi * swell_dir / 360)
    row["swell_dir_cos"] = np.cos(2 * np.pi * swell_dir / 360)
    row["is_weekend"] = int(target_dt.weekday() >= 5)

    wx_raw = predictors.get("weather_condition", "")
    wx_simple, is_night = simplify_weather_condition(wx_raw)
    row["is_night"] = int(is_night)
    for cat in sorted(weather_categories):
        if cat == sorted(weather_categories)[0]:
            continue  # drop_first matches training
        row[f"wx_{cat}"] = 1.0 if wx_simple == cat else 0.0

    df_row = pd.DataFrame([row])
    df_row[numeric_cols] = (df_row[numeric_cols] - train_mean) / train_std
    df_row = df_row.reindex(columns=template_columns, fill_value=0.0)
    df_row["const"] = 1.0
    return df_row


def main():
    args = parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = (datetime.now() + timedelta(days=1)).date()

    hours = args.hours.split(",") if args.hours else DEFAULT_HOURS

    print(f"Fetching live forecast (today + tomorrow) for Jack's...")
    by_hour = sp.build_predictor_map()
    if not by_hour:
        print("ERROR: no forecast data returned — check network/Surfline API status.")
        return

    print("Loading training data and fitting production models on all available rows...")
    X, y, df, numeric_cols = load_and_prepare()
    weather_categories = set(df["weather_simple"].unique())

    train_mean = X[numeric_cols].mean()
    train_std = X[numeric_cols].std().replace(0, 1)
    X_std = X.copy()
    X_std[numeric_cols] = (X_std[numeric_cols] - train_mean) / train_std

    median_model, lower_model, upper_model, mean_model = train_production_models(X_std, y, weather_categories)

    print(f"\n=== Surfer count prediction for {target_date} ===\n")
    for hh_mm in hours:
        h, m = map(int, hh_mm.strip().split(":"))
        target_dt = datetime.combine(target_date, datetime.min.time()).replace(hour=h, minute=m)
        hour_key = target_dt.replace(minute=0, second=0, microsecond=0)
        predictors = by_hour.get(hour_key)

        if predictors is None:
            print(f"{hh_mm}: no forecast data available for this hour (outside the live "
                  f"today+tomorrow window, or Surfline didn't return this hour)")
            continue

        feat_row = build_feature_row(target_dt, predictors, numeric_cols, weather_categories,
                                      train_mean, train_std, X_std.columns)

        lower_pred = max(0.0, float(lower_model.predict(feat_row)[0]))
        point_pred = max(lower_pred, float(median_model.predict(feat_row)[0]))
        upper_pred = max(point_pred, float(upper_model.predict(feat_row)[0]))
        mean_pred = max(0.0, float(mean_model.predict(feat_row)[0]))

        conditions = (
            f"rating={predictors.get('rating_key','?')} tide={predictors.get('tide_ft','?')}ft "
            f"surf={predictors.get('surf_min_ft','?')}-{predictors.get('surf_max_ft','?')}ft "
            f"swell={predictors.get('primary_swell_height_ft','?')}ft@{predictors.get('primary_swell_period_s','?')}s "
            f"wind={predictors.get('wind_speed_mph','?')}mph {predictors.get('weather_condition','?')}"
        )
        divergence_flag = ""
        if point_pred > 0:
            pct_diff = (mean_pred - point_pred) / point_pred
            if abs(pct_diff) > 0.25:
                divergence_flag = f"  [mean/median diverge {pct_diff:+.0%} — check for growing skew]"
        print(f"{hh_mm}: ~{point_pred:.0f} surfers [median] (80% range: {lower_pred:.0f}-{upper_pred:.0f})  "
              f"mean-based: ~{mean_pred:.0f}{divergence_flag}")
        print(f"        conditions: {conditions}")

    print(f"\nNote: the primary point estimate is the MEDIAN model — guaranteed to fall inside "
          f"its own range, which a mean-based estimate isn't for this right-skewed target. The "
          f"mean-based estimate (~6.15 MAE vs ~6.96 for median — see fit_surfer_count_model.py) "
          f"is shown alongside it as a diagnostic: watch for the two drifting apart over time as "
          f"more data comes in and models improve, which would signal changing skew in the "
          f"underlying crowd-size distribution. The 80% range's true empirical coverage varies by "
          f"dataset snapshot — check fit_surfer_count_model.py's latest output rather than "
          f"trusting a fixed number here. Treat all of this as a directional estimate, not a "
          f"precise count.")


if __name__ == "__main__":
    main()
