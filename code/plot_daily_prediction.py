"""
plot_daily_prediction.py
--------------------------
Generates a daily surfer-count prediction chart (point estimate = median GBT
model, 33%/66% quantile bands, tide overlay, wave-energy bars, weather-coded
markers, night-hour shading, model/detector info footer) and saves it to
data/charts/surfer_count_YYYY-MM-DD.png.

Meant to run once per day (not tied to the twice-weekly clip-collection
cron — this only needs the live Surfline forecast + the existing trained
model, no new clips required).

Usage:
    python code/plot_daily_prediction.py                # today
    python code/plot_daily_prediction.py --date 2026-08-29
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_clips as gc  # noqa: E402 — needed for get_light_window() (real dawn/dusk)
import get_surf_predictors as sp  # noqa: E402
from fit_surfer_count_model import load_and_prepare, fit_quantile_model_robust  # noqa: E402
from predict_surf_count import build_feature_row, MEAN_KWARGS  # noqa: E402
from build_training_features import simplify_weather_condition  # noqa: E402
import pytz  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = _PROJECT_ROOT / "data" / "charts"

# Real, verified detector stats — pulled directly from the actual YOLOv8s training
# log for the production model (data/model_out/20251013/train/runs/detect/train13/
# results.csv, final epoch 60), not estimated. "Specificity" isn't a standard
# object-detection metric (no fixed universe of negative boxes to measure against,
# unlike binary classification) — recall is the direct analog to sensitivity.
DETECTOR_PRECISION = 0.87843
DETECTOR_RECALL = 0.80618  # = sensitivity

# Dataset-wide hour range the model has ANY training examples for (used only to flag
# extrapolated hours below) — NOT the same thing as "is it light on this specific day",
# which varies by ~1hr+ across seasons and is computed per-day via get_light_window()
# instead (a fixed 5-20 filter here previously showed a confident-looking prediction
# for 5am on a day whose real dawn was 6:10am — the model doesn't know today's specific
# dawn time, only the coarse is_night flag, so it happily extrapolated).
TRAINED_HOUR_MIN, TRAINED_HOUR_MAX = 5, 20

WEATHER_COLORS = {"CLEAR": "#DDA43A", "CLOUDY_OVERCAST": "#7F8C8D", "RAIN": "#2E86AB", "FOG": "#95A5A6"}
WEATHER_MARKERS = {"CLEAR": "o", "CLOUDY_OVERCAST": "s", "RAIN": "^", "FOG": "D"}
WEATHER_ABBREV = {"CLEAR": "clear", "CLOUDY_OVERCAST": "cloudy", "RAIN": "rain", "FOG": "fog"}
READABLE_NAMES = {
    "tide_ft": "Tide", "hour_cos": "Time of day", "hour_sin": "Time of day",
    "is_weekend": "Weekend", "energy_nearshore_kj": "Wave energy (nearshore)",
    "energy_offshore_kj": "Wave energy (offshore)", "temperature_f": "Temperature",
    "real_temperature_f": "Temperature (observed)", "real_humidity_pct": "Humidity (observed)",
    "real_cloud_cover_pct": "Cloud cover (observed)", "real_pressure_mb": "Pressure (observed)",
    "consistency_wave_count": "Wave consistency", "primary_swell_height_ft": "Swell height",
    "is_night": "Night", "pressure_mb": "Pressure", "wind_speed_mph": "Wind speed",
    "wind_gust_mph": "Wind gust", "rating_value": "Surf rating",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Target date YYYY-MM-DD (default: today, local)")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()

    X, y, df, numeric_cols = load_and_prepare()
    weather_categories = set(df["weather_simple"].unique())
    train_mean = X[numeric_cols].mean()
    train_std = X[numeric_cols].std().replace(0, 1)
    X_std = X.copy()
    X_std[numeric_cols] = (X_std[numeric_cols] - train_mean) / train_std

    Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
    importance_model = HistGradientBoostingRegressor(loss="poisson", **MEAN_KWARGS).fit(Xi_train, yi_train)
    perm = permutation_importance(importance_model, Xi_test, yi_test, n_repeats=15,
                                    random_state=42, scoring="neg_mean_absolute_error")
    top_predictors = pd.Series(perm.importances_mean, index=Xi_test.columns).sort_values(ascending=False).head(5)

    q17_model, _ = fit_quantile_model_robust(X_std, y, 0.17)
    q335_model, _ = fit_quantile_model_robust(X_std, y, 0.335)
    median_model, _ = fit_quantile_model_robust(X_std, y, 0.5)
    q665_model, _ = fit_quantile_model_robust(X_std, y, 0.665)
    q83_model, _ = fit_quantile_model_robust(X_std, y, 0.83)

    by_hour = sp.build_predictor_map()
    local_tz = pytz.timezone(gc.LOCATION["timezone"])
    dawn, dusk = gc.get_light_window(target_date, local_tz)
    print(f"Real dawn/dusk for {target_date}: {dawn.strftime('%-I:%M %p')} - {dusk.strftime('%-I:%M %p')}")
    day_hours = sorted(hk for hk in by_hour if hk.date() == target_date and dawn.hour <= hk.hour <= dusk.hour)
    if not day_hours:
        print(f"No forecast data available for {target_date} (outside the live today+tomorrow window).")
        return

    records = []
    for hk in day_hours:
        predictors = by_hour[hk]
        feat_row = build_feature_row(hk, predictors, numeric_cols, weather_categories,
                                      train_mean, train_std, X_std.columns)
        q17 = max(0.0, float(q17_model.predict(feat_row)[0]))
        q335 = max(0.0, float(q335_model.predict(feat_row)[0]))
        point = max(0.0, float(median_model.predict(feat_row)[0]))
        q665 = max(0.0, float(q665_model.predict(feat_row)[0]))
        q83 = max(0.0, float(q83_model.predict(feat_row)[0]))
        q335 = max(q335, q17)
        point = max(point, q335)
        q665 = max(q665, point)
        q83 = max(q83, q665)
        weather_simple, is_night = simplify_weather_condition(predictors.get("weather_condition", ""))
        tide_ft = float(predictors.get("tide_ft", 0) or 0)
        energy_nearshore_kj = float(predictors.get("energy_nearshore_kj", 0) or 0)
        in_training_range = TRAINED_HOUR_MIN <= hk.hour <= TRAINED_HOUR_MAX
        records.append(dict(hour=hk, point=point, q17=q17, q335=q335, q665=q665, q83=q83,
                             weather_simple=weather_simple, is_night=is_night,
                             tide_ft=tide_ft, energy_nearshore_kj=energy_nearshore_kj,
                             in_training_range=in_training_range))

    d = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y_top = d["q83"].max() * 1.18
    ax.set_ylim(bottom=0, top=y_top)

    for _, row in d.iterrows():
        if row["is_night"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor="#2C3E50", alpha=0.08, hatch="//", edgecolor="#2C3E50", linewidth=0, zorder=0)
    for _, row in d.iterrows():
        if not row["in_training_range"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor="#B0413E", alpha=0.06, hatch="xx", edgecolor="#B0413E", linewidth=0, zorder=0)

    bar_width = timedelta(minutes=22, seconds=30)
    energy_max = d["energy_nearshore_kj"].max()
    energy_scale = (y_top * 0.20) / energy_max if energy_max > 0 else 0
    bar_heights = d["energy_nearshore_kj"] * energy_scale
    ax.bar(d["hour"], bar_heights, width=bar_width, color="#C97B3D", alpha=0.35,
           zorder=1, label="Wave energy, nearshore (kJ)")
    for x, h, val in zip(d["hour"], bar_heights, d["energy_nearshore_kj"]):
        ax.annotate(f"{val:.0f}", (x, h), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=6.5, color="#8A5623", rotation=90, va="bottom")

    ax.fill_between(d["hour"], d["q17"], d["q83"], color="#4C72B0", alpha=0.18, label="66% range")
    ax.fill_between(d["hour"], d["q335"], d["q665"], color="#4C72B0", alpha=0.35, label="33% range")
    ax.plot(d["hour"], d["point"], color="#1B3B6F", linewidth=2, zorder=3, label="Median")

    for wx in ["CLEAR", "CLOUDY_OVERCAST", "RAIN", "FOG"]:
        sub = d[d["weather_simple"] == wx]
        if len(sub) == 0:
            ax.scatter([], [], color=WEATHER_COLORS[wx], marker=WEATHER_MARKERS[wx],
                       s=100, edgecolor="white", linewidth=1, label=WEATHER_ABBREV[wx])
            continue
        point_alpha = [1.0 if r else 0.35 for r in sub["in_training_range"]]
        ax.scatter(sub["hour"], sub["point"], color=WEATHER_COLORS[wx], marker=WEATHER_MARKERS[wx],
                   s=100, zorder=4, edgecolor="white", linewidth=1, alpha=point_alpha, label=WEATHER_ABBREV[wx])

    for _, row in d.iterrows():
        label = WEATHER_ABBREV.get(row["weather_simple"], row["weather_simple"])
        if row["is_night"]:
            label += "\n(night)"
        if not row["in_training_range"]:
            label += "\n(no training\ndata this hour)"
        ax.annotate(label, (row["hour"], row["point"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    color="#B0413E" if not row["in_training_range"] else "#333333")

    seen_names, predictor_lines = [], []
    for feat in top_predictors.index:
        label = READABLE_NAMES.get(feat, feat)
        if label in seen_names:
            continue
        seen_names.append(label)
        predictor_lines.append(f"  • {label}")

    predictors_str = ", ".join(seen_names)
    info_text = (
        f"Model: gradient-boosted trees (quantile regression), point estimate = median model  "
        f"|  Top predictors (live fit): {predictors_str}\n"
        f"Surfer detector (YOLOv8s, train13 — actual training log): "
        f"precision {DETECTOR_PRECISION:.1%}, recall {DETECTOR_RECALL:.1%}"
    )
    fig.text(0.5, 0.01, info_text, fontsize=8, ha="center", va="bottom", color="#333333")

    ax2 = ax.twinx()
    ax2.plot(d["hour"], d["tide_ft"], color="#1B998B", linestyle="--", linewidth=2, zorder=2, label="Tide (ft)")
    ax2.set_ylabel("Tide (ft)", color="#1B998B")
    ax2.tick_params(axis="y", colors="#1B998B")
    tide_pad = (d["tide_ft"].max() - d["tide_ft"].min()) * 0.15 or 0.5
    ax2.set_ylim(bottom=d["tide_ft"].min() - tide_pad, top=d["tide_ft"].max() + tide_pad)
    ax2.grid(False)

    ax.set_title(f"Predicted surfer count — {target_date.strftime('%A, %B %d, %Y')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted surfer count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")

    night_patch = Patch(facecolor="#2C3E50", alpha=0.08, hatch="//", edgecolor="#2C3E50", label="Night hours")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2 + [night_patch], labels1 + labels2 + ["Night hours"],
              loc="upper left", ncol=2, fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"surfer_count_{target_date.isoformat()}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
