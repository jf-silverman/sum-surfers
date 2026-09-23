"""
plot_daily_prediction.py
--------------------------
Generates the daily surfer-count prediction chart (point estimate = median
GBT model, a continuous 10-90% prediction-interval fan built from 9 real
fitted quantile models rendered as a smooth gradient, an 80% range side
table, tide overlay, weather-coded markers, night-hour shading,
model/detector info footer, per-hour 1-5 crowd level) to
data/charts/surfer_count_YYYY-MM-DD.png, a 7-day crowd outlook
(data/charts/latest_week.png — a day x hour grid of crowd levels, see
generate_week_chart for why it is a separate chart rather than a wider
daily one), plus a detection-review image (real bounding boxes + labels on the day's
~8am crop, with the
model's predicted range/median for that hour overlaid) to
data/charts/latest_detection.gif. Both get a stable, git-tracked "latest"
copy and are embedded in README.md between the DAILY_CHART markers.

Meant to run once per day (not tied to the twice-weekly clip-collection
cron — the chart only needs the live Surfline forecast + the existing
trained model; the detection image needs today's own clip, which the
clip-collection cron already downloads separately).

Usage:
    python code/plot_daily_prediction.py                # forecast for tomorrow, detection image from today
    python code/plot_daily_prediction.py --date 2026-08-29   # forecast + detection both pinned to one date

By default (no --date) the forecast chart/table target tomorrow (so the
evening cron run shows tomorrow's forecast, not a same-day one running out
of remaining hours) while the detection image still uses today's own ~8am
crop — the two dates are independent unless --date pins both to the same
day (e.g. for backfill/testing a specific past date).
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_clips as gc  # noqa: E402 — needed for get_light_window() (real dawn/dusk)
import get_surf_predictors as sp  # noqa: E402
import detect_surfers as ds  # noqa: E402
from fit_surfer_count_model import (  # noqa: E402
    load_and_prepare, fit_quantile_model_robust, split_by_day,
)
from predict_surf_count import build_feature_row, add_tide_daylight_features, MEAN_KWARGS  # noqa: E402
from build_training_features import simplify_weather_condition  # noqa: E402
from crowd_rating import (  # noqa: E402
    CROWD_LEVELS, RATING_QUANTILE, band_text, crowd_level, level_info, rating_from_quantiles,
)
import pytz  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = _PROJECT_ROOT / "data" / "charts"

# Real, verified detector stats — pulled directly from the actual YOLOv8s training
# log for the production model (data/model_out/20251013/train/runs/detect/train13/
# results.csv, final epoch 60), not estimated. "Specificity" isn't a standard
# object-detection metric (no fixed universe of negative boxes to measure against,
# unlike binary classification) — recall is the direct analog to sensitivity.
# Validation metrics for the checkpoint actually deployed: best.pt of the
# 2026-09-21 fog retrain, which is epoch 46 of 60 (training log,
# data/model_out/20260921_fog/train/results.csv). Adopted 2026-09-21, replacing
# the October 2025 model (0.85634 / 0.82047, its epoch 51). Measured on a
# harder validation split than the previous model's — it now includes hazy fog
# frames — so the two pairs are not directly comparable; see PROJECT_HISTORY.md
# 2026-09-21 for the like-for-like whole-frame count comparison.
DETECTOR_PRECISION = 0.88461
DETECTOR_RECALL = 0.81342  # = sensitivity

# Dataset-wide hour range the model has ANY training examples for (used only to flag
# extrapolated hours below) — NOT the same thing as "is it light on this specific day",
# which varies by ~1hr+ across seasons and is computed per-day via get_light_window()
# instead (a fixed 5-20 filter here previously showed a confident-looking prediction
# for 5am on a day whose real first light was 6:10am — the model doesn't know today's specific
# first-light time, only the coarse is_night flag, so it happily extrapolated).
TRAINED_HOUR_MIN, TRAINED_HOUR_MAX = 5, 20

# Prediction-interval quantile levels the fan chart is built from -- 9 real
# fitted GBT quantile models (0.50 = median = point estimate), rendered as a
# continuous gradient by interpolating between them (see main()). Chosen as
# a 10%-90% span (an 80% central prediction interval) per Joel's request,
# in place of the old fixed 33%/66% bands.
# Just the three levels the chart actually draws: the 80% interval's two
# edges and the median. It used to fit nine and render a 40-band gradient;
# Joel found that too busy (2026-09-16), and nine fits cost nine models a run.
FAN_LEVELS = [0.10, 0.50, 0.90]

# How many days past today the week chart covers. The daily chart still
# targets tomorrow alone; this is the outlook chart's span (tomorrow through
# tomorrow+6). The predictor fetch asks for FORECAST_DAYS + 1 days because
# Surfline counts `days` from today inclusive.
FORECAST_DAYS = 7

WEATHER_COLORS = {"CLEAR": "#f2c14e", "CLOUDY_OVERCAST": "#9aa0a6", "RAIN": "#4fa3d1", "FOG": "#c9c9c9"}
WEATHER_MARKERS = {"CLEAR": "o", "CLOUDY_OVERCAST": "s", "RAIN": "^", "FOG": "D"}
WEATHER_ABBREV = {"CLEAR": "clear", "CLOUDY_OVERCAST": "cloudy", "RAIN": "rain", "FOG": "fog"}

# Dark theme — matches analysis/weekday_weekend_patterns/weekday_weekend_*.png (aqua blue /
# lime green on black), with a couple of complementary colors added for the
# extra series this chart needs (wave-energy bars, tide line, warning hatches).
BG_COLOR = "black"
AXES_BG = "#111111"
GRID_COLOR = "#333333"
TEXT_COLOR = "white"
MUTED_TEXT = "#bbbbbb"
AQUA = "#3ab4c9"      # primary — median line / quantile bands
LIME = "#9de35a"      # tide line, wave-energy bars
CORAL = "#ff6f61"     # out-of-training-range warning hatch/labels
NIGHT_COLOR = "#7a7aa8"  # night-hour shading
# Crowd levels 1-5 are ordered magnitude, not five separate things, so they get
# a single-hue sequential ramp rather than five hues — a categorical palette
# here would say "different", where the data says "more". The hue is AQUA, the
# chart's existing primary, stepped monotonically in lightness so it still
# reads as a ramp in grayscale or with any colour vision deficiency; level 4 is
# AQUA itself. Dark background, so the ramp runs dark (quiet) to bright
# (packed) — the busiest hours are the brightest cells.
CROWD_COLORS = {1: "#0f3139", 2: "#17596a", 3: "#22869c", 4: "#3ab4c9", 5: "#8fe0ef"}
# Ink on top of each of those, chosen for contrast against it.
CROWD_TEXT_COLORS = {1: "#e8f6f9", 2: "#e8f6f9", 3: "#04202a", 4: "#04202a", 5: "#04202a"}
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


# Refitting every night was churn, not learning. Measured 2026-09-23: adding one
# day of data (14 rows, 0.9% of the table) moved predictions by 0.78 surfers on
# average and up to 3.35, with 33 of 101 hours moving by more than a surfer —
# about 15% of the model's own error, for reasons unrelated to conditions. The
# models are now fitted on a schedule and reused in between, so a forecast only
# changes when the weather does or when the model is deliberately refreshed.
MODEL_CACHE_PATH = _PROJECT_ROOT / "data" / "model_cache" / "chart_models.joblib"
REFIT_AFTER_DAYS = 7          # refresh weekly...
REFIT_AFTER_NEW_ROWS = 100    # ...or sooner if this much new data has arrived


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Target date YYYY-MM-DD (default: today, local)")
    p.add_argument("--refit", action="store_true",
                   help="Refit the forecast models now, ignoring the cache")
    p.add_argument("--no-cache", action="store_true",
                   help="Fit fresh without reading or writing the cache")
    return p.parse_args()


def cached_models_are_usable(cache, n_rows, feature_names):
    """Whether a cached fit can still be used for today's chart.

    Refuses a cache whose feature columns differ from the current ones: a
    silently mismatched feature order would produce plausible-looking nonsense
    rather than an error.
    """
    if cache is None:
        return False, "no cache yet"
    if list(cache.get("feature_names", [])) != list(feature_names):
        return False, "feature columns changed"
    age_days = (datetime.now() - datetime.fromisoformat(cache["fitted_at"])).days
    if age_days >= REFIT_AFTER_DAYS:
        return False, f"cache is {age_days} days old"
    new_rows = n_rows - cache.get("n_rows", 0)
    if new_rows >= REFIT_AFTER_NEW_ROWS:
        return False, f"{new_rows} new rows since the fit"
    return True, f"fitted {age_days}d ago on {cache['n_rows']} rows, {new_rows} new since"


def load_model_cache():
    if not MODEL_CACHE_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_CACHE_PATH)
    except Exception as e:
        print(f"  WARNING: could not read the model cache ({type(e).__name__}: {e}); refitting.")
        return None


def save_model_cache(payload):
    try:
        MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, MODEL_CACHE_PATH)
    except Exception as e:
        print(f"  WARNING: could not write the model cache ({type(e).__name__}: {e})")


def main():
    args = parse_args()
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        detection_date = target_date
    else:
        today = datetime.now().date()
        target_date = today + timedelta(days=1)
        detection_date = today
    print(f"Forecast target date: {target_date} | Detection image date: {detection_date}")

    X, y, df, numeric_cols = load_and_prepare()
    weather_categories = set(df["weather_simple"].unique())
    train_mean = X[numeric_cols].mean()
    train_std = X[numeric_cols].std().replace(0, 1)
    X_std = X.copy()
    X_std[numeric_cols] = (X_std[numeric_cols] - train_mean) / train_std

    cache = None if (args.no_cache or args.refit) else load_model_cache()
    usable, why = cached_models_are_usable(cache, len(X_std), X_std.columns)
    if usable:
        print(f"  Reusing the cached forecast models ({why}). "
              f"Pass --refit to force a new fit.")
        quantile_models = cache["quantile_models"]
        top_predictors = cache["top_predictors"]
        fitted_at = datetime.fromisoformat(cache["fitted_at"])
        trained_rows = cache["n_rows"]
    else:
        print(f"  Fitting forecast models ({why}).")
        # Day-grouped forward split, not a random one — see split_by_day (B21).
        train_idx, test_idx = split_by_day(df, test_size=0.2, scheme="forward")
        Xi_train, Xi_test = X_std.iloc[train_idx], X_std.iloc[test_idx]
        yi_train, yi_test = y.iloc[train_idx], y.iloc[test_idx]
        importance_model = HistGradientBoostingRegressor(loss="poisson", **MEAN_KWARGS).fit(Xi_train, yi_train)
        perm = permutation_importance(importance_model, Xi_test, yi_test, n_repeats=15,
                                        random_state=42, scoring="neg_mean_absolute_error")
        top_predictors = pd.Series(perm.importances_mean, index=Xi_test.columns).sort_values(ascending=False).head(5)

    # Fan-chart quantiles: 9 real fitted models at 10%-90% (step 10), rendered
    # as a continuous-looking gradient by interpolating between them at plot
    # time (see the fill loop below) rather than fitting dozens of models.
    # FAN_LEVELS[4] (0.50) is the median / point estimate.
    # allow_degenerate=True: with ~12% of hours at zero surfers the true 10th
    # percentile is 0.00, so that level collapses to a flat zero by rights, not
    # by mis-fitting (see fit_surfer_count_model.ConstantQuantileModel). Before
    # 2026-09-16 that raised and killed the whole chart — the 09-15 run produced
    # nothing at all. A display chart can draw a flat band honestly; calibration
    # code still uses the default and still raises.
        quantile_models = {level: fit_quantile_model_robust(X_std, y, level, allow_degenerate=True)[0]
                           for level in FAN_LEVELS}
        fitted_at = datetime.now()
        trained_rows = len(X_std)
        if not args.no_cache:
            save_model_cache({"quantile_models": quantile_models,
                              "top_predictors": top_predictors,
                              "feature_names": list(X_std.columns),
                              "n_rows": trained_rows,
                              "fitted_at": fitted_at.isoformat(timespec="seconds")})

    degenerate_levels = [lv for lv, m in quantile_models.items() if getattr(m, "degenerate", False)]
    if degenerate_levels:
        print(f"  Degenerate quantile level(s): {degenerate_levels} — drawn as flat bands.")

    # One fetch covers both charts: today through today+FORECAST_DAYS (Surfline
    # counts `days` from today inclusive, so +1). Deliberately a single
    # build_predictor_map() call rather than one per chart — these endpoints
    # have been returning intermittent Cloudflare 403s, and asking for a wider
    # window costs exactly the same number of requests as asking for two days.
    by_hour = add_tide_daylight_features(sp.build_predictor_map(days=FORECAST_DAYS + 1))
    local_tz = pytz.timezone(gc.LOCATION["timezone"])

    def predict_for_hour(hk):
        """Runs the fitted quantile models for one by_hour key. Shared by the
        chart's day loop, the week chart, and the detection image's own-hour
        lookup below, so a prediction for any hour in by_hour (today through
        today+FORECAST_DAYS) is always computed the same way regardless of
        which date it's for."""
        predictors = by_hour.get(hk)
        if predictors is None:
            return None
        feat_row = build_feature_row(hk, predictors, numeric_cols, weather_categories,
                                      train_mean, train_std, X_std.columns)
        # Force monotonicity (each level >= the previous) -- independently fit
        # quantile models have no built-in guarantee they won't cross.
        quantiles = {}
        running_min = 0.0
        for level in FAN_LEVELS:
            val = max(running_min, float(quantile_models[level].predict(feat_row)[0]))
            quantiles[level] = val
            running_min = val
        point = quantiles[0.50]
        weather_simple, is_night = simplify_weather_condition(predictors.get("weather_condition", ""))
        tide_ft = float(predictors.get("tide_ft", 0) or 0)
        in_training_range = TRAINED_HOUR_MIN <= hk.hour <= TRAINED_HOUR_MAX
        # Which model features this hour actually HAS. Further out in the
        # forecast, or after a failed endpoint fetch, a feature can simply be
        # absent; the GBT still returns a number (it takes the "missing"
        # branch at every split using that feature), so nothing warns you.
        # Counting them here is what lets the charts and the CSV say which
        # hours are standing on less than the model was trained on, instead
        # of degrading quietly. Measured 2026-09-23: at days=8 every endpoint
        # is complete on every day, so this normally reads zero — it is the
        # alarm, not the routine case.
        missing = [c for c in numeric_cols if pd.isna(feat_row.iloc[0].get(c, np.nan))]
        level, level_value = rating_from_quantiles(quantiles)
        return dict(hour=hk, point=point, quantiles=quantiles,
                    weather_simple=weather_simple, is_night=is_night,
                    tide_ft=tide_ft, in_training_range=in_training_range,
                    missing_predictors=missing,
                    crowd_level=level, crowd_level_value=level_value,
                    crowd_level_low=crowd_level(quantiles[FAN_LEVELS[0]]),
                    crowd_level_high=crowd_level(quantiles[FAN_LEVELS[-1]]))

    def predict_nearest_hour(date_, hour, minute):
        """Finds the by_hour key on date_ closest to hour:minute and predicts
        for it — used for the detection image, whose crop time (e.g. 7:56)
        won't exactly match an on-the-hour by_hour key."""
        candidates = [hk for hk in by_hour if hk.date() == date_]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda hk: abs((hk.hour * 60 + hk.minute) - (hour * 60 + minute)))
        return predict_for_hour(nearest)

    dawn, dusk = gc.get_light_window(target_date, local_tz)
    print(f"First light / last light for {target_date}: "
          f"{dawn.strftime('%-I:%M %p')} - {dusk.strftime('%-I:%M %p')}")
    day_hours = sorted(hk for hk in by_hour if hk.date() == target_date and dawn.hour <= hk.hour <= dusk.hour)
    if not day_hours:
        print(f"No forecast data available for {target_date} (outside the live today+tomorrow window).")
        return

    records = [r for r in (predict_for_hour(hk) for hk in day_hours) if r is not None]
    d = pd.DataFrame(records)
    save_forecast_record(target_date, records)

    fig = plt.figure(figsize=(14, 6.5), facecolor=BG_COLOR)
    # The table panel carries four columns since the crowd level joined it
    # (2026-09-23) — at the old 3.2:1 it clipped its own right-hand column.
    gs = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.1], wspace=0.11)
    ax = fig.add_subplot(gs[0], facecolor=AXES_BG)
    ax_table = fig.add_subplot(gs[1], facecolor=AXES_BG)

    y_top = d["quantiles"].apply(lambda q: q[0.90]).max() * 1.18
    ax.set_ylim(bottom=0, top=y_top)

    for _, row in d.iterrows():
        if row["is_night"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor=NIGHT_COLOR, alpha=0.20, hatch="//", edgecolor=NIGHT_COLOR, linewidth=0, zorder=0)
    for _, row in d.iterrows():
        if not row["in_training_range"] or row["missing_predictors"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor=CORAL, alpha=0.12, hatch="xx", edgecolor=CORAL, linewidth=0, zorder=0)

    # One shaded 80% prediction interval, q0.10 to q0.90. This replaced a
    # 40-band alpha gradient interpolated across nine fitted quantiles
    # (2026-09-16): it looked like more information than the model has, and
    # the extra bands were reading as busy rather than informative.
    ax.fill_between(d["hour"],
                    d["quantiles"].apply(lambda q: q[0.10]),
                    d["quantiles"].apply(lambda q: q[0.90]),
                    color=AQUA, alpha=0.22, linewidth=0, zorder=2)
    fan_patch = Patch(facecolor=AQUA, alpha=0.22, label="80% prediction interval")
    ax.plot(d["hour"], d["point"], color=AQUA, linewidth=2.5, zorder=3, label="Median")

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
        if row["missing_predictors"]:
            label += f"\n({len(row['missing_predictors'])} predictors\nmissing)"
        ax.annotate(label, (row["hour"], row["point"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    color=CORAL if (not row["in_training_range"] or row["missing_predictors"])
                    else MUTED_TEXT)

    seen_names, predictor_lines = [], []
    for feat in top_predictors.index:
        label = READABLE_NAMES.get(feat, feat)
        if label in seen_names:
            continue
        seen_names.append(label)
        predictor_lines.append(f"  • {label}")

    predictors_str = ", ".join(seen_names)
    # Record which data this run's models were fit on. The forecast models are
    # refit from scratch every run against a table that grows nightly, so
    # without this stamp a published chart is not attributable to any
    # particular snapshot and an odd-looking older chart can't be diagnosed
    # after the fact.
    info_text = (
        f"Model: gradient-boosted trees (quantile regression), point estimate = median model  "
        f"|  Top predictors: {predictors_str}\n"
        f"Model fitted {fitted_at:%Y-%m-%d} on {trained_rows:,} detection-hours "
        f"({df['date'].min()} to {df['date'].max()}); reused since, refit weekly  "
        f"|  Surfer detector (YOLOv8s, Sept 2026 fog retrain — actual training log): "
        f"precision {DETECTOR_PRECISION:.1%}, recall {DETECTOR_RECALL:.1%}"
    )
    info_text += "\n" + crowd_key_text()
    incomplete = [r for r in records if r["missing_predictors"]]
    if incomplete:
        # Never silent: an hour the model scored on fewer inputs than it was
        # trained on is a weaker forecast, and saying so is the same courtesy
        # the "no training data this hour" hatch already extends.
        worst = max(len(r["missing_predictors"]) for r in incomplete)
        info_text += (f"\n{len(incomplete)} of {len(records)} hours are missing up to {worst} "
                      f"predictor(s) — those hours are marked and are weaker forecasts")
    if degenerate_levels:
        # Say so on the chart itself: a band pinned flat is a real statement
        # about the data, and a reader should not mistake it for a fitted curve.
        zero_pct = (y == 0).mean()
        info_text += (
            f"\n{', '.join(f'{lv:.0%}' for lv in degenerate_levels)} band flat: "
            f"{zero_pct:.0%} of recorded hours had no surfers, so that percentile is 0"
        )
    fig.text(0.5, 0.01, info_text, fontsize=8, ha="center", va="bottom", color=MUTED_TEXT)

    ax2 = ax.twinx()
    ax2.set_facecolor(AXES_BG)
    ax2.plot(d["hour"], d["tide_ft"], color=LIME, linestyle="--", linewidth=2, zorder=2, label="Tide (ft)")
    ax2.set_ylabel("Tide (ft)", color=LIME)
    ax2.tick_params(axis="y", colors=LIME)
    tide_pad = (d["tide_ft"].max() - d["tide_ft"].min()) * 0.15 or 0.5
    ax2.set_ylim(bottom=d["tide_ft"].min() - tide_pad, top=d["tide_ft"].max() + tide_pad)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)

    ax.set_title(f"Predicted surfer count — {target_date.strftime('%A, %B %d, %Y')}", color=TEXT_COLOR)
    ax.set_xlabel("Time", color=TEXT_COLOR)
    ax.set_ylabel("Predicted surfer count", color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.tick_params(axis="both", colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    night_patch = Patch(facecolor=NIGHT_COLOR, alpha=0.20, hatch="//", edgecolor=NIGHT_COLOR, label="Night hours")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(handles1 + handles2 + [fan_patch, night_patch], labels1 + labels2 + [fan_patch.get_label(), "Night hours"],
                        loc="upper left", ncol=2, fontsize=8, facecolor=AXES_BG, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    ax.grid(alpha=0.25, color=GRID_COLOR)

    # Table panel: hour -> point prediction -> 80% range, rendered as part of
    # the same figure/image rather than a separate markdown table, so chart and
    # table always render side by side. The point column was added 2026-09-16;
    # before that the table showed only the range, which left the reader no
    # single number to act on.
    ax_table.axis("off")
    ax_table.set_title("Predicted count, 80% range, crowd level", fontsize=10, pad=10, color=TEXT_COLOR)
    # The Crowd column is read off the RATING_QUANTILE reading of the same
    # fitted quantiles, not off the Predicted column beside it — see
    # code/crowd_rating.py for why, and code/eval_crowd_rating.py for the
    # held-out numbers. The two can therefore disagree by a level, which is
    # the point: the median under-reports crowded hours.
    cell_text = [[row["hour"].strftime("%-I:%M %p"),
                  f"{row['point']:.0f}",
                  f"{row['quantiles'][0.10]:.0f}–{row['quantiles'][0.90]:.0f}",
                  f"{row['crowd_level']} {level_info(row['crowd_level'])['name']}"]
                 for _, row in d.iterrows()]
    table_h = min(0.95, 0.062 * (len(cell_text) + 1))
    tbl = ax_table.table(cellText=cell_text, colLabels=["Time", "Count", "80% range", "Crowd"],
                          cellLoc="center", bbox=[0.0, 0.96 - table_h, 1.0, table_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    crowd_col = 3
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor(AQUA)
            cell.set_text_props(color="black", weight="bold")
        elif c == crowd_col:
            lvl = int(d.iloc[r - 1]["crowd_level"])
            cell.set_facecolor(CROWD_COLORS[lvl])
            cell.set_text_props(color=CROWD_TEXT_COLORS[lvl], weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1a1a1a")
            cell.set_text_props(color=TEXT_COLOR)
        else:
            cell.set_facecolor(AXES_BG)
            cell.set_text_props(color=TEXT_COLOR)

    # tight_layout's rect is ignored on this figure (it warns that the twinx and
    # table axes are incompatible), so the bottom margin is set explicitly
    # afterwards. Needed once the degenerate-band note added a third footer
    # line, which ran straight into the "Time" axis label.
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(bottom=0.10 + 0.03 * (info_text.count("\n") - 1))

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"surfer_count_{target_date.isoformat()}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")

    # Stable, git-tracked path for the README embed — overwritten daily rather than
    # accumulating a new tracked file every day (the dated file above stays local/
    # untracked, matching the rest of data/'s convention).
    latest_path = CHARTS_DIR / "latest.png"
    fig.savefig(latest_path, dpi=150, facecolor=fig.get_facecolor())

    # The week outlook reuses the models already fitted above — it is a second
    # chart, not a second fit, and it costs no extra API requests either (the
    # single predictor fetch above already covers the whole horizon).
    made_date = datetime.now().date()
    week_start = target_date if args.date else made_date + timedelta(days=1)
    day_records = collect_week_records(week_start, by_hour, predict_for_hour, local_tz)
    week_result = generate_week_chart(made_date, day_records, len(df))
    save_week_forecast_record(made_date, day_records)

    detection_capture = generate_detection_gif(detection_date)
    update_readme(target_date, detection_capture,
                  week_days=len(day_records) if week_result else 0,
                  week_start=week_start)


def find_nearest_hour_crop(target_date, target_hour=8, lookback_days=7):
    """Finds the predictions.csv row closest to target_hour on the most recent date
    at or before target_date (searching back up to lookback_days) with quality_ok=True
    and a crop image still on disk. Falls back to earlier dates rather than returning
    None outright -- local_pipeline.sh runs twice a week, not daily, so target_date
    itself (usually "today") frequently has no crop yet; without the fallback, the
    README's detection image would disappear entirely between runs instead of just
    showing the most recent one available. Returns the row dict, or None only if
    nothing usable exists anywhere in the lookback window."""
    with open(ds.PREDS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    def hour_distance(r):
        h, m = map(int, r["time_local"].split(":"))
        return abs((h * 60 + m) - target_hour * 60)

    for days_back in range(lookback_days + 1):
        check_date = target_date - timedelta(days=days_back)
        rows = [r for r in all_rows if r["date"] == check_date.isoformat() and r["quality_ok"] == "True"]
        rows.sort(key=hour_distance)
        for r in rows:
            if (ds.CROPS_DIR / r["filename"]).exists():
                return r
    return None


# --- Detection animation settings (2026-09-16) -------------------------------
# Replaced the single ~8am still with an animation of a whole day of frames.
# SIDE_CROP_FRAC trims this fraction off each end: the camera's full 1280px
# strip makes surfers tiny once GitHub scales it to page width, so the outer
# 40% is traded away to see the remaining 60% properly. GIF_UPSCALE then takes
# the 768px crop past GitHub's ~880px content column so it renders full width
# instead of being letterboxed at its natural size.
# Full width, no side crop (Joel's call, 2026-09-22, after trying 0.20-0.30).
# Cropping does make surfers bigger — GitHub scales the image to its ~880px
# column regardless, so apparent size is 880 over the kept width — but it drops
# real detections off the sides: measured against 2026-09-21's 186 detections,
# trimming 30% off each end left only 56% of them in view. That produced frames
# whose printed count disagreed with what the pipeline actually counted, and in
# one case a frame reading "0 surfers detected" for an hour that had one. Small
# surfers are the better trade than a picture that contradicts the data.
#
# GIF_UPSCALE does not change on-screen size, since GitHub scales to its column
# either way; it only keeps the image sharp on high-DPI displays. 1.5x puts the
# full 1280px strip at 1920px, comfortably above 2x that column.
SIDE_CROP_FRAC = 0.0
GIF_UPSCALE = 1.25
# Each hour appears twice: bare frame, then the same frame with boxes. The bare
# one holds longer because that is the half asking the viewer to do something —
# find the specks — while the reveal only has to be read.
GIF_LOOK_MS = 2500
GIF_REVEAL_MS = 2000
GIF_MAX_COLORS = 256          # full palette: box pixels are alpha-blended (see BOX_ALPHA),
                              # so their exact color varies with the water beneath and a
                              # smaller palette starts discarding them
# Boxes are drawn slightly translucent so they sit on the water rather than
# hovering over it. Kept high: at lower opacity the blended greens drift far
# enough from the reserved palette entry that GIF quantization starts snapping
# them toward gray, which is the bug this whole path was fixed for once already.
BOX_ALPHA = 0.90
# Box color, defined once in both spaces: cv2 draws in BGR, the GIF palette
# reserves it in RGB. Keeping a single source for it is what guarantees the
# drawn pixels and the reserved palette entry are the same color.
BOX_COLOR_RGB = (157, 227, 90)        # lime green, matching LIME "#9de35a"
BOX_COLOR_BGR = BOX_COLOR_RGB[::-1]
# Colors that must survive quantization exactly, whatever the frame contains.
GIF_RESERVED_COLORS = (BOX_COLOR_RGB, (255, 255, 255), (235, 235, 235), (0, 0, 0))
# GitHub does not autoplay an animated GIF in a README — it shows the first
# frame with a small play button in the top-right corner, which readers miss.
# The first frame therefore carries a callout pointing at it.
GIF_PLAY_CALLOUT = "Click Play Here"
# Clearance from the right edge, as a FRACTION of image width rather than a
# pixel count. GitHub's play button is a fixed size in CSS pixels while the
# image is scaled to fit its column, so a fixed pixel offset buys less and less
# clearance the wider the image gets — a 90px offset that worked at 1536px wide
# still left the arrow partly under the button at full width. A fraction scales
# with the image and holds at any zoom level.
PLAY_BUTTON_CLEARANCE_FRAC = 0.13
GIF_LOOKBACK_DAYS = 14
# A day also has to be busy enough to be worth showing: more than
# BUSY_COUNT_MIN surfers in at least MIN_BUSY_FRACTION of its frames. Added
# 2026-09-16 after the animation landed on a foggy, near-empty day (2026-09-15:
# 13% of frames above 2 surfers) which demonstrated nothing about detection. A
# day that misses the bar leaves the existing animation in place rather than
# replacing it with a worse one.
BUSY_COUNT_MIN = 2
MIN_BUSY_FRACTION = 0.60
# A day needs at least this many usable frames to be worth animating. The
# scheduled run is at 20:30, by which point the current day is complete, but an
# off-hours run would otherwise pick up a half-collected day — a 7am test run
# produced a one-frame "animation".
MIN_GIF_FRAMES = 6


def find_day_crops(detection_date, lookback_days=GIF_LOOKBACK_DAYS):
    """All quality_ok crops for the most recent day at or before detection_date.

    Falls back to earlier days rather than returning nothing, for the same
    reason the single-frame version did: a run can happen before that day's
    clips exist, and an empty animation in the README is worse than yesterday's.
    Returns (date, [row, ...]) in time order, or (None, []).
    """
    with open(ds.PREDS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    for days_back in range(lookback_days + 1):
        check_date = detection_date - timedelta(days=days_back)
        rows = [r for r in all_rows
                if r["date"] == check_date.isoformat() and r["quality_ok"] == "True"
                and (ds.CROPS_DIR / r["filename"]).exists()]
        if not rows:
            continue
        if len(rows) < MIN_GIF_FRAMES:
            print(f"  {check_date}: only {len(rows)} usable frame(s), need {MIN_GIF_FRAMES} — skipping.")
            continue
        counts = [float(r["surfer_count"]) for r in rows if r["surfer_count"] not in ("", "None")]
        busy = sum(1 for c in counts if c > BUSY_COUNT_MIN)
        share = busy / len(counts) if counts else 0.0
        if share < MIN_BUSY_FRACTION:
            print(f"  {check_date}: only {share:.0%} of frames have more than {BUSY_COUNT_MIN} "
                  f"surfers, need {MIN_BUSY_FRACTION:.0%} — skipping.")
            continue
        rows.sort(key=lambda r: r["time_local"])
        print(f"  {check_date}: {len(rows)} frames, {share:.0%} above {BUSY_COUNT_MIN} surfers — using this day.")
        return check_date, rows
    return None, []


def render_detection_frame(img_path, model, draw_boxes=True, boxes=None):
    """One frame: cropped, upscaled, and optionally annotated with real boxes.

    `draw_boxes=False` returns the identical pixels with nothing drawn, which is
    what the animation's "look first" half needs — the two halves of a pair must
    differ only by the boxes, or the eye tracks the change in the image instead
    of the change in the annotation.

    Boxes come from the production path (tiling, cross-tile NMS,
    false-positive filtering) via run_inference_with_boxes, not a
    reimplementation. Cropping happens first and drawing second, so box
    outlines and text are drawn at final resolution rather than being
    upscaled into blurry lines.
    """
    # `boxes` lets a caller run inference once and render the frame twice — the
    # paired animation needs the same detections drawn and not drawn, and
    # inference is by far the expensive part of this function.
    if boxes is None:
        boxes = ds.run_inference_with_boxes(model, img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0

    h, w = img.shape[:2]
    x0, x1 = int(w * SIDE_CROP_FRAC), int(w * (1 - SIDE_CROP_FRAC))
    cropped = img[:, x0:x1]
    out = cv2.resize(cropped, None, fx=GIF_UPSCALE, fy=GIF_UPSCALE, interpolation=cv2.INTER_CUBIC)

    overlay = out.copy()
    visible = 0
    for bx1, by1, bx2, by2, conf in boxes:
        # Drop boxes the crop removed; clip ones it cuts through.
        if bx2 < x0 or bx1 > x1:
            continue
        visible += 1
        p1 = (int(max(bx1 - x0, 0) * GIF_UPSCALE), int(by1 * GIF_UPSCALE))
        p2 = (int(min(bx2 - x0, x1 - x0) * GIF_UPSCALE), int(by2 * GIF_UPSCALE))
        # Drawn onto an overlay that is composited at BOX_ALPHA below. An
        # earlier version used 0.65 here, which pushed the blended green far
        # enough toward the water beneath that GIF quantization dropped it on
        # some frames and the boxes rendered gray. 0.90 stays close enough to
        # the reserved palette entry to survive, which the regeneration check
        # verifies frame by frame.
        if draw_boxes:
            cv2.rectangle(overlay, p1, p2, BOX_COLOR_BGR, 2)
    if visible and draw_boxes:
        out = cv2.addWeighted(overlay, BOX_ALPHA, out, 1.0 - BOX_ALPHA, 0)
    return out, visible


FORECASTS_DIR = _PROJECT_ROOT / "data" / "forecasts"


def crowd_key_text(n_rows=None):
    """One footer line defining what each crowd level means in surfers."""
    parts = "  ".join(f"{b['level']} {b['name']} {band_text(b['level'], units=False)}"
                      for b in CROWD_LEVELS)
    basis = f" of {n_rows:,} recorded hours" if n_rows else ""
    return (f"Crowd level (surfers): {parts}  —  equal fifths{basis}, read off the forecast's "
            f"{RATING_QUANTILE:.0%} percentile, not its median (code/crowd_rating.py)")


def forecast_row(target_date, r):
    """One CSV row for one forecast hour.

    Shared by the per-day record and the week record so the two files have the
    same column meanings. Columns added 2026-09-23 (crowd_*, missing_*) are
    appended after the originals; code/email_daily_report.py reads this file by
    column name, so extra columns are additive and do not disturb it.
    """
    q = r["quantiles"]
    return {
        "date": target_date.isoformat(),
        "hour_local": r["hour"].strftime("%H:%M"),
        "predicted": round(r["point"], 2),
        "lower_q10": round(q[FAN_LEVELS[0]], 2),
        "upper_q90": round(q[FAN_LEVELS[-1]], 2),
        "weather_simple": r["weather_simple"],
        "tide_ft": round(r["tide_ft"], 2),
        "in_training_range": r["in_training_range"],
        "forecast_made_at": datetime.now().isoformat(timespec="seconds"),
        # The rating and what it was read off — keeping the value means a later
        # audit can tell a level-4 that only just cleared the band edge from one
        # well inside it, without refitting anything.
        "crowd_level": r["crowd_level"],
        "crowd_label": level_info(r["crowd_level"])["name"],
        "crowd_band": band_text(r["crowd_level"]),
        "crowd_level_quantile": RATING_QUANTILE,
        "crowd_level_value": round(r["crowd_level_value"], 2),
        "crowd_level_low": r["crowd_level_low"],
        "crowd_level_high": r["crowd_level_high"],
        # Empty in the normal case; a semicolon-joined list of model features
        # this hour had no value for when it is not.
        "n_missing_predictors": len(r["missing_predictors"]),
        "missing_predictors": ";".join(r["missing_predictors"]),
    }


def save_forecast_record(target_date, records):
    """Writes the day's forecast to `data/forecasts/forecast_<date>.csv`.

    Until 2026-09-22 a forecast existed only as pixels in a PNG, so there was no
    way to ask later how well it did — the evening report would have had to refit
    a model and call the result "what we predicted", which it would not be. One
    row per hour, written when the forecast is made.

    Never overwrites an existing file: the point of the record is what was
    predicted *before* the day happened, and a later run on the same date would
    have more data and quietly improve history.
    """
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FORECASTS_DIR / f"forecast_{target_date.isoformat()}.csv"
    if out.exists():
        return out

    pd.DataFrame([forecast_row(target_date, r) for r in records]).to_csv(out, index=False)
    print(f"Saved forecast record ({len(records)} hours) to {out}")
    return out


def save_week_forecast_record(made_date, day_records):
    """Writes the whole 7-day outlook to `data/forecasts/week_<made>.csv`.

    A separate file, named for the day the forecast was MADE, rather than
    seven `forecast_<date>.csv` files. Those are deliberately write-once (see
    save_forecast_record): if this run wrote one for six days out, the run on
    the eve of that day — the one with the freshest data, and the one
    `code/email_daily_report.py` scores against — would find the file already
    there and skip it, quietly replacing tomorrow's forecast with a week-old
    one. The per-day record keeps meaning "the last forecast before the day";
    the week record is its own artifact, with `lead_days` on every row so
    forecast skill by lead time can be measured later.
    """
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FORECASTS_DIR / f"week_{made_date.isoformat()}.csv"
    rows = []
    for day, records in day_records:
        for r in records:
            row = forecast_row(day, r)
            row["lead_days"] = (day - made_date).days
            rows.append(row)
    if not rows:
        return None
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved week forecast record ({len(rows)} hours, {len(day_records)} days) to {out}")
    return out


WEEK_CHART_NAME = "latest_week.png"


def collect_week_records(start_date, by_hour, predict_for_hour, local_tz):
    """[(date, [hour record, ...]), ...] for FORECAST_DAYS days from start_date.

    Days the predictor fetch didn't reach are left out rather than filled with
    anything — the caller reports the shortfall instead of drawing a day the
    forecast does not have.
    """
    out = []
    for offset in range(FORECAST_DAYS):
        day = start_date + timedelta(days=offset)
        dawn, dusk = gc.get_light_window(day, local_tz)
        hours = sorted(hk for hk in by_hour
                       if hk.date() == day and dawn.hour <= hk.hour <= dusk.hour)
        records = [r for r in (predict_for_hour(hk) for hk in hours) if r is not None]
        if records:
            out.append((day, records))
    return out


def generate_week_chart(made_date, day_records, n_train_rows):
    """The 7-day outlook, as its OWN chart rather than a widened daily chart.

    Why separate: the daily chart already carries an hourly median line, a
    shaded 80% band, a tide curve on a second axis, per-hour weather markers
    and labels, night shading and a side table, at a width the README embeds
    fixed. Seven days of that is roughly a hundred hourly points and seven
    tide curves in the same space — the fan and the labels would be illegible
    long before the week was readable. The two charts also answer different
    questions: "what does tomorrow look like hour by hour" wants the interval
    and the tide; "which day this week should I go" wants one comparable
    number per day-hour. So the week gets the form that question needs — a
    day x hour grid of crowd levels, the count printed in each cell — and the
    daily chart is left alone.

    Returns (out_path, notes) where notes is the list of caveat strings shown
    on the chart, or None if there is nothing to draw.
    """
    if not day_records:
        print("No forecast days available for the week chart — skipping.")
        return None

    all_hours = sorted({r["hour"].hour for _, records in day_records for r in records})
    hour_cols = list(range(all_hours[0], all_hours[-1] + 1))
    n_rows, n_cols = len(day_records), len(hour_cols)

    fig = plt.figure(figsize=(15, 7.6), facecolor=BG_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.1], wspace=0.04)
    ax = fig.add_subplot(gs[0], facecolor=AXES_BG)
    ax_table = fig.add_subplot(gs[1], facecolor=AXES_BG)

    flagged = 0
    for row_i, (day, records) in enumerate(day_records):
        by_hour_num = {r["hour"].hour: r for r in records}
        for col_i, hour in enumerate(hour_cols):
            r = by_hour_num.get(hour)
            if r is None:
                # Outside this day's light window — the day is shorter than the
                # widest day on the chart. Left as bare background.
                continue
            level = r["crowd_level"]
            ax.add_patch(plt.Rectangle((col_i + 0.03, row_i + 0.05), 0.94, 0.90,
                                       facecolor=CROWD_COLORS[level], edgecolor=AXES_BG,
                                       linewidth=1.5, zorder=2))
            low_confidence = (not r["in_training_range"]) or r["missing_predictors"]
            if low_confidence:
                flagged += 1
                # Same coral cross-hatch the daily chart uses for an hour the
                # model has no training data for, extended to cover an hour
                # scored on fewer predictors than the model was trained on.
                ax.add_patch(plt.Rectangle((col_i + 0.03, row_i + 0.05), 0.94, 0.90,
                                           facecolor="none", hatch="xx", edgecolor=CORAL,
                                           linewidth=0, zorder=3))
            ax.text(col_i + 0.5, row_i + 0.5, f"{r['point']:.0f}",
                    ha="center", va="center", fontsize=9, zorder=4,
                    weight="bold", color=CROWD_TEXT_COLORS[level])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels([datetime(2000, 1, 1, h).strftime("%-I%p").lower() for h in hour_cols])
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels([d.strftime("%a %-d %b") for d, _ in day_records])
    ax.tick_params(axis="both", colors=TEXT_COLOR, length=0)
    ax.set_title(f"Crowd outlook — {day_records[0][0].strftime('%a %-d %b')} to "
                 f"{day_records[-1][0].strftime('%a %-d %b %Y')}", color=TEXT_COLOR)
    ax.set_xlabel("Hour (first light to last light)", color=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(False)

    handles = [Patch(facecolor=CROWD_COLORS[b["level"]], edgecolor=AXES_BG,
                     label=f"{b['level']} {b['name']} {band_text(b['level'], units=False)}")
               for b in CROWD_LEVELS]
    handles.append(Patch(facecolor="none", hatch="xx", edgecolor=CORAL,
                         label="lower confidence (see footer)"))
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                       ncol=6, fontsize=8, facecolor=AXES_BG, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # Side table: one line per day, for the "which day should I go" read.
    ax_table.axis("off")
    ax_table.set_title("Busiest hour each day", fontsize=10, pad=10, color=TEXT_COLOR)
    cell_text, busiest_levels = [], []
    for day, records in day_records:
        busiest = max(records, key=lambda r: r["crowd_level_value"])
        cell_text.append([day.strftime("%a %-d"),
                          busiest["hour"].strftime("%-I%p").lower(),
                          f"{busiest['point']:.0f}",
                          f"{busiest['crowd_level']} {level_info(busiest['crowd_level'])['name']}"])
        busiest_levels.append(busiest["crowd_level"])
    # Explicit bbox rather than loc="upper center": the auto-sized table is laid
    # out to its text width and left a wide empty gutter beside it, squeezing
    # the grid that is the actual chart.
    table_h = min(0.92, 0.085 * (len(cell_text) + 1))
    tbl = ax_table.table(cellText=cell_text, colLabels=["Day", "Peak", "Count", "Crowd"],
                         cellLoc="center", bbox=[0.0, 0.95 - table_h, 1.0, table_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor(AQUA)
            cell.set_text_props(color="black", weight="bold")
        elif c == 3:
            lvl = busiest_levels[r - 1]
            cell.set_facecolor(CROWD_COLORS[lvl])
            cell.set_text_props(color=CROWD_TEXT_COLORS[lvl], weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1a1a1a")
            cell.set_text_props(color=TEXT_COLOR)
        else:
            cell.set_facecolor(AXES_BG)
            cell.set_text_props(color=TEXT_COLOR)

    notes = []
    short = FORECAST_DAYS - len(day_records)
    if short:
        notes.append(f"{short} of the {FORECAST_DAYS} requested days had no forecast data "
                     f"and are not shown")
    incomplete = [r for _, records in day_records for r in records if r["missing_predictors"]]
    if incomplete:
        worst = max(len(r["missing_predictors"]) for r in incomplete)
        notes.append(f"{len(incomplete)} hour(s) are missing up to {worst} predictor(s) and are "
                     f"hatched — the model scored them on less than it was trained on")
    else:
        notes.append("every hour shown has the full predictor set")
    if flagged and not incomplete:
        notes.append(f"{flagged} hatched hour(s) fall outside the hours the model has training "
                     f"data for")

    footer = (
        f"Each cell is the predicted surfer count for that hour; its colour is the crowd level.  "
        f"Numbers this far out move — day 7 is the same model on a week-old view of the weather.\n"
        f"{crowd_key_text(n_train_rows)}\n"
        + "  |  ".join(notes)
    )
    fig.text(0.5, 0.045, footer, fontsize=7.5, ha="center", va="bottom", color=MUTED_TEXT)
    # Explicit margins, not tight_layout: the table axes makes tight_layout warn
    # and then lay the footer over the x-axis label and the legend.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.26)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / WEEK_CHART_NAME
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(CHARTS_DIR / f"week_{made_date.isoformat()}.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved week chart ({len(day_records)} days) to {out_path}")
    return out_path, notes


def quantize_to_shared_palette(frames):
    """One palette for every frame, with the box green reserved exactly.

    GIF is palette-based, so each frame's colors get reduced to at most 256
    entries. The previous code called `convert("P", palette=ADAPTIVE)` per
    frame, which despite the comment on it built a *separate* palette for each
    one: a thin green box line covers very few pixels, so on frames whose water
    happened to be more varied the green was dropped as an unimportant color
    and snapped to the nearest gray. That is why some frames showed green boxes
    and others gray.

    Here the palette is built once from every frame together, and the colors
    that must be exact — the box green, the banner text, black — are appended
    afterwards rather than being left to survive on merit. Dithering is off so
    flat colors stay flat instead of being stippled from neighbouring entries.
    """
    reserved = list(GIF_RESERVED_COLORS)
    montage = Image.new("RGB", (frames[0].width, sum(f.height for f in frames)))
    y = 0
    for f in frames:
        montage.paste(f, (0, y))
        y += f.height

    base = montage.quantize(colors=max(GIF_MAX_COLORS - len(reserved), 2),
                            method=Image.Quantize.MEDIANCUT)
    palette = base.getpalette()[: 3 * (GIF_MAX_COLORS - len(reserved))]
    for color in reserved:
        palette.extend(color)
    palette.extend([0, 0, 0] * (256 - len(palette) // 3))

    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(palette)
    return [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]


def add_play_callout(canvas):
    """Draws "Click Play Here -->" at the top right of the first frame.

    GitHub renders an animated GIF in a README as a still first frame with a
    small play button in the top-right corner, and readers do not reliably
    notice it. The arrow points at where that button sits.
    """
    h, w = canvas.shape[:2]
    text = f"{GIF_PLAY_CALLOUT}  -->"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 12
    # GitHub overlays its own play button in the top-right corner, which sat on
    # top of the arrow. Pull the whole callout left so the arrow points AT that
    # button instead of being covered by it.
    x = max(w - tw - pad * 3 - int(w * PLAY_BUTTON_CLEARANCE_FRAC), pad)
    y = pad + th

    # Dark plate behind the text so it reads over bright water or sky.
    cv2.rectangle(canvas, (x - pad, y - th - pad), (min(x + tw + pad, w - 1), y + pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(canvas, text, (x, y), font, scale, BOX_COLOR_BGR, thickness, cv2.LINE_AA)
    return canvas


def keep_existing_gif():
    """(date, n_frames) for the animation already on disk, or None.

    Read from a sidecar written next to the GIF, so a run that finds no
    qualifying day can leave the file alone and still caption it correctly.
    """
    meta_path = CHARTS_DIR / "latest_detection.json"
    if not (CHARTS_DIR / "latest_detection.gif").exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        return datetime.strptime(meta["date"], "%Y-%m-%d").date(), int(meta["n_frames"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def generate_detection_gif(detection_date):
    """Animates a full day of detections, one frame per clip, 1s each.

    Replaces the single ~8am still (2026-09-16): one frame showed one hour's
    crowd, while a day of them shows the shape the forecast model is actually
    trying to predict — empty at first light, building through the morning, thinning
    out again. Returns (date, n_frames) or None.
    """
    day, rows = find_day_crops(detection_date)
    if not rows:
        # Keep whatever is already published rather than dropping the animation
        # from the README: a stale-but-representative day beats no image, and
        # beats a flat day that shows nothing. The sidecar carries the date the
        # existing file covers, so the caption stays truthful about which day
        # is on screen.
        existing = keep_existing_gif()
        if existing:
            print(f"No day met the bar at or before {detection_date} — keeping the "
                  f"existing animation ({existing[0]}, {existing[1]} frames).")
        else:
            print(f"No day met the bar at or before {detection_date} and no existing "
                  f"animation to keep — skipping.")
        return existing

    model = ds.load_model()
    frames, durations = [], []
    for row in rows:
        img_path = ds.CROPS_DIR / row["filename"]
        detections = ds.run_inference_with_boxes(model, img_path)
        boxed, count = render_detection_frame(img_path, model, boxes=detections)
        if boxed is None:
            continue
        clean, _ = render_detection_frame(img_path, model, draw_boxes=False,
                                          boxes=detections)

        hh, mm = map(int, row["time_local"].split(":")[:2])
        stamp = datetime(day.year, day.month, day.day, hh, mm).strftime("%-I:%M %p")

        # Each hour shows twice: the bare frame first, so the reader can hunt
        # for the dark specks themselves, then the same frame with the boxes
        # drawn. Joel's idea (2026-09-22) — it turns the animation from
        # something to watch into something to play along with, and it shows
        # honestly how hard these are to see before the model marks them.
        for is_boxed, image in ((False, clean), (True, boxed)):
            # Banner below the image, so the labels never cover water that might
            # contain a surfer the reader is trying to spot.
            banner_h = 46
            h, w = image.shape[:2]
            canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
            canvas[:h] = image
            # ASCII only: cv2's Hershey fonts have no glyph for an em dash and
            # silently render it as "???".
            label = (f"{stamp}   |   {count} surfers detected" if is_boxed
                     else f"{stamp}   |   how many surfers can you spot?")
            cv2.putText(canvas, label, (10, h + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
            if not frames:
                add_play_callout(canvas)
            frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
            durations.append(GIF_LOOK_MS if not is_boxed else GIF_REVEAL_MS)

    if not frames:
        print("No frames rendered — skipping detection animation.")
        return None

    palette_frames = quantize_to_shared_palette(frames)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CHARTS_DIR / "latest_detection.gif"
    palette_frames[0].save(latest_path, save_all=True, append_images=palette_frames[1:],
                           duration=durations, loop=0, optimize=True, disposal=2)
    n_hours = len(frames) // 2
    (CHARTS_DIR / "latest_detection.json").write_text(
        json.dumps({"date": day.isoformat(), "n_frames": n_hours}) + "\n")
    dated_path = CHARTS_DIR / f"detection_{day.isoformat()}.gif"
    dated_path.write_bytes(latest_path.read_bytes())
    size_mb = latest_path.stat().st_size / 1e6
    print(f"Saved detection animation ({n_hours} hours, {len(frames)} frames, "
          f"{size_mb:.1f} MB) to {dated_path}")
    return day, n_hours


README_START_MARKER = "<!-- DAILY_CHART_START -->"
README_END_MARKER = "<!-- DAILY_CHART_END -->"
# One rewritten block holding the animation above the forecast chart. The static
# hero image at the top of the README is deliberately OUTSIDE these markers, so
# the nightly rewrite never touches it.


def update_readme(target_date, detection_capture=None, week_days=0, week_start=None):
    """Replaces the marked section of README.md with the latest detection image
    (if one was generated) + chart-with-table image. Idempotent — safe to run
    daily; only the content between the markers changes."""
    readme_path = _PROJECT_ROOT / "README.md"
    readme = readme_path.read_text()
    if README_START_MARKER not in readme or README_END_MARKER not in readme:
        print(f"WARNING: README.md markers not found — skipping README update. "
              f"Add {README_START_MARKER} / {README_END_MARKER} to enable this.")
        return

    # Static caption text, rewritten into the README on every run. Kept
    # verbatim from what's currently in README.md so the daily automation
    # preserves it instead of overwriting it.
    DETECTION_CAPTION = (
        "Each green box below contains a surfer, according to the object "
        "detection model.  Look carefully and you may find "
        "additional surfers that the model missed or other objects which are "
        "misclassified as surfers - like the wind sock at the bottom center "
        "of the photo.  Birds, reflections, people on the beach and sun glare "
        "fool it too: [what isn't a surfer](docs/non_surfer_objects.md) "
        "catalogs each one, how to tell it apart, and what the pipeline does "
        "about it.  **Each hour is shown twice: first the bare frame, so you "
        "can hunt for the surfers yourself as small dark specks, then the same "
        "frame with the model's boxes drawn.**  Try it before the boxes appear - "
        "it shows how hard these are to see, which is the whole problem the "
        "model is solving.  Each frame shows the camera's full width, the same "
        "strip the pipeline counts, so the number printed is the whole count "
        "for that hour.\n"
    )
    FORECAST_CAPTION = (
        "Once enough hours and days were gathered along with weather and surf "
        "conditions, a surf count prediction model was built to forecast how "
        "many surfers would be present at each hour for the coming day.  This "
        "is useful for surfers to plan to avoid busy times or at least know "
        "what to expect.  The forecast still runs about one surfer low on "
        "average, and it misses by about five surfers in a typical hour - "
        "crowds depend on plenty of things the weather and surf data never "
        "see.  The detector underneath it used to undercount badly in fog and "
        "glare, which made the forecast worse; that was fixed in September "
        "2026 by labeling those conditions and retraining, so what remains is "
        "the prediction model's own error.  The right-hand column rates each "
        "hour from 1 (near-empty) to 5 (packed), using the five equal slices "
        "of every hour the camera has counted so far.\n"
    )

    detection_block = ""
    if (CHARTS_DIR / "latest_detection.gif").exists() and detection_capture is not None:
        capture_day, _n_frames = detection_capture
        capture_str = capture_day.strftime("%A, %B %d, %Y")
        detection_block = (
            f"#### A Full Day of Surfer Detections: {capture_str}\n\n"
            f"{DETECTION_CAPTION}"
            f"![Detections through {capture_str}](data/charts/latest_detection.gif)\n\n"
        )

    WEEK_CAPTION = (
        "The same model, run out to a week. Every hour gets a crowd level from "
        "1 (near-empty) to 5 (packed); the levels are the five equal slices of "
        "every hour the camera has counted so far, so level 3 is literally an "
        "average hour and level 5 is the busiest fifth of them. The number in "
        "each cell is the predicted count, and the colour is the level. The "
        "level is not read off that number: the forecast pulls toward the "
        "middle, so the middle of its range under-calls busy hours, and the "
        "level comes from a point higher up the range that was measured to "
        "catch them. Surf and weather data run the full seven days, but a "
        "forecast seven days out is still a forecast seven days out - read the "
        "far right of the grid as a shape, not a number.\n"
    )

    week_block = ""
    if week_days and week_start is not None:
        week_range = (f"{week_start.strftime('%B %d')} - "
                      f"{(week_start + timedelta(days=week_days - 1)).strftime('%B %d, %Y')}")
        week_block = (
            f"#### The Week Ahead: {week_range}\n\n"
            f"{WEEK_CAPTION}"
            f"![Crowd outlook for the week ahead](data/charts/{WEEK_CHART_NAME})\n\n"
        )

    target_date_str = target_date.strftime("%A, %B %d, %Y")
    section = (
        f"{README_START_MARKER}\n"
        f"{detection_block}"
        f"#### The Surfer Crowd Forecast for: {target_date_str}\n\n"
        f"{FORECAST_CAPTION}"
        f"![Latest daily prediction chart](data/charts/latest.png)\n\n"
        f"{week_block}"
        f"{README_END_MARKER}"
    )

    before, _, rest = readme.partition(README_START_MARKER)
    _, _, after = rest.partition(README_END_MARKER)
    readme_path.write_text(before + section + after)
    print("Updated README.md daily chart section.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        # Exit 3 marks a deterministic modelling failure, as opposed to the
        # transient network errors daily_chart.sh retries. Retrying this kind
        # just burns 50 minutes and fails identically — which is what happened
        # on 2026-09-15.
        print(f"DETERMINISTIC FAILURE (not retryable): {e}", file=sys.stderr)
        sys.exit(3)
