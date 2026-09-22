"""
plot_daily_prediction.py
--------------------------
Generates the daily surfer-count prediction chart (point estimate = median
GBT model, a continuous 10-90% prediction-interval fan built from 9 real
fitted quantile models rendered as a smooth gradient, an 80% range side
table, tide overlay, weather-coded markers, night-hour shading,
model/detector info footer) to data/charts/surfer_count_YYYY-MM-DD.png,
plus a detection-review image (real bounding boxes + labels on the day's
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
from fit_surfer_count_model import load_and_prepare, fit_quantile_model_robust  # noqa: E402
from predict_surf_count import build_feature_row, add_tide_daylight_features, MEAN_KWARGS  # noqa: E402
from build_training_features import simplify_weather_condition  # noqa: E402
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

    Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
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
    degenerate_levels = [lv for lv, m in quantile_models.items() if getattr(m, "degenerate", False)]
    if degenerate_levels:
        print(f"  Degenerate quantile level(s): {degenerate_levels} — drawn as flat bands.")

    by_hour = add_tide_daylight_features(sp.build_predictor_map())
    local_tz = pytz.timezone(gc.LOCATION["timezone"])

    def predict_for_hour(hk):
        """Runs the fitted quantile models for one by_hour key. Shared by the
        chart's day loop and the detection image's own-hour lookup below, so a
        prediction for any hour in by_hour (today OR tomorrow, since DAYS=2)
        is always computed the same way regardless of which date it's for."""
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
        return dict(hour=hk, point=point, quantiles=quantiles,
                    weather_simple=weather_simple, is_night=is_night,
                    tide_ft=tide_ft, in_training_range=in_training_range)

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
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0], facecolor=AXES_BG)
    ax_table = fig.add_subplot(gs[1], facecolor=AXES_BG)

    y_top = d["quantiles"].apply(lambda q: q[0.90]).max() * 1.18
    ax.set_ylim(bottom=0, top=y_top)

    for _, row in d.iterrows():
        if row["is_night"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor=NIGHT_COLOR, alpha=0.20, hatch="//", edgecolor=NIGHT_COLOR, linewidth=0, zorder=0)
    for _, row in d.iterrows():
        if not row["in_training_range"]:
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
        ax.annotate(label, (row["hour"], row["point"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    color=CORAL if not row["in_training_range"] else MUTED_TEXT)

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
        f"|  Top predictors (live fit): {predictors_str}\n"
        f"Refit this run on {len(df):,} detection-hours, {df['date'].min()} to {df['date'].max()}  "
        f"|  Surfer detector (YOLOv8s, Sept 2026 fog retrain — actual training log): "
        f"precision {DETECTOR_PRECISION:.1%}, recall {DETECTOR_RECALL:.1%}"
    )
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
    ax_table.set_title("Predicted count and 80% range", fontsize=10, pad=10, color=TEXT_COLOR)
    cell_text = [[row["hour"].strftime("%-I:%M %p"),
                  f"{row['point']:.0f}",
                  f"{row['quantiles'][0.10]:.0f}–{row['quantiles'][0.90]:.0f}"]
                 for _, row in d.iterrows()]
    tbl = ax_table.table(cellText=cell_text, colLabels=["Time", "Predicted", "Range"],
                          cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor(AQUA)
            cell.set_text_props(color="black", weight="bold")
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

    detection_capture = generate_detection_gif(detection_date)
    update_readme(target_date, detection_capture)


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
GIF_UPSCALE = 1.5
GIF_FRAME_MS = 1000
GIF_MAX_COLORS = 128          # palette size — the strip is mostly water, so this is plenty
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


def render_detection_frame(img_path, model):
    """One annotated frame: real detection boxes, cropped, upscaled, labelled.

    Boxes come from the production path (tiling, cross-tile NMS,
    false-positive filtering) via run_inference_with_boxes, not a
    reimplementation. Cropping happens first and drawing second, so box
    outlines and text are drawn at final resolution rather than being
    upscaled into blurry lines.
    """
    boxes = ds.run_inference_with_boxes(model, img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0

    h, w = img.shape[:2]
    x0, x1 = int(w * SIDE_CROP_FRAC), int(w * (1 - SIDE_CROP_FRAC))
    cropped = img[:, x0:x1]
    out = cv2.resize(cropped, None, fx=GIF_UPSCALE, fy=GIF_UPSCALE, interpolation=cv2.INTER_CUBIC)

    visible = 0
    for bx1, by1, bx2, by2, conf in boxes:
        # Drop boxes the crop removed; clip ones it cuts through.
        if bx2 < x0 or bx1 > x1:
            continue
        visible += 1
        p1 = (int(max(bx1 - x0, 0) * GIF_UPSCALE), int(by1 * GIF_UPSCALE))
        p2 = (int(min(bx2 - x0, x1 - x0) * GIF_UPSCALE), int(by2 * GIF_UPSCALE))
        # Drawn at full opacity, directly onto the frame. These used to be
        # composited at 65% via addWeighted, which blended every box toward the
        # gray water under it — and a blended, frame-dependent green is exactly
        # the kind of rare color GIF palette quantization throws away, so boxes
        # came out green on some frames and gray on others. One exact color,
        # reserved in the shared palette below, renders identically everywhere.
        cv2.rectangle(out, p1, p2, BOX_COLOR_BGR, 2)
        cv2.putText(out, f"{conf:.2f}", (p1[0], max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR_BGR, 2, cv2.LINE_AA)
    return out, visible


FORECASTS_DIR = _PROJECT_ROOT / "data" / "forecasts"


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

    rows = []
    for r in records:
        q = r["quantiles"]
        rows.append({
            "date": target_date.isoformat(),
            "hour_local": r["hour"].strftime("%H:%M"),
            "predicted": round(r["point"], 2),
            "lower_q10": round(q[FAN_LEVELS[0]], 2),
            "upper_q90": round(q[FAN_LEVELS[-1]], 2),
            "weather_simple": r["weather_simple"],
            "tide_ft": round(r["tide_ft"], 2),
            "in_training_range": r["in_training_range"],
            "forecast_made_at": datetime.now().isoformat(timespec="seconds"),
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved forecast record ({len(rows)} hours) to {out}")
    return out


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
    frames = []
    for row in rows:
        frame, count = render_detection_frame(ds.CROPS_DIR / row["filename"], model)
        if frame is None:
            continue

        # Banner below the image, so the labels never cover water that might
        # contain a surfer the reader is trying to spot.
        banner_h = 46
        h, w = frame.shape[:2]
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        hh, mm = map(int, row["time_local"].split(":")[:2])
        stamp = datetime(day.year, day.month, day.day, hh, mm).strftime("%-I:%M %p")
        # ASCII only: cv2's Hershey fonts have no glyph for an em dash and
        # silently render it as "???".
        cv2.putText(canvas, f"{stamp}   |   {count} surfers detected", (10, h + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        if not frames:
            add_play_callout(canvas)
        frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

    if not frames:
        print("No frames rendered — skipping detection animation.")
        return None

    palette_frames = quantize_to_shared_palette(frames)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CHARTS_DIR / "latest_detection.gif"
    palette_frames[0].save(latest_path, save_all=True, append_images=palette_frames[1:],
                           duration=GIF_FRAME_MS, loop=0, optimize=True, disposal=2)
    (CHARTS_DIR / "latest_detection.json").write_text(
        json.dumps({"date": day.isoformat(), "n_frames": len(frames)}) + "\n")
    dated_path = CHARTS_DIR / f"detection_{day.isoformat()}.gif"
    dated_path.write_bytes(latest_path.read_bytes())
    size_mb = latest_path.stat().st_size / 1e6
    print(f"Saved detection animation ({len(frames)} frames, {size_mb:.1f} MB) to {dated_path}")
    return day, len(frames)


README_START_MARKER = "<!-- DAILY_CHART_START -->"
README_END_MARKER = "<!-- DAILY_CHART_END -->"


def update_readme(target_date, detection_capture=None):
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
        "detection model.  Each number above a box indicates the probability "
        "that the object is a surfer.  Look carefully and you may find "
        "additional surfers that the model missed or other objects which are "
        "misclassified as surfers - like the wind sock at the bottom center "
        "of the photo.  Birds, reflections, people on the beach and sun glare "
        "fool it too: [what isn't a surfer](docs/non_surfer_objects.md) "
        "catalogs each one, how to tell it apart, and what the pipeline does "
        "about it.  Each frame is one hour of that day, one second apart, and "
        "shows the camera's full width - the same strip the pipeline counts, "
        "so the number printed on each frame is the whole count for that hour.  "
        "The surfers are small at this width; open the image on its own to see "
        "them properly.\n"
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
        "the prediction model's own error.\n"
    )

    detection_block = ""
    if (CHARTS_DIR / "latest_detection.gif").exists() and detection_capture is not None:
        capture_day, n_frames = detection_capture
        capture_str = capture_day.strftime("%A, %B %d, %Y")
        detection_block = (
            f"#### A Full Day of Surfer Detections: {capture_str}\n\n"
            f"{DETECTION_CAPTION}"
            f"![Detections through {capture_str}](data/charts/latest_detection.gif)\n\n"
        )

    target_date_str = target_date.strftime("%A, %B %d, %Y")
    section = (
        f"{README_START_MARKER}\n"
        f"{detection_block}"
        f"#### The Surfer Crowd Forecast for: {target_date_str}\n\n"
        f"{FORECAST_CAPTION}"
        f"![Latest daily prediction chart](data/charts/latest.png)\n\n"
        f"{README_END_MARKER}"
    )

    before, _, rest = readme.partition(README_START_MARKER)
    _, _, after = rest.partition(README_END_MARKER)
    new_readme = before + section + after
    readme_path.write_text(new_readme)
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
