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
data/charts/latest_detection.png. Both get a stable, git-tracked "latest"
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_clips as gc  # noqa: E402 — needed for get_light_window() (real dawn/dusk)
import get_surf_predictors as sp  # noqa: E402
import detect_surfers as ds  # noqa: E402
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

# Prediction-interval quantile levels the fan chart is built from -- 9 real
# fitted GBT quantile models (0.50 = median = point estimate), rendered as a
# continuous gradient by interpolating between them (see main()). Chosen as
# a 10%-90% span (an 80% central prediction interval) per Joel's request,
# in place of the old fixed 33%/66% bands.
FAN_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

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
    quantile_models = {level: fit_quantile_model_robust(X_std, y, level)[0] for level in FAN_LEVELS}

    by_hour = sp.build_predictor_map()
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
        # Predict all 9 fan levels, then force monotonicity (each level's value
        # >= the previous one's) -- independently fit quantile models have no
        # built-in guarantee they won't cross, same reasoning as the old
        # 5-quantile chaining this replaces.
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
    print(f"Real dawn/dusk for {target_date}: {dawn.strftime('%-I:%M %p')} - {dusk.strftime('%-I:%M %p')}")
    day_hours = sorted(hk for hk in by_hour if hk.date() == target_date and dawn.hour <= hk.hour <= dusk.hour)
    if not day_hours:
        print(f"No forecast data available for {target_date} (outside the live today+tomorrow window).")
        return

    records = [r for r in (predict_for_hour(hk) for hk in day_hours) if r is not None]
    d = pd.DataFrame(records)

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

    # Continuous-looking prediction-interval fan: interpolate between the 9
    # real fitted quantiles (FAN_LEVELS, 10%-90%) at each hour to get a much
    # finer grid of levels, then draw many thin stacked bands whose alpha
    # peaks at the median and fades toward the 10%/90% edges -- a smooth
    # gradient built from real model output, not fit from dozens of models.
    RENDER_LEVELS = np.linspace(FAN_LEVELS[0], FAN_LEVELS[-1], 41)  # 40 bands
    quantile_matrix = np.array([
        np.interp(RENDER_LEVELS, FAN_LEVELS, [q[lv] for lv in FAN_LEVELS])
        for q in d["quantiles"]
    ])  # shape (n_hours, len(RENDER_LEVELS))
    MAX_BAND_ALPHA, MIN_BAND_ALPHA = 0.55, 0.04
    for i in range(len(RENDER_LEVELS) - 1):
        level_center = (RENDER_LEVELS[i] + RENDER_LEVELS[i + 1]) / 2
        dist_from_median = abs(level_center - 0.50) / 0.40  # 0 at median, 1 at the 10%/90% edge
        alpha = MAX_BAND_ALPHA - dist_from_median * (MAX_BAND_ALPHA - MIN_BAND_ALPHA)
        ax.fill_between(d["hour"], quantile_matrix[:, i], quantile_matrix[:, i + 1],
                         color=AQUA, alpha=alpha, linewidth=0, zorder=2)
    # One representative legend entry for the whole gradient (can't label 40
    # individual bands) -- details go in README's "How to read this chart".
    fan_patch = Patch(facecolor=AQUA, alpha=0.35, label="10-90% prediction interval (darker = more likely)")
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
    info_text = (
        f"Model: gradient-boosted trees (quantile regression), point estimate = median model  "
        f"|  Top predictors (live fit): {predictors_str}\n"
        f"Surfer detector (YOLOv8s, train13 — actual training log): "
        f"precision {DETECTOR_PRECISION:.1%}, recall {DETECTOR_RECALL:.1%}"
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

    # Table panel: hour -> 80% range (10th-90th percentile) only (no median —
    # Joel asked for range without the point estimate here), rendered as part
    # of the same figure/image rather than a separate markdown table, so
    # chart and table always render side by side.
    ax_table.axis("off")
    ax_table.set_title("80% Range", fontsize=10, pad=10, color=TEXT_COLOR)
    cell_text = [[row["hour"].strftime("%-I:%M %p"), f"{row['quantiles'][0.10]:.0f}–{row['quantiles'][0.90]:.0f}"]
                 for _, row in d.iterrows()]
    tbl = ax_table.table(cellText=cell_text, colLabels=["Time", "Range"],
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

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"surfer_count_{target_date.isoformat()}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")

    # Stable, git-tracked path for the README embed — overwritten daily rather than
    # accumulating a new tracked file every day (the dated file above stays local/
    # untracked, matching the rest of data/'s convention).
    latest_path = CHARTS_DIR / "latest.png"
    fig.savefig(latest_path, dpi=150, facecolor=fig.get_facecolor())

    detection_capture = generate_detection_image(detection_date, predict_nearest_hour)
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


def generate_detection_image(detection_date, predict_nearest_hour):
    """Draws real detection boxes + confidence labels on the most recent ~8am
    crop at or before detection_date (the actual production tiling/NMS/false-
    positive-filter pipeline via detect_surfers.run_inference_with_boxes —
    not a simplified re-implementation), with the model's own predicted
    range/median for that same hour (via predict_nearest_hour, independent
    of whatever date the forecast chart is for) overlaid as text.
    find_nearest_hour_crop falls back to earlier dates (local_pipeline.sh
    runs twice a week, not daily, so detection_date itself — usually
    "today" — frequently has no crop yet) rather than skipping outright, so
    this only returns None if nothing usable exists within its lookback
    window at all."""
    row = find_nearest_hour_crop(detection_date, target_hour=8)
    if row is None:
        print(f"No usable ~8am crop found at or before {detection_date} — skipping detection image.")
        return None

    img_path = ds.CROPS_DIR / row["filename"]
    model = ds.load_model()
    boxes = ds.run_inference_with_boxes(model, img_path)

    img = cv2.imread(str(img_path))
    # Draw boxes/labels on a copy, then alpha-blend back so they read as
    # translucent overlays rather than opaque marks on the surf photo.
    overlay = img.copy()
    BOX_COLOR = (90, 227, 157)  # BGR — lime green (matches LIME "#9de35a")
    for x1, y1, x2, y2, conf in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(overlay, p1, p2, BOX_COLOR, 1)
        label = f"{conf:.2f}"
        cv2.putText(overlay, label, (p1[0], max(p1[1] - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)
    BOX_ALPHA = 0.60
    img = cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0)

    # Model's predicted range/median for this exact hour — looked up directly
    # (not from a chart-date-scoped table), so it's correct even when the
    # forecast chart is for a different date (e.g. tomorrow) than this image.
    # Uses row["date"] (the crop's actual date, which find_nearest_hour_crop
    # may have fallen back to an earlier day than detection_date) rather than
    # detection_date itself -- by_hour only covers today+tomorrow's live
    # forecast, so a past fallback date correctly comes back "not available"
    # instead of silently showing a different day's predicted range.
    row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
    hh, mm = map(int, row["time_local"].split(":"))
    pred_row = predict_nearest_hour(row_date, hh, mm)
    if pred_row is not None:
        q = pred_row["quantiles"]
        pred_text = f"Predicted: {pred_row['point']:.0f} (80% range {q[0.10]:.0f}-{q[0.90]:.0f})"
    else:
        pred_text = "Predicted range/median: not available for this hour"

    detected_text = f"Detected: {len(boxes)} surfer(s) at {row['time_local']}"
    legend_text = (
        f"Box = detected surfer, label = model confidence (0-1). "
        f"Confidence threshold: {ds.CONF_THRESH:.3f} (boxes below this are dropped)."
    )

    # Black banner strip below the image so text never overlaps real image content.
    banner_h = 62
    h, w = img.shape[:2]
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[:h] = img
    cv2.putText(canvas, detected_text, (8, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (211, 211, 211), 1, cv2.LINE_AA)
    cv2.putText(canvas, pred_text, (8, h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (211, 211, 211), 1, cv2.LINE_AA)
    cv2.putText(canvas, legend_text, (8, h + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

    dated_path = CHARTS_DIR / f"detection_{row_date.isoformat()}.png"
    cv2.imwrite(str(dated_path), canvas)
    latest_path = CHARTS_DIR / "latest_detection.png"
    cv2.imwrite(str(latest_path), canvas)
    print(f"Saved detection image to {dated_path}")
    return row["date"], row["time_local"]


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
        "of the photo.\n"
    )
    FORECAST_CAPTION = (
        "Once enough hours and days were gathered along with weather and surf "
        "conditions, a surf count prediction model was built to forecast how "
        "many surfers would be present at each hour for the coming day.  This "
        "is useful for surfers to plan to avoid busy times or at least know "
        "what to expect.  Recently the predictions have been low compared to "
        "actual counts, so more images are being collected to improve the "
        "object detection model's ability to find surfers in a variety of "
        "light and water conditions, like fog or choppy water surfaces.\n"
    )

    detection_block = ""
    if (CHARTS_DIR / "latest_detection.png").exists() and detection_capture is not None:
        capture_date, capture_time = detection_capture
        capture_dt = datetime.strptime(f"{capture_date} {capture_time}", "%Y-%m-%d %H:%M")
        capture_str = capture_dt.strftime("%A, %B %d, %Y, %-I:%M %p")
        detection_block = (
            f"#### A Recent Surfer Detection Count: {capture_str}\n\n"
            f"{DETECTION_CAPTION}"
            f"![Latest detection review](data/charts/latest_detection.png)\n\n"
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
    main()
