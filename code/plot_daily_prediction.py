"""
plot_daily_prediction.py
--------------------------
Generates the daily surfer-count prediction chart (point estimate = median
GBT model, 33%/66% quantile bands rendered as a side table, tide overlay,
wave-energy bars, weather-coded markers, night-hour shading, model/detector
info footer) to data/charts/surfer_count_YYYY-MM-DD.png, plus a detection-
review image (real bounding boxes + labels on the day's ~8am crop, with the
model's predicted range/median for that hour overlaid) to
data/charts/latest_detection.png. Both get a stable, git-tracked "latest"
copy and are embedded in README.md between the DAILY_CHART markers.

Meant to run once per day (not tied to the twice-weekly clip-collection
cron — the chart only needs the live Surfline forecast + the existing
trained model; the detection image needs today's own clip, which the
clip-collection cron already downloads separately).

Usage:
    python code/plot_daily_prediction.py                # today
    python code/plot_daily_prediction.py --date 2026-08-29
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

WEATHER_COLORS = {"CLEAR": "#f2c14e", "CLOUDY_OVERCAST": "#9aa0a6", "RAIN": "#4fa3d1", "FOG": "#c9c9c9"}
WEATHER_MARKERS = {"CLEAR": "o", "CLOUDY_OVERCAST": "s", "RAIN": "^", "FOG": "D"}
WEATHER_ABBREV = {"CLEAR": "clear", "CLOUDY_OVERCAST": "cloudy", "RAIN": "rain", "FOG": "fog"}

# Dark theme — matches data/charts/one_off/weekday_weekend_*.png (aqua blue /
# lime green on black), with a couple of complementary colors added for the
# extra series this chart needs (wave-energy bars, tide line, warning hatches).
BG_COLOR = "black"
AXES_BG = "#111111"
GRID_COLOR = "#333333"
TEXT_COLOR = "white"
MUTED_TEXT = "#bbbbbb"
AQUA = "#3ab4c9"      # primary — median line / quantile bands
LIME = "#9de35a"      # tide line
AMBER = "#f2a950"     # wave-energy bars (complementary warm accent)
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

    fig = plt.figure(figsize=(14, 6.5), facecolor=BG_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0], facecolor=AXES_BG)
    ax_table = fig.add_subplot(gs[1], facecolor=AXES_BG)

    y_top = d["q83"].max() * 1.18
    ax.set_ylim(bottom=0, top=y_top)

    for _, row in d.iterrows():
        if row["is_night"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor=NIGHT_COLOR, alpha=0.20, hatch="//", edgecolor=NIGHT_COLOR, linewidth=0, zorder=0)
    for _, row in d.iterrows():
        if not row["in_training_range"]:
            ax.axvspan(row["hour"] - timedelta(minutes=30), row["hour"] + timedelta(minutes=30),
                       facecolor=CORAL, alpha=0.12, hatch="xx", edgecolor=CORAL, linewidth=0, zorder=0)

    bar_width = timedelta(minutes=22, seconds=30)
    energy_max = d["energy_nearshore_kj"].max()
    energy_scale = (y_top * 0.20) / energy_max if energy_max > 0 else 0
    bar_heights = d["energy_nearshore_kj"] * energy_scale
    ax.bar(d["hour"], bar_heights, width=bar_width, color=AMBER, alpha=0.45,
           zorder=1, label="Wave energy, nearshore (kJ)")
    for x, h, val in zip(d["hour"], bar_heights, d["energy_nearshore_kj"]):
        ax.annotate(f"{val:.0f}", (x, h), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=6.5, color=AMBER, rotation=90, va="bottom")

    ax.fill_between(d["hour"], d["q17"], d["q83"], color=AQUA, alpha=0.18, label="66% range")
    ax.fill_between(d["hour"], d["q335"], d["q665"], color=AQUA, alpha=0.38, label="33% range")
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
    legend = ax.legend(handles1 + handles2 + [night_patch], labels1 + labels2 + ["Night hours"],
                        loc="upper left", ncol=2, fontsize=8, facecolor=AXES_BG, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    ax.grid(alpha=0.25, color=GRID_COLOR)

    # Table panel: hour -> 33% range only (no median — Joel asked for range without
    # the point estimate here), rendered as part of the same figure/image rather than
    # a separate markdown table, so chart and table always render side by side.
    ax_table.axis("off")
    ax_table.set_title("33% Range", fontsize=10, pad=10, color=TEXT_COLOR)
    cell_text = [[row["hour"].strftime("%-I:%M %p"), f"{row['q335']:.0f}–{row['q665']:.0f}"]
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

    generate_detection_image(target_date, d)
    update_readme()


def find_nearest_hour_crop(target_date, target_hour=8):
    """Finds the predictions.csv row for target_date closest to target_hour (default
    8am) with quality_ok=True and a crop image still on disk. Returns the row dict, or
    None if no usable row exists for that date (e.g. the hour hasn't happened yet)."""
    with open(ds.PREDS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["date"] == target_date.isoformat() and r["quality_ok"] == "True"]
    if not rows:
        return None
    def hour_distance(r):
        h, m = map(int, r["time_local"].split(":"))
        return abs((h * 60 + m) - target_hour * 60)
    rows.sort(key=hour_distance)
    for r in rows:
        if (ds.CROPS_DIR / r["filename"]).exists():
            return r
    return None


def generate_detection_image(target_date, day_predictions_df):
    """Draws real detection boxes + confidence labels on the day's ~8am crop (the
    actual production tiling/NMS/false-positive-filter pipeline via
    detect_surfers.run_inference_with_boxes — not a simplified re-implementation),
    with the model's predicted range/median for that same hour overlaid as text.
    Skipped (not an error) if today's ~8am clip hasn't been captured/processed yet."""
    row = find_nearest_hour_crop(target_date, target_hour=8)
    if row is None:
        print(f"No usable ~8am crop for {target_date} yet — skipping detection image.")
        return

    img_path = ds.CROPS_DIR / row["filename"]
    model = ds.load_model()
    boxes = ds.run_inference_with_boxes(model, img_path)

    img = cv2.imread(str(img_path))
    # Draw boxes/labels on a copy, then alpha-blend back so they read as
    # translucent overlays rather than opaque marks on the surf photo.
    overlay = img.copy()
    BOX_COLOR = (40, 40, 220)  # BGR — red
    for x1, y1, x2, y2, conf in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(overlay, p1, p2, BOX_COLOR, 2)
        label = f"{conf:.2f}"
        cv2.putText(overlay, label, (p1[0], max(p1[1] - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)
    BOX_ALPHA = 0.75
    img = cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0)

    # Model's predicted range/median for this exact hour, if it's in the already-
    # computed day_predictions_df (it normally will be, since 8am is within the
    # dawn-dusk window) — falls back to "not available" text rather than silently
    # omitting it or computing something inconsistent from a different data source.
    hh = int(row["time_local"].split(":")[0])
    match = day_predictions_df[day_predictions_df["hour"].dt.hour == hh]
    if len(match):
        m = match.iloc[0]
        pred_text = f"Predicted: {m['point']:.0f} (33% range {m['q335']:.0f}-{m['q665']:.0f})"
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
    cv2.putText(canvas, detected_text, (8, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, pred_text, (8, h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, legend_text, (8, h + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    dated_path = CHARTS_DIR / f"detection_{target_date.isoformat()}.png"
    cv2.imwrite(str(dated_path), canvas)
    latest_path = CHARTS_DIR / "latest_detection.png"
    cv2.imwrite(str(latest_path), canvas)
    print(f"Saved detection image to {dated_path}")


README_START_MARKER = "<!-- DAILY_CHART_START -->"
README_END_MARKER = "<!-- DAILY_CHART_END -->"


def update_readme():
    """Replaces the marked section of README.md with the latest detection image
    (if one was generated) + chart-with-table image. Idempotent — safe to run
    daily; only the content between the markers changes."""
    readme_path = _PROJECT_ROOT / "README.md"
    readme = readme_path.read_text()
    if README_START_MARKER not in readme or README_END_MARKER not in readme:
        print(f"WARNING: README.md markers not found — skipping README update. "
              f"Add {README_START_MARKER} / {README_END_MARKER} to enable this.")
        return

    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    detection_block = ""
    if (CHARTS_DIR / "latest_detection.png").exists():
        detection_block = "![Latest detection review](data/charts/latest_detection.png)\n\n"

    section = (
        f"{README_START_MARKER}\n"
        f"## Today's Surfer Count Prediction\n\n"
        f"_Last updated: {generated_at}_\n\n"
        f"{detection_block}"
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
