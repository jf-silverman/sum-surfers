"""
build_training_features.py
---------------------------
Phase 1 of the surfer-count modeling plan (see PROJECT_HISTORY.md, 2026-08
"Modeling plan scoped" entry): joins predictions.csv (target: surfer_count)
with surfline_predictors.csv (features: tide/swell/wind/energy/weather/
rating) on filename, restricted to quality_ok=True rows (frames the
pipeline itself flagged as too dark/foggy to trust are excluded — the
whole point of the quality gate), and adds time-of-day/day-of-week/month
features derived from date + time_local.

Not lossy: every quality_ok=True row that has a matching predictors row is
kept, even if some predictor fields are blank (temperature_f/weather_
condition/pressure_mb/consistency_wave_count are structurally missing for
dates backfilled via the browser-HAR method — see backfill_predictors_
from_har.py's docstring). Whether to drop those columns, impute, or
restrict to complete cases is a Phase 2 modeling decision, not baked in
here.

Usage:
    python code/build_training_features.py
    python code/build_training_features.py --out data/training_features.csv
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = _PROJECT_ROOT / "data" / "predictions.csv"
PREDICTORS_CSV = _PROJECT_ROOT / "data" / "surfline_predictors.csv"
DEFAULT_OUT_CSV = _PROJECT_ROOT / "data" / "training_features.csv"

PREDICTOR_FEATURE_COLS = [
    "rating_value", "tide_ft", "surf_min_ft", "surf_max_ft",
    "primary_swell_height_ft", "primary_swell_period_s", "primary_swell_direction_deg",
    "wind_speed_mph", "wind_direction_deg", "wind_gust_mph",
    "energy_offshore_kj", "energy_nearshore_kj",
    "weather_condition", "temperature_f", "pressure_mb", "consistency_wave_count",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Simplified weather scheme (2026-08-28): Surfline's raw weather_condition has 21
# distinct values, most with under 20 occurrences — too sparse to be useful as
# individual categories (fit_surfer_count_model.py was collapsing all of them into
# a single generic OTHER bucket, discarding rain signal entirely). This strips the
# NIGHT_ prefix into its own boolean (is_night) and merges the rest into 4 broader,
# better-populated buckets: CLEAR (clear/mostly clear), CLOUDY_OVERCAST (mostly
# cloudy/overcast/cloudy), RAIN (all shower/rain/drizzle variants), FOG (fog/mist —
# stays its own category per Joel's request even though the merged count is still
# small, ~3 rows; fine for a tree model, would be unstable for a GLM).
WEATHER_SIMPLE_MAP = {
    "CLEAR": "CLEAR", "MOSTLY_CLEAR": "CLEAR",
    "MOSTLY_CLOUDY": "CLOUDY_OVERCAST", "OVERCAST": "CLOUDY_OVERCAST", "CLOUDY": "CLOUDY_OVERCAST",
    "LIGHT_SHOWERS": "RAIN", "BRIEF_SHOWERS": "RAIN", "LIGHT_RAIN": "RAIN",
    "BRIEF_SHOWERS_POSSIBLE": "RAIN", "RAIN": "RAIN", "DRIZZLE": "RAIN",
    "FOG": "FOG", "MIST": "FOG",
}


def simplify_weather_condition(raw):
    """Returns (weather_simple, is_night) from a raw Surfline weather_condition
    string (e.g. 'NIGHT_MOSTLY_CLEAR' -> ('CLEAR', True)). Unmapped/blank values
    return ('OTHER', is_night) as a safety net, not silently dropped."""
    if not raw:
        return "OTHER", False
    is_night = raw.startswith("NIGHT_")
    base = raw[len("NIGHT_"):] if is_night else raw
    return WEATHER_SIMPLE_MAP.get(base, "OTHER"), is_night


OUT_HEADER = [
    "filename", "date", "time_local",
    "surfer_count", "used_multiframe",
    "hour_local", "day_of_week", "is_weekend", "month",
    "weather_simple", "is_night",
] + PREDICTOR_FEATURE_COLS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT_CSV), help=f"Output CSV path (default: {DEFAULT_OUT_CSV})")
    return p.parse_args()


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_target_count(pred_row):
    """Prefer the multi-frame-averaged count (frame_count_mean, rounded — same
    convention detect_surfers.py's run_inference_multi() uses live) over the
    legacy single-frame surfer_count, wherever multi-frame data is available.
    backfill_multiframe_counts.py deliberately never overwrites surfer_count
    itself (non-destructive by design), so without this the model would keep
    training on stale single-frame values even after a multi-frame backfill."""
    frame_mean = pred_row.get("frame_count_mean", "").strip()
    if frame_mean:
        return round(float(frame_mean))
    return int(pred_row["surfer_count"])


def build_row(pred_row, predictor_row):
    dt = datetime.strptime(f"{pred_row['date']} {pred_row['time_local']}", "%Y-%m-%d %H:%M")
    weather_simple, is_night = simplify_weather_condition(predictor_row.get("weather_condition", ""))
    out = {
        "filename": pred_row["filename"],
        "date": pred_row["date"],
        "time_local": pred_row["time_local"],
        "surfer_count": resolve_target_count(pred_row),
        "used_multiframe": pred_row.get("frame_count_mean", "").strip() != "",
        "hour_local": dt.hour,
        "day_of_week": DAY_NAMES[dt.weekday()],
        "is_weekend": dt.weekday() >= 5,
        "month": dt.month,
        "weather_simple": weather_simple,
        "is_night": is_night,
    }
    for col in PREDICTOR_FEATURE_COLS:
        out[col] = predictor_row.get(col, "")
    return out


def main():
    args = parse_args()
    out_csv = Path(args.out)

    preds = load_rows(PREDS_CSV)
    predictors_by_filename = {r["filename"]: r for r in load_rows(PREDICTORS_CSV)}

    quality_ok = [r for r in preds if r["quality_ok"] == "True"]
    matched = [(r, predictors_by_filename[r["filename"]]) for r in quality_ok if r["filename"] in predictors_by_filename]
    unmatched = [r for r in quality_ok if r["filename"] not in predictors_by_filename]

    print(f"quality_ok=True rows in predictions.csv: {len(quality_ok)}")
    print(f"matched to a surfline_predictors.csv row: {len(matched)}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} quality_ok row(s) have no predictor match "
              f"(dates: {sorted({r['date'] for r in unmatched})})")

    rows_out = [build_row(pr, pd_) for pr, pd_ in matched]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_HEADER)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} row(s) to {out_csv}")


if __name__ == "__main__":
    main()
