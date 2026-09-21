"""
build_training_features.py
---------------------------
Phase 1 of the surfer-count modeling plan (see docs/PROJECT_HISTORY.md, 2026-08
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

import pytz

import get_clips as gc
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
PREDICTORS_CSV = _PROJECT_ROOT / "data" / "predictor_vars" / "surfline_predictors.csv"
OPENMETEO_CSV = _PROJECT_ROOT / "data" / "predictor_vars" / "openmeteo_weather.csv"
DEFAULT_OUT_CSV = _PROJECT_ROOT / "data" / "training_features.csv"

PREDICTOR_FEATURE_COLS = [
    "rating_value", "tide_ft", "surf_min_ft", "surf_max_ft",
    "primary_swell_height_ft", "primary_swell_period_s", "primary_swell_direction_deg",
    "wind_speed_mph", "wind_direction_deg", "wind_gust_mph",
    "energy_offshore_kj", "energy_nearshore_kj",
    "weather_condition", "temperature_f", "pressure_mb", "consistency_wave_count",
]

# Real observed historical weather (Open-Meteo archive, not a forecast) — see
# backfill_openmeteo_weather.py. real_humidity_pct is a validated (if imperfect)
# proxy for the fog/blur conditions the image-quality gate already flags.
OPENMETEO_FEATURE_COLS = [
    "real_temperature_f", "real_humidity_pct", "real_cloud_cover_pct",
    "real_weather_code", "real_pressure_mb",
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


# Derived tide/daylight features (added 2026-09-16). This spot surfs better on
# a lower tide, roughly under GOOD_TIDE_MAX_FT, so a day with a long low-tide
# window during daylight gives people more chances to go. That is a property of
# the whole day, not of the hour a frame was taken in, so no per-row column
# (tide_ft included) carries it — a tree can split tide_ft at 3.5 for the
# current hour, but cannot see that the window lasts eight hours today and two
# tomorrow.
GOOD_TIDE_MAX_FT = 3.5


def daylight_hours_for(date_str, _cache={}):
    """(first_hour, last_hour) of real daylight for a date, from dawn and dusk.

    A fixed 6:00-19:00 window caps the count at 14 hours, which silently
    undercounts summer: first light to last light here runs past 15 hours near the solstice,
    so a day whose tide never came up could never score above 14. Uses the same
    get_light_window() the clip collector uses, so "daylight" means one thing
    across the project. Cached per date — astral is cheap but this is called
    once per row.
    """
    if date_str not in _cache:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        tz = pytz.timezone(gc.LOCATION["timezone"])
        dawn, dusk = gc.get_light_window(d, tz)
        _cache[date_str] = (dawn.hour, dusk.hour)
    return _cache[date_str]
TIDE_FEATURE_COLS = ["good_tide_hours", "good_tide_frac", "good_tide_hours_left"]


OUT_HEADER = [
    "filename", "date", "time_local",
    "surfer_count", "used_multiframe",
    "hour_local", "day_of_week", "is_weekend", "month",
    "weather_simple", "is_night",
] + PREDICTOR_FEATURE_COLS + OPENMETEO_FEATURE_COLS + TIDE_FEATURE_COLS


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


def build_row(pred_row, predictor_row, openmeteo_row):
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
    for col in OPENMETEO_FEATURE_COLS:
        out[col] = (openmeteo_row or {}).get(col, "")
    return out


def add_tide_daylight_features(rows):
    """Per-day tide/daylight summaries, attached to every row of that day.

    Counted over the daylight hours actually observed that date rather than a
    full astronomical day: the pipeline samples roughly hourly from first
    light to last light, so the observed hours are very nearly the daylight hours, and a
    fraction alongside the raw count keeps a short collection day from looking
    like a bad-tide day. Rows whose tide is blank are skipped in the counting
    but still receive the day's values.
    """
    by_date = {}
    for r in rows:
        try:
            hour = int(r["time_local"][:2])
            tide = float(r["tide_ft"])
        except (ValueError, KeyError, TypeError):
            continue
        first_hour, last_hour = daylight_hours_for(r["date"])
        if not (first_hour <= hour <= last_hour):
            continue
        # Keyed by hour, not appended: some days carry several clips for the
        # same hour (the 60-second variability study collected 31 frames on
        # 2026-08-27), and counting rows produced 20 "daylight hours" in a
        # 14-hour window.
        by_date.setdefault(r["date"], {})[hour] = tide

    good_by_date, frac_by_date = {}, {}
    for date_str, hours in by_date.items():
        good = sorted(h for h, tide in hours.items() if tide < GOOD_TIDE_MAX_FT)
        good_by_date[date_str] = good
        frac_by_date[date_str] = len(good) / len(hours) if hours else ""

    for r in rows:
        good = good_by_date.get(r["date"])
        if good is None:
            r["good_tide_hours"] = r["good_tide_frac"] = r["good_tide_hours_left"] = ""
            continue
        r["good_tide_hours"] = len(good)
        r["good_tide_frac"] = round(frac_by_date[r["date"]], 4)
        try:
            hour = int(r["time_local"][:2])
            # Good-tide daylight hours still ahead at this moment — the version a
            # surfer deciding whether to go now would actually care about.
            r["good_tide_hours_left"] = sum(1 for h in good if h >= hour)
        except (ValueError, KeyError):
            r["good_tide_hours_left"] = ""
    return rows


def main():
    args = parse_args()
    out_csv = Path(args.out)

    preds = load_rows(PREDS_CSV)
    predictors_by_filename = {r["filename"]: r for r in load_rows(PREDICTORS_CSV)}
    openmeteo_by_filename = {r["filename"]: r for r in load_rows(OPENMETEO_CSV)} if OPENMETEO_CSV.exists() else {}

    quality_ok = [r for r in preds if r["quality_ok"] == "True"]
    matched = [(r, predictors_by_filename[r["filename"]]) for r in quality_ok if r["filename"] in predictors_by_filename]
    unmatched = [r for r in quality_ok if r["filename"] not in predictors_by_filename]

    print(f"quality_ok=True rows in predictions.csv: {len(quality_ok)}")
    print(f"matched to a surfline_predictors.csv row: {len(matched)}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} quality_ok row(s) have no predictor match "
              f"(dates: {sorted({r['date'] for r in unmatched})})")

    openmeteo_missing = sum(1 for r, _ in matched if r["filename"] not in openmeteo_by_filename)
    if openmeteo_missing:
        print(f"NOTE: {openmeteo_missing} row(s) have no Open-Meteo match — "
              f"run backfill_openmeteo_weather.py to fill these in.")

    rows_out = [build_row(pr, pd_, openmeteo_by_filename.get(pr["filename"])) for pr, pd_ in matched]
    rows_out = add_tide_daylight_features(rows_out)
    with_tide = sum(1 for r in rows_out if r["good_tide_hours"] != "")
    print(f"tide/daylight features attached to {with_tide} of {len(rows_out)} row(s) "
          f"(good tide = under {GOOD_TIDE_MAX_FT} ft, daylight = first light to last light per date)")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_HEADER)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} row(s) to {out_csv}")


if __name__ == "__main__":
    main()
