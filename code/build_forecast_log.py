"""
One file holding every forecast this project has made and what actually happened.

Until now the two halves lived apart: forecasts in `data/forecasts/forecast_<date>.csv`
(one file per day, written when the forecast is made) and counts in
`data/predictions/predictions.csv`. Answering "how have the forecasts been doing"
meant joining them by hand every time, and nothing accumulated.

This builds `data/forecasts/forecast_log.csv`: one row per forecast hour, with
the prediction, the band, the actual count, the error, and whether the actual
fell inside the band. It is **rebuilt from the sources**, not appended to, so it
can never drift from them — the per-day forecast files stay the record of what
was predicted before the day happened, and this is a view over them.

Rows are kept for hours that have no actual yet (a forecast for tomorrow), with
empty actual/error columns, so the file also shows what is currently outstanding.

Usage:
    python code/build_forecast_log.py
    python code/build_forecast_log.py --summary   # also print accuracy so far
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECASTS_DIR = _PROJECT_ROOT / "data" / "forecasts"
PREDICTIONS_CSV = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT_CSV = FORECASTS_DIR / "forecast_log.csv"

# Columns every forecast file must have. Anything else it carries (a crowd
# rating, weather, tide) is passed through untouched, so the log keeps working
# when the forecast format gains fields.
REQUIRED = ["date", "hour_local", "predicted", "lower_q10", "upper_q90"]


def load_forecasts():
    files = sorted(FORECASTS_DIR.glob("forecast_*.csv"))
    files = [f for f in files if f.name != OUT_CSV.name]
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"  WARNING: {f.name} lacks {missing} — skipped")
            continue
        df["forecast_file"] = f.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED)
    return pd.concat(frames, ignore_index=True)


def load_actuals():
    """Hourly counts keyed by (date, hour) — the mean where an hour has two clips."""
    df = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    df = df[(df["quality_ok"].astype(str) == "True") & df["surfer_count"].notna()].copy()
    df["hour"] = df["time_local"].astype(str).str.slice(0, 2).astype(int)
    grouped = (df.groupby(["date", "hour"])
                 .agg(actual=("surfer_count", "mean"),
                      actual_clips=("surfer_count", "size"))
                 .reset_index())
    return grouped


def build():
    fc = load_forecasts()
    if fc.empty:
        print("No forecast files found — nothing to build yet.")
        return None

    fc["hour"] = fc["hour_local"].astype(str).str.slice(0, 2).astype(int)
    actuals = load_actuals()
    merged = fc.merge(actuals, on=["date", "hour"], how="left")

    merged["error"] = merged["actual"] - merged["predicted"]
    merged["abs_error"] = merged["error"].abs()
    merged["inside_band"] = merged.apply(
        lambda r: "" if pd.isna(r["actual"])
        else bool(r["lower_q10"] <= r["actual"] <= r["upper_q90"]), axis=1)
    merged["band_width"] = merged["upper_q90"] - merged["lower_q10"]

    lead = []
    for _, r in merged.iterrows():
        made = str(r.get("forecast_made_at", "") or "")
        try:
            made_date = datetime.fromisoformat(made).date()
            target = datetime.strptime(r["date"], "%Y-%m-%d").date()
            lead.append((target - made_date).days)
        except (ValueError, TypeError):
            lead.append("")
    merged["lead_days"] = lead

    front = ["date", "hour_local", "hour", "predicted", "lower_q10", "upper_q90",
             "band_width", "actual", "actual_clips", "error", "abs_error",
             "inside_band", "lead_days"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest].sort_values(["date", "hour"])

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    scored = merged["actual"].notna().sum()
    print(f"Wrote {len(merged)} forecast hour(s) to {OUT_CSV}")
    print(f"  {scored} already have an actual count; {len(merged) - scored} still outstanding")
    return merged


def summarize(df):
    scored = df[df["actual"].notna()]
    if scored.empty:
        print("\nNothing scored yet — no forecast hour has an actual count against it.")
        return
    inside = scored["inside_band"].astype(str).eq("True").sum()
    print(f"\nAccuracy so far, on {len(scored)} scored hour(s):")
    print(f"  mean absolute error : {scored['abs_error'].mean():.2f} surfers")
    print(f"  bias                : {scored['error'].mean():+.2f} "
          f"({'forecast runs low' if scored['error'].mean() > 0 else 'forecast runs high'})")
    print(f"  inside the 80% band : {inside} of {len(scored)} ({inside / len(scored):.0%})")
    print(f"  mean band width     : {scored['band_width'].mean():.1f} surfers")
    if scored["date"].nunique() > 1:
        print("\n  By day:")
        for date, g in scored.groupby("date"):
            hit = g["inside_band"].astype(str).eq("True").sum()
            print(f"    {date}: {len(g):>2} hours, MAE {g['abs_error'].mean():>5.2f}, "
                  f"{hit}/{len(g)} in band")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", action="store_true", help="print accuracy so far")
    args = p.parse_args()
    df = build()
    if df is not None and args.summary:
        summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
