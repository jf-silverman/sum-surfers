"""
build_retrospective_forecast.py
-------------------------------
Rebuilds what the forecast WOULD have said for a past date, using only data
that existed before it.

`data/forecasts/forecast_log.csv` only begins around 2026-09, so review sets
generated from the archive (back to 2025-10) have no forecast to be scored
against. This fills that gap, which is the only way to measure forecast skill
in winter, spring and summer -- seasons the model is fitted on least and has
never been evaluated in.

WHAT MAKES THIS HONEST, AND WHAT DOES NOT

Two input families, with opposite properties, and the difference is the whole
design:

  Surfline predictors -- faithfully reproducible. Verified 2026-10-06: values
  the nightly run stored live on 2026-09-26 match what the historical endpoint
  returns now to 0.002 mph on wind and exactly on rating. Surfline's archive
  preserves the forecast AS ISSUED, so no special handling is needed.

  Open-Meteo `real_*` weather -- NOT reproducible from the archive. The archive
  returns what was OBSERVED, and handing a model observations for the day it is
  forecasting is the perfect-prog trap: it would flatter every number produced
  here. This script therefore fetches the day-1-ahead forecast as issued, from
  the previous-runs API, and substitutes it for the target date's `real_*`
  columns (Joel's decision, 2026-10-06).

Training rows keep their archive values, deliberately. That is not an
oversight: production trains on the Open-Meteo archive and serves with the
forecast API, so reproducing production means reproducing that asymmetry. The
mismatch itself is logged separately in docs/model_and_feature_ideas.md.

LIMITS, which must travel with any number this produces

  * DAY-AHEAD ONLY. Surfline's archive returns the near-term forecast, not what
    was predicted 5-7 days out, so longer leads cannot be reconstructed. Never
    quote a figure from here as a 7-day result.
  * The training cutoff is enforced per target date, so a date early in the
    record is forecast by a model with very little history. The output records
    `n_train` for exactly this reason -- a 2025-10 forecast is not comparable
    to a 2026-08 one.
  * Retrospective and live forecasts should be scored in SEPARATE strata. They
    are built differently and pooling them hides that.

Usage:
    python code/build_retrospective_forecast.py --dates 2025-10-25,2026-03-12
    python code/build_retrospective_forecast.py --missing-from-reviews
    python code/build_retrospective_forecast.py --missing-from-reviews --write
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

import fit_surfer_count_model as fsc  # noqa: E402
from predict_surf_count import MEAN_KWARGS  # noqa: E402

REVIEW_CSV = _PROJECT_ROOT / "data" / "reviews" / "review_all.csv"
OUT_CSV = _PROJECT_ROOT / "data" / "forecasts" / "retrospective_forecasts.csv"

LAT, LON, TZ = 36.9577, -121.9688, "America/Los_Angeles"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# Open-Meteo variable -> the training column it replaces. "_previous_day1" is
# the run issued one day before the target, i.e. a day-ahead forecast.
WX_MAP = {
    "temperature_2m_previous_day1": "real_temperature_f",
    "relative_humidity_2m_previous_day1": "real_humidity_pct",
    "cloud_cover_previous_day1": "real_cloud_cover_pct",
    "surface_pressure_previous_day1": "real_pressure_mb",
}
MIN_TRAIN_DAYS = 10


def fetch_forecast_weather(date):
    """Day-1-ahead forecast for one date, keyed by local hour -> {col: value}."""
    r = requests.get(PREV_RUNS_URL, params={
        "latitude": LAT, "longitude": LON, "start_date": date, "end_date": date,
        "hourly": ",".join(WX_MAP), "timezone": TZ}, timeout=40)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j.get("reason", "open-meteo error"))
    h = j["hourly"]
    out = {}
    for i, t in enumerate(h["time"]):
        hour = datetime.fromisoformat(t).hour
        row = {}
        for src, dst in WX_MAP.items():
            v = h[src][i]
            if v is None:
                continue
            # The archive stores Fahrenheit; the API returns Celsius.
            row[dst] = round(v * 9 / 5 + 32, 1) if dst == "real_temperature_f" else v
        if row:
            out[hour] = row
    return out


def forecast_one_date(date, X, y, df, numeric_cols, wx):
    """Fit on everything strictly before `date`, predict that date's hours."""
    day = df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    train_mask = (day < date).values
    test_mask = (day == date).values
    n_train_days = day[train_mask].nunique()
    if test_mask.sum() == 0:
        return None, f"no rows for {date} in training_features.csv"
    if n_train_days < MIN_TRAIN_DAYS:
        return None, f"only {n_train_days} prior day(s) — below the {MIN_TRAIN_DAYS}-day floor"

    X_te = X[test_mask].copy()
    hours = pd.to_datetime(df.loc[test_mask, "time_local"], format="%H:%M").dt.hour.values
    # Swap in forecast-as-issued weather for the target date only.
    swapped = 0
    for col in set(WX_MAP.values()):
        if col not in X_te.columns:
            continue
        vals = [wx.get(h, {}).get(col) for h in hours]
        if any(v is not None for v in vals):
            X_te[col] = [v if v is not None else old
                         for v, old in zip(vals, X_te[col].values)]
            swapped += 1

    X_tr = X[train_mask]
    X_tr_s, X_te_s = fsc.standardize(X_tr, X_te, numeric_cols)
    # The production chart's point estimate is the 0.50 quantile model, so use
    # the same thing rather than a Poisson mean — they are not interchangeable.
    model, _ = fsc.fit_quantile_model_robust(X_tr_s, y[train_mask], 0.50,
                                             allow_degenerate=True)
    pred = np.maximum(model.predict(X_te_s), 0.0)
    return pd.DataFrame({
        "date": date,
        "time_local": df.loc[test_mask, "time_local"].values,
        "filename": df.loc[test_mask, "filename"].values,
        "retro_forecast": np.round(pred, 1),
        "detector_count": df.loc[test_mask, "surfer_count"].values,
        "n_train_rows": int(train_mask.sum()),
        "n_train_days": int(n_train_days),
        "wx_cols_swapped": swapped,
    }), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", help="comma-separated YYYY-MM-DD")
    ap.add_argument("--missing-from-reviews", action="store_true",
                    help="every review-sheet date whose forecast column is blank")
    ap.add_argument("--representative", type=int, metavar="N",
                    help="Sample N days at random, stratified by month, to MEASURE "
                         "forecast skill. Use this and not --missing-from-reviews for "
                         "any accuracy figure — see the sampling note below.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--write", action="store_true",
                    help="also fill the review sheet's blank forecast cells")
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--label", default="", help="tag written into the output rows")
    args = ap.parse_args()

    if args.representative:
        # WHY A SEPARATE SAMPLE (2026-10-08). The six days in the review sheet
        # were chosen for DETECTOR review -- deliberately the most crowded and
        # foggiest in the corpus -- so they average 23.3 surfers/hour against a
        # corpus mean of 14.1, and two sit at the 99th percentile. Forecast
        # error is worst exactly at crowd extremes, so skill measured on them is
        # pessimistic by construction: 11.27 MAE against about 4.1 for the live
        # day-ahead forecast. Those 82 forecasts are still valid FOR THOSE DAYS,
        # and they are what the review sheet needs; they are simply not a sample
        # anything can be generalised from. A month-stratified random draw is.
        tf = pd.read_csv(_PROJECT_ROOT / "data" / "training_features.csv")
        d = tf["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        all_days = pd.Series(sorted(d.unique()))
        # A day needs MIN_TRAIN_DAYS of history before it, so the earliest are
        # not eligible -- that is a real limit, not a sampling choice.
        eligible = all_days[all_days.index >= MIN_TRAIN_DAYS]
        by_month = eligible.groupby(eligible.str[:7])
        per = max(1, args.representative // max(by_month.ngroups, 1))
        rng = np.random.default_rng(args.seed)
        picked = []
        for _, grp in by_month:
            g = list(grp)
            picked += list(rng.choice(g, size=min(per, len(g)), replace=False))
        # Top up to N from whatever is left, so a thin month does not shrink the draw.
        leftover = [x for x in eligible if x not in picked]
        if len(picked) < args.representative and leftover:
            picked += list(rng.choice(leftover,
                                      size=min(args.representative - len(picked), len(leftover)),
                                      replace=False))
        dates = sorted(picked)[:args.representative]
    elif args.missing_from_reviews:
        rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
        rv = rv[rv["date"].str.match(r"^\d{4}-\d{2}-\d{2}$")]
        dates = sorted(rv.loc[rv["forecast"].str.strip() == "", "date"].unique())
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        sys.exit("Give --dates or --missing-from-reviews.")
    print(f"{len(dates)} date(s): {', '.join(dates)}\n")

    X, y, df, numeric_cols = fsc.load_and_prepare()
    frames, problems = [], []
    for i, d in enumerate(dates):
        try:
            wx = fetch_forecast_weather(d)
        except Exception as e:
            problems.append(f"{d}: weather fetch failed — {type(e).__name__}: {e}")
            continue
        out, err = forecast_one_date(d, X, y, df, numeric_cols, wx)
        if err:
            problems.append(f"{d}: {err}")
            continue
        frames.append(out)
        print(f"  {d}  {len(out):2d} hours   trained on {out.n_train_days.iloc[0]:3d} days "
              f"/ {out.n_train_rows.iloc[0]:4d} rows   {out.wx_cols_swapped.iloc[0]}/4 weather cols forecast")
        if i < len(dates) - 1:
            time.sleep(1.5)

    if problems:
        print("\nSKIPPED:")
        for p in problems:
            print(f"  {p}")
    if not frames:
        sys.exit("\nNothing produced.")

    res = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n{len(res)} hour(s) -> {out}")

    if args.write:
        rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
        key = res.set_index("filename")["retro_forecast"].astype(str)
        blank = rv["forecast"].str.strip() == ""
        fill = rv["filename"].map(key)
        n = int((blank & fill.notna()).sum())
        rv.loc[blank & fill.notna(), "forecast"] = fill[blank & fill.notna()]
        rv.to_csv(REVIEW_CSV, index=False)
        print(f"filled {n} blank forecast cell(s) in {REVIEW_CSV.name}")
        print("NOTE: these are RETROSPECTIVE, day-ahead only. Score them in a "
              "separate stratum from live forecasts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
