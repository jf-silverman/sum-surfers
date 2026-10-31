"""
backfill_tide_residual.py
-------------------------
Pulls the NON-TIDAL RESIDUAL at the nearest NOAA tide gauge: observed water
level minus the harmonic tide prediction, hourly, for the whole project record.

Why this exists. The `tide_ft` feature comes from Surfline's `tides` forecast
endpoint, which is a HARMONIC prediction — astronomy. By construction it holds
no non-tidal signal, so anything that raises the actual water level without
changing the astronomy is invisible to all 34 model features.

That is not hypothetical as of October 2026. A coastally trapped Kelvin wave,
forced by the developing El Nino, is sitting on the California coast and raising
the baseline. Measured at Monterey, the daily-mean residual rose from 0.51 ft in
late September to 1.33 ft on 2026-10-10 (peak hourly 1.83 ft) before relaxing to
about 0.79 ft by 2026-10-15. Comparing like months to control for the seasonal
cycle and datum drift, October 2026 averaged 0.995 ft against October 2025's
0.237 ft — a 0.76 ft year-over-year rise. CPC's September discussion put >90% on
a very strong El Nino for winter 2026-27, so more pulses are the base case.

It also silently corrupts three features we already have. The `good_tide_*`
family thresholds PREDICTED tide at build_training_features.GOOD_TIDE_MAX_FT.
Share of daylight hours actually under that threshold:

                   by predicted tide    by observed level    overstated by
    Oct 2025            39.2%                32.8%              6.5 pp
    Oct 2026            41.7%                19.4%             22.2 pp

So in the current event the good-tide window is claimed at more than twice its
real size. Correcting that needs the residual, which is what this script fetches.

GAUGE CHOICE. Santa Cruz (9413745) has no MLLW datum and is not a full station,
so this uses Monterey (9413450), about 25 miles across the bay. For a coastally
trapped wave the cross-shore scale is 12-20 miles and the along-shore
propagation is ~150 miles/day, so at the daily resolution that matters here the
two share the same residual. This is the right call for the slow, large-scale
signal and the wrong one for anything local and fast (harbor seiche, a river
plume, a local wind setup) — those are real at Monterey and not at Pleasure
Point, and they are part of why the residual is reported with its own spread
rather than as a single number.

THE FORECASTING CONSTRAINT, which matters for how the columns are defined.
NOAA publishes no forecast of the non-tidal residual, so at a 1-7 day lead the
residual is NOT KNOWN. Two consequences built into the output:

  * The trend column uses a TRAILING window, never a centred one. A centred
    window needs future observations, so a model trained on it could not be
    served, and backtests would quietly leak.
  * Serving this feature live means persisting the last observed residual
    forward. That is defensible — the signal varies over weeks, and the
    detrended day-to-day deviation has sd 0.147 ft — but it is an assumption,
    and anything using these columns at forecast time should say so.

Columns written:

    date, hour_local        local clock (lst_ldt), matching predictions.csv
    observed_ft             measured water level, MLLW
    predicted_ft            harmonic prediction, MLLW
    residual_ft             observed - predicted: the quantity of interest
    residual_trail30_ft     trailing 30-day mean, the slow trend (causal)
    residual_dev_ft         residual_ft - residual_trail30_ft, the part NOT
                            confounded with the seasonal/event trend, and the
                            component that could identify an effect on its own
    quality                 v = verified, p = preliminary

NOT DONE HERE, deliberately. This writes a predictor file and nothing else. It
does not add features to the training table, change the good_tide_* definitions,
or touch the nightly pipeline. Each of those changes production behaviour and is
a separate decision.

API limits that shape the request loop (all per request, from CO-OPS):
hourly_height 365 days, water_level 31 days, predictions 365 days. Verified
hourly data lags real time, so recent dates come from the preliminary 6-minute
product and are averaged to the hour.

Usage:
    python code/backfill_tide_residual.py
    python code/backfill_tide_residual.py --start 2025-09-01
    python code/backfill_tide_residual.py --station 9413450 --refresh-days 45
"""

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT_CSV = _PROJECT_ROOT / "data" / "predictor_vars" / "tide_residual.csv"

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION = "9413450"          # Monterey, CA — see GAUGE CHOICE above
DATUM = "MLLW"               # same datum Surfline's tide heights use
TREND_DAYS = 30              # trailing window for residual_trail30_ft
# Preliminary data gets revised into verified data, so the most recent stretch
# is re-fetched on every run rather than trusted from the existing file.
DEFAULT_REFRESH_DAYS = 45
DEFAULT_START = date(2025, 9, 1)

CSV_HEADER = [
    "date", "hour_local", "observed_ft", "predicted_ft", "residual_ft",
    "residual_trail30_ft", "residual_dev_ft", "quality",
]

# Per-request day limits imposed by CO-OPS, by product.
CHUNK_DAYS = {"hourly_height": 365, "water_level": 31, "predictions": 365}


MIN_DAY_HOURS = 12           # a day needs this many hours to give a daily mean


def daily_dev_means(path=OUT_CSV):
    """{date: mean residual_dev_ft over that day's hours}, days with enough hours."""
    by_day = {}
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                by_day.setdefault(date.fromisoformat(r["date"]), []).append(float(r["residual_dev_ft"]))
            except (ValueError, KeyError):
                continue
    return {d: sum(v) / len(v) for d, v in by_day.items() if len(v) >= MIN_DAY_HOURS}


def prior_dev_mean(target, means):
    """The model feature `tide_resid_dev_prior_ft` for a day being forecast.

    The mean detrended residual of the latest day with data BEFORE `target`.
    At the evening run that is today for tomorrow's forecast, and still today
    for days 2-7 (carried forward, since NOAA forecasts no residual). In
    training, with the full record, it is almost always the day before. Same
    rule both ways, so the model trains on what it will be served.
    """
    earlier = [d for d in means if d < target]
    return round(means[max(earlier)], 4) if earlier else ""


def fetch(product, start, end, station):
    """One CO-OPS request. Returns [(datetime, value, quality_flag), ...].

    CO-OPS answers an out-of-range or empty window with HTTP 200 and an `error`
    object rather than a status code, so the body has to be checked. A window
    with no data is normal (a gauge outage, or a date past the verified lag) and
    returns empty rather than raising.
    """
    params = {
        "station": station, "product": product,
        "begin_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d"),
        "datum": DATUM, "units": "english", "time_zone": "lst_ldt",
        "format": "json", "application": "sum-surfers",
    }
    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        print(f"    {product} {start}..{end}: {body['error']['message'].strip()}")
        return []
    out = []
    for row in body.get("data") or body.get("predictions") or []:
        try:
            t = datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
            v = float(row["v"])
        except (KeyError, ValueError):
            continue
        out.append((t, v, row.get("q", "p" if product == "water_level" else "v")))
    return out


def fetch_range(product, start, end, station):
    """fetch() across as many chunks as the product's day limit requires."""
    step = timedelta(days=CHUNK_DAYS[product])
    rows, cur = [], start
    while cur <= end:
        stop = min(cur + step - timedelta(days=1), end)
        rows += fetch(product, cur, stop, station)
        cur = stop + timedelta(days=1)
    return rows


def to_hourly(rows):
    """Average sub-hourly samples onto the hour. {datetime_on_the_hour: (v, q)}.

    The preliminary product is 6-minute data whatever `interval` is requested,
    so averaging is not optional. Verified hourly_height arrives already hourly
    and passes through unchanged.
    """
    buckets = {}
    for t, v, q in rows:
        key = t.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append((v, q))
    # "v" only if every sample in the hour is verified -- one preliminary
    # sample makes the hour preliminary, which is the conservative label.
    return {k: (sum(v for v, _ in vs) / len(vs),
                "v" if all(q == "v" for _, q in vs) else "p")
            for k, vs in buckets.items()}


def existing_rows():
    if not OUT_CSV.exists():
        return {}
    out = {}
    with open(OUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(f"{row['date']} {row['hour_local']}", "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                continue
            out[t] = row
    return out


def project_start():
    """First date in predictions.csv, so the pull covers the whole record."""
    if not PREDS_CSV.exists():
        return DEFAULT_START
    dates = []
    with open(PREDS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("date") or "").strip()
            if d:
                dates.append(d)
    if not dates:
        return DEFAULT_START
    return min(datetime.strptime(min(dates), "%Y-%m-%d").date(), DEFAULT_START)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD; default is the project's first "
                                    "prediction date (or 2025-09-01, whichever is earlier)")
    ap.add_argument("--end", help="YYYY-MM-DD; default today")
    ap.add_argument("--station", default=STATION)
    ap.add_argument("--refresh-days", type=int, default=DEFAULT_REFRESH_DAYS,
                    help="re-fetch this many recent days even if already stored, "
                         "because preliminary data is later revised to verified")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    start = (datetime.strptime(args.start, "%Y-%m-%d").date() if args.start
             else project_start())

    have = existing_rows()
    # Incremental: fetch from the gap onward, but always re-fetch the recent
    # window. Without the second part the file would keep preliminary values
    # forever, since a date already present would never be revisited.
    if have and not args.start:
        newest = max(have).date()
        start = max(start, min(newest, end) - timedelta(days=args.refresh_days))
    # The trend column needs TREND_DAYS of history before the first output row,
    # so pull that much extra and use it for context only.
    warm = start - timedelta(days=TREND_DAYS)

    print(f"Station {args.station}, {warm} .. {end} "
          f"({TREND_DAYS}-day warm-up before {start}), datum {DATUM}")

    print("  observed (verified hourly):")
    obs = to_hourly(fetch_range("hourly_height", warm, end, args.station))
    print(f"    {len(obs)} hour(s)")
    # Verified data lags, so top up the tail from the preliminary product.
    missing_tail = max(obs).date() + timedelta(days=1) if obs else warm
    if missing_tail <= end:
        print(f"  observed (preliminary, {missing_tail} .. {end}):")
        prelim = to_hourly(fetch_range("water_level", missing_tail, end, args.station))
        print(f"    {len(prelim)} hour(s)")
        obs.update(prelim)

    print("  harmonic predictions:")
    pred = to_hourly(fetch_range("predictions", warm, end, args.station))
    print(f"    {len(pred)} hour(s)")

    shared = sorted(set(obs) & set(pred))
    if not shared:
        print("No overlapping observed/predicted hours — nothing written.", file=sys.stderr)
        return 1

    resid = {t: obs[t][0] - pred[t][0] for t in shared}

    rows, skipped_warmup = [], 0
    for t in shared:
        window = [resid[u] for u in shared
                  if t - timedelta(days=TREND_DAYS) < u <= t]
        # A trailing mean over a nearly empty window is not a trend. Rows
        # inside the warm-up are dropped rather than written with a trend
        # fitted on a handful of hours.
        if t.date() < start or len(window) < TREND_DAYS * 24 * 0.5:
            skipped_warmup += 1
            continue
        trail = sum(window) / len(window)
        rows.append({
            "date": t.strftime("%Y-%m-%d"), "hour_local": t.strftime("%H:%M"),
            "observed_ft": round(obs[t][0], 3), "predicted_ft": round(pred[t][0], 3),
            "residual_ft": round(resid[t], 3),
            "residual_trail30_ft": round(trail, 3),
            "residual_dev_ft": round(resid[t] - trail, 3),
            "quality": obs[t][1],
        })

    # Merge over anything already stored, so a run with a narrow window updates
    # the recent rows instead of truncating the file to just that window.
    merged = dict(have)
    for row in rows:
        merged[datetime.strptime(f"{row['date']} {row['hour_local']}", "%Y-%m-%d %H:%M")] = row
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for t in sorted(merged):
            w.writerow({k: merged[t].get(k, "") for k in CSV_HEADER})

    n_prelim = sum(1 for r in merged.values() if r.get("quality") == "p")
    print(f"\nWrote {len(merged)} row(s) to {OUT_CSV} "
          f"({len(rows)} fetched this run, {n_prelim} still preliminary, "
          f"{skipped_warmup} warm-up hour(s) used for the trend but not written).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
