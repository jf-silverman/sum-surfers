"""
Does the stretch of low tide still AHEAD of an hour predict its crowd?

Joel's idea (2026-10-25): an hour with half an hour of sub-3 ft tide left and
the tide rising fast should draw fewer surfers than one with the same tide
height and a long, slowly dropping low still to come. A long low-tide window
in daylight gives people time to get there.

For every counted frame:
  low_left_h    hours from the frame's time until the tide next rises to
                THRESH ft or last light comes, whichever is first; 0 if the
                tide is at or above THRESH now
  wait_low_h    if the tide is at or above THRESH now, hours until it next
                drops below it within daylight (capped at the hours of daylight
                left; 0 if it is below now)

The tide curve is the day's own hourly tide_ft values (Surfline's prediction,
one per daylight hour) interpolated linearly to the minute; last light comes
from get_clips.get_light_window(), the same window the collector uses.
Tested at 3.0 ft (Joel's number) and at 3.5 ft (the good_tide_* cut-off).

Production already has good_tide_hours_left: the count of whole daylight
hours under 3.5 ft later in the day, contiguous or not. These columns differ in
being contiguous, minute-resolution and measured from the frame's own time.

Same week-grouped folds as test_amplitude_and_tide.py, plus date-shuffled
placebos (each date gets another date's columns at the nearest time of day)
for the best arm.

Usage:
    python analysis/forecast_amplitude/test_low_tide_window.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "code"))
import fit_surfer_count_model as fit  # noqa: E402
import get_clips as gc  # noqa: E402
from test_amplitude_and_tide import PROD, GOOD  # noqa: E402

OUT = HERE / "low_tide_window_results.csv"
SEEDS = range(8)


def last_light(date_str, _cache={}):
    if date_str not in _cache:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        tz = pytz.timezone(gc.LOCATION["timezone"])
        _, dusk = gc.get_light_window(d, tz)
        _cache[date_str] = dusk.hour + dusk.minute / 60
    return _cache[date_str]


def window_cols(df, thresh):
    t_row = df.time_local.str[:2].astype(int) + df.time_local.str[3:5].astype(int) / 60
    low_left = np.zeros(len(df))
    wait = np.zeros(len(df))
    for date, idx in df.groupby("date").groups.items():
        g = df.loc[idx]
        pts = g.groupby("hour_local").tide_ft.first().sort_index()
        hrs, tide = pts.index.to_numpy(float), pts.to_numpy(float)
        end = last_light(date)
        grid = np.arange(hrs[0], max(end, hrs[-1]) + 1 / 60, 1 / 60)
        if len(hrs) >= 2:
            curve = np.interp(grid, hrs, tide)
            # beyond the last hourly value, extend the last segment's slope
            past = grid > hrs[-1]
            slope = (tide[-1] - tide[-2]) / (hrs[-1] - hrs[-2])
            curve[past] = tide[-1] + slope * (grid[past] - hrs[-1])
        else:
            curve = np.full_like(grid, tide[0])
        for i in idx:
            t = t_row[i]
            j = int(np.clip(round((t - grid[0]) * 60), 0, len(grid) - 1))
            ahead = (grid >= grid[j]) & (grid <= end)
            c, gx = curve[ahead], grid[ahead]
            if len(c) == 0:
                continue
            if c[0] < thresh:
                above = np.nonzero(c >= thresh)[0]
                stop = gx[above[0]] if len(above) else end
                low_left[df.index.get_loc(i)] = max(stop - gx[0], 0)
            else:
                below = np.nonzero(c < thresh)[0]
                wait[df.index.get_loc(i)] = (gx[below[0]] - gx[0]) if len(below) else max(end - gx[0], 0)
    night = (df.is_night == 1).to_numpy()
    low_left[night] = 0
    wait[night] = 0
    s = f"{thresh:g}".replace(".", "p")
    return pd.DataFrame({f"low_left_{s}": low_left, f"wait_low_{s}": wait}, index=df.index)


def oof(X, y, extra, drop, folds, numeric_cols):
    Xa = pd.concat([X.drop(columns=drop), extra], axis=1)
    num = [c for c in numeric_cols if c not in drop] + list(extra.columns)
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        Xtr, Xte = fit.standardize(Xa.iloc[tr], Xa.iloc[te], num)
        p[te] = HistGradientBoostingRegressor(**PROD).fit(Xtr, y.iloc[tr]).predict(Xte)
    return np.clip(p, 0, None)


def placebo(df, cols, seed):
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(df.date.unique()))
    perm = dict(zip(dates, rng.permutation(dates)))
    t = df.time_local.str[:2].astype(int) * 60 + df.time_local.str[3:5].astype(int)
    key = pd.concat([df[["date"]], t.rename("t"), cols], axis=1)
    by = {d: g for d, g in key.groupby("date")}
    rows = []
    for d, tt in zip(df.date, t):
        g = by[perm[d]]
        rows.append(g.iloc[np.abs(g.t.to_numpy() - tt).argmin()][cols.columns].to_numpy(float))
    out = pd.DataFrame(rows, columns=cols.columns, index=df.index)
    out[(df.is_night == 1).to_numpy()] = 0.0
    return out


def main():
    X, y, df, numeric_cols = fit.load_and_prepare()
    w3, w35 = window_cols(df, 3.0), window_cols(df, 3.5)
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V").to_numpy()
    folds = list(GroupKFold(n_splits=10).split(X, y, groups=week))
    yt, day = y.to_numpy(), (df.is_night == 0).to_numpy()
    tide = df.tide_ft.to_numpy()

    d = df[day]
    print(f"{day.sum()} daylight frames. Below 3 ft: {(w3.low_left_3[day] > 0).mean():.0%}; "
          f"of those, low tide left median {np.median(w3.low_left_3[day][w3.low_left_3[day] > 0]):.1f} h")
    lo = day & (tide < 3)
    q = pd.qcut(w3.low_left_3[lo], 4, duplicates="drop")
    print("\nFrames below 3 ft, mean count by hours of low tide left (weekday / weekend):")
    t = pd.DataFrame({"left": q, "wk": df.is_weekend[lo], "n": yt[lo]})
    print(t.pivot_table(index="left", columns="wk", values="n", aggfunc=["mean", "size"], observed=True).round(1))

    arms = {
        "production": (w3[[]], []),
        "+ low_left 3ft": (w3[["low_left_3"]], []),
        "+ low_left, wait 3ft": (w3, []),
        "+ low_left 3.5ft": (w35[["low_left_3p5"]], []),
        "+ low_left, wait 3.5ft": (w35, []),
        "+ low_left, wait 3ft - good_tide_*": (w3, GOOD),
    }
    preds = {n: oof(X, y, e, dr, folds, numeric_cols) for n, (e, dr) in arms.items()}
    base = np.abs(preds["production"] - yt)[day]
    rows = []
    for n, p in preds.items():
        e = np.abs(p - yt)[day]
        diff = pd.Series(e - base).groupby(week[day]).mean()
        ye, pe = yt[day], p[day]
        rows.append({"arm": n, "MAE": e.mean(),
                     "bias actual 30+": (pe - ye)[ye >= 30].mean(),
                     "bias low tide <1.5": (pe - ye)[tide[day] < 1.5].mean(),
                     "weeks better": "" if n == "production" else f"{int((diff < 0).sum())}/{len(diff)}",
                     "p": np.nan if n == "production" else stats.wilcoxon(diff).pvalue})
    res = pd.DataFrame(rows).set_index("arm")
    print()
    print(res.round(3).to_string())

    best = res.drop("production").MAE.idxmin()
    extra, drop = arms[best]
    pl = []
    for s in SEEDS:
        p = oof(X, y, placebo(df, extra, s), drop, folds, numeric_cols)
        pl.append(np.abs(p - yt)[day].mean())
    pl = np.array(pl)
    real = res.loc[best, "MAE"]
    print(f"\nPlacebo for '{best}': real {real:.3f}, placebos mean {pl.mean():.3f} "
          f"(sd {pl.std(ddof=1):.3f}, range {pl.min():.3f}-{pl.max():.3f}); "
          f"real beats {int((real < pl).sum())}/{len(pl)}, {(real - pl.mean()) / pl.std(ddof=1):+.1f} sd")
    res["placebo_mean_for_best"] = np.where(res.index == best, pl.mean(), np.nan)
    res.round(3).to_csv(OUT)


if __name__ == "__main__":
    main()
