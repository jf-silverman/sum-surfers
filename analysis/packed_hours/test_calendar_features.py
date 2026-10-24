"""
Do local calendar and inland-heat features explain the packed hours the
forecast misses?

Adds day-level candidates to the production feature set and compares
out-of-fold error against the unchanged model, on the same ten week-grouped
folds as packed_hours_oof.py:

  holiday       a day most local workplaces are closed: the federal holidays
                Californians actually get off, plus the day after
                Thanksgiving (a California state holiday). Columbus Day and
                the holidays-package oddities (Susan B. Anthony Day, Diwali)
                are left out on purpose.
  long_weekend  any day inside a run of 3+ consecutive days off (weekend
                plus an adjacent holiday)
  school_out    a weekday when students in Live Oak SD or Santa Cruz City
                Schools are out (local_school_calendar.csv, transcribed from
                the official district calendars)
  inland_tmax   San Jose daily maximum, deg F (Open-Meteo archive, saved in
                san_jose_tmax.json). The "over the hill" heat that sends
                Santa Clara Valley to the coast.

Scored on daylight hours. Each arm's change in error is also tested week by
week (paired across the calendar weeks, since the hours within a week are
not independent).

Usage:
    python analysis/packed_hours/test_calendar_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import fit_surfer_count_model as fit  # noqa: E402

GBT = dict(loss="poisson", max_iter=300, learning_rate=0.05, max_depth=4,
           l2_regularization=1.0, random_state=42)
DAYS_OFF = {"New Year's Day", "Martin Luther King Jr. Day", "Presidents' Day",
            "Memorial Day", "Juneteenth National Independence Day", "Independence Day",
            "Independence Day (observed)", "Labor Day", "Veterans Day", "Thanksgiving Day",
            "Day After Thanksgiving", "Christmas Day"}


def holiday_set(years):
    h = holidays.US(state="CA", years=years)
    return {d for d, name in h.items() if name in DAYS_OFF}


def school_out_set():
    cal = pd.read_csv(HERE / "local_school_calendar.csv", comment="#")
    out = set()
    for _, r in cal.iterrows():
        for d in pd.date_range(r.start, r.end):
            if d.dayofweek < 5:
                out.add(d.date())
    return out


def day_features(dates):
    days = pd.to_datetime(pd.Series(sorted(set(dates))))
    span = pd.date_range(days.min() - pd.Timedelta(days=4), days.max() + pd.Timedelta(days=4))
    hol = holiday_set(sorted({d.year for d in span}))
    off = pd.Series([(d.dayofweek >= 5) or (d.date() in hol) for d in span], index=span)
    # length of the run of consecutive days off each day sits in
    run_id = (off != off.shift()).cumsum()
    run_len = off.groupby(run_id).transform("size").where(off, 0)
    so = school_out_set()
    t = json.load(open(HERE / "san_jose_tmax.json"))["daily"]
    tmax = dict(zip(t["time"], t["temperature_2m_max"]))
    f = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in span],
        "holiday": [int(d.date() in hol) for d in span],
        "long_weekend": (run_len >= 3).astype(int).to_numpy(),
        "school_out": [int(d.dayofweek < 5 and d.date() in so) for d in span],
    })
    f["inland_tmax"] = f.date.map(tmax)
    return f


ARMS = {
    "production features": [],
    "+ holiday, long weekend": ["holiday", "long_weekend"],
    "+ school out": ["school_out"],
    "+ inland heat": ["inland_tmax"],
    "+ all four": ["holiday", "long_weekend", "school_out", "inland_tmax"],
}


def main():
    X, y, df, numeric_cols = fit.load_and_prepare()
    days = day_features(df.date)
    df = df.merge(days, on="date", how="left")
    assert df.inland_tmax.notna().all(), "a training date has no San Jose temperature"
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V")
    folds = list(GroupKFold(n_splits=10).split(X, y, groups=week))
    day_mask = (df.is_night == 0).to_numpy()

    preds = {}
    for arm, extra in ARMS.items():
        Xa = pd.concat([X, df[extra]], axis=1) if extra else X
        num = numeric_cols + [c for c in extra if c == "inland_tmax"]
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            Xtr, Xte = fit.standardize(Xa.iloc[tr], Xa.iloc[te], num)
            oof[te] = HistGradientBoostingRegressor(**GBT).fit(Xtr, y.iloc[tr]).predict(Xte)
        preds[arm] = oof

    d = df[day_mask].copy()
    yt = y[day_mask].to_numpy()
    base_err = np.abs(preds["production features"][day_mask] - yt)
    packed = yt >= 30
    longwk = d.long_weekend.to_numpy() == 1
    rows = []
    for arm, p in preds.items():
        p = p[day_mask]
        err = np.abs(p - yt)
        wk = pd.DataFrame({"week": week[day_mask].to_numpy(), "diff": err - base_err}).groupby("week")["diff"].mean()
        better = int((wk < 0).sum())
        pval = stats.wilcoxon(wk).pvalue if arm != "production features" else np.nan
        rows.append({"arm": arm, "MAE": err.mean(),
                     "packed bias": (p[packed] - yt[packed]).mean(),
                     "packed MAE": err[packed].mean(),
                     "long-weekend MAE": err[longwk].mean(),
                     "weeks better": f"{better}/{len(wk)}" if arm != "production features" else "",
                     "p (weeks)": pval})
    r = pd.DataFrame(rows).set_index("arm")
    print(f"{day_mask.sum()} daylight hours, {d.date.nunique()} days, {week[day_mask].nunique()} weeks; "
          f"{packed.sum()} packed hours (30+); {longwk.sum()} hours on {d[longwk].date.nunique()} long-weekend days; "
          f"{(d.school_out == 1).sum()} school-out weekday hours on {d[d.school_out == 1].date.nunique()} days")
    print(r.round(3).to_string())
    r.round(3).to_csv(HERE / "calendar_feature_test.csv")


if __name__ == "__main__":
    main()
