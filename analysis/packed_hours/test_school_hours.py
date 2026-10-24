"""
Do local high-school dismissal times and Cabrillo College's semester show up
in the crowd, beyond what the model already knows?

Weekdays only, holidays excluded: every flag below is 0 on weekends and on
the California days off used in test_calendar_features.py.

  hs_session     a student day at Santa Cruz City Schools (Harbor, Soquel and
                 Santa Cruz High), from local_school_calendar.csv plus the
                 first and last student days: 2025-08-07 to 2026-05-28 and
                 from 2026-08-06.
  after_school   hs_session and the hour is after dismissal. Harbor High's
                 bell schedule (identical in 2024-25 and 2026-27), and Soquel High's
                 2026-27 schedule, both let out at
                 3:35 pm Mon/Tue/Thu/Fri and 2:10 pm Wednesday, so the flag
                 starts at the 4pm hour, and at the 3pm hour on Wednesday.
  cypress_out    hs_session and the hour is after Cypress High lets out. Cypress
                 (2039 Merrill St, 1.5 km from the camera and the closest high
                 school) ends classes at 1:25 pm Mon/Tue/Thu/Fri with an
                 optional tutorial to 2:20, and at 1:10 pm Wednesday (its
                 2026-27 bell schedule), so the flag starts at the 2pm hour, and
                 at 1pm on Wednesday. Cypress is a County Office of Education
                 school; its own calendar was not checked, so the SCCS student
                 days stand in for it.
  delta_out      hs_session and the hour is after Delta High's morning group
                 leaves (12:10 pm Mon-Thu, 11:25 am Friday, so from the 12pm
                 hour). Its afternoon group leaves at 3:55 pm (1:25 pm Friday),
                 which after_school already covers Mon-Thu. Delta is on the
                 Cabrillo campus in Aptos, 5.3 km away, and enrols about 114.
  cabrillo       a Cabrillo College class day: fall 2025 Aug 25 - Dec 13,
                 spring 2026 Jan 26 - May 23 (break Mar 23-28), fall 2026
                 Aug 24 - Dec 12, less its listed holidays (from Cabrillo's
                 Dates and Deadlines pages).

Three looks, all on the week-grouped out-of-fold residuals from
packed_hours_oof.py (actual minus hindcast, so positive = model too low):
  1. residual by hour, school weekdays against non-school weekdays
  2. Wednesday early release: Wednesday minus the other weekdays at the 3pm
     hour, school weeks against non-school weeks (a difference in
     differences, so the Wednesday effect itself cancels)
  3. the features added to the model, same folds, as in
     test_calendar_features.py

Usage:
    python analysis/packed_hours/test_school_hours.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "code"))
import fit_surfer_count_model as fit  # noqa: E402
from test_calendar_features import GBT, holiday_set  # noqa: E402

HS_YEARS = [("2025-08-07", "2026-05-28"), ("2026-08-06", "2027-05-27")]
CABRILLO_TERMS = [("2025-08-25", "2025-12-13"), ("2026-01-26", "2026-05-23"),
                  ("2026-08-24", "2026-12-12")]
CABRILLO_OFF = ["2025-09-01", "2025-09-26", "2025-11-10", "2025-11-27", "2025-11-28",
                "2026-01-19", "2026-02-13", "2026-02-16", "2026-03-31", "2026-05-25",
                "2026-09-07", "2026-09-25", "2026-11-11", "2026-11-26", "2026-11-27"]
CABRILLO_OFF += [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-03-23", "2026-03-28")]


def in_ranges(day, ranges):
    return any(a <= day <= b for a, b in ranges)


def flags(df):
    d = pd.to_datetime(df.date)
    hol = {x.strftime("%Y-%m-%d") for x in holiday_set([2025, 2026, 2027])}
    cal = pd.read_csv(HERE / "local_school_calendar.csv", comment="#")
    cal = cal[cal.district == "SCCS"]
    sccs_off = {x.strftime("%Y-%m-%d") for _, r in cal.iterrows()
                for x in pd.date_range(r.start, r.end)}
    workday = (d.dt.dayofweek < 5) & ~df.date.isin(hol)
    hs = workday & df.date.map(lambda x: in_ranges(x, HS_YEARS)) & ~df.date.isin(sccs_off)
    first_free = np.where(d.dt.dayofweek == 2, 15, 16)          # Wednesday lets out at 2:10
    out = pd.DataFrame(index=df.index)
    out["hs_session"] = hs.astype(int)
    out["after_school"] = (hs & (df.hour_local >= first_free)).astype(int)
    out["cypress_out"] = (hs & (df.hour_local >= np.where(d.dt.dayofweek == 2, 13, 14))).astype(int)
    out["delta_out"] = (hs & (df.hour_local >= 12)).astype(int)
    out["cabrillo"] = (workday & df.date.map(lambda x: in_ranges(x, CABRILLO_TERMS))
                       & ~df.date.isin(CABRILLO_OFF)).astype(int)
    out["workday"] = workday.astype(int)
    return out


def main():
    oof = pd.read_csv(HERE / "packed_hours_oof.csv")
    oof = pd.concat([oof, flags(oof)], axis=1)
    w = oof[(oof.workday == 1) & (oof.is_night == 0)].copy()
    w["dow"] = pd.to_datetime(w.date).dt.dayofweek
    print(f"Workday daylight hours: {len(w)} on {w.date.nunique()} days "
          f"({w[w.hs_session == 1].date.nunique()} high-school days, "
          f"{w[w.hs_session == 0].date.nunique()} not; "
          f"{w[w.cabrillo == 1].date.nunique()} Cabrillo class days)\n")

    print("1. Mean residual (actual minus hindcast) by hour, workdays")
    t = w.pivot_table(index="hs_session", columns="hour_local", values="resid", aggfunc="mean")
    t.index = ["school out", "school in"]
    print(t.round(1).to_string())
    t = w.pivot_table(index="hs_session", columns="hour_local", values="surfer_count", aggfunc="mean")
    t.index = ["school out (mean count)", "school in (mean count)"]
    print(t.round(1).to_string(), "\n")

    print("2. Wednesday early release, 3pm hour: Wednesday minus other weekdays")
    h3 = w[w.hour_local == 15]
    for s, name in [(1, "school weeks"), (0, "non-school weeks")]:
        x = h3[h3.hs_session == s]
        a, b = x[x.dow == 2], x[x.dow != 2]
        print(f"   {name:16s} residual {a.resid.mean() - b.resid.mean():+5.1f}  "
              f"count {a.surfer_count.mean() - b.surfer_count.mean():+5.1f}  "
              f"(Wed n={len(a)}, other n={len(b)})")
    x = h3[h3.hs_session == 1]
    print(f"   school weeks, Wed vs others residual: p={stats.mannwhitneyu(x[x.dow == 2].resid, x[x.dow != 2].resid).pvalue:.2f}\n")

    print("3. As model features (same week-grouped folds)")
    X, y, df, numeric_cols = fit.load_and_prepare()
    f = flags(df)
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V")
    folds = list(GroupKFold(n_splits=10).split(X, y, groups=week))
    arms = {"production features": [], "+ hs_session, after_school": ["hs_session", "after_school"],
            "+ cypress_out": ["hs_session", "cypress_out"],
            "+ delta_out": ["hs_session", "delta_out"],
            "+ cabrillo": ["cabrillo"],
            "+ all": ["hs_session", "after_school", "cypress_out", "delta_out", "cabrillo"]}
    day = (df.is_night == 0).to_numpy()
    wk = (f.workday == 1).to_numpy() & day
    yt = y.to_numpy()
    preds = {}
    for arm, extra in arms.items():
        Xa = pd.concat([X, f[extra]], axis=1) if extra else X
        p = np.full(len(y), np.nan)
        for tr, te in folds:
            Xtr, Xte = fit.standardize(Xa.iloc[tr], Xa.iloc[te], numeric_cols)
            p[te] = HistGradientBoostingRegressor(**GBT).fit(Xtr, y.iloc[tr]).predict(Xte)
        preds[arm] = p
    base = np.abs(preds["production features"] - yt)
    rows = []
    for arm, p in preds.items():
        e = np.abs(p - yt)
        diff = pd.Series(e[day] - base[day]).groupby(week[day].to_numpy()).mean()
        after = wk & (df.hour_local.to_numpy() >= 12)
        rows.append({"arm": arm, "MAE daylight": e[day].mean(), "MAE workdays": e[wk].mean(),
                     "MAE workday 12pm+": e[after].mean(),
                     "weeks better": f"{int((diff < 0).sum())}/{len(diff)}" if arm != "production features" else "",
                     "p (weeks)": stats.wilcoxon(diff).pvalue if arm != "production features" else np.nan})
    print(pd.DataFrame(rows).set_index("arm").round(3).to_string())


if __name__ == "__main__":
    main()
