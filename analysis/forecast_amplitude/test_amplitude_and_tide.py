"""
Is the production crowd model held back too much, and is there a better way to
give it the tide?

The forecast swings less than the crowd does. Over every counted hour the
hindcast's standard deviation is 9.7 against 12.8 for the counts, and a day's
hindcast spans 25 surfers where the counts span 35, so low-tide peaks come in
short and high-tide hours a little high (analysis/packed_hours/). Two possible
causes are tested here, each against the production model on identical folds:

  capacity   the production GBT (Poisson, 300 trees at learning rate 0.05,
             depth 4, L2 1.0, default 20 rows per leaf) may be shrunk toward
             the mean. Arms loosen one setting at a time, try a squared-error
             loss, and weight the busy hours.
  tide       production gives the model tide_ft (Surfline's predicted height
             at the hour) and three day-level counts of daylight hours under
             3.5 ft (good_tide_*). The tide arms add the shape of the day's
             tide, which the height alone does not carry:
               tide_slope      ft per hour, + rising, - falling
               tide_vs_low     height above the day's daylight low, and
                               signed hours from that low
               day_low_range   how low the day's daylight low gets, and the
                               day's daylight range
             plus an arm that drops the good_tide_* columns, which permutation
             importance already scored as worthless.

The shape columns come from the tide values on the day's own rows (about 14
per day, first to last light). Tide is astronomical, so every one of them is
known in advance and usable in a forecast.

Folds: GroupKFold(10) by ISO week, as in packed_hours_oof.py. Scored on
daylight hours. "weeks better" is the share of held-out weeks whose MAE beats
production, with a Wilcoxon signed-rank p on the per-week differences.

Usage:
    python analysis/forecast_amplitude/test_amplitude_and_tide.py
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
sys.path.insert(0, str(HERE.parents[1] / "code"))
import fit_surfer_count_model as fit  # noqa: E402

OUT = HERE / "amplitude_and_tide_results.csv"
PROD = dict(loss="poisson", max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=42)
GOOD = ["good_tide_hours", "good_tide_frac", "good_tide_hours_left"]


def tide_shape(df):
    """Per-row tide shape columns from the day's own daylight rows."""
    out = pd.DataFrame(index=df.index)
    day = df[df.is_night == 0]
    # one tide value per (date, hour); rows within an hour share it
    th = day.groupby(["date", "hour_local"]).tide_ft.first().reset_index()
    th = th.sort_values(["date", "hour_local"])
    g = th.groupby("date")
    nxt_t, prv_t = g.tide_ft.shift(-1), g.tide_ft.shift(1)
    nxt_h, prv_h = g.hour_local.shift(-1), g.hour_local.shift(1)
    # central difference where both neighbours exist, one-sided at the ends
    th["tide_slope"] = np.where(
        nxt_t.notna() & prv_t.notna(), (nxt_t - prv_t) / (nxt_h - prv_h),
        np.where(nxt_t.notna(), (nxt_t - th.tide_ft) / (nxt_h - th.hour_local),
                 (th.tide_ft - prv_t) / (th.hour_local - prv_h)))
    low = th.loc[th.groupby("date").tide_ft.idxmin(), ["date", "hour_local", "tide_ft"]]
    low.columns = ["date", "low_hour", "day_low"]
    rng = th.groupby("date").tide_ft.agg(lambda s: s.max() - s.min()).rename("day_range")
    th = th.merge(low, on="date").merge(rng, on="date")
    th["tide_above_low"] = th.tide_ft - th.day_low
    th["hours_from_low"] = th.hour_local - th.low_hour
    m = df[["date", "hour_local"]].merge(th, on=["date", "hour_local"], how="left")
    m.index = df.index
    for c in ["tide_slope", "tide_above_low", "hours_from_low", "day_low", "day_range"]:
        out[c] = m[c].fillna(0.0)   # night rows: no daylight shape
    return out


def main():
    X, y, df, numeric_cols = fit.load_and_prepare()
    shape = tide_shape(df)
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V").to_numpy()
    folds = list(GroupKFold(n_splits=10).split(X, y, groups=week))
    yt = y.to_numpy()
    dayrow = (df.is_night == 0).to_numpy()
    tide = df.tide_ft.to_numpy()
    busy_w = np.where(yt >= 25, 2.0, 1.0)

    arms = {
        "production": (PROD, [], [], None),
        "600 trees": ({**PROD, "max_iter": 600}, [], [], None),
        "1000 trees": ({**PROD, "max_iter": 1000}, [], [], None),
        "learning rate 0.1": ({**PROD, "learning_rate": 0.1}, [], [], None),
        "depth 6": ({**PROD, "max_depth": 6}, [], [], None),
        "no L2": ({**PROD, "l2_regularization": 0.0}, [], [], None),
        "5 rows per leaf": ({**PROD, "min_samples_leaf": 5}, [], [], None),
        "squared-error loss": ({**PROD, "loss": "squared_error"}, [], [], None),
        "busy hours x2 weight": (PROD, [], [], busy_w),
        "+ tide_slope": (PROD, ["tide_slope"], [], None),
        "+ tide_vs_low": (PROD, ["tide_above_low", "hours_from_low"], [], None),
        "+ day_low_range": (PROD, ["day_low", "day_range"], [], None),
        "+ all tide shape": (PROD, ["tide_slope", "tide_above_low", "hours_from_low",
                                    "day_low", "day_range"], [], None),
        "- good_tide_*": (PROD, [], GOOD, None),
        "+ tide shape - good_tide_*": (PROD, ["tide_slope", "tide_above_low", "hours_from_low",
                                              "day_low", "day_range"], GOOD, None),
    }

    preds = {}
    for name, (params, add, drop, w) in arms.items():
        Xa = pd.concat([X.drop(columns=drop), shape[add]], axis=1)
        num = [c for c in numeric_cols if c not in drop] + add
        p = np.full(len(yt), np.nan)
        for tr, te in folds:
            Xtr, Xte = fit.standardize(Xa.iloc[tr], Xa.iloc[te], num)
            m = HistGradientBoostingRegressor(**params)
            m.fit(Xtr, yt[tr], sample_weight=None if w is None else w[tr])
            p[te] = np.clip(m.predict(Xte), 0, None)
        preds[name] = p
        print(f"  fitted {name}")

    d = df.loc[dayrow, ["date"]].copy()
    base = np.abs(preds["production"] - yt)[dayrow]
    rows = []
    for name, p in preds.items():
        pe, ye = p[dayrow], yt[dayrow]
        e = np.abs(pe - ye)
        d["a"], d["p"] = ye, pe
        rngs = d.groupby("date").agg(a=("a", lambda s: s.max() - s.min()),
                                     p=("p", lambda s: s.max() - s.min()),
                                     n=("a", "size"))
        rngs = rngs[rngs.n >= 10]
        diff = pd.Series(e - base).groupby(week[dayrow]).mean()
        lo, hi = tide[dayrow] < 1.5, tide[dayrow] > 3.5
        rows.append({
            "arm": name,
            "MAE": e.mean(),
            "MAE actual 30+": e[ye >= 30].mean(),
            "bias actual 30+": (pe - ye)[ye >= 30].mean(),
            "bias low tide": (pe - ye)[lo].mean(),
            "bias high tide": (pe - ye)[hi].mean(),
            "sd ratio": pe.std() / ye.std(),
            "day range ratio": (rngs.p / rngs.a).median(),
            "weeks better": "" if name == "production" else f"{int((diff < 0).sum())}/{len(diff)}",
            "p": np.nan if name == "production" else stats.wilcoxon(diff).pvalue,
        })
    res = pd.DataFrame(rows).set_index("arm")
    res.round(3).to_csv(OUT)
    n30 = int((yt[dayrow] >= 30).sum())
    print(f"\n{dayrow.sum()} daylight hours on {df[dayrow].date.nunique()} days, "
          f"{n30} with 30+ surfers; low tide < 1.5 ft: {(tide[dayrow] < 1.5).sum()}, "
          f"high > 3.5 ft: {(tide[dayrow] > 3.5).sum()}\n")
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string())


if __name__ == "__main__":
    main()
