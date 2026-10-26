"""
Placebo for the "+ all tide shape" arm of test_amplitude_and_tide.py.

Each placebo gives every date the tide-shape columns of a different, randomly
chosen date at the same hour (falling back to that date's nearest hour), so the
columns keep realistic values and their within-day structure but lose their
link to the day they describe. If the real columns beat most placebos, the gain
is the tide's shape, not the extra columns.

Usage:
    python analysis/forecast_amplitude/placebo_tide_shape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "code"))
import fit_surfer_count_model as fit  # noqa: E402
from test_amplitude_and_tide import PROD, tide_shape  # noqa: E402

COLS = ["tide_slope", "tide_above_low", "hours_from_low", "day_low", "day_range"]
SEEDS = range(8)


def oof_mae(X, y, extra, folds, numeric_cols, day, week):
    Xa = pd.concat([X, extra], axis=1)
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        Xtr, Xte = fit.standardize(Xa.iloc[tr], Xa.iloc[te], numeric_cols + list(extra.columns))
        p[te] = HistGradientBoostingRegressor(**PROD).fit(Xtr, y.iloc[tr]).predict(Xte)
    e = np.abs(np.clip(p, 0, None) - y.to_numpy())[day]
    return e.mean(), pd.Series(e).groupby(week[day]).mean()


def main():
    X, y, df, numeric_cols = fit.load_and_prepare()
    shape = tide_shape(df)
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V").to_numpy()
    folds = list(GroupKFold(n_splits=10).split(X, y, groups=week))
    day = (df.is_night == 0).to_numpy()

    base, bw = oof_mae(X, y, shape[[]], folds, numeric_cols, day, week)
    real, rw = oof_mae(X, y, shape[COLS], folds, numeric_cols, day, week)
    print(f"production {base:.3f}   real tide shape {real:.3f} ({real - base:+.3f})")

    key = pd.concat([df[["date", "hour_local"]], shape[COLS]], axis=1)
    lut = key.groupby(["date", "hour_local"])[COLS].first()
    dates = np.array(sorted(df.date.unique()))
    out = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        perm = dict(zip(dates, rng.permutation(dates)))
        rows = []
        for d, h in zip(df.date, df.hour_local):
            src = lut.loc[perm[d]]
            hh = src.index[np.abs(src.index.to_numpy() - h).argmin()]
            rows.append(src.loc[hh].to_numpy())
        fake = pd.DataFrame(rows, columns=COLS, index=df.index)
        fake[df.is_night.to_numpy() == 1] = 0.0
        m, _ = oof_mae(X, y, fake, folds, numeric_cols, day, week)
        out.append(m)
        print(f"  placebo seed {s}: {m:.3f} ({m - base:+.3f})")
    out = np.array(out)
    print(f"\nplacebos: mean {out.mean():.3f} (sd {out.std(ddof=1):.3f}); "
          f"real beats {int((real < out).sum())}/{len(out)}; "
          f"real vs placebo mean {real - out.mean():+.3f} "
          f"= {(real - out.mean()) / out.std(ddof=1):+.1f} sd")


if __name__ == "__main__":
    main()
