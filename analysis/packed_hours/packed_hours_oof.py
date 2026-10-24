"""
Out-of-fold predictions for every counted hour, for studying the packed hours.

The scored forecast log covers one month. To study hours with 30+ surfers
there needs to be more of them, so this refits the production point model
(the same Poisson GBT settings as fit_surfer_count_model.py, on the same
features via its load_and_prepare()) ten times, each time holding out a
different tenth of the calendar weeks, and keeps each hour's prediction from
the fit that did not see its week.

These are hindcasts: the inputs are the conditions recorded for the hour, not
the forecast of them made the evening before. So a miss here is something the
predictors cannot explain even when the conditions are known exactly, which is
the question: what do packed hours have that the inputs don't carry?

Writes packed_hours_oof.csv next to this script.

Usage:
    python analysis/packed_hours/packed_hours_oof.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import fit_surfer_count_model as fit  # noqa: E402

OUT = Path(__file__).resolve().parent / "packed_hours_oof.csv"
GBT = dict(loss="poisson", max_iter=300, learning_rate=0.05, max_depth=4,
           l2_regularization=1.0, random_state=42)


def main():
    X, y, df, numeric_cols = fit.load_and_prepare()
    week = pd.to_datetime(df.date).dt.strftime("%G-W%V")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=10).split(X, y, groups=week):
        Xtr, Xte = fit.standardize(X.iloc[tr], X.iloc[te], numeric_cols)
        m = HistGradientBoostingRegressor(**GBT).fit(Xtr, y.iloc[tr])
        oof[te] = m.predict(Xte)
    out = df.copy()
    out["oof_pred"] = oof
    out["resid"] = out.surfer_count - out.oof_pred          # positive = under-forecast
    out.to_csv(OUT, index=False)
    d = out[out.is_night == 0]
    print(f"{len(out)} rows, {out.date.nunique()} days; daylight MAE "
          f"{d.resid.abs().mean():.2f}, bias (forecast minus actual) {-d.resid.mean():+.2f}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
