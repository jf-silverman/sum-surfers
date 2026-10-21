"""
Does a hurdle model fix the forecast's quiet-hour over-prediction? (B19)

Two separate findings point at the same cause. The lower band edge collapsed to
a constant 0.0 on the full record (2026-10-13 through 10-20), which is why the
band moved q0.10 -> q0.15. And the re-exported release showed where the error
actually lives: on held-out hours with 0-5 actual surfers the model runs +4.01
mean bias, against -14.74 on hours with 30 or more. The headline +1.02 is those
cancelling. Raising the lower edge made that worse, not better -- the production
band now misses 33.6% low against a 15% target -- so the edge level is not the
lever.

The suspected cause is that one quantile model is being asked to describe two
different things at once: whether anyone is out at all, and how many are out
when someone is. The empty-hour share is strongly seasonal (8.0% in 2026-07,
19.5% in 2026-10, 33.8% in 2025-11, 38.9% in 2025-12), so this gets worse
through the winter. This script fits the alternative -- a hurdle model -- and
scores both on the same held-out split:

    P(count > 0)                     classifier
    quantiles of count | count > 0   quantile GBTs on the positive rows only

and composes them back into predictive quantiles. For a target quantile q, if
the probability of an empty hour already exceeds q the prediction is 0;
otherwise it is the conditional quantile at (q - p_zero) / (1 - p_zero),
interpolated across a grid of conditional levels.

Judged on pinball loss (the proper scoring rule for quantiles -- coverage alone
can be bought with a floor), plus coverage of the production band, one-sided
miss rates, and bias by actual-count bucket, which is the thing that needs to
move.

Split: whole days, train on the past and test on the most recent days
(`split_by_day(scheme="forward")`). The random row split this script used
first leaked the test day's own crowd level into training (B21) and flattered
both models.

Usage:
    python code/eval_hurdle_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_surfer_count_model as fit  # noqa: E402

# 0.15 is in the list because it is the edge production actually draws
# (fit.BAND_LOWER_Q); 0.10 stays so the old band can still be compared.
QUANTILES = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
INTERVALS = ((0.40, 0.60, "20%"), (0.30, 0.70, "40%"), (0.20, 0.80, "60%"),
             (fit.BAND_LOWER_Q, fit.BAND_UPPER_Q, fit.BAND_LABEL), (0.10, 0.90, "80%"))
COUNT_BUCKETS = ((0, 5), (5, 15), (15, 30), (30, 10**6))


def pinball(y_true, y_pred, q):
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_baseline(X_train, y_train, X_test):
    """Today's approach: one quantile GBT per level, fit on all rows."""
    preds = {}
    for q in QUANTILES:
        # allow_degenerate: the collapse at the low tail is the thing under
        # test, so the baseline has to be allowed to exhibit it rather than
        # raise. Any collapse is printed by the fitter.
        model, _leaf = fit.fit_quantile_model_robust(
            X_train, y_train, q, allow_degenerate=True)
        preds[q] = np.asarray(model.predict(X_test), dtype=float)
    return preds


def fit_hurdle(X_train, y_train, X_test):
    """P(count>0) x quantiles of count given count>0."""
    is_pos = (y_train > 0).astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=42)
    clf.fit(X_train, is_pos)
    p_zero = 1.0 - clf.predict_proba(X_test)[:, 1]

    pos_mask = y_train > 0
    X_pos, y_pos = X_train[pos_mask], y_train[pos_mask]
    print(f"  Hurdle: {int(is_pos.sum())} positive of {len(y_train)} training rows "
          f"({1 - is_pos.mean():.1%} empty); conditional models fit on the positives.")

    # The conditional level that composes back to an unconditional q differs per
    # ROW, because p_zero does, so no single conditional model serves everyone.
    # Fit the conditional quantile on a grid of levels and interpolate per row.
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    cond_stack = np.vstack([
        np.asarray(fit.fit_quantile_model_robust(
            X_pos, y_pos, float(g), allow_degenerate=True)[0].predict(X_test), dtype=float)
        for g in grid])                                       # (n_grid, n_test)
    cond_stack = np.maximum.accumulate(cond_stack, axis=0)    # monotone in level

    preds = {}
    for q in QUANTILES:
        adj = (q - p_zero) / np.maximum(1.0 - p_zero, 1e-9)   # conditional level per row
        out = np.zeros(len(p_zero))
        inside = adj > 0
        cols = np.where(inside)[0]
        a = adj[inside]
        # Linear interpolation between the two bracketing grid levels. Clipping
        # the weight also handles a > 0.95 and a < 0.05 by holding the end model.
        lo = np.clip(np.searchsorted(grid, a, side="right") - 1, 0, len(grid) - 2)
        w = np.clip((a - grid[lo]) / (grid[lo + 1] - grid[lo]), 0.0, 1.0)
        out[inside] = (1.0 - w) * cond_stack[lo, cols] + w * cond_stack[lo + 1, cols]
        preds[q] = np.maximum(out, 0.0)
    return preds, p_zero


def enforce_monotone(preds):
    qs = sorted(preds)
    stack = np.vstack([preds[q] for q in qs])
    stack = np.maximum.accumulate(stack, axis=0)
    return {q: stack[i] for i, q in enumerate(qs)}


def report(name, preds, y_test):
    preds = enforce_monotone(preds)
    y = np.asarray(y_test, dtype=float)
    print(f"\n=== {name} ===")

    total_pin = 0.0
    print(f"  {'quantile':>9} {'pinball':>9} {'mean pred':>10}")
    for q in QUANTILES:
        pl = pinball(y, preds[q], q)
        total_pin += pl
        print(f"  {q:>9.2f} {pl:>9.4f} {preds[q].mean():>10.2f}")
    print(f"  {'mean':>9} {total_pin / len(QUANTILES):>9.4f}")

    print(f"\n  {'interval':>9} {'coverage':>9} {'target':>7} {'below':>7} {'above':>7}")
    for lo, hi, label in INTERVALS:
        below = float(np.mean(y < preds[lo]))
        above = float(np.mean(y > preds[hi]))
        cov = 1.0 - below - above
        print(f"  {label:>9} {cov:>8.1%} {hi - lo:>7.0%} {below:>7.1%} {above:>7.1%}")

    lo = preds[fit.BAND_LOWER_Q]
    pinned = float(np.mean(lo < 1))
    nonempty = y > 0
    pinned_nonempty = float(np.mean(lo[nonempty] < 1))
    print(f"\n  Lower edge (q={fit.BAND_LOWER_Q}) under 1 surfer: {pinned:.1%} of hours, "
          f"{pinned_nonempty:.1%} of hours that had surfers")
    print(f"  Median-quantile MAE: {np.mean(np.abs(y - preds[0.50])):.2f}, "
          f"bias {np.mean(preds[0.50] - y):+.2f}")

    # The thing that has to move: the median's bias on near-empty hours.
    print(f"\n  {'actual':>9} {'n':>5} {'MAE':>7} {'bias':>8}")
    for lo_b, hi_b in COUNT_BUCKETS:
        m = (y >= lo_b) & (y < hi_b)
        if not m.any():
            continue
        label = f"{lo_b}-{hi_b}" if hi_b < 10**6 else f"{lo_b}+"
        print(f"  {label:>9} {int(m.sum()):>5} {np.mean(np.abs(preds[0.50][m] - y[m])):>7.2f} "
              f"{np.mean(preds[0.50][m] - y[m]):>+8.2f}")
    return total_pin / len(QUANTILES), pinned, pinned_nonempty


def rolling_origin(X, y, df, n_folds=4, fold_days=15):
    """Score both models on several successive forward windows.

    One forward split is one month of weather, and the band verdict turned on
    coverage, which is noisier than a mean. This refits on everything before
    each window and scores the window, so a verdict has to hold more than once.
    """
    days = np.array(sorted(set(pd.to_datetime(df["date"]).dt.date.values)))
    day_of = pd.to_datetime(df["date"]).dt.date.values
    rows = []
    for k in range(n_folds, 0, -1):
        end = len(days) - (k - 1) * fold_days
        test_days = set(days[end - fold_days:end])
        if end - fold_days <= 0:
            continue
        train_days = set(days[:end - fold_days])
        tr = np.array([d in train_days for d in day_of])
        te = np.array([d in test_days for d in day_of])
        yt = y[te]
        base = enforce_monotone(fit_baseline(X[tr], y[tr], X[te]))
        hurd = enforce_monotone(fit_hurdle(X[tr], y[tr], X[te])[0])
        row = {"window": f"{min(test_days)}..{max(test_days)}", "n": int(te.sum()),
               "empty": float(np.mean(yt == 0))}
        for name, pr in (("base", base), ("hurdle", hurd)):
            row[f"{name}_pin"] = float(np.mean(
                [pinball(yt, pr[q], q) for q in QUANTILES]))
            below = float(np.mean(yt < pr[fit.BAND_LOWER_Q]))
            above = float(np.mean(yt > pr[fit.BAND_UPPER_Q]))
            row[f"{name}_cov"] = 1.0 - below - above
            row[f"{name}_below"] = below
        rows.append(row)

    print(f"\n=== Rolling origin: {len(rows)} forward windows of ~{fold_days} days ===")
    print(f"  {'window':>24} {'n':>5} {'empty':>7} | {'pinball base/hurdle':>21} | "
          f"{fit.BAND_LABEL + ' coverage':>19} | {'misses low':>15}")
    for r in rows:
        print(f"  {r['window']:>24} {r['n']:>5} {r['empty']:>7.1%} | "
              f"{r['base_pin']:>9.4f} {r['hurdle_pin']:>11.4f} | "
              f"{r['base_cov']:>8.1%} {r['hurdle_cov']:>10.1%} | "
              f"{r['base_below']:>6.1%} {r['hurdle_below']:>8.1%}")
    print(f"  {'mean':>24} {'':>5} {'':>7} | "
          f"{np.mean([r['base_pin'] for r in rows]):>9.4f} "
          f"{np.mean([r['hurdle_pin'] for r in rows]):>11.4f} | "
          f"{np.mean([r['base_cov'] for r in rows]):>8.1%} "
          f"{np.mean([r['hurdle_cov'] for r in rows]):>10.1%} | "
          f"{np.mean([r['base_below'] for r in rows]):>6.1%} "
          f"{np.mean([r['hurdle_below'] for r in rows]):>8.1%}")
    wins = sum(1 for r in rows if r["hurdle_below"] < r["base_below"])
    print(f"  Hurdle has the smaller low-miss rate in {wins}/{len(rows)} windows; "
          f"lower pinball in "
          f"{sum(1 for r in rows if r['hurdle_pin'] < r['base_pin'])}/{len(rows)}")
    return rows


def main():
    X, y, df, _numeric = fit.load_and_prepare()
    train_idx, test_idx = fit.split_by_day(df, test_size=0.2, scheme="forward")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    dates = pd.to_datetime(df["date"]).dt.date.values
    print(f"\nTrain {len(y_train)} rows ({dates[train_idx].min()} to "
          f"{dates[train_idx].max()}), test {len(y_test)} rows "
          f"({dates[test_idx].min()} to {dates[test_idx].max()}); "
          f"empty hours: train {np.mean(y_train == 0):.1%}, "
          f"test {np.mean(y_test == 0):.1%}")

    base = fit_baseline(X_train, y_train, X_test)
    hurdle, p_zero = fit_hurdle(X_train, y_train, X_test)

    b_pin, b_pinned, b_pinned_ne = report("Baseline: one quantile model per level", base, y_test)
    h_pin, h_pinned, h_pinned_ne = report("Hurdle: P(any) x count | any", hurdle, y_test)

    print("\n=== Verdict ===")
    print(f"  Mean pinball loss: baseline {b_pin:.4f} vs hurdle {h_pin:.4f} "
          f"({(h_pin - b_pin) / b_pin:+.1%})")
    print(f"  Lower bound under 1: baseline {b_pinned:.1%} vs hurdle {h_pinned:.1%}")
    print(f"  ... on hours with surfers: baseline {b_pinned_ne:.1%} vs hurdle {h_pinned_ne:.1%}")

    # Bootstrap the pinball difference so a small gap is not over-read.
    rng = np.random.default_rng(0)
    bm = enforce_monotone(base)
    hm = enforce_monotone(hurdle)
    per_row = np.zeros(len(y_test))
    for q in QUANTILES:
        for preds, sign in ((hm, 1.0), (bm, -1.0)):
            d = y_test - preds[q]
            per_row += sign * np.maximum(q * d, (q - 1) * d) / len(QUANTILES)
    boots = [per_row[rng.integers(0, len(per_row), len(per_row))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  Hurdle minus baseline pinball: {per_row.mean():+.4f}, "
          f"95% bootstrap CI [{lo:+.4f}, {hi:+.4f}] (negative favours hurdle)")
    print(f"  P(empty) on test rows: mean {p_zero.mean():.1%}, "
          f"actual empty rate {np.mean(y_test == 0):.1%}")

    if "--no-rolling" not in sys.argv:
        rolling_origin(X, y, df)


if __name__ == "__main__":
    main()
