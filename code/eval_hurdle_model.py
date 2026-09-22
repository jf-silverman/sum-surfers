"""
Does a hurdle model fix the forecast's pinned lower bound? (B19)

After the September 2026 recount, the 10th-percentile model predicts under one
surfer for 72% of held-out hours, including 69% of hours that do have surfers.
The 80% band is almost never undershot (4.6% against a 10% target), so part of
its coverage comes from a floor at zero rather than from the model knowing
anything. Only the upper edge carries information.

The suspected cause is that one quantile model is being asked to describe two
different things at once: whether anyone is out at all (15% of hours are empty),
and how many are out when someone is. This script fits the alternative — a
hurdle model — and scores both on the same held-out split:

    P(count > 0)                     classifier
    quantiles of count | count > 0   quantile GBTs on the positive rows only

and composes them back into predictive quantiles. For a target quantile q, if
the probability of an empty hour already exceeds q the prediction is 0;
otherwise it is the conditional quantile at (q - p_zero) / (1 - p_zero).

Judged on pinball loss (the proper scoring rule for quantiles — coverage alone
can be bought with a floor), plus coverage, one-sided miss rates, and how often
the lower bound is informative.

Usage:
    python code/eval_hurdle_model.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_surfer_count_model as fit  # noqa: E402

QUANTILES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
INTERVALS = ((0.40, 0.60, "20%"), (0.30, 0.70, "40%"), (0.20, 0.80, "60%"), (0.10, 0.90, "80%"))


def pinball(y_true, y_pred, q):
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_baseline(X_train, y_train, X_test):
    """Today's approach: one quantile GBT per level, fit on all rows."""
    preds = {}
    for q in QUANTILES:
        model, _leaf = fit.fit_quantile_model_robust(X_train, y_train, q)
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

    preds = {}
    for q in QUANTILES:
        # Conditional level that composes back to the unconditional quantile q.
        # Vectorised per row, so one model cannot serve every row; fit the
        # conditional quantile on a grid and interpolate per row instead.
        preds[q] = None
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    cond = {}
    for g in grid:
        model, _leaf = fit.fit_quantile_model_robust(X_pos, y_pos, float(g))
        cond[g] = np.asarray(model.predict(X_test), dtype=float)
    cond_stack = np.vstack([cond[g] for g in grid])          # (n_grid, n_test)

    for q in QUANTILES:
        adj = (q - p_zero) / np.maximum(1.0 - p_zero, 1e-9)   # conditional level per row
        out = np.zeros(len(p_zero))
        inside = adj > 0
        idx = np.clip(np.searchsorted(grid, adj[inside]) - 0, 0, len(grid) - 1)
        out[inside] = cond_stack[idx, np.where(inside)[0]]
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

    lo10 = preds[0.10]
    pinned = float(np.mean(lo10 < 1))
    nonempty = y > 0
    pinned_nonempty = float(np.mean(lo10[nonempty] < 1))
    print(f"\n  Lower bound (q=0.10) under 1 surfer: {pinned:.1%} of hours, "
          f"{pinned_nonempty:.1%} of hours that had surfers")
    print(f"  Median-quantile MAE: {np.mean(np.abs(y - preds[0.50])):.2f}")
    return total_pin / len(QUANTILES), pinned, pinned_nonempty


def main():
    X, y, df, _numeric = fit.load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    print(f"\nTrain {len(y_train)} rows, test {len(y_test)} rows; "
          f"empty hours: train {np.mean(y_train == 0):.1%}, test {np.mean(y_test == 0):.1%}")

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


if __name__ == "__main__":
    main()
