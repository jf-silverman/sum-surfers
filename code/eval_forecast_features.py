"""
eval_forecast_features.py
-------------------------
Measures what each feature group in `forecast_features.py` is worth against
the live model, and whether an ensemble of a cross-sectional model and a
temporal-feature model beats either one alone.

Nothing here writes to the repo or touches the live pipeline. It reads
`data/training_features.csv`, reuses `fit_surfer_count_model.load_and_prepare`
so the baseline feature matrix is exactly the production one, and reuses
`fit_surfer_count_model.split_by_day` so every number is on the forward
day-grouped split that replaced the leaky random split (B21).

Three evaluations, because they answer different questions:

  headline   One forward split -- train on the earliest 80% of days, test on
             the most recent 20%. This is the number to quote, and it is the
             one the 7.60 baseline refers to.
  folds      Five rolling origins, each training on everything before a cut
             day and testing on the block of days after it. A single split of
             24 test days is a thin basis for a 0.5 MAE claim; the folds say
             whether an improvement is consistent or one lucky fortnight.
  stability  Ten origins over the lean feature set, with a paired t-test on
             per-origin MAE. This is the one that settled the question.

Adoption bar: >= 0.5 MAE better than baseline on the headline split AND
better in a majority of folds.

Result as of 2026-09-23: nothing clears the bar. The best combination gains
0.64 MAE on the headline split but wins only 2 of 5 folds, and over ten
origins its advantage is +0.011 MAE with paired t p=0.953 -- that is, no
effect at all. The headline gain is concentrated in two of twenty-four test
days. Read the verdict and stability blocks together; the headline number
alone is misleading here, which is the main thing this script exists to show.

Usage:
    python code/eval_forecast_features.py
    python code/eval_forecast_features.py --folds 5 --origins 10 --quiet
"""

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit_surfer_count_model as F  # noqa: E402
from forecast_features import LEAN_FEATURES, build_feature_groups  # noqa: E402

# Exactly the production point-estimate model (fit_surfer_count_model.py:188).
# Held fixed throughout: this round asks what the FEATURES are worth, so
# changing hyperparameters at the same time would confound the two.
GBT_KWARGS = dict(
    loss="poisson",
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    l2_regularization=1.0,
    random_state=42,
)

ADOPT_MAE_GAIN = 0.5


def make_model():
    return HistGradientBoostingRegressor(**GBT_KWARGS)


def forward_folds(df, n_folds=5, test_frac=0.12):
    """Rolling origins, always train-past / test-future, never splitting a day.

    Same principle as split_by_day(scheme="forward"), applied at several cut
    points instead of one. Yields (train_idx, test_idx) positional arrays.
    """
    days = np.array(sorted(pd.to_datetime(df["date"]).dt.date.unique()))
    n = len(days)
    block = max(int(round(n * test_frac)), 1)
    out = []
    for i in range(n_folds):
        start = int(round(n * (0.5 + 0.1 * i)))
        test_days = set(days[start:start + block])
        if not test_days:
            continue
        d = pd.to_datetime(df["date"]).dt.date.values
        mask = np.array([x in test_days for x in d])
        train = np.where(d < min(test_days))[0]
        if len(train) < 200 or mask.sum() < 20:
            continue
        out.append((train, np.where(mask)[0]))
    return out


def fit_predict(X, y, train_idx, test_idx):
    model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
    return model.predict(X.iloc[test_idx])


def score(y_true, pred):
    return dict(
        mae=mean_absolute_error(y_true, pred),
        bias=float(np.mean(pred - y_true)),
    )


def assemble(X_base, groups, names):
    """Baseline matrix plus the named groups' columns."""
    parts = [X_base] + [groups[n] for n in names]
    return pd.concat(parts, axis=1)


def evaluate(X, y, df, folds, head_split):
    tr, te = head_split
    head = score(y.iloc[te], fit_predict(X, y, tr, te))
    fold_maes = []
    for f_tr, f_te in folds:
        fold_maes.append(mean_absolute_error(y.iloc[f_te], fit_predict(X, y, f_tr, f_te)))
    return head, fold_maes


def fmt_folds(maes):
    return "[" + " ".join(f"{m:5.2f}" for m in maes) + f"] mean {np.mean(maes):5.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--origins", type=int, default=10,
                    help="rolling origins for the stability check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.quiet:
        warnings.filterwarnings("ignore")

    X_base, y, df, _ = F.load_and_prepare()
    X_base = X_base.drop(columns=["const"])  # tree model; the intercept is dead weight
    X_base = X_base.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)

    head_split = F.split_by_day(df, test_size=0.2, scheme="forward")
    folds = forward_folds(df, n_folds=args.folds)
    print(f"\nHeadline split: {len(head_split[0])} train rows / {len(head_split[1])} test rows "
          f"({df.iloc[head_split[1]].date.nunique()} test days, forward, whole days)")
    print(f"Rolling folds: {len(folds)} origins, test sizes "
          f"{[len(t) for _, t in folds]}\n")

    groups = build_feature_groups(df)
    for name, g in groups.items():
        miss = g.isna().mean().mean()
        print(f"  group {name:6s} {g.shape[1]:2d} cols, {100 * miss:4.1f}% NaN "
              f"(HistGBT handles NaN natively; lags are left unfilled across gaps)")

    # ---- baseline --------------------------------------------------------
    base_head, base_folds = evaluate(X_base, y, df, folds, head_split)
    print("\n" + "=" * 78)
    print("BASELINE (production features, production model)")
    print("=" * 78)
    print(f"  headline MAE {base_head['mae']:.3f}   bias {base_head['bias']:+.3f}")
    print(f"  folds        {fmt_folds(base_folds)}")

    # ---- each group alone, then every combination ------------------------
    print("\n" + "=" * 78)
    print("FEATURE GROUPS vs baseline (negative delta = better)")
    print("=" * 78)
    print(f"  {'features':28s} {'MAE':>7s} {'delta':>7s} {'bias':>7s}  "
          f"{'fold MAEs':>34s}  {'won':>5s}")
    results = {}
    names = list(groups)
    combos = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    for combo in combos:
        X = assemble(X_base, groups, list(combo))
        head, fmaes = evaluate(X, y, df, folds, head_split)
        won = sum(1 for a, b in zip(fmaes, base_folds) if a < b)
        results[combo] = dict(head=head, folds=fmaes, won=won, X=X)
        label = "+".join(combo)
        print(f"  {label:28s} {head['mae']:7.3f} {head['mae'] - base_head['mae']:+7.3f} "
              f"{head['bias']:+7.3f}  {fmt_folds(fmaes):>34s}  {won}/{len(fmaes)}")

    # ---- ensemble --------------------------------------------------------
    # Joel's framing: combine the cross-sectional model with a temporal one
    # rather than replacing either. Component A is the production model on
    # production features. Component B is the same learner on production
    # features PLUS every candidate group -- so the ensemble is averaging two
    # views of the same data, not two different learners.
    print("\n" + "=" * 78)
    print("ENSEMBLE: production model blended with an all-features model")
    print("=" * 78)
    all_combo = tuple(names)
    X_all = results[all_combo]["X"]

    tr, te = head_split
    pa = fit_predict(X_base, y, tr, te)
    pb = fit_predict(X_all, y, tr, te)
    yt = y.iloc[te]
    print(f"  {'blend w (weight on temporal model)':38s} {'MAE':>7s} {'bias':>7s}")
    best = None
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        s = score(yt, (1 - w) * pa + w * pb)
        tag = {0.0: "  (= baseline)", 1.0: "  (= all-features)"}.get(w, "")
        print(f"  w={w:<36.2f} {s['mae']:7.3f} {s['bias']:+7.3f}{tag}")
        if best is None or s["mae"] < best[1]:
            best = (w, s["mae"])
    print(f"\n  best blend on the headline split: w={best[0]:.2f}, MAE {best[1]:.3f}")
    print("  NOTE: that w was chosen by looking at the test set, so it is an")
    print("  upper bound, not a deployable result. The fold table below picks")
    print("  w=0.5 blind instead.")

    ens_folds = []
    for f_tr, f_te in folds:
        qa = fit_predict(X_base, y, f_tr, f_te)
        qb = fit_predict(X_all, y, f_tr, f_te)
        ens_folds.append(mean_absolute_error(y.iloc[f_te], 0.5 * qa + 0.5 * qb))
    ens_head = score(yt, 0.5 * pa + 0.5 * pb)
    won = sum(1 for a, b in zip(ens_folds, base_folds) if a < b)
    print(f"\n  fixed 50/50 blend: headline MAE {ens_head['mae']:.3f} "
          f"({ens_head['mae'] - base_head['mae']:+.3f}), bias {ens_head['bias']:+.3f}")
    print(f"                     folds {fmt_folds(ens_folds)}  won {won}/{len(ens_folds)}")
    comp_a, comp_b = base_head["mae"], results[all_combo]["head"]["mae"]
    print(f"  components: production {comp_a:.3f}, all-features {comp_b:.3f}")
    print(f"  ensemble beats both components: {ens_head['mae'] < min(comp_a, comp_b)}")

    # ---- stability check -------------------------------------------------
    # The decisive test. The headline split is 24 days; a 0.5 MAE difference
    # over 24 days is well inside what one unusual fortnight can produce, so
    # the headline alone cannot settle anything. This sweeps many more origins
    # and asks whether the improvement is there at all.
    print("\n" + "=" * 78)
    print("STABILITY: many rolling origins, lean feature set vs baseline")
    print("=" * 78)
    X_lean = pd.concat([X_base, pd.concat(list(groups.values()), axis=1)[LEAN_FEATURES]], axis=1)
    days = np.array(sorted(pd.to_datetime(df["date"]).dt.date.unique()))
    dv = pd.to_datetime(df["date"]).dt.date.values
    n = len(days)
    base_o, lean_o = [], []
    print(f"  {'cut day':12s} {'ntest':>5s} {'base':>7s} {'lean':>7s} {'delta':>7s}")
    for i in range(args.origins):
        start = int(n * 0.45) + i * 6
        test_days = set(days[start:start + 8])
        if len(test_days) < 8:
            break
        f_tr = np.where(dv < min(test_days))[0]
        f_te = np.where(np.isin(dv, list(test_days)))[0]
        if len(f_tr) < 200 or len(f_te) < 20:
            continue
        a = mean_absolute_error(y.iloc[f_te], fit_predict(X_base, y, f_tr, f_te))
        b = mean_absolute_error(y.iloc[f_te], fit_predict(X_lean, y, f_tr, f_te))
        base_o.append(a)
        lean_o.append(b)
        print(f"  {str(min(test_days)):12s} {len(f_te):5d} {a:7.3f} {b:7.3f} {b - a:+7.3f}")
    base_o, lean_o = np.array(base_o), np.array(lean_o)
    if len(base_o) >= 3:
        from scipy import stats  # noqa: PLC0415
        pval = stats.ttest_rel(base_o, lean_o).pvalue
        print(f"\n  over {len(base_o)} origins: base {base_o.mean():.3f}  "
              f"lean {lean_o.mean():.3f}  delta {lean_o.mean() - base_o.mean():+.3f}")
        print(f"  lean better in {int((lean_o < base_o).sum())}/{len(base_o)} origins, "
              f"paired t p={pval:.3f}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"VERDICT (bar: >= {ADOPT_MAE_GAIN} MAE better AND a majority of folds)")
    print("=" * 78)
    any_pass = False
    for combo, r in sorted(results.items(), key=lambda kv: kv[1]["head"]["mae"]):
        gain = base_head["mae"] - r["head"]["mae"]
        majority = r["won"] > len(r["folds"]) / 2
        ok = gain >= ADOPT_MAE_GAIN and majority
        any_pass |= ok
        print(f"  {'PASS' if ok else 'fail'}  {'+'.join(combo):28s} "
              f"gain {gain:+.3f}  folds won {r['won']}/{len(r['folds'])}")
    if not any_pass:
        print("\n  Nothing clears the bar. On this data the temporal and lunar")
        print("  features do not add enough to justify shipping them.")


if __name__ == "__main__":
    main()
