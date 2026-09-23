"""
fit_surfer_count_model.py
--------------------------
Phase 2 of the surfer-count modeling plan (see docs/PROJECT_HISTORY.md): fits
Poisson and negative-binomial GLMs on data/training_features.csv (built by
build_training_features.py) and reports which one fits better, plus honest
out-of-sample accuracy via a held-out test split.

Why Poisson/NegBin instead of linear regression: surfer_count is a
non-negative count with a right-skewed distribution (lots of low counts,
occasional 40-70+ crowds) — a count-regression GLM respects that shape
where OLS would not (it could predict negative counts, and assumes
constant-variance Gaussian errors that don't match count data).

Cyclical features (hour_local, month, wind_direction_deg,
primary_swell_direction_deg) are sin/cos-encoded rather than used as raw
linear values, since e.g. 359 degrees and 0 degrees are adjacent in
reality but maximally far apart as raw numbers — standard practice for
circular predictors in regression.

weather_condition (19 raw categories) is collapsed to categories with
>=20 occurrences; rarer ones are bucketed into OTHER to avoid fitting
near-singleton dummy variables on a ~1000-row dataset.

Usage:
    python code/fit_surfer_count_model.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = _PROJECT_ROOT / "data" / "training_features.csv"


def load_and_prepare():
    df = pd.read_csv(FEATURES_CSV)

    numeric_cols = [
        "temperature_f", "pressure_mb", "rating_value", "tide_ft",
        "surf_min_ft", "surf_max_ft", "primary_swell_height_ft", "primary_swell_period_s",
        "wind_speed_mph", "wind_gust_mph", "energy_offshore_kj", "energy_nearshore_kj",
        "consistency_wave_count",
        # Real observed historical weather (Open-Meteo archive, not a forecast — see
        # backfill_openmeteo_weather.py). real_humidity_pct is a validated (if
        # imperfect) proxy for the fog/blur conditions the quality gate already flags.
        "real_temperature_f", "real_humidity_pct", "real_cloud_cover_pct", "real_pressure_mb",
        # Derived in build_training_features.py (2026-09-16): how much of the
        # day's daylight sits under the tide this spot surfs best on. Day-level
        # information that no per-hour column carries — tide_ft says what the
        # tide is now, not how long the good window lasts.
        "good_tide_hours", "good_tide_frac", "good_tide_hours_left",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric_cols + ["surfer_count"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} row(s) with missing numeric features ({len(df)} remain)")

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction_deg"] / 360)
    df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction_deg"] / 360)
    df["swell_dir_sin"] = np.sin(2 * np.pi * df["primary_swell_direction_deg"] / 360)
    df["swell_dir_cos"] = np.cos(2 * np.pi * df["primary_swell_direction_deg"] / 360)

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_night"] = df["is_night"].astype(int)

    # weather_simple (2026-08-28): the curated 4-category scheme from
    # build_training_features.py's simplify_weather_condition() — CLEAR/
    # CLOUDY_OVERCAST/RAIN/FOG, with day/night pulled out separately as is_night —
    # replaces the old approach of collapsing whatever raw categories had <20
    # occurrences into a generic OTHER, which was silently discarding all rain
    # signal (every individual rain-family category had well under 20 rows).
    print(f"weather_simple categories: {sorted(df['weather_simple'].unique())} "
          f"(counts: {df['weather_simple'].value_counts().to_dict()})")
    wx_dummies = pd.get_dummies(df["weather_simple"], prefix="wx", drop_first=True, dtype=float)

    feature_cols = numeric_cols + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "wind_dir_sin", "wind_dir_cos", "swell_dir_sin", "swell_dir_cos",
        "is_weekend", "is_night",
    ]
    X = pd.concat([df[feature_cols], wx_dummies], axis=1)
    X = sm.add_constant(X)
    y = df["surfer_count"].astype(float)

    return X, y, df, numeric_cols


def standardize(X_train, X_test, numeric_cols):
    """Z-score the raw-scale numeric columns (e.g. energy_offshore_kj in the
    thousands vs. sin/cos terms in [-1,1]) using train-set statistics only, to
    avoid leaking test-set information and to stabilize the NB MLE optimizer,
    which (unlike GLM's IRLS solver) is much more sensitive to feature scale."""
    mean = X_train[numeric_cols].mean()
    std = X_train[numeric_cols].std().replace(0, 1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = (X_train[numeric_cols] - mean) / std
    X_test[numeric_cols] = (X_test[numeric_cols] - mean) / std
    return X_train, X_test


def fit_and_report(X_train, y_train, X_test, y_test):
    results = {}

    poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    pred_poisson = poisson_model.predict(X_test)
    mae_p = mean_absolute_error(y_test, pred_poisson)
    rmse_p = mean_squared_error(y_test, pred_poisson) ** 0.5
    resid_pearson = poisson_model.resid_pearson
    dispersion = (resid_pearson ** 2).sum() / poisson_model.df_resid
    results["poisson"] = dict(model=poisson_model, mae=mae_p, rmse=rmse_p, aic=poisson_model.aic, dispersion=dispersion)

    print("\n=== Poisson GLM ===")
    print(f"AIC: {poisson_model.aic:.1f}")
    print(f"Pearson chi2 / df_resid (dispersion ratio): {dispersion:.2f}  "
          f"(should be ~1.0 for Poisson to be appropriate; >>1 means overdispersed)")
    print(f"Held-out test MAE: {mae_p:.2f}  RMSE: {rmse_p:.2f}")

    # MLE-based NB2, jointly estimates the dispersion parameter alpha rather than
    # fixing it (sm.GLM's NegativeBinomial family defaults to alpha=1.0 unless told
    # otherwise, which understates/overstates dispersion depending on the true value —
    # standard practice (Cameron & Trivedi) is to estimate alpha, not assume it).
    # Warm-start from the (already-converged) Poisson coefficients, plus an alpha
    # guess from the Poisson dispersion ratio — the NB log-likelihood surface is
    # much better-behaved near the right answer than from statsmodels' default
    # all-zeros start, given 30 features.
    start_params = np.append(poisson_model.params.values, max(dispersion - 1, 0.1))
    # Try successively more robust optimizers rather than giving up on the first
    # failure. bfgs from the Poisson warm start is fastest and usually enough;
    # nm (Nelder-Mead, derivative-free) and powell are slower but cope better
    # with an awkward likelihood surface. Previously a single bfgs attempt that
    # failed raised outright, which killed the entire script — including the GBT
    # results and permutation importance below, the parts actually used in
    # production — over one of three comparison models not converging.
    negbin_model = None
    for method, kwargs in (("bfgs", dict(maxiter=1000, gtol=1e-8)),
                           ("nm", dict(maxiter=5000)),
                           ("powell", dict(maxiter=5000))):
        try:
            candidate = NegativeBinomial(y_train, X_train, loglike_method="nb2").fit(
                start_params=start_params, disp=0, method=method, **kwargs
            )
        except Exception as e:  # noqa: BLE001 — any optimizer blow-up is just "this method failed"
            print(f"  NegBin MLE via {method} raised {type(e).__name__}, trying next method...")
            continue
        if candidate.mle_retvals.get("converged"):
            negbin_model = candidate
            if method != "bfgs":
                print(f"  NegBin MLE converged via fallback optimizer '{method}'.")
            break
        print(f"  NegBin MLE via {method} did not converge, trying next method...")

    print("\n=== Negative Binomial GLM (alpha estimated via MLE) ===")
    if negbin_model is None:
        # Report it as unavailable and keep going. The NB numbers genuinely
        # wouldn't be trustworthy, but that's a reason to omit them from the
        # comparison, not to discard the GBT work in the rest of the script.
        print("  DID NOT CONVERGE with any optimizer (bfgs/nm/powell) — negative-binomial")
        print("  results are omitted from the comparison below rather than reported untrustworthy.")
        results["negbin"] = None
    else:
        pred_negbin = negbin_model.predict(X_test)
        mae_nb = mean_absolute_error(y_test, pred_negbin)
        rmse_nb = mean_squared_error(y_test, pred_negbin) ** 0.5
        results["negbin"] = dict(model=negbin_model, mae=mae_nb, rmse=rmse_nb, aic=negbin_model.aic)
        print(f"Estimated alpha (dispersion): {negbin_model.params['alpha']:.3f}")
        print(f"AIC: {negbin_model.aic:.1f}")
        print(f"Held-out test MAE: {mae_nb:.2f}  RMSE: {rmse_nb:.2f}")

    # GBT: no distributional assumption, captures nonlinearities/interactions the
    # GLMs can't (e.g. tide effect that differs by time of day) — same train/test
    # split and feature set for a fair comparison. Poisson loss since the target
    # is a non-negative count, same reasoning as the GLMs above.
    gbt_model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.05,
        max_depth=4, l2_regularization=1.0, random_state=42,
    ).fit(X_train, y_train)
    pred_gbt = gbt_model.predict(X_test)
    mae_gbt = mean_absolute_error(y_test, pred_gbt)
    rmse_gbt = mean_squared_error(y_test, pred_gbt) ** 0.5
    results["gbt"] = dict(model=gbt_model, mae=mae_gbt, rmse=rmse_gbt)

    print("\n=== Gradient-Boosted Trees (Poisson loss) ===")
    print(f"Held-out test MAE: {mae_gbt:.2f}  RMSE: {rmse_gbt:.2f}")

    print("\n=== Comparison ===")
    # NegBin is omitted throughout if it failed to converge (see above) rather
    # than reported with untrustworthy numbers.
    has_nb = results.get("negbin") is not None
    if has_nb:
        better_aic = "Negative Binomial" if negbin_model.aic < poisson_model.aic else "Poisson"
        print(f"Lower AIC (GLMs only, GBT has no AIC): {better_aic} "
              f"(Poisson={poisson_model.aic:.1f}, NegBin={negbin_model.aic:.1f})")
    else:
        print(f"Lower AIC: Poisson={poisson_model.aic:.1f} (NegBin unavailable — did not converge)")

    mae_table = {"Poisson": mae_p, "GBT": mae_gbt}
    rmse_table = {"Poisson": rmse_p, "GBT": rmse_gbt}
    if has_nb:
        mae_table["Negative Binomial"] = mae_nb
        rmse_table["Negative Binomial"] = rmse_nb
    nb_mae = f"  NegBin={mae_nb:.2f}" if has_nb else "  NegBin=n/a"
    nb_rmse = f"  NegBin={rmse_nb:.2f}" if has_nb else "  NegBin=n/a"
    print(f"Held-out MAE — Poisson={mae_p:.2f}{nb_mae}  GBT={mae_gbt:.2f}  "
          f"(best: {min(mae_table, key=mae_table.get)})")
    print(f"Held-out RMSE — Poisson={rmse_p:.2f}{nb_rmse}  GBT={rmse_gbt:.2f}  "
          f"(best: {min(rmse_table, key=rmse_table.get)})")

    # Feature importance via permutation (model-agnostic, comparable across all three)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(gbt_model, X_test, y_test, n_repeats=15, random_state=42, scoring="neg_mean_absolute_error")
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mae_increase": perm.importances_mean,
    }).sort_values("importance_mae_increase", ascending=False)
    print("\n=== GBT permutation feature importance (MAE increase when shuffled) ===")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(importance_df.head(15).to_string(index=False))

    fit_quantile_intervals(X_train, y_train, X_test, y_test)

    return results


QUANTILE_BASE_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=0.0, random_state=42)
# Escalating min_samples_leaf candidates for fit_quantile_model_robust(). Empirically,
# no single fixed hyperparameter combo is safe here — see that function's docstring.
QUANTILE_MIN_LEAF_CANDIDATES = (20, 30, 40, 50, 75, 100, 150, 200)
QUANTILE_DEGENERATE_STD_THRESHOLD = 0.5


class ConstantQuantileModel:
    """Stand-in for a quantile level that is genuinely degenerate in this data.

    Not a workaround for a bad fit. With ~12% of hours at exactly zero surfers,
    the true 10th percentile of surfer_count IS 0.00, so "predict 0 everywhere"
    is the correct answer at that level rather than a failed one — verified
    2026-09-16 by sweeping min_samples_leaf from 20 to 600, where every value
    returns a flat zero except one fluke at 400 that tops out at 2 surfers.
    Callers that need a real, varying model (calibration, coverage reporting)
    must keep the default and let the error raise; only display code that can
    honestly draw a flat band should opt in.
    """

    def __init__(self, value, quantile):
        self.value = float(value)
        self.quantile = quantile
        self.degenerate = True

    def predict(self, X):
        return np.full(len(X), self.value)


def fit_quantile_model_robust(X_train, y_train, quantile, base_kwargs=None,
                               min_leaf_candidates=QUANTILE_MIN_LEAF_CANDIDATES,
                               allow_degenerate=False):
    """Fits a quantile-loss GBT with a self-check against silent collapse to a
    near-constant prediction — a real, repeatedly-observed failure mode for this
    dataset's extreme (zero-inflated) quantiles, NOT something one fixed
    hyperparameter combo can be trusted to avoid permanently.

    Discovered twice now: first with l2_regularization=1.0 (fixed by l2=0.0 +
    max_depth=3), then again after adding features (weather_simple/is_night) —
    same l2=0.0/depth=3 combo re-collapsed on the updated dataset. Root cause:
    with ~11% of rows at surfer_count==0, "always predict 0" already nearly
    minimizes the pinball loss at low quantiles, so any config that doesn't push
    hard enough on individual splits settles into that trivial optimum instead.
    A follow-up sweep showed min_samples_leaf is the more reliable lever than
    l2/depth, but the collapse boundary is NOT monotonic in it either (leaf=50
    can work while leaf=75 collapses again) — so instead of hardcoding a value
    that's only verified to work today, this fits with escalating min_samples_leaf
    values and keeps the first one whose TRAINING predictions actually vary
    (std > QUANTILE_DEGENERATE_STD_THRESHOLD), raising a clear error if every
    candidate collapses rather than silently returning a trivial model that would
    look deceptively fine in an aggregate coverage number (see below)."""
    kwargs = dict(base_kwargs or QUANTILE_BASE_KWARGS)
    for min_leaf in min_leaf_candidates:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                                               min_samples_leaf=min_leaf, **kwargs).fit(X_train, y_train)
        train_std = model.predict(X_train).std()
        if train_std > QUANTILE_DEGENERATE_STD_THRESHOLD:
            return model, min_leaf
    if allow_degenerate:
        # Report the constant the collapsed fit actually settled on rather than
        # assuming zero, so a future collapse at some other value is visible.
        constant = float(model.predict(X_train).mean())
        print(f"  NOTE: quantile={quantile} is degenerate in this data — every min_samples_leaf "
              f"candidate collapsed to ~{constant:.2f}. Using a constant {constant:.2f} for this level.")
        return ConstantQuantileModel(constant, quantile), None
    raise RuntimeError(
        f"Quantile GBT (quantile={quantile}) collapsed to a near-constant prediction "
        f"for every min_samples_leaf candidate tried {min_leaf_candidates} — needs "
        f"manual investigation (e.g. widen min_leaf_candidates, or the quantile may "
        f"genuinely be unlearnable given how zero-inflated this dataset is at that "
        f"tail — see this function's docstring). Refusing to silently return a "
        f"trivial model."
    )


def fit_quantile_intervals(X_train, y_train, X_test, y_test, lower_q=0.1, upper_q=0.9):
    """80% prediction intervals via GBT quantile regression (separate models for the
    10th/50th/90th percentiles) — much lighter-weight than a full Bayesian refit, and
    gives real per-prediction uncertainty bands instead of a single point estimate.
    Reports empirical coverage AND each tail's individual miss rate (should land near
    lower_q / 1-upper_q respectively) — an aggregate coverage number alone can hide a
    degenerate model: a lower bound that's always 0 trivially never excludes anything
    from below (counts are never negative), so all "coverage" credit can come from the
    upper bound alone while looking deceptively fine in aggregate. See
    fit_quantile_model_robust()'s docstring for the full history of this failure mode
    (found and re-found twice) and why a self-checking fit is used instead of a fixed
    hyperparameter combo. Note extreme quantiles (tried q=0.05) aren't learnable at all
    given how zero-inflated this dataset is, so lower_q/upper_q defaults stay at
    0.1/0.9 rather than pushed wider."""
    lower_model, lower_leaf = fit_quantile_model_robust(X_train, y_train, lower_q)
    median_model, median_leaf = fit_quantile_model_robust(X_train, y_train, 0.5)
    upper_model, upper_leaf = fit_quantile_model_robust(X_train, y_train, upper_q)
    print(f"Quantile models converged with min_samples_leaf: lower={lower_leaf} median={median_leaf} upper={upper_leaf}")

    pred_lower = np.clip(lower_model.predict(X_test), 0, None)
    pred_median = np.clip(median_model.predict(X_test), 0, None)
    pred_upper = np.clip(upper_model.predict(X_test), 0, None)
    # Enforce monotonicity (lower <= median <= upper) — quantile models are fit
    # independently, so crossing is a known possible artifact.
    pred_upper = np.maximum(pred_upper, pred_lower)
    pred_median = np.clip(pred_median, pred_lower, pred_upper)

    covered = (y_test.values >= pred_lower) & (y_test.values <= pred_upper)
    coverage = covered.mean()
    below_frac = (y_test.values < pred_lower).mean()
    above_frac = (y_test.values > pred_upper).mean()
    mean_width = (pred_upper - pred_lower).mean()
    median_mae = mean_absolute_error(y_test, pred_median)

    print(f"\n=== GBT quantile prediction intervals ({int((upper_q - lower_q) * 100)}%, "
          f"{lower_q:.0%}-{upper_q:.0%}) ===")
    print(f"Empirical coverage on held-out test set: {coverage:.1%} (target: {upper_q - lower_q:.0%})")
    print(f"  below-lower miss rate: {below_frac:.1%} (target {lower_q:.0%}) — "
          f"reported separately from the upper tail so a degenerate/trivial bound "
          f"can't hide behind a deceptively OK-looking aggregate number")
    print(f"  above-upper miss rate: {above_frac:.1%} (target {1 - upper_q:.0%})")
    print(f"Mean interval width: {mean_width:.1f} surfers")
    print(f"Median-quantile model MAE: {median_mae:.2f} (cross-check vs. main GBT point model)")

    print("\nSample predictions (actual vs. [lower, median, upper]):")
    sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(10, len(y_test)), replace=False)
    for i in sample_idx:
        print(f"  actual={y_test.values[i]:.0f}  ->  [{pred_lower[i]:.1f}, {pred_median[i]:.1f}, {pred_upper[i]:.1f}]")


def split_by_day(df, test_size=0.2, scheme="forward", random_state=42):
    """Train/test indices that never split a day across both sides.

    Replaces `train_test_split(..., random_state=42)`, which leaked badly here
    (B21). At roughly 13 rows per day, a random 20% split puts hours from the
    same day on BOTH sides: the model learns that specific day's crowd level
    from its own hours and is then scored on the rest of that same day.
    Measured cost of that leak on this data, same model and features:

        random 80/20            MAE 5.53   <- what was reported
        whole days held out     MAE 6.75
        train past / test future MAE 7.57  <- what a day-ahead forecast faces

    `scheme`:
      "forward" — train on the earliest days, test on the most recent. This is
        the number to quote, because it is the only one that matches how the
        model is actually used: fit on the past, predict a day that has not
        happened. It also has to survive real level shifts (Oct-Nov day-means
        of 12.1 and 9.4 against Jul-Aug at 18.9 and 18.1).
      "grouped" — random days held out. Removes the same-day leak but still
        lets the model see the future. Useful to separate the two effects.

    Returns (train_idx, test_idx) as positional integer arrays.
    """
    import numpy as np  # noqa: PLC0415

    days = pd.to_datetime(df["date"]).dt.date.values
    unique_days = np.array(sorted(set(days)))
    n_test = max(int(round(len(unique_days) * test_size)), 1)

    if scheme == "forward":
        test_days = set(unique_days[-n_test:])
    elif scheme == "grouped":
        rng = np.random.default_rng(random_state)
        test_days = set(rng.choice(unique_days, size=n_test, replace=False))
    else:
        raise ValueError(f"unknown scheme {scheme!r} (use 'forward' or 'grouped')")

    mask = np.array([d in test_days for d in days])
    return np.where(~mask)[0], np.where(mask)[0]


def main():
    X, y, df, numeric_cols = load_and_prepare()

    train_idx, test_idx = split_by_day(df, test_size=0.2, scheme="forward")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train, X_test = standardize(X_train, X_test, numeric_cols)
    train_days = pd.to_datetime(df["date"]).dt.date.iloc[train_idx]
    test_days = pd.to_datetime(df["date"]).dt.date.iloc[test_idx]
    print(f"\nSplit: forward in time, whole days (see split_by_day / B21) — "
          f"train through {train_days.max()}, test from {test_days.min()}")
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Target (surfer_count) — train mean: {y_train.mean():.2f}, var: {y_train.var():.2f} "
          f"(var >> mean is the classic sign of overdispersion favoring negative binomial)")
    print(f"Note: {numeric_cols} are z-score standardized (train-set mean/std) for optimizer "
          f"stability — their IRRs below are 'per 1 std-dev increase', not per raw unit.")

    results = fit_and_report(X_train, y_train, X_test, y_test)

    # Falls back to Poisson if NegBin didn't converge (results["negbin"] is None).
    nb = results.get("negbin")
    best_key = "negbin" if nb is not None and nb["aic"] < results["poisson"]["aic"] else "poisson"
    best_model = results[best_key]["model"]
    print(f"\n=== {best_key.upper()} coefficients (as incidence rate ratios, exp(coef)) ===")
    coef_params = best_model.params.drop("alpha", errors="ignore")  # alpha is a dispersion param, not a rate-ratio coefficient
    irr = np.exp(coef_params)
    pvals = best_model.pvalues.drop("alpha", errors="ignore")
    summary_df = pd.DataFrame({"IRR": irr, "p_value": pvals}).sort_values("p_value")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(summary_df)


if __name__ == "__main__":
    main()
