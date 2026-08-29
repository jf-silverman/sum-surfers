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
    negbin_model = NegativeBinomial(y_train, X_train, loglike_method="nb2").fit(
        start_params=start_params, disp=0, method="bfgs", maxiter=1000, gtol=1e-8
    )
    if not negbin_model.mle_retvals["converged"]:
        raise RuntimeError("Negative binomial MLE did not converge — results below would not be trustworthy")
    pred_negbin = negbin_model.predict(X_test)
    mae_nb = mean_absolute_error(y_test, pred_negbin)
    rmse_nb = mean_squared_error(y_test, pred_negbin) ** 0.5
    results["negbin"] = dict(model=negbin_model, mae=mae_nb, rmse=rmse_nb, aic=negbin_model.aic)

    print("\n=== Negative Binomial GLM (alpha estimated via MLE) ===")
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
    better_aic = "Negative Binomial" if negbin_model.aic < poisson_model.aic else "Poisson"
    print(f"Lower AIC (GLMs only, GBT has no AIC): {better_aic} "
          f"(Poisson={poisson_model.aic:.1f}, NegBin={negbin_model.aic:.1f})")
    mae_table = {"Poisson": mae_p, "Negative Binomial": mae_nb, "GBT": mae_gbt}
    best_mae_name = min(mae_table, key=mae_table.get)
    print(f"Held-out MAE — Poisson={mae_p:.2f}  NegBin={mae_nb:.2f}  GBT={mae_gbt:.2f}  "
          f"(best: {best_mae_name})")
    rmse_table = {"Poisson": rmse_p, "Negative Binomial": rmse_nb, "GBT": rmse_gbt}
    best_rmse_name = min(rmse_table, key=rmse_table.get)
    print(f"Held-out RMSE — Poisson={rmse_p:.2f}  NegBin={rmse_nb:.2f}  GBT={rmse_gbt:.2f}  "
          f"(best: {best_rmse_name})")

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


def fit_quantile_model_robust(X_train, y_train, quantile, base_kwargs=None,
                               min_leaf_candidates=QUANTILE_MIN_LEAF_CANDIDATES):
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


def main():
    X, y, df, numeric_cols = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = standardize(X_train, X_test, numeric_cols)
    print(f"\nTrain rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Target (surfer_count) — train mean: {y_train.mean():.2f}, var: {y_train.var():.2f} "
          f"(var >> mean is the classic sign of overdispersion favoring negative binomial)")
    print(f"Note: {numeric_cols} are z-score standardized (train-set mean/std) for optimizer "
          f"stability — their IRRs below are 'per 1 std-dev increase', not per raw unit.")

    results = fit_and_report(X_train, y_train, X_test, y_test)

    best_key = "negbin" if results["negbin"]["aic"] < results["poisson"]["aic"] else "poisson"
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
