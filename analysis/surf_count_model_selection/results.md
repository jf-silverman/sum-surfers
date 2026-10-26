# Model-family comparison — results (2026-09-11)

Run: `python analysis/surf_count_model_selection/compare_model_families.py`
(~5 min). Selection by 5-fold CV on the train split only; held-out rows
scored once at the end.

## Cross-validated MAE (train, 5-fold)

| model | CV MAE |
|---|---|
| GBT poisson, best of 72-combination grid | **6.201** |
| GBT on sqrt(y) | 6.213 |
| **GBT poisson, current production params** | **6.238** |
| GBT on log1p(y) | 6.339 |
| GBT squared_error | 6.461 |
| GBT absolute_error | 6.585 |
| ExtraTrees (400) | 6.587 |
| RandomForest (400) | 6.798 |
| Ridge (linear) | 8.802 |
| PoissonRegressor (linear) | 9.078 |
| Predict the train mean | 11.559 |

## Held-out test, never used for selection

| model | MAE | bias | 0-4 bias | 30+ bias |
|---|---|---|---|---|
| GBT poisson (current) | **6.326** | +0.010 | +3.85 | **-10.31** |
| GBT poisson (best grid) | 6.385 | -0.113 | +3.94 | -10.87 |
| GBT on log1p(y) | 6.394 | -1.662 | +2.71 | -12.63 |
| GBT absolute_error | 6.741 | -0.431 | +4.29 | -13.48 |
| RandomForest (400) | 6.778 | +0.944 | +5.78 | -10.58 |

Best grid parameters: `max_iter=300, learning_rate=0.03, max_depth=6,
l2_regularization=1.0`.

## What this says

1. **Hyperparameter tuning is exhausted.** 72 combinations improved CV MAE
   by 0.037 (0.6%), and that winner is *worse* on held-out data (6.385 vs
   6.326) — inside the noise. The hand-picked production parameters were
   already at the plateau. There is nothing left to win here.
2. **No model family helps.** sqrt(y) ties, every other loss and transform
   is worse, both forest ensembles are worse, and the linear models are far
   worse (8.8-9.1 against a 11.6 mean-baseline — they recover less than
   half the signal the GBT does). Poisson GBT is the right family and is
   already being used.
3. **Nothing fixes the crowd bias.** Every variant tested lands between
   **-10.31 and -13.48** on 30+ hours. The under-prediction of crowded hours
   is not a hyperparameter artifact, not a loss-function artifact, and not a
   target-scale artifact. It survives every algorithm change.
4. **It is not target noise either.** Within-clip measurement noise
   (`frame_count_stdev` across the three frames of the same clip, n=1,144)
   averages **0.97 surfers** — 0.37 on quiet hours, 1.66 on 30+. The target
   is measured far more precisely than the model predicts it, so the ~6.3
   gap is genuine unexplained variance, not a noisy label.

**Conclusion: the limit is information, not algorithm.** Every model
converges on MAE ~6.2-6.4 against a mean-baseline of 11.6, which is what it
looks like when the available predictors have been fully exploited. Effort
spent on model class, tuning, or target transforms will not move this.
Effort spent on new predictors might — the swell re-backfill is the
precedent: correcting one feature moved period's correlation with count from
r=+0.015 to +0.158 and lifted nearshore energy to the #2 feature.

## Regularized linear models and a blend, week-held-out (2026-10-26)

`test_linear_models.py` (kept local; `linear_models_results.csv`). Same 10
week-grouped folds as the packed-hours tests, daylight rows, production GBT
6.401. Each linear model is fitted on the raw features and on "shaped" ones:
raw plus a tide spline and one-hot hour, hour × weekend, month and rating,
which a linear model needs to bend where the crowd does.

| Model | MAE | Bias at actual 30+ | Weeks better than GBT |
|---|---|---|---|
| Ridge / Lasso / ElasticNet, raw | 8.18 / 8.34 / 8.34 | −16 to −17 | 2-4 of 29 |
| Poisson GLM, raw | 8.78 | −16.7 | 3/29 |
| Ridge on log count, raw | 9.35 | −17.8 | 3/29 |
| Ridge / Lasso / ElasticNet, shaped | 7.03 / 7.32 / 7.25 | −13 to −14.5 | 8-9/29 |
| Ridge / Lasso on log count, shaped | 6.78 / 7.27 | −13.9 / −17.9 | 9 and 7/29 |
| **Poisson GLM, shaped** | **6.54** | **−9.65** | 12/29 (p = 0.52) |
| GBT given the shaped inputs | 6.30 | −11.8 | 18/29 (p = 0.051) |
| **0.5 GBT + 0.5 Poisson GLM, shaped** | **6.18** | −10.8 | **20/29 (p = 0.012)** |
| 0.7 GBT + 0.3 Poisson GLM, shaped | 6.20 | −11.3 | **23/29 (p < 0.001)** |
| 0.3 GBT + 0.7 Poisson GLM, shaped | 6.26 | −10.4 | 19/29 (p = 0.27) |
| 0.5 GBT + 0.5 Ridge, shaped | 6.39 | −12.5 | 16/29 (p = 0.51) |

Lasso does no better than Ridge, and fitting the log count is worse than
fitting the count. The Poisson GLM on shaped inputs is the best linear model
and has the smallest packed-hour shortfall of any single model tested so far.
On its own it trails the GBT, but averaged with it, it beats the GBT
week by week, at every weight from 0.5 to 0.7 GBT. Its errors are different
from the GBT's in a useful way; Ridge's are not. The blend
is better in 6 of the 9 months on record (worse in December, March and July).

Caveats: the pairing was chosen after seeing the single-model results (the
weight sweep and the Ridge partners are the check on that); the scores are
hindcasts on recorded conditions; and the packed-hour shortfall narrows only
from −12.0 to about −11. A forward-in-time check (fit only on earlier dates)
would come before any production change.

### Forward in time (`blend_forward_results.csv`, `blend_forward_by_month.csv`)

`test_blend_forward.py` (kept local) predicts each calendar week only from
models fitted on dates before it, refitting weekly, once 20 counted days are
available (24 weeks, 129 days, from 2025-11-10):

| Model | MAE | Bias at actual 30+ | Weeks better than GBT | Days better |
|---|---|---|---|---|
| GBT (production) | 6.97 | −14.9 | | |
| Poisson GLM, shaped | 7.63 | −11.2 | 8/24 (p = 0.053, worse) | 51/129 |
| 0.5 GBT + 0.5 Poisson GLM | 6.95 | −13.0 | 14/24 (p = 0.73) | 67/129 |
| 0.7 GBT + 0.3 Poisson GLM | 6.87 | −13.8 | 15/24 (p = 0.19) | 71/129 |

Most of the week-held-out gain does not survive. The GLM is poor while
history is short (December 9.58 against the GBT's 4.00): its one-hot month and
hour × weekend terms have little or nothing to learn from early on. Restricted
to the 15 weeks with 60+ training days (from 2026-07-16), the 0.7/0.3 blend is
6.79 against 6.94, better in 11 of 15 weeks, p = 0.095; half and half 6.81,
p = 0.30. A small gain that may firm up as the record grows, not yet a case
for changing production.
