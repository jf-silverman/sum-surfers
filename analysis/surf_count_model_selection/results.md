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
