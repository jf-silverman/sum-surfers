# What standard practice says this project is missing

Two research passes run 2026-10-03, one on object-detection evaluation and
human-in-the-loop annotation, one on count forecasting. This is my assessment of
what came back, with the load-bearing claims checked against the repo. Where a
claim was wrong I say so; where it corrected something I had written, I say that
too.

The short version: **the detector is evaluated better than I expected and the
forecast is evaluated worse.** The detector's remaining gaps are about what the
numbers mean. The forecast had no reference point at all.

---

## Verified before relaying

| claim | verdict |
|---|---|
| No climatology / persistence / seasonal-naive / MASE anywhere in the repo | **true** — confirmed by grep, now fixed |
| `MODEL_CARD.md` contradicts itself on checkpoint selection | **true** — says epoch 46 "chosen for best mAP@0.5:0.95", then argues that metric is wrong for this task |
| FP/FN cancellation hides box errors | **true** — 2026-09-23: 13 box mistakes netted to −5; one hour cancelled exactly |
| `forecast_log.csv` holds ~1,022 rows of per-lead predictors | **overstated** — 1,022 total rows but only **196** carry predictor values, added 2026-10-01 |
| The project does not refit on a schedule | **false** — `REFIT_AFTER_DAYS = 7`, `REFIT_AFTER_NEW_ROWS = 100` |

---

## Two corrections to things I had written

**The `r = +0.900` framing was backwards.** I described band width tracking the
predicted level as a defect — "it encodes count magnitude, not input
uncertainty". For count data it *should*: a Poisson-like mean-variance
relationship forces sd ≈ √μ, and quantile regression is meant to be
heteroscedastic. The real defect is narrower — width is conditioned on level but
**not on lead**, so it carries no horizon information. My wording invited the
wrong fix (decorrelate width from level) rather than the right one (add a
lead-conditional component). Corrected in `MODEL_CARD.md`.

**Rejecting recency weighting did not close the drift question.** Weighting
changes what the trees learn, and a tree cannot extrapolate past its training
range at all — which is exactly why it failed once the crowd level drifted below
that range. An additive offset applied *outside* the model has no such limit.
These are different objects and the rejection of one is not evidence against the
other. Measured below: it isn't.

---

## What was built in response

**Naive baselines** (`code/eval_baselines.py`). On the same forward split, 378
test rows, mean count 13.87:

| reference | n | MAE | skill |
|---|---|---|---|
| persistence (yesterday, same hour) | 320 | **8.09** | — strongest naive |
| climatology (hour × weekend) | 378 | 10.82 | −34% |
| trailing 7-day mean level | 378 | 11.28 | −39% |
| seasonal naive (7 days ago) | 356 | 17.26 | −113% |
| **the model** | 378 | **6.79** | **+16%** |

The model does beat the strongest naive, by 16%. That is a much more modest
claim than a bare MAE of 7.6, and it is the claim that was missing.

**Persistence at 8.09 is the result worth keeping.** Yesterday's count at the
same hour very nearly matches a 34-feature gradient-boosted model. It also shows
the earlier rejection of previous-day *lag features* never closed the question of
whether yesterday carries signal: a tree can fail to exploit a feature while a
naive forecast built on it scores well.

**Target noise floor: 0.68 surfers** — the spread between one extracted frame and
the 3-frame mean of its clip, over 1,750 clips. No forecast can beat that; it is
measurement noise in the target.

**Post-hoc recalibration, tested out-of-sample**
(`analysis/forecast_recalibration/`). Fitted only on earlier target dates, 625
rows over 10 dates:

| correction | MAE | change |
|---|---|---|
| global affine (a + b·pred) | **4.471** | **−0.276, p < 0.001** |
| per-lead affine | 4.547 | −0.199, p = 0.001 |
| global EWMA offset, 3-day half-life | 4.567 | −0.179, p = 0.001 |
| per-lead EWMA offset, 7-day | 4.646 | −0.100, p = 0.067 |

**All four help, and the simplest helps most.** Set against every in-model
feature idea tried here — all flat or worse, best −0.09 and not significant —
this is the first thing to move MAE significantly. Not wired into production:
adopting it changes every forecast published.

**An `uncertain` column** on the review sets, for calls the reviewer cannot
confidently make. Scored twice, counting them as errors and excluding them; the
gap is the irreducible ambiguity budget, which is what bounds how far whitewater
recall can go.

---

## Detection: what is worth doing, in order

1. **Keep the two headline numbers apart.** Validation says 88.5 / 81.3 (IoU-0.5
   matching, on tiles, on a split enriched with fog and glare). The per-box review
   says 98.1 / 95.8 (human "is there a surfer under this box", whole frames, one
   ordinary October day). Human box-identity judgement is far more lenient than
   IoU ≥ 0.5. **The review number must not migrate into the model card as the
   detector's precision and recall.** Expect it to fall as harder days enter.
2. **Report the cancellation rate, not just count error.** Count MAE 0.64/hour
   against a box-error rate of 6.0% of truth. Count accuracy is never evidence
   that boxes are good.
3. **Hourly-aggregate detector MAE, then a forecast-sensitivity test.** Inject
   noise of the measured magnitude into the target and refit. If detector noise
   contributes little of the forecast's error, further detector work has no
   headroom — the strongest possible argument for where to spend effort.
4. **Confidence intervals on the review.** Wilson gives precision ≈ [95.2, 99.3]
   and recall ≈ [92.2, 97.8] on day one, both ignoring that boxes cluster within
   frames, so treat them as lower bounds on width. For ±3pp with clustering,
   budget 2–3 whole days; for ±2pp, 5–7.
5. **Stratify which days get reviewed** — fog, glare, high-count, low-light,
   winter, ordinary — and report per-stratum rather than pooling into one global
   figure.
6. **Change the checkpoint-selection rule** before the next training run: pick on
   whole-frame count MAE through the production path, not mAP@0.5:0.95. Record
   whether it picks a different epoch; if not, that closes the question.
7. **`frame_count_stdev` is a free disagreement signal** already logged and not
   used for labelling selection. Counts vary by up to 7 surfers between frames
   1.5 s apart; high-stdev frames are the ambiguous ones.

**Explicitly not worth doing here:** TIDE (two of six error classes are
structurally zero at one class, and localisation error is irrelevant to
counting), GAME (wrong partition for a 1280×180 strip; the existing vertical-band
breakdown already does the job), an uncertainty-sampling active-learning loop
(the benchmark literature does not support it, and the existing targeted
fog/glare rounds have a measured 39-point win), amodal/visible dual boxes
(CrowdHuman-scale machinery for 172 frames).

**Where the literature contradicts itself:** active learning (thousands of papers
report gains; benchmark studies report no method reliably beats random, some
significantly worse); ignore regions (simultaneously standard practice in
crowded-human benchmarks and the subject of a 2024 critique arguing they produce
unreliable evaluation); mAP's predictive value (a comparability anchor, a poor
selection objective, and papers rarely say so plainly).

---

## Forecasting: what is worth doing, in order

1. **Baselines** — done above. Was the biggest gap.
2. **Post-hoc recalibration** — measured above. Decide whether to adopt.
3. **Perfect prog vs MOS.** Training on observed predictors and serving forecast
   predictors has a name — *perfect prognosis* — and is the known-inferior
   approach. The standard replacement, MOS, regresses on archived forecasts with
   one equation per lead. Full per-lead MOS is not fittable at 125 days, but the
   affine recalibration above is its small-sample form, and `forecast_log.csv`
   now carries the per-lead predictors needed to go further later.
4. **Conformalize the intervals.** Two quantile GBTs with no calibration step is
   exactly the setup conformalized quantile regression exists to fix; a
   lead-grouped (Mondrian) calibration set would replace the hand-fitted
   inflation factor in `calibrate_lead_bands.py` with a coverage guarantee for
   similar effort. What is there now is an uncalibrated version of the same idea.
5. **Measure predictor degradation by lead** from the `pred_*` columns. Right now
   "inputs degrade with lead" is an assumption; tide is known to drift 0.000 ft
   at seven days, so the degradation is entirely on the weather and surf side and
   is worth quantifying per predictor.
6. **Promote pinball loss / CRPS and a PIT histogram** to the headline beside
   MAE, and split every metric empty vs non-empty — 15% of hours are empty and
   easy zeros flatter MAE. `eval_hurdle_model.py` already does a version of this.
7. **Relative neighbouring-break conditions.** The only genuinely new information
   channel identified, with a real mechanism (surfers substitute across spots)
   and no proxy among the current 34 features. Nearby breaks share swell, so the
   predictor must be *relative* — Jack's rating minus the mean of 3–4 neighbours —
   and the same Surfline endpoints already called work for any `spotId`.

**Not worth doing at ~125 days:** per-lead MOS models, holiday and school-calendar
features *as learned predictors* (4–6 holidays in the record will fit as noise —
add them as annotations on the scored record instead, and revisit with a second
year), a state-space or GAM rewrite, further model-family search, MinT
reconciliation, Google Trends (it is established for park visitation, but that is
nowcasting; future search volume does not exist at a 1–7 day lead).

**One framing correction for `works_cited.md`:** the claim that no published work
forecasts surfer counts at a break survives, but a commercial analogue exists —
Surfline Premium ships crowd-prediction features, and Pine Forecast sells the
same product shape for ski resorts. Narrowing the claim to *published literature*
and naming the commercial precedent is a stronger framing, not a weaker one.

---

## The sentence worth remembering

The project's own conclusion that it is "predictor-limited, not
algorithm-limited" is probably right about features and probably wrong as a
whole. The three largest measured errors — the −7 bias at day 7, the sign-flipping
level lag, and 57% coverage at long lead — are **post-processing failures, not
predictor failures**, and until today none had been attacked with the standard
tool. The rejection list in `MODEL_CARD.md` rules out a set of *in-model
features*; it was never evidence about corrections applied outside the model.
