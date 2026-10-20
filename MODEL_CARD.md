# Model Card — Sum Surfers

Two models, used in sequence: a **detector** that counts surfers in a camera
frame, and a **forecast model** that predicts how many will be out in a given
hour. Every number below was measured, not estimated; where a figure has a
known caveat it is stated next to the figure rather than in a footnote.

Last updated 2026-10-17.

---

## 1. Detector — surfer counting from a frame

### Intended use
Counting surfers in cropped frames from one fixed shoreline camera at Jack's /
Pleasure Point, Santa Cruz, during daylight. It produces an hourly count that
feeds the forecast model.

### Not intended for
Other camera angles, other breaks, other distances, or anything where a
miscount carries a cost. It has only ever been trained and measured on this one
view. Counts are approximate by design: the question this project asks is "how
crowded is it" — not "exactly how many people are in the water".

### Architecture and inference
- **YOLOv8s** (Ultralytics), single class `surfer`, CPU inference.
- Each frame is split into **4 horizontal tiles, 376 × 180 px, 20% overlap**
  (a surfer is a handful of pixels in a 1280-px-wide strip; within a tile the
  same surfer is proportionally much larger).
- Cross-tile duplicates removed by **NMS at IoU 0.45**, then **containment
  suppression at 0.7** (intersection ÷ smaller box area), which catches the
  nested-box case NMS misses — a small box inside a large one shares little
  area relative to the larger one and survives ordinary NMS. Measured against
  box-level truth in October 2026, this suppression costs more than it buys:
  only **10 of 183 held-out false positives are a second box on an
  already-counted surfer**, and loosening the rule in dense neighbourhoods lifts
  recall. Both a crowd-indexed confidence threshold and density-adaptive
  containment are implemented and **off by default** pending a truth-protocol
  question — see `analysis/crowd_threshold/results.md`.
- **Confidence threshold 0.195.** Re-swept across 0.05–0.60 on 76 held-out
  frames in September 2026 and left unchanged: count MAE bottoms out at 0.200
  and F1 is flat from 0.125 to 0.425. A box-level hand review of everything the
  band 0.10–0.195 would admit now puts the measured floor at **0.150** — see
  Performance below. Unchanged pending the clear-day overcount re-check.

### Training data
The production checkpoint (`data/model_out/20260921_fog/`) was trained on
tiled exports: **384 training tiles / 2,522 boxes**, 176 validation tiles /
1,131 boxes, 128 test tiles / 612 boxes. Labels were drawn by hand in CVAT
across several batches deliberately targeting failure conditions — fog, glare,
winter light, and surfer pose.

### Performance
On the validation set, at the deployed checkpoint (**epoch 46**, chosen for best
mAP@0.5:0.95 — validation DFL loss rises after ~epoch 40 while training loss
keeps falling):

| metric | value |
|---|---|
| precision | 88.5% |
| recall | 81.3% |
| mAP@0.5 | 86.8% |
| mAP@0.5:0.95 | 38.5% |

mAP@0.5:0.95 being far below mAP@0.5 is expected: it averages over much
stricter box-overlap requirements than this task needs. A box on the right
surfer counts that surfer whether or not it hugs the outline.

**Against human counts**, which matters more than validation boxes. Updated
2026-10-09 on a far larger review: **220 frames hand-counted box by box across
16 days spanning a full year, 3,880 real surfers.**

| metric | value |
|---|---|
| precision | **98.7%** |
| recall | **91.9%** |
| count error | **1.47 surfers** per frame (bias −1.23) |

These are higher than the validation figures above because they ask a different
question — is this box on a surfer — rather than requiring a close overlap with
a labelled box.

**The undercount is real but modest, and varies by period.** Fitted across all
220 frames, `detector = 0.938 × truth − 0.10` (r = 0.986), a 6.2% proportional
undercount. By period, though:

| period | n | peak frame | slope | 95% CI |
|---|---:|---:|---:|---|
| summer 2026 | 31 | 73 | 0.996 | 0.945–1.047 |
| autumn/winter 2025–26 | 51 | 54 | 0.961 | 0.922–1.000 |
| Sept–Oct 2026 | 138 | 54 | 0.866 | 0.834–0.897 |

Two of the three include 1.0 — including the summer days, which hold the most
crowded frames in the corpus. **An earlier estimate of 13.5%, fitted on the
Sept–Oct window alone, was an artifact of that window**; correcting counts by
it would have inflated every count by roughly 8%. No correction is applied.

**Where the misses go.** Of 316 missed surfers, **286 had no box at all** and 30
were inside a box drawn around several people. Of 46 false positives, 35 were on
something that is not a surfer and 11 were a duplicate box. By pose: 223 prone,
40 sitting, 42 unknown, 10 standing — lying flat is what defeats it.

**Most of the missed surfers were seen and discarded.** Re-running inference at
a 0.02 confidence floor shows that dropping `CONF_THRESH` from 0.195 to 0.10
adds 197 boxes and closes the count gap almost exactly (3,900 against 3,880),
and those boxes appear where the misses were (frames with more no-box misses
gain more boxes, rho = +0.425, p = 4.7e-11).

**Scored by hand, 2026-10-17 — half of them are surfers the count is missing.**
All 187 of those boxes that can actually reach `predictions.csv` were judged
box-by-box (`analysis/conf_band_decision/`). Counting a box as a hit only when
it adds a surfer the count does not already have — a second box on an
already-counted surfer inflates the count exactly as a box on open water does —
the band is **92 real against 95 spurious, 49.2% precision**, against 91.8% for
what production already emits. Of the 92 recovered, **64 are prone and side-on
prone (34) is the biggest group, not end-on (16)**.

**The measured floor is 0.150.** Against a current undercount of 2.14
surfers/frame on these 113 frames, it cuts mean |count error| by **0.212**
(95% CI −0.389 to −0.035, Wilcoxon p = 0.023, 39 frames better against 27 worse)
— the largest improvement in the sweep and the only one clearing p < 0.05 with
room, sitting inside the 0.15–0.18 range where admitted-box precision peaks at
0.55–0.58. Everything below it buys nothing measurable: the extra 105 boxes from
0.15 down to 0.10 are 47 recovered against 58 spurious, the count improvement
does not move (−0.204, p = 0.233), and admitted-box precision decays to 0.49.
Even at its best this recovers about a tenth of the counting error — 197 of the
242 missing surfers on these frames are still missing at 0.15.

**The threshold has not been changed, and the re-check said no.** The review
frames are biased toward undercounting — all of them were chosen because they
had misses — so the clear-day overcount set was re-checked
(`analysis/conf_thresh_recheck/`): the two hand-count sets disagree in sign at
every threshold, and a flat cut costs +0.056 mean error on the emptiest
quartile while buying −0.246 on the busiest. The threshold is not a dial on
accuracy but one that trades empty-frame precision for crowded-frame recall.

A **count-indexed threshold** removes that objection — below a count of 10 it is
0.195, so the emptiest quartile is untouched — and together with density-adaptive
containment it moves box-level |error| 2.31 → 1.85 (p = 0.002) and `reviewdays`
1.50 → 1.30 (p = 0.010), against `spotcheck` 1.23 → 1.58 (p = 0.030). It is built
and off. What blocks adoption is no longer the instrument but the truth data: at
matched crowd levels, box-level CVAT annotation and `reviewdays` agree on a ratio
of 0.924 and 0.926 while `spotcheck` reads 1.021, and the only 2 frames in both
count sets have the box-by-box protocol reading +2 and +1 above the direct count.
Settling that needs ~30 frames hand-counted under one protocol.

The cheaper move is unblocked: **28 of the 95 spurious boxes sit in static zones** (18 tree bough on
the right edge, 9 shore head at the bottom centre, 1 wind sock), and masking
those lifts the band to 57.9% precision at no cost in recall, at the current
threshold or any lower one.

### Known failure modes
- **Birds**, and in one measured case a **bird's reflection**, counted as
  surfers. Low-flying birds over water produce a detached reflection below them.
- **People on the beach** and, occasionally, one surfer split into two
  side-by-side boxes. The split-box case is **rarer than it looks**: on the 61
  held-out labelled frames only 22 of 1,394 kept boxes overlap another at
  IoU ≥ 0.3, and the false-positive rate is flat against local crowding (0.129 at
  zero neighbours, 0.158 at five or more). Boxes piling onto a cluster are mostly
  real adjacent surfers. The beach case has a measured shape: the ROI is rows
  420–600 of the 720-row frame, and the bluff path runs at y 590–660, so only
  the top of a head enters the image and the box is always clipped by the ROI's
  bottom edge. A zone rule on that signature — centre in x 400–680, bottom edge
  at or past y 179, confidence under 0.50 — was hand-checked against all 24
  boxes it fires on in a 300-frame sample: **19 shore heads, 1 duplicate box,
  4 real surfers, a 5:1 trade** at 0.080 boxes/frame. Nothing separates the 4
  from the 19 on confidence, size or position, so the rule cannot be tuned, only
  taken or left. **Adopted 2026-10-17** as `BLUFF_MASK`, alongside `TREE_MASK`
  and `FLAG_MASK` — see `analysis/shore_head_zone/`. It removes 0.063 false
  counts/frame at a cost of 0.013 real surfers, which runs slightly against a
  count that is already low; it is in because shore heads are a systematic bias
  that recurs at the same bluff positions rather than noise that averages out.
- **Whitewater band (bottom of frame)** has the weakest recall: 78%, against 86%
  mid-frame and 84% top. Not a threshold problem — dropping that band's
  threshold to 0.08 moves recall only to 80%. It needs labeled whitewater.
- **Fog and glare** were the dominant failure until September 2026. Retraining
  on labeled examples took held-out hazy-frame recall from **59% to 98%**.
- [`docs/non_surfer_objects.md`](docs/non_surfer_objects.md) catalogs each
  false-positive class with its visual tell.

### Image-quality gate
Frames are rejected before counting if mean brightness < 75.4 or Laplacian
variance < 12.7, plus a third rule for frames outside the first-light/last-light
window with Laplacian variance > 180 — amplified night sensor noise is
higher-frequency than real water. Glare fraction is recorded but deliberately
not gated: gating on it discarded countable crowded frames.

---

## 2. Forecast model — predicting hourly crowd

### Intended use
Predicting the surfer count for a given hour at this one spot, up to seven days
ahead, from publicly available surf and weather forecasts.

### Architecture
`HistGradientBoostingRegressor` with Poisson loss for the point estimate, plus
two quantile models for the prediction interval. **This released bundle ships
the q0.10 / q0.90 pair**, and every interval figure in this card is measured on
that pair. The live forecast moved its lower edge to **q0.15 on 2026-10-20** — a
75% central interval — because the 10th percentile of hourly surfer count has
become exactly zero as the share of empty hours rises seasonally; see the README
section "Why the prediction range is 75% and not 80%". Two consequences for
anyone re-exporting this bundle: the interval numbers below describe an 80% band
that production no longer draws, and the q0.10 fit here is now marginal — on the
current record it survives only at `min_samples_leaf` 200, the last candidate
`fit_quantile_model_robust` tries, so the next export may raise rather than
silently ship a flat zero. That is the intended behaviour: calibration and
release code keep `allow_degenerate=False`, and only display code opts in. The target is
`round(frame_count_mean)` — the rounded mean of the three frames cut at
1.0/2.5/4.0 s. The Poisson objective is load-bearing: squared
error costs **+0.43 MAE** on 16 rolling origins (p < 0.05, better on only 3 of
16). This confirms the September 5-fold comparison (6.238 against 6.461) on a
forward split, the earlier one having used the random-hour split later found to
leak. **34 features**:
surf height, swell height/period/direction, wind speed/gust/direction, tide,
rating, wave energy, consistency, observed and forecast weather, plus cyclical
hour/month encodings, `is_weekend` and `is_night`.

### Performance
Held out by **whole days, forward in time** — fit on the earliest days, scored
on the most recent, so no day appears on both sides:

| metric | value |
|---|---|
| MAE | **7.60 surfers** |
| RMSE | 10.43 |
| mean bias | −1.84 (predicted below actual) |
| 80% interval coverage | **71.6%** |

Measured on 352 held-out hours (2026-08-29 → 2026-09-21) against 1,242 training
rows.

**Against naive baselines** (`code/eval_baselines.py`, same forward split, 378
test rows as the record has since grown): the model's MAE of 6.79 beats the
strongest naive reference — persistence, yesterday's count at the same hour, at
**8.09** — by **16%**. Climatology by hour × weekend scores 10.82, a trailing
7-day level 11.28, seasonal naive 17.26. The **target noise floor is about 1.5
surfers** at typical crowd levels — the error in the production target
(`round(frame_count_mean)`, the rounded mean of the 3 frames cut at 1.0/2.5/4.0 s)
against its hour's true level, measured on three dense hours sampled at 1 fps
(10,862 frames, 36 clips, hour means 5.4 / 12.1 / 24.7). No forecast can beat
that. Read the headline MAE between those two numbers, not on its own.

The floor is close to **additive** rather than proportional to crowd size:
MAE ≈ 0.95 + 0.045 × count (r = 0.643, p < 0.0001, slope 95% CI 0.027–0.064),
so about 1.0 on a quiet hour and 2.2 on a crowded one. The matched pair that
shows this is 2026-10-13 08:00 against 2026-08-28 08:00 — same clock hour, same
camera, so light and geometry are held constant while the mean count halves from
24.7 to 12.1. Per-frame error did not move (2.07 against 2.08) where a
proportional model predicted about 1.0.

*(An earlier figure of **0.68** is withdrawn. It compared each frame to a 3-frame
mean **containing that frame**, which deflates the result by exactly 3/2 — the
closed form, confirmed at 1.50 across all 1,903 clips — and used frames 1.5 s
apart when counts decorrelate only over about 60 s, so the reference was nearly
the same observation. It also was not stable: the same definition returns 0.61,
1.00 and 0.69 on the three dense hours. See
[`analysis/count_variability/noise_floor/`](analysis/count_variability/noise_floor/).)*

*(The calibration section below reports 71.9% for the same interval on the
same rows. Both are right: the released bundle enforces monotonicity between the
shipped q0.10/q0.90 pair alone, while the calibration chart fits a nine-level
quantile ladder and enforces it across all of them, which nudges the upper bound
and catches two more rows out of 352.)*

**This number got worse on 2026-09-23 without the model changing.** Accuracy had
been measured with a random 80/20 split of *hours*; at ~13 hours per day that
puts hours from the same day on both sides, so the model learned a day's crowd
level from that day's own hours and was scored on the rest of it. Splitting by
whole day costs ~1.2 MAE, and holding out the most *recent* days rather than
random ones costs ~0.8 more. The published figure is now the one a real
day-ahead forecast faces.

**Live tracking** (updated 2026-10-01): 568 scored forecast-hours across
2026-09-23 → 2026-10-01. Day-ahead MAE **4.50**, 83% band coverage; across all
lead times MAE 5.0 and coverage 76%.

**The direction of the bias is not stable, and that matters more than its size.**
The backtest above shows the model predicting 1.84 surfers BELOW actual. The live
record since shows it predicting **1.87 surfers ABOVE** actual — nearly the same
magnitude, opposite sign, 68% of hours over-forecast. The model is not biased in
a fixed direction; it is **lagging the level**. Mean counts fell from 15.9
surfers/hour in the backtest window to 11.5 in the live window, and a model
trained mostly on busier months keeps forecasting the old level. Measured
independently across 16 rolling origins, forecast bias tracks the train/test
level gap at **r = −0.811**.

Day-ahead alone the over-prediction is small and uniform: mean −0.82, median
−1.07, 60% of hours over-forecast (t = −2.03, p = 0.044), and it over-forecasts
in every predicted quartile (−0.69 / −1.26 / −1.02 / −0.31) rather than
squashing busy hours toward the middle. It grows with lead time, reaching about
−7 at seven days out, which is the same staleness amplified by more regression
toward the training mean.

**Read any bias figure here as a snapshot of one period, not a property of the
model.** See [`data/forecasts/forecast_log.csv`](data/forecasts/forecast_log.csv)
for the current record.

### Known limitations
- **The intervals do not widen with lead time, so coverage collapses further
  out.** Band width tracks the predicted level at r = +0.900 — which is correct
  for count data, where a Poisson-like mean-variance relationship forces
  sd ≈ √μ, and is not the defect. The defect is that width is conditioned on
  level but **not on lead**, so it carries no horizon information: at longer lead
  the forecast regresses toward the middle and the band narrows with it, exactly
  where the inputs are least reliable. Measured
  coverage ran 85% at one day out to 57% at seven. Partly corrected since
  2026-10-01 by `code/calibrate_lead_bands.py`, which widens each lead's band by
  a factor fitted on the scored record; the correction is gentle (1.02 to 1.21)
  and its slope is not yet significant.
- **Prediction intervals are overconfident at every level** — measured coverage
  15/26/36/71.9% against nominal 20/40/60/80%. The 80% band misses 9.4% low and
  18.8% high against a 10% target on each side; the high edge is where it loses.
- **The count distribution is zero-inflated, and seasonally so**, which the
  single-model-per-quantile design does not represent. Empty hours ran 7.6% in
  2026-08 and 19.5% in 2026-10, against 33.8% and 38.9% recorded in 2025-11 and
  2025-12. Past roughly a tenth, the true 10th percentile is zero and the lowest
  quantile levels stop carrying information — which is why the live band's lower
  edge moved to q0.15. A hurdle model (P(anyone out) times the count quantiles
  fit on non-empty hours only, `code/eval_hurdle_model.py`) scores better on
  pinball loss on 5 of 5 splits but moves 80% coverage the wrong way on 3 of 5,
  so it is measured and not adopted. An automatic check reports when empty hours
  pass 30%, which is the point that trade is meant to be revisited.
- **It is predictor-limited, not algorithm-limited.** Every model family and a
  72-combination hyperparameter grid landed within noise of each other.
  Measured and rejected on rolling origins: previous-day lags, lunar tide
  structure, day-of-week one-hots, raw hour, deeper trees, weekend×hour
  interactions (re-tested 2026-10-17 on all 101 origins after the box plot
  showed the weekend gap is confined to 08:00–17:00: +0.010 to +0.029 MAE, no
  arm beating the add-a-column null — the tree ensemble already represents the
  interaction by splitting on one feature inside a branch of another),
  within-day rank features, tide bands, tide-window structure,
  and training on the **unrounded** 3-frame mean instead of the rounded one
  (+0.07 MAE, p = 0.25, and inside a placebo null built by permuting the same
  rounding deltas — 3 of 8 random seeds did as well or better). Rounding
  displaces the target by 0.188 surfers on average, against a target whose own
  measurement noise is about 1.5, so there was nothing there to recover.
  Day-level aggregate predictors are the only idea to have moved MAE the right
  way (−0.09), and that is not significant either.
- **How finely these tests can resolve anything.** The per-origin MAE delta has
  a standard deviation of 0.27–0.36, so the standard error of a mean over 16
  rolling origins is about ±0.09 and over all 101 about ±0.03. Sixteen origins
  resolves roughly 0.25 MAE, not 0.1. The weekend×hour test demonstrated this
  directly on 2026-10-17: the same arm read −0.084 at 16 origins and +0.101 at
  30 before settling at +0.029 over all 101. The rejections above stand — a
  null that survives a noisy test is still a null — but a magnitude quoted from
  a 16-origin run carries about ±0.09, and 16 should not be used again for a
  claim under about 0.25. See `analysis/forecast_weekend_hour/results.md`.
- **The apparent under-call at crowded hours is mostly a measurement artifact.**
  Binned by actual count the top quintile shows −15.7; binned by *predicted*
  count the same rows show −2.2, and the calibration slope is 1.02 (95% CI
  0.92–1.12). Conditioning on the outcome produces the first number for any
  imperfect model.
- **Counts come from the detector, not humans**, so the forecast inherits the
  detector's error (~1.3 surfers per frame) on both sides of training.
- **One spot, ~125 days, no full year.** October–December exist only for 2025
  and March–September only for 2026, so season and year are perfectly
  confounded. Nothing here supports a year-over-year claim.

---

## Reproducing these numbers

The detector weights and a held-out evaluation set are committed; see
[Run It Yourself](README.md#run-it-yourself). `data/model_release/` ships the
fitted forecast model, the 352 held-out rows it never saw, and the metadata
above, so the reported accuracy can be recomputed rather than taken on trust.

Full method and history: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) and
[`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md).

## License
Code AGPL-3.0 ([LICENSE](LICENSE)); data and weights CC BY-SA 4.0
([LICENSE-DATA](LICENSE-DATA)).
