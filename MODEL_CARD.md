# Model Card — Sum Surfers

Two models, used in sequence: a **detector** that counts surfers in a camera
frame, and a **forecast model** that predicts how many will be out in a given
hour. Every number below was measured, not estimated; where a figure has a
known caveat it is stated next to the figure rather than in a footnote.

Last updated 2026-09-26.

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
  area relative to the larger one and survives ordinary NMS.
- **Confidence threshold 0.195.** Re-swept across 0.05–0.60 on 76 held-out
  frames in September 2026 and left unchanged: count MAE bottoms out at 0.200
  and F1 is flat from 0.125 to 0.425.

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

**Against human counts**, which matters more than validation boxes: across 43
hand-counted frames the detector is off by **1.26 surfers on average** (mean
error +0.37). On the nine frames with 25 or more real surfers, mean error is
**−0.22** — no systematic undercount at the crowded end, which was not true of
the previous model.

### Known failure modes
- **Birds**, and in one measured case a **bird's reflection**, counted as
  surfers. Low-flying birds over water produce a detached reflection below them.
- **People on the beach** and, occasionally, one surfer split into two
  side-by-side boxes.
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
two quantile models (q0.10 / q0.90) for the prediction interval. **34 features**:
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
| mean bias | −1.84 |
| 80% interval coverage | **71.6%** |

Measured on 352 held-out hours (2026-08-29 → 2026-09-21) against 1,242 training
rows. *(The calibration section below reports 71.9% for the same interval on the
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

**Live tracking**, which is a different and easier sample: on 83 scored
forecast-hours the MAE is 4.26 with 73% band coverage. Too few hours and too few
days to be the headline; see
[`data/forecasts/forecast_log.csv`](data/forecasts/forecast_log.csv).

### Known limitations
- **Prediction intervals are overconfident at every level** — measured coverage
  15/26/36/71.9% against nominal 20/40/60/80%. The 80% band misses 9.4% low and
  18.8% high against a 10% target on each side; the high edge is where it loses.
- **It is predictor-limited, not algorithm-limited.** Every model family and a
  72-combination hyperparameter grid landed within noise of each other.
  Measured and rejected on rolling origins: previous-day lags, lunar tide
  structure, day-of-week one-hots, raw hour, deeper trees, weekend×hour
  interactions, within-day rank features, tide bands, tide-window structure.
  Day-level aggregate predictors are the only idea to have moved MAE the right
  way (−0.09), and that is not significant either.
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
