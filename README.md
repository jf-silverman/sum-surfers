# Sum Surfers

Automated surfer counting from a beach camera.

## What This Project Does, In Plain Terms

There's a live camera pointed at a surf spot in Santa Cruz, California.
This project watches that camera and tries to answer a simple question:
**how many surfers are out there, and how many will there be later
today?**

Here's the basic idea, step by step:

1. **Grab video.** A few times a day, the project downloads a short clip
   from the camera.
2. **Pull out a picture.** From each clip, it grabs one still image and
   crops it down to just the part of the water where surfers actually
   are.
3. **Count the surfers in the picture.** This is the hard part, and it's
   done by a computer-vision model — a program that has been trained on
   thousands of hand-labeled examples to recognize what a surfer in the
   water looks like, and draw a box around each one it finds. This is
   the same basic kind of technology used in things like self-driving
   cars (spotting pedestrians) or photo apps (finding faces). The more
   examples the model has seen, the better it gets at telling a real
   surfer apart from, say, a bird, a shadow, or a whitecap.
4. **Keep score over time.** Every count gets logged with the date, time,
   and conditions (tide, weather, etc.), building up a running history
   of how many surfers show up throughout the day and across the year.
5. **Make a forecast.** Using that history, a second, simpler model
   looks for patterns — for example, "it's a weekend, the tide is
   dropping, and it's sunny" tends to mean more surfers — and uses those
   patterns to predict roughly how crowded the spot will be later today,
   hour by hour. It's the same general idea as a weather forecast: not a
   guarantee, just an educated, data-backed guess with a plausible range
   attached to it.

The rest of this README goes into the technical details for anyone who
wants them, but that's the whole project in a nutshell.

## Technical Overview

This project downloads short video clips around daylight hours, extracts 3 cropped frames per clip (roughly 1.5-3 seconds apart), runs a [YOLOv8](docs/HOW_IT_WORKS.md#main-resources) object detector on tiled sections of each frame, and stores per-clip surfer counts averaged across those frames.

<!-- DAILY_CHART_START -->
## A Recent Surfer Detection Count: Thursday, September 03, 2026, 7:56 AM

![Latest detection review](data/charts/latest_detection.png)

## The Surfer Crowd Forecast for: Tuesday, September 08, 2026

![Latest daily prediction chart](data/charts/latest.png)

<!-- DAILY_CHART_END -->

### How to Read the Daily Chart

- **Aqua line** — the model's single best-guess ("median") count for each
  hour.
- **Shaded gradient + side table ("80% Range")** — the model's
  [prediction interval](docs/HOW_IT_WORKS.md#glossary): the range the
  real count is expected to fall in on most days, shown both as a
  gradient around the line (darker = more likely, fading out toward the
  10%/90% edges) and as plain numbers in the table. **It isn't perfectly
  calibrated** — see [Model Calibration](#model-calibration) below for
  the real, measured accuracy of this range, not just the claimed one.
- **Weather markers** (circle/square/triangle/diamond) — the model's
  predicted weather condition for that hour, plotted at the median
  count.
- **Green dashed line (right-hand axis)** — predicted tide height, in
  feet.
- **Hatched band / "no training data" label** — flags hours with little
  or no real training data behind them (night hours, or hours outside
  the model's normal range) — treat those points as a rough
  extrapolation, not a confident prediction.
- **Caption** — the real predictors driving that day's forecast, plus
  the detector's actual [precision and recall](docs/HOW_IT_WORKS.md#glossary)
  from its real training log (not estimated).

See [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a deeper walkthrough of
the detection pipeline and a glossary of every term used on this page,
and [PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) for the full history
of how the forecast model was built, tested, and tuned.

## What This Repo Does

1. Downloads a short video clip from the camera for each roughly
   9-minute window during daylight hours (real dawn to dusk for that
   date, not fixed clock times), into a dated folder.
2. Extracts 3 cropped [regions of interest](docs/HOW_IT_WORKS.md#glossary)
   from each clip — a primary frame plus 2 "side" frames a few seconds
   apart — so one count isn't at the mercy of a single unlucky frame
   (someone briefly hidden behind a wave, a bird flying through, etc.).
3. Checks the primary frame's brightness and blur before running
   detection; frames too dark or too blurred (fog, dusk, a wet lens) are
   skipped entirely rather than counted wrong. See
   [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the exact thresholds.
4. Splits each frame into 4 overlapping horizontal tiles — small,
   distant surfers are easier for the detector to find within a tile
   than scattered across one wide frame — runs
   [YOLOv8](docs/HOW_IT_WORKS.md#main-resources) on each tile, then
   merges detections that land on a tile boundary back into one count
   per frame.
5. Averages the 3 per-frame counts into one per-clip count, and appends
   the result — plus the 3 raw per-frame counts, for later analysis —
   to `data/predictions/predictions.csv`.

See [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a full walkthrough of the
detection pipeline and a glossary of every term used in this repo,
[PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) for how it was built and
tuned over time, and [PROJECT_FILES.md](docs/PROJECT_FILES.md) for a map
of what every file in this repo is for.

## Detector Training Metrics

Real per-epoch training log for the production YOLOv8s surfer detector
(`data/model_out/20251013/`, 60 epochs — see `code/train_model.py` for
the retraining path). Precision/recall/mAP are computed on the held-out
val split each epoch, not the training data.

![YOLOv8s detector training metrics — loss, precision, recall, mAP over 60 epochs](analysis/detector_training_metrics/detector_training_metrics.png)

Final epoch: precision 87.8%, recall 80.6%, mAP@0.5 84.4%, mAP@0.5:0.95
37.3% — the same real numbers cited in the daily chart's caption
(`code/plot_daily_prediction.py`'s `DETECTOR_PRECISION`/`DETECTOR_RECALL`).

## Surfer Count Prediction Model

A separate modeling pipeline on top of `data/predictions/predictions.csv` +
`data/predictor_vars/surfline_predictors.csv`, built in three phases (see
[`PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) for the full story,
including two real bugs found and fixed along the way):

- `code/backfill_openmeteo_weather.py` — adds real observed historical
  weather (Open-Meteo archive API, free/no-auth) to
  `data/predictor_vars/openmeteo_weather.csv`.
- `code/build_training_features.py` — joins predictions (target) with
  predictors (features) for `quality_ok=True` rows, adds derived
  time-of-day/day-of-week/month features. Writes `data/training_features.csv`.
- `code/fit_surfer_count_model.py` — fits and compares Poisson GLM,
  negative-binomial GLM, and gradient-boosted trees (the best performer,
  ~6 surfer MAE), plus GBT quantile-regression prediction intervals.
- `code/predict_surf_count.py` — pulls live tomorrow's forecast and outputs
  a prediction with an 80% range:
    ```bash
    python code/predict_surf_count.py                          # tomorrow, default hours
    python code/predict_surf_count.py --date 2026-08-28 --hours 07:10,12:00
    ```
- `code/demo_predictions.py` — shows N random held-out predictions
  alongside the actual count and conditions, for eyeballing model behavior.
- `code/plot_daily_prediction.py` + `code/daily_chart.sh` — generates a daily
  prediction chart (median line + a continuous 10-90% prediction-interval
  gradient with a side-by-side 80%-range table, tide, weather, night
  shading) and a detection-review image (real boxes/labels on the day's
  ~8am crop with the predicted range overlaid), auto-committed to the top
  of this README. Run via its own daily cron entry, independent of the
  twice-weekly clip pipeline. See "How to Read the Daily Chart" below for
  what everything on it means.

Caveat: held-out MAE is ~6 surfers on a typical count of ~15 — treat
outputs as directional estimates, not precise counts.

### Model Calibration

Empirical coverage of the GBT quantile prediction intervals, checked
directly against a held-out test split (`analysis/surf_count_model_calibration/plot_calibration.py`)
rather than trusted from the nominal target:

![Prediction-interval calibration for the surf-count model](analysis/surf_count_model_calibration/calibration_plot.png)

The narrower intervals (20-60%) run a few points under their nominal
target — mildly overconfident, in the normal direction for this kind of
model. The wide 80% interval instead measures *above* nominal (~83%
actual vs. 80% claimed): its lower bound is the 10th-percentile model,
and since ~12% of real held-out counts are genuinely 0, that bound
floors at 0 and can't be undershot, which mechanically inflates coverage
at that one point rather than reflecting better-than-usual calibration.
This is a real re-check of a number reported earlier
([`PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md)) as "closer to 65% than
80%" — re-run today on the current codebase and current data, it comes
out differently (see PROJECT_HISTORY.md for the investigation).

### Exploratory Findings

Tide and weekend/weekday are the two strongest predictors of surfer count
at this spot (see [`PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) for the full
GBT permutation-importance breakdown). A closer look at the weekend effect:

#### Weekday vs. Weekend

1. The weekend-to-weekday ratio varies noticeably by month — from about
   1.16-1.18x in March, May, and July 2026 up to nearly 2x in November
   2025 (also elevated, ~1.5x, in October/December 2025 and August
   2026). Some months show a much more pronounced weekend effect than
   others. We'll have to see if the trend holds once we have data from
   every month — several calendar months are still completely
   unrepresented in the dataset so far.
2. Weekends are also more variable day-to-day than weekdays: standard
   deviation 6.7 vs 5.6 surfers.

![Mean surfer count by month, weekday vs weekend](analysis/weekday_weekend_patterns/weekday_weekend_by_month_2026-08-28.png)

#### Daily Mean Count Kernel Density Estimate (KDE)

1. Each curve is normalized to its own group (n=61 weekdays, n=31
   weekends) — the taller weekday peak isn't a sample-size artifact.
2. Weekday counts cluster closer to their mean (std 5.6 vs 6.7), giving
   it a taller, narrower peak.
3. Weekends span a wider range (min 3.8 to max 33.8 vs weekday's 1.5 to
   28.0).

![Distribution of daily mean surfer counts, weekday vs weekend KDE](analysis/weekday_weekend_patterns/weekday_weekend_kde_2026-08-28.png)

## Pipeline Scripts

- `code/local_pipeline.sh`
  - The entry point. Downloads clips, extracts crops, checks local clip
    storage, runs detection, pulls the surf-forecast predictors below,
    records a success timestamp.
- `code/get_clips.py`
  - Downloads clips between real dawn and dusk — "civil twilight," the
    point the sky is light enough to see by, not full sunrise/sunset —
    using the camera provider's own live light forecast for today, or a
    backup calculation (with corrected camera coordinates) for backfill
    days.
  - Uses the provider's own native clip windows and can backfill up to
    the previous 5 days.
- `code/get_cropped_frame.py`
  - Reads downloaded clips and saves 3 cropped JPG frames each (primary +
    2 side frames), for multi-frame count averaging.
- `code/detect_surfers.py`
  - Checks the primary frame's brightness/blur before running detection,
    skipping all 3 frames if it's too dark or too foggy to reliably count
    (see [`HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)).
  - Runs [YOLO](docs/HOW_IT_WORKS.md#main-resources) on 4 horizontal
    overlapping tiles per frame, deduplicates boxes, filters out known
    static false positives, averages the 3 per-frame counts, and writes
    the result plus raw per-frame data.
- `code/backfill_multiframe_counts.py`
  - Manual, one-off script that backfills `frame_count_*` for existing
    `predictions.csv` rows whose raw clip is still on disk — never touches
    the original `surfer_count`/`confidence_avg`.
- `code/get_surf_predictors.py`
  - Pulls weather, rating, tide, swell, wind, wave-energy, and consistency
    data for Jack's from the surf-forecast provider's public API and
    appends to `data/predictor_vars/surfline_predictors.csv`, matched to
    `predictions.csv` rows by filename. Forward-looking only (today + tomorrow).
- `code/backfill_historical_predictors.py`
  - Manual, one-off script (not run by `local_pipeline.sh`) that backfills
    the same predictor fields for past dates, using the provider's
    historical API. See the script's docstring for usage and safety notes
    before running it.
- `code/manage_clips.py`
  - Emails a warning if local clip storage exceeds `CLIPS_DIR_LIMIT_GB`.
- `code/send_email.py`
  - Shared Gmail SMTP sender used for storage warnings.

## Schedule

`code/local_pipeline.sh` (clip collection + detection) and
`code/daily_chart.sh` (the daily prediction chart) each run automatically on
their own recurring schedule on the machine hosting the pipeline —
`local_pipeline.sh` a couple times a week, `daily_chart.sh` once a day.
Both are safe to run manually any time; see each script for details.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Add secrets to `.env`.
4. Run the pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r docs/requirements.txt
cp .env.example .env
bash code/local_pipeline.sh
```

## Required Environment Variables

Create `.env` from `.env.example` and set:

- `SURFLINE_CAMERA_ID`
- `SURFLINE_ACCESS_TOKEN`

Optional:

- `MODEL_PATH` (override default YOLO weights path)
- `SMTP_USER` / `SMTP_APP_PASSWORD` / `EMAIL_TO` (Gmail App Password, for storage-warning emails)
- `CLIPS_DIR_LIMIT_GB` (local clip storage warning threshold, default 2.0)
- `DETECT_MODE` / `DETECT_RECENT_DAYS` / `DETECT_START_DATE` (detection scope)
- `SURFLINE_HISTORICAL_TOKEN` (only for `backfill_historical_predictors.py`,
  never read by the scheduled pipeline — see that script's docstring)

## Data Locations

- Clips: `data/not_needed_in_repo/surf_clips`
- Crops: `data/j_shore_cam/surf_crops`
- Predictions: `data/predictions/predictions.csv` — one row per clip:
  `date, time_local, filename, surfer_count, confidence_avg, quality_ok,
  quality_reason, brightness, lap_var, human_count, frame_count_1,
  frame_count_2, frame_count_3, frame_count_mean, frame_count_stdev`.
  `surfer_count`/`confidence_avg` are left blank when `quality_ok` is
  `False` (detection was skipped). `human_count` is filled in manually
  over time. `frame_count_*` are the 3 raw per-frame counts and their
  mean/stdev that `surfer_count` is averaged from.
- Predictor variables (weather/rating/tide/swell/wind/energy/consistency
  for Jack's, plus observed Open-Meteo weather): `data/predictor_vars/`
- Human-review datasets (image batches + a `review_counts.csv` to fill
  in), one subfolder per dataset: `data/reviews/`
- Model weights default:
  `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`

## Analysis

One-off analyses (not part of the scheduled pipeline) each get their own
folder under `analysis/`, holding whatever mix of CSVs, charts, and
plotting/analysis scripts that investigation produced — see
[`PROJECT_FILES.md`](docs/PROJECT_FILES.md) for what's in each one.

