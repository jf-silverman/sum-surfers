# Project File Descriptions

A map of what every file and folder in this repo is for. See `README.md`
for how to run the pipeline and `CLAUDE.md` (local-only, gitignored) for
deeper architecture notes.

## Root

| Path | Purpose |
|---|---|
| `README.md` | Setup and usage instructions. |
| `HOW_IT_WORKS.md` | Plain-language walkthrough of the detection pipeline (capture → crop → quality gate → tile → detect → filter → count), plus a glossary of CV/ML terms used across this repo. |
| `PROJECT_HISTORY.md` | Chronological record of how the project was built and tuned, plus an "Open engineering leads" section for unresolved findings. |
| `CLAUDE.md` | Local-only architecture notes for Claude Code (gitignored, never committed). |
| `PROJECT_FILES.md` | This file. |
| `.env` | Secrets and config (gitignored). Copy from `.env.example`. |
| `.env.example` | Template for `.env` — required and optional variables. |
| `.gitignore` | Excludes secrets, OS files, caches, large clip video, and `archive/`'s raw clips. |

## `code/` — pipeline scripts

| Path | Purpose |
|---|---|
| `local_pipeline.sh` | Entry point, run via cron. Chains all six steps below and records a success timestamp. |
| `get_clips.py` | Downloads Surfline clips for the camera (Pleasure Point, Santa Cruz — coordinates corrected 2026-08-28, see `PROJECT_HISTORY.md`) between real dawn and dusk (civil twilight, via `get_light_window()` — prefers Surfline's own live `sunlight` forecast for today, falls back to astral for backfill days); backfills up to `CLIP_LOOKBACK_DAYS`. Emails a warning (via `send_email.py`) if any download fails with HTTP 401 (expired `SURFLINE_ACCESS_TOKEN`). Supports a one-shot clip-duration override via `data/.clip_duration_override` (consumed/deleted on read, so it can never silently persist past one run). |
| `get_cropped_frame.py` | Extracts 3 ROI-cropped frames per downloaded clip (a primary frame at 2.5s, plus 2 "side" frames 1.5s before/after) for multi-frame count averaging — see `HOW_IT_WORKS.md`. |
| `detect_surfers.py` | Checks each crop's image quality (brightness + Laplacian variance) before detection, skipping frames too dark/foggy to reliably count; runs tiled YOLOv8 inference on all 3 frames, de-duplicates boxes via NMS, filters out known static false positives (tree bough, flag), averages the 3 per-frame counts, and appends results to `data/predictions.csv`. |
| `get_surf_predictors.py` | Pulls weather, rating, tide, swell, wind, wave-energy, and consistency data for Jack's from Surfline's forecast API; appends to `data/surfline_predictors.csv`. Forward-looking only (today + tomorrow), runs every scheduled pipeline execution. |
| `manage_clips.py` | Local clip-storage manager. `--check` (used by cron) emails a warning past `CLIPS_DIR_LIMIT_GB`; interactive mode offers deletion by age. |
| `send_email.py` | Shared Gmail SMTP + App Password sender used for storage-warning emails. |
| `backfill_tide.py` | One-off (not part of the scheduled pipeline): estimated historical tide height via NOAA hi/lo + cosine interpolation, before the Surfline tide endpoint was found. Its output (`tide_backfill.csv`) has been archived; kept here for reference. |
| `backfill_historical_predictors.py` | One-off, manually-run (not part of the scheduled pipeline): backfills the same predictor fields as `get_surf_predictors.py` for past dates in `predictions.csv`, using Surfline's `start=YYYY-MM-DD` param and a personal, premium-account session token (never committed — see the script's own docstring). Jittered pauses between requests to avoid bot detection; run a `--dry-run` first to preview. |
| `backfill_multiframe_counts.py` | One-off, manually-run: for `predictions.csv` rows whose raw clip is still on disk, extracts any missing side frames and fills in `frame_count_*` columns — never touches the original `surfer_count`/`confidence_avg`. Local/compute-only (no network, no rate limits). |
| `backfill_predictors_from_har.py` | One-off, manually-run: parses HAR files exported from clicking through Surfline's own Historical view (an alternative to the token-based script's live requests). Filters each response down to only the day matching its own `start=` param (day-of only, not the multi-day forecast window the API actually returns), with a narrow 1-day-out fallback for dates that were never directly clickable. Cannot get `weather_condition`/`temperature_f`/`pressure_mb`/`consistency_wave_count` — the Historical view never calls those endpoints. |
| `backfill_openmeteo_weather.py` | Adds real observed historical weather (Open-Meteo archive API, free/no-auth) for every `predictions.csv` row to `data/openmeteo_weather.csv` — separate from `surfline_predictors.csv`, non-destructive. Includes `real_humidity_pct`, a validated proxy for the quality gate's fog/blur detections. |
| `build_training_features.py` | Phase 1 of the surfer-count modeling plan: joins `predictions.csv` (target) with `surfline_predictors.csv` and `openmeteo_weather.csv` (features) on filename, restricted to `quality_ok=True` rows, adds `hour_local`/`day_of_week`/`is_weekend`/`month`. Target uses `round(frame_count_mean)` (multi-frame average) wherever available, falling back to the legacy single-frame `surfer_count` otherwise — see `resolve_target_count()` and the `used_multiframe` output column. Also derives `weather_simple` (CLEAR/CLOUDY_OVERCAST/RAIN/FOG, merged from Surfline's 21 raw categories) and `is_night` via `simplify_weather_condition()`, shared with `predict_surf_count.py` for live forecasts. Writes `data/training_features.csv`. |
| `fit_surfer_count_model.py` | Phase 2: fits and compares Poisson GLM, negative-binomial GLM (MLE-estimated dispersion), and gradient-boosted trees on `data/training_features.csv`, plus GBT quantile-regression prediction intervals via `fit_quantile_model_robust()` — a self-checking fitter that escalates `min_samples_leaf` until the quantile model's predictions actually vary, since no fixed hyperparameter combo has proven safe against silent collapse to a constant prediction (found and re-found — see `PROJECT_HISTORY.md`'s 2026-08-27 and 2026-08-28 entries). `demo_predictions.py` and `predict_surf_count.py` both use this same shared fitter. |
| `demo_predictions.py` | Shows N random held-out predictions (point + 80% interval) alongside the actual count and main predictor conditions, for eyeballing model behavior rather than only trusting aggregate metrics. |
| `predict_surf_count.py` | Phase 3: pulls live forward-looking predictors (today+tomorrow, no auth token needed) via `get_surf_predictors.py`'s `build_predictor_map()`, trains production GBT quantile models (10th/50th/90th) plus a separate mean-based (Poisson loss) model on all of `training_features.csv`, and outputs a prediction with an 80% range for one or more times. Primary point estimate is the median model — guarantees it always falls inside its own range, unlike a mean-based estimate on this right-skewed target. The mean-based estimate is shown alongside it as a diagnostic (auto-flagged if the two diverge >25%) so growing skew stays visible as the dataset grows — see `PROJECT_HISTORY.md`'s 2026-08-28 entries. |
| `plot_daily_prediction.py` | Generates the daily prediction chart (median + 33%/66% bands, tide, wave-energy bars, weather-coded markers, night shading, model/detector info footer) to `data/charts/surfer_count_YYYY-MM-DD.png`, plus overwrites the git-tracked `data/charts/latest.png` + `latest_table.md` (hour → 33% range only) and rewrites the marked daily-chart section of `README.md`. Filters to each day's *real* dawn/dusk (via `get_clips.py`'s `get_light_window()`), not a fixed hour range — see `PROJECT_HISTORY.md`'s 2026-08-28 entry for the pre-dawn extrapolation bug this fixed. Run daily via `daily_chart.sh`. |
| `daily_chart.sh` | Cron wrapper for `plot_daily_prediction.py`, independent of `local_pipeline.sh`'s twice-weekly schedule — needs only the live forecast + existing model, not new clips. Also stages/commits/pushes `data/charts/latest.png` + `latest_table.md` + `README.md` if changed (non-fatal on git/network failure — see `PROJECT_HISTORY.md`'s 2026-08-28 entry). |
| `detect_surfers_v2.ipynb` | Earlier notebook version of the detection logic, superseded by `detect_surfers.py`. Kept for reference. |
| `__pycache__/` | Python bytecode cache. Gitignored, regenerated automatically. |

## `data/` — active pipeline output and training assets

| Path | Purpose |
|---|---|
| `predictions.csv` | Main dataset: one row per crop with `date, time_local, filename, surfer_count, confidence_avg, quality_ok, quality_reason, brightness, lap_var, human_count, frame_count_1, frame_count_2, frame_count_3, frame_count_mean, frame_count_stdev`. `surfer_count`/`confidence_avg` are blank when `quality_ok` is `False` (detection was skipped). `human_count` is manually filled in over time. `frame_count_*` hold the 3 per-frame counts multi-frame averaging is based on — see `HOW_IT_WORKS.md`. Not tracked in git. |
| `fog_review/` | Scratch review artifacts from the 2026-08 model-performance review (annotated image batches + CSVs used to hand-label usable/unusable frames and derive the image-quality gate's thresholds). Not tracked in git — see `PROJECT_HISTORY.md` for what each batch found. |
| `model_review_50/` | The original 50-image manual spot-check (`review_counts.csv`) that kicked off the 2026-08 model-performance review. Not tracked in git. |
| `surfline_predictors.csv` | Weather/rating/tide/swell/wind/energy/consistency predictors per `predictions.csv` row (matched by filename), from `get_surf_predictors.py` (forward-looking, ongoing), `backfill_historical_predictors.py`, and `backfill_predictors_from_har.py` (past dates, manual). As of 2026-08-27, every `predictions.csv` row has a matching predictor row — no known gap. |
| `openmeteo_weather.csv` | Real observed historical weather (Open-Meteo archive API — ERA5 reanalysis, not a forecast) per `predictions.csv` row, from `backfill_openmeteo_weather.py`. Separate from `surfline_predictors.csv`, non-destructive. `real_humidity_pct` is a validated proxy for fog/blur conditions — see `PROJECT_HISTORY.md`'s 2026-08-28 entry. |
| `charts/` | Daily prediction chart PNGs. Dated files (`surfer_count_YYYY-MM-DD.png`) are local-only/untracked, matching the rest of `data/`. `latest.png` + `latest_table.md` ARE git-tracked (overwritten, not accumulated, daily) — the exception to that convention, specifically so GitHub's rendered README can embed the current one. |
| `training_features.csv` | Output of `build_training_features.py` — joined predictions + predictors + derived time features, restricted to `quality_ok=True` rows. Input to `fit_surfer_count_model.py`. Not tracked in git. |
| `.clip_duration_override` | One-shot override file for `get_clips.py`'s clip duration (seconds, plain integer). Consumed and deleted the moment it's read, so it can never silently persist past a single run. Not tracked in git (ephemeral). |
| `external/` | **Gitignored, contains sensitive data.** HAR files used to discover the historical-predictor API mechanism — contain a real Surfline session auth token. Not tracked in git; review/delete when no longer needed. |
| `j_shore_cam/surf_crops/` | Cropped JPG frames produced by `get_cropped_frame.py` — 3 per clip (unsuffixed primary + `_side1`/`_side2`), the images `detect_surfers.py` runs inference on. |
| `not_needed_in_repo/surf_clips/` | Raw downloaded video clips from `get_clips.py`. Gitignored — not needed in the repo, only locally for cropping. |
| `model_out/20251013/` | Trained YOLOv8 weights and training/validation run logs. `train/runs/detect/train13/weights/best.pt` is the weights path `detect_surfers.py` uses by default. |
| `cvat_out_coco/` | CVAT annotation export (COCO format) used to train the current model — `splits/` is the untiled train/val/test split, `splits_tiled/` matches the 4-tile training setup. Same commit date as `model_out/20251013`. |
| `cvat_out_yolo_rebuilt/` | Same training data rebuilt into YOLO format (`images/`, `labels/`, `data.yaml`) for training. |
| `local_pipeline.log` | Cron run output (gitignored, `*.log`). |
| `.last_local_success` | UTC timestamp of the last successful pipeline run (gitignored). |

## `archive/` — one-offs kept for reference, not part of the active pipeline

Mirrors the original `data/` paths. Everything here predates the current
pipeline's naming convention and was superseded once `surf_crops` /
`surf_clips` became the standard output.

| Path | Purpose |
|---|---|
| `data/j_shore_cam/jacks_20250719_crops/`, `jacks_20250719_frames/` | Earliest manual test capture (crops + uncropped frames), before the pipeline existed. |
| `data/j_shore_cam/jacks_20250807-0811_crops/`, `jacks_20250810_crops/` | Later manual test captures, still pre-pipeline. |
| `data/not_needed_in_repo/jacks_20250719_clips/`, `jacks_20250807-0811_clips/`, `jacks_20250810_clips/` | Raw clips matching the folders above. Gitignored, same as the active clips folder. |
| `data/cvat_out_dataset_2025_09_24_coco 1.0.zip` | Earlier zipped CVAT COCO export, superseded by the unzipped `data/cvat_out_coco/`. |
| `data/cvat_out_yolo_detect_1.0.zip` | Earlier zipped CVAT YOLO export, superseded by the unzipped `data/cvat_out_yolo_rebuilt/`. |
| `data/tide_backfill.csv` | Output of `code/backfill_tide.py` — NOAA-based historical tide estimates, superseded by Surfline data going forward. |
