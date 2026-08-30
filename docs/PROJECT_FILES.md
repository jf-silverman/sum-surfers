# Project File Descriptions

A map of what every file and folder in this repo is for. See `README.md`
for how to run the pipeline and `CLAUDE.md` (local-only, gitignored) for
deeper architecture notes.

## Root

| Path | Purpose |
|---|---|
| `README.md` | Setup and usage instructions. Stays at repo root (rather than `docs/`) so GitHub auto-renders it on the repo homepage. |
| `CLAUDE.md` | Local-only architecture notes for Claude Code (gitignored, never committed). |
| `.env` | Secrets and config (gitignored). Copy from `.env.example`. |
| `.env.example` | Template for `.env` — required and optional variables. |
| `.gitignore` | Excludes secrets, OS files, caches, large clip video, and `archive/`'s raw clips. |

## `docs/` — project documentation

Moved out of the repo root 2026-08-28 to keep the root uncluttered (`README.md`
stays at root so GitHub still renders it on the repo homepage).

| Path | Purpose |
|---|---|
| `docs/HOW_IT_WORKS.md` | Plain-language walkthrough of the detection pipeline (capture → crop → quality gate → tile → detect → filter → count), plus a glossary of CV/ML terms used across this repo. |
| `docs/PROJECT_HISTORY.md` | Chronological record of how the project was built and tuned. |
| `docs/bugs.md` | Known defects/false-positive-prone behaviors and their open/resolved status (gitignored — split out of `PROJECT_HISTORY.md`'s old "Open engineering leads" section 2026-08-28; still recoverable from repo history). |
| `docs/model_and_feature_ideas.md` | Enhancements/possible future work that aren't bugs, split into "Model Ideas" (prediction model/predictors) and "Feature Ideas" (detection pipeline) subsections (gitignored — same split as `bugs.md`; renamed from `feature_ideas.md` 2026-08-28). |
| `docs/misc_notes.md` | Small operational notes (env/cron/GCP-history caveats), split out of `README.md`'s old "Notes" section (gitignored, 2026-08-28). |
| `docs/PROJECT_FILES.md` | This file. |
| `docs/requirements.txt` | Pinned Python dependencies (added 2026-08-28 after a security review flagged there was previously no way to reproduce/audit the environment). Install with `pip install -r docs/requirements.txt`. |

## `code/` — pipeline scripts

| Path | Purpose |
|---|---|
| `local_pipeline.sh` | Entry point, run via cron. Chains all six steps below and records a success timestamp. |
| `get_clips.py` | Downloads Surfline clips for the camera (Pleasure Point, Santa Cruz — coordinates corrected 2026-08-28, see `PROJECT_HISTORY.md`) between real dawn and dusk (civil twilight, via `get_light_window()` — prefers Surfline's own live `sunlight` forecast for today, falls back to astral for backfill days); backfills up to `CLIP_LOOKBACK_DAYS`. Emails a warning (via `send_email.py`) if any download fails with HTTP 401 (expired `SURFLINE_ACCESS_TOKEN`). Supports a one-shot clip-duration override via `data/.clip_duration_override` (consumed/deleted on read, so it can never silently persist past one run). |
| `get_cropped_frame.py` | Extracts 3 ROI-cropped frames per downloaded clip (a primary frame at 2.5s, plus 2 "side" frames 1.5s before/after) for multi-frame count averaging — see `HOW_IT_WORKS.md`. |
| `detect_surfers.py` | Checks each crop's image quality (brightness + Laplacian variance) before detection, skipping frames too dark/foggy to reliably count; runs tiled YOLOv8 inference on all 3 frames, de-duplicates boxes via NMS, filters out known static false positives (tree bough, flag), averages the 3 per-frame counts, and appends results to `data/predictions/predictions.csv`. Also exposes `run_inference_with_boxes()` — same pipeline, returns the surviving boxes instead of discarding them, used by `plot_daily_prediction.py`'s detection-review image. |
| `get_surf_predictors.py` | Pulls weather, rating, tide, swell, wind, wave-energy, and consistency data for Jack's from Surfline's forecast API; appends to `data/predictor_vars/surfline_predictors.csv`. Forward-looking only (today + tomorrow), runs every scheduled pipeline execution. `build_predictor_map()` (shared with `predict_surf_count.py`) also merges in live `real_temperature_f`/`real_humidity_pct`/`real_cloud_cover_pct`/`real_weather_code`/`real_pressure_mb` from Open-Meteo's forecast API — added 2026-08-28, see `PROJECT_HISTORY.md`. |
| `manage_clips.py` | Local clip-storage manager. `--check` (used by cron) emails a warning past `CLIPS_DIR_LIMIT_GB`; interactive mode offers deletion by age. |
| `send_email.py` | Shared Gmail SMTP + App Password sender used for storage-warning emails. |
| `backfill_tide.py` | One-off (not part of the scheduled pipeline): estimated historical tide height via NOAA hi/lo + cosine interpolation, before the Surfline tide endpoint was found. Its output (`tide_backfill.csv`) has been archived; kept here for reference. |
| `backfill_historical_predictors.py` | One-off, manually-run (not part of the scheduled pipeline): backfills the same predictor fields as `get_surf_predictors.py` for past dates in `predictions.csv`, using Surfline's `start=YYYY-MM-DD` param and a personal, premium-account session token (never committed — see the script's own docstring). Jittered pauses between requests to avoid bot detection; run a `--dry-run` first to preview. |
| `backfill_multiframe_counts.py` | One-off, manually-run: for `predictions.csv` rows whose raw clip is still on disk, extracts any missing side frames and fills in `frame_count_*` columns — never touches the original `surfer_count`/`confidence_avg`. Local/compute-only (no network, no rate limits). |
| `backfill_predictors_from_har.py` | One-off, manually-run: parses HAR files exported from clicking through Surfline's own Historical view (an alternative to the token-based script's live requests). Filters each response down to only the day matching its own `start=` param (day-of only, not the multi-day forecast window the API actually returns), with a narrow 1-day-out fallback for dates that were never directly clickable. Cannot get `weather_condition`/`temperature_f`/`pressure_mb`/`consistency_wave_count` — the Historical view never calls those endpoints. |
| `backfill_openmeteo_weather.py` | Adds real observed historical weather (Open-Meteo archive API, free/no-auth) for every `predictions.csv` row to `data/predictor_vars/openmeteo_weather.csv` — separate from `surfline_predictors.csv`, non-destructive. Includes `real_humidity_pct`, a validated proxy for the quality gate's fog/blur detections. |
| `build_training_features.py` | Phase 1 of the surfer-count modeling plan: joins `predictions.csv` (target) with `surfline_predictors.csv` and `openmeteo_weather.csv` (features) on filename, restricted to `quality_ok=True` rows, adds `hour_local`/`day_of_week`/`is_weekend`/`month`. Target uses `round(frame_count_mean)` (multi-frame average) wherever available, falling back to the legacy single-frame `surfer_count` otherwise — see `resolve_target_count()` and the `used_multiframe` output column. Also derives `weather_simple` (CLEAR/CLOUDY_OVERCAST/RAIN/FOG, merged from Surfline's 21 raw categories) and `is_night` via `simplify_weather_condition()`, shared with `predict_surf_count.py` for live forecasts. Writes `data/training_features.csv`. |
| `fit_surfer_count_model.py` | Phase 2: fits and compares Poisson GLM, negative-binomial GLM (MLE-estimated dispersion), and gradient-boosted trees on `data/training_features.csv`, plus GBT quantile-regression prediction intervals via `fit_quantile_model_robust()` — a self-checking fitter that escalates `min_samples_leaf` until the quantile model's predictions actually vary, since no fixed hyperparameter combo has proven safe against silent collapse to a constant prediction (found and re-found — see `PROJECT_HISTORY.md`'s 2026-08-27 and 2026-08-28 entries). `demo_predictions.py` and `predict_surf_count.py` both use this same shared fitter. |
| `demo_predictions.py` | Shows N random held-out predictions (point + 80% interval) alongside the actual count and main predictor conditions, for eyeballing model behavior rather than only trusting aggregate metrics. |
| `predict_surf_count.py` | Phase 3: pulls live forward-looking predictors (today+tomorrow, no auth token needed) via `get_surf_predictors.py`'s `build_predictor_map()`, trains production GBT quantile models (10th/50th/90th) plus a separate mean-based (Poisson loss) model on all of `training_features.csv`, and outputs a prediction with an 80% range for one or more times. Primary point estimate is the median model — guarantees it always falls inside its own range, unlike a mean-based estimate on this right-skewed target. The mean-based estimate is shown alongside it as a diagnostic (auto-flagged if the two diverge >25%) so growing skew stays visible as the dataset grows — see `PROJECT_HISTORY.md`'s 2026-08-28 entries. |
| `plot_daily_prediction.py` | Generates the daily prediction chart (median + 33%/66% bands rendered with a side-by-side 33%-range table in the same image, tide, wave-energy bars, weather-coded markers, night shading, model/detector info footer) to `data/charts/surfer_count_YYYY-MM-DD.png` + git-tracked `data/charts/latest.png`. Also generates a detection-review image (real boxes/confidence labels on the day's ~8am crop via `detect_surfers.run_inference_with_boxes()`, with the predicted range/median overlaid) to `data/charts/latest_detection.png`. Rewrites the marked daily-chart section of `README.md` with both. Filters to each day's *real* dawn/dusk (via `get_clips.py`'s `get_light_window()`), not a fixed hour range — see `PROJECT_HISTORY.md`'s 2026-08-28 entries. Run daily via `daily_chart.sh`. |
| `daily_chart.sh` | Cron wrapper for `plot_daily_prediction.py`, independent of `local_pipeline.sh`'s twice-weekly schedule — needs only the live forecast + existing model, not new clips. Also stages/commits/pushes `data/charts/latest.png` + `latest_detection.png` + `README.md` individually if changed (non-fatal on git/network failure, and a missing detection image can't block the others — see `PROJECT_HISTORY.md`'s 2026-08-28 entries). |
| `train_model.py` | Rebuilds the surfer-detection YOLOv8 model from a CVAT COCO export: tiles images/annotations to match production inference tiling exactly (imports `TILE_W`/`TILE_H`/`STEP_X`/`NUM_TILES` directly from `detect_surfers.py` so the two can't drift apart), converts to YOLO format, fine-tunes from a COCO-pretrained checkpoint, and validates on the held-out val split with real, printed precision/recall/mAP. Added 2026-08-29 to replace `detect_surfers_v2.ipynb` as the retraining path — rerun this (not the notebook) whenever there's a new or larger CVAT export. Each run writes to fresh, dated output folders (`data/cvat_out_coco/splits_tiled_<run-name>/`, `data/cvat_out_yolo_rebuilt_<run-name>/`, `data/model_out/<run-name>/`) rather than overwriting the current production model — nothing production-facing changes until `MODEL_PATH` is manually pointed at the new weights. See its own docstring (`python code/train_model.py --help`) for usage. |
| `detect_surfers_v2.ipynb` | The original YOLOv8 model training notebook that produced the current `model_out/20251013/` weights — superseded by `train_model.py` as the retraining path (2026-08-29), kept for its historical training log/plots (real epoch losses, model summary, final metrics — do not re-run cells in place, it would overwrite that record). Its early exploratory-detection-logic cells also predate and are superseded by `detect_surfers.py` (the production inference pipeline). |
| `__pycache__/` | Python bytecode cache. Gitignored, regenerated automatically. |

## `data/` — active pipeline output and training assets

Reorganized 2026-08-29 into `predictions/`, `predictor_vars/`, and
`reviews/` subfolders (previously flat CSVs and separately-named review
folders directly under `data/`) — see `PROJECT_HISTORY.md`'s 2026-08-29
entry for the full before/after mapping.

| Path | Purpose |
|---|---|
| `predictions/predictions.csv` | Main dataset: one row per crop with `date, time_local, filename, surfer_count, confidence_avg, quality_ok, quality_reason, brightness, lap_var, human_count, frame_count_1, frame_count_2, frame_count_3, frame_count_mean, frame_count_stdev`. `surfer_count`/`confidence_avg` are blank when `quality_ok` is `False` (detection was skipped). `human_count` is manually filled in over time. `frame_count_*` hold the 3 per-frame counts multi-frame averaging is based on — see `HOW_IT_WORKS.md`. Not tracked in git. |
| `predictor_vars/surfline_predictors.csv` | Weather/rating/tide/swell/wind/energy/consistency predictors per `predictions.csv` row (matched by filename), from `get_surf_predictors.py` (forward-looking, ongoing), `backfill_historical_predictors.py`, and `backfill_predictors_from_har.py` (past dates, manual). As of 2026-08-27, every `predictions.csv` row has a matching predictor row — no known gap. |
| `predictor_vars/openmeteo_weather.csv` | Real observed historical weather (Open-Meteo archive API — ERA5 reanalysis, not a forecast) per `predictions.csv` row, from `backfill_openmeteo_weather.py`. Separate from `surfline_predictors.csv`, non-destructive. `real_humidity_pct` is a validated proxy for fog/blur conditions — see `PROJECT_HISTORY.md`'s 2026-08-28 entry. |
| `reviews/count_60sec_var/` | Human-review dataset (in progress): 8 sets x 10 images spanning the `analysis/frame_timing_variability/` clips, with `review_counts.csv` to fill in hand counts against — will validate whether wider frame spacing is actually more accurate, not just less noisy. Not tracked in git. Renamed from `data/count_review/`. |
| `reviews/fog_quality/` | Review artifacts from the 2026-08 model-performance review (annotated image batches + CSVs used to hand-label usable/unusable frames and derive the image-quality gate's thresholds). Not tracked in git — see `PROJECT_HISTORY.md` for what each batch found. Renamed from `data/fog_review/`. |
| `reviews/model_spotcheck_50/` | The original 50-image manual spot-check (`review_counts.csv`) that kicked off the 2026-08 model-performance review. Not tracked in git. Renamed from `data/model_review_50/`. |
| `charts/` | Daily prediction chart + detection-review image PNGs. Dated files (`surfer_count_YYYY-MM-DD.png`, `detection_YYYY-MM-DD.png`) are local-only/untracked, matching the rest of `data/`. `latest.png` + `latest_detection.png` ARE git-tracked (overwritten, not accumulated, daily) — the exception to that convention, specifically so GitHub's rendered README can embed the current ones. |
| `training_features.csv` | Output of `build_training_features.py` — joined predictions + predictors + derived time features, restricted to `quality_ok=True` rows. Input to `fit_surfer_count_model.py`. Stays directly under `data/` (a derived join of the two folders above, not raw source data). Not tracked in git. |
| `.clip_duration_override` | One-shot override file for `get_clips.py`'s clip duration (seconds, plain integer). Consumed and deleted the moment it's read, so it can never silently persist past a single run. Not tracked in git (ephemeral). |
| `external/` | **Gitignored, contains sensitive data.** HAR files used to discover the historical-predictor API mechanism — contain a real Surfline session auth token. Not tracked in git; review/delete when no longer needed. |
| `j_shore_cam/surf_crops/` | Cropped JPG frames produced by `get_cropped_frame.py` — 3 per clip (unsuffixed primary + `_side1`/`_side2`), the images `detect_surfers.py` runs inference on. |
| `not_needed_in_repo/surf_clips/` | Raw downloaded video clips from `get_clips.py`. Gitignored — not needed in the repo, only locally for cropping. |
| `model_out/20251013/` | Trained YOLOv8 weights and training/validation run logs. `train/runs/detect/train13/weights/best.pt` is the weights path `detect_surfers.py` uses by default. `train/yolov8s.pt` (22.5MB, verified 2026-08-29) is the COCO-pretrained base checkpoint training started from — despite `detect_surfers_v2.ipynb` loading it via the path `../data/model_out/yolov8s.pt`, it actually landed one level deeper, under `train/`. To rebuild the model, point at this file (or let `ultralytics` auto-download a fresh copy if it's ever missing). |
| `cvat_out_coco/` | CVAT annotation export (COCO format) used to train the current model — `splits/` is the untiled train/val/test split, `splits_tiled/` matches the 4-tile training setup. Same commit date as `model_out/20251013`. |
| `cvat_out_yolo_rebuilt/` | Same training data rebuilt into YOLO format (`images/`, `labels/`, `data.yaml`) for training. |
| `local_pipeline.log` | Cron run output (gitignored, `*.log`). |
| `.last_local_success` | UTC timestamp of the last successful pipeline run (gitignored). |

## `analysis/` — one-off investigations

New top-level folder 2026-08-29 (previously scattered as loose CSVs/PNGs
directly under `data/` plus scripts under `code/`). Each subfolder is a
self-contained one-off analysis, not part of the scheduled pipeline: its
own CSVs, charts, and the script(s) that produced them, so a future
investigation like this can be found and rerun as a unit.

| Path | Purpose |
|---|---|
| `frame_timing_variability/` | 2026-08-27 study of detector-count variability at different second-to-second lags and averaging-window widths, using 1-frame-per-second extraction (quality-gated + detected) across 14 real 60-second clips spanning dawn to dusk. `analyze_frame_timing.py` extracts+detects (writes `frame_variability_analysis.csv`); `summarize_frame_variability.py` computes the lag/averaging-window analysis (writes `frame_variability_analysis_summary.png`); `plot_hourly_counts_strip.py` charts every per-second count by clip start time (`hourly_60sec_counts_2026-08-27.png`). Real numbers only, no fabricated stats — see `PROJECT_HISTORY.md`'s 2026-08-27/28 entries. |
| `hourly_variability_8to9am/` | 2026-08-28 study of within-hour crowd variability, using 12 back-to-back 5-minute clips (8-9am). `pull_hourly_variability_clips.py` downloads the clips (to `data/not_needed_in_repo/hourly_variability/`, unmoved); `analyze_hourly_variability.py` samples 5 frames/clip (sparse, `hourly_variability_analysis.csv`); `analyze_hourly_variability_full.py` extracts+detects 1 frame/second across the full hour (~3600 frames, "60x60", full production quality gate — `hourly_variability_full.csv`); `plot_hourly_full_strip.py` charts the full-density per-second counts with mean ± std error (`hourly_full_counts_2026-08-28.png`). |
| `weekday_weekend_patterns/` | Weekday-vs-weekend surfer-count comparison, reading `data/training_features.csv` (not copied here — it's shared production data). `plot_daily_counts.py` regenerates a monthly grouped bar chart and a KDE distribution, both dark-themed to match the daily prediction chart; the two PNGs ARE git-tracked, embedded in `README.md`'s Exploratory Findings section. |

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
