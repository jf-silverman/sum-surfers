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
| `get_clips.py` | Downloads Surfline clips for the camera between sunrise-30min and sunset+30min; backfills up to `CLIP_LOOKBACK_DAYS`. |
| `get_cropped_frame.py` | Extracts one ROI-cropped frame per downloaded clip. |
| `detect_surfers.py` | Checks each crop's image quality (brightness + Laplacian variance) before detection, skipping frames too dark/foggy to reliably count; runs tiled YOLOv8 inference on the rest, de-duplicates boxes via NMS, filters out known static false positives (tree bough, flag), and appends results to `data/predictions.csv`. |
| `get_surf_predictors.py` | Pulls weather, condition rating, tide, and swell data for Jack's from Surfline's forecast API; appends to `data/surfline_predictors.csv`. Forward-looking only — no historical backfill. |
| `manage_clips.py` | Local clip-storage manager. `--check` (used by cron) emails a warning past `CLIPS_DIR_LIMIT_GB`; interactive mode offers deletion by age. |
| `send_email.py` | Shared Gmail SMTP + App Password sender used for storage-warning emails. |
| `backfill_tide.py` | One-off (not part of the scheduled pipeline): estimated historical tide height via NOAA hi/lo + cosine interpolation, before the Surfline tide endpoint was found. Its output (`tide_backfill.csv`) has been archived; kept here for reference. |
| `detect_surfers_v2.ipynb` | Earlier notebook version of the detection logic, superseded by `detect_surfers.py`. Kept for reference. |
| `__pycache__/` | Python bytecode cache. Gitignored, regenerated automatically. |

## `data/` — active pipeline output and training assets

| Path | Purpose |
|---|---|
| `predictions.csv` | Main dataset: one row per crop with `date, time_local, filename, surfer_count, confidence_avg, quality_ok, quality_reason, brightness, lap_var`. `surfer_count`/`confidence_avg` are blank when `quality_ok` is `False` (detection was skipped — see `HOW_IT_WORKS.md`). Not tracked in git. |
| `fog_review/` | Scratch review artifacts from the 2026-08 model-performance review (annotated image batches + CSVs used to hand-label usable/unusable frames and derive the image-quality gate's thresholds). Not tracked in git — see `PROJECT_HISTORY.md` for what each batch found. |
| `model_review_50/` | The original 50-image manual spot-check (`review_counts.csv`) that kicked off the 2026-08 model-performance review. Not tracked in git. |
| `surfline_predictors.csv` | Weather/rating/tide/swell predictors per `predictions.csv` row (matched by filename), from `get_surf_predictors.py`. |
| `j_shore_cam/surf_crops/` | Cropped JPG frames produced by `get_cropped_frame.py` — the images `detect_surfers.py` runs inference on. |
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
