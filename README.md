# Sum Surfers

Automated surfer counting from Surfline clips.

This project downloads short clips around daylight hours, extracts a cropped frame per clip, runs YOLOv8 inference on tiled images, and stores per-frame surfer counts.

## What This Repo Does

1. Pulls Surfline clips into dated folders.
2. Extracts one ROI frame from each clip.
3. Runs tiled YOLOv8 inference and deduplicates boxes across tile boundaries.
4. Appends predictions to `data/predictions.csv`.

Runs entirely on a local machine via cron — no cloud VM involved.

## Pipeline Scripts

- `code/local_pipeline.sh`
  - The entry point. Downloads clips, extracts crops, checks local clip
    storage, runs detection, records a success timestamp.
- `code/get_clips.py`
  - Downloads clips between sunrise-30 minutes and sunset+30 minutes.
  - Uses the nearest Surfline clip windows and can backfill up to the previous 5 days.
- `code/get_cropped_frame.py`
  - Reads downloaded clips and saves cropped JPG frames.
- `code/detect_surfers.py`
  - Runs YOLO on 4 horizontal overlapping tiles and writes counts.
- `code/manage_clips.py`
  - Emails a warning if local clip storage exceeds `CLIPS_DIR_LIMIT_GB`.
- `code/send_email.py`
  - Shared Gmail SMTP sender used for storage warnings.

## Schedule

Runs twice a week via cron (Tue/Thu):

```cron
0 19 * * 2,4 caffeinate -i /path/to/sum-surfers/code/local_pipeline.sh >> /path/to/sum-surfers/data/local_pipeline.log 2>&1
```

`caffeinate -i` keeps the laptop awake for the run; if the laptop is asleep or off at the scheduled time, the run is skipped.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Add secrets to `.env`.
4. Run the pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests astral pytz opencv-python-headless ultralytics torch torchvision
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

## Data Locations

- Clips: `data/not_needed_in_repo/surf_clips`
- Crops: `data/j_shore_cam/surf_crops`
- Predictions: `data/predictions.csv`
- Model weights default:
  `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`

## Notes

- `.env` is ignored by git and should never be committed.
- macOS: cron requires Full Disk Access for `/usr/sbin/cron`, **and** for the
  actual Python interpreter binary your `.venv` resolves to (two separate
  grants under System Settings → Privacy & Security → Full Disk Access).
- This project previously ran on a GCP VM (and briefly a hybrid laptop+VM
  split). That's gone as of 2026-07-24 — detection ran on CPU either way, so
  the cloud hop added cost with no benefit.
