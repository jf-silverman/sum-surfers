# Sum Surfers

Automated surfer counting from Surfline clips.

This project downloads short clips around daylight hours, extracts a cropped frame per clip, runs YOLOv8 inference on tiled images, and stores per-frame surfer counts.

## What This Repo Does

1. Pulls Surfline clips into dated folders.
2. Extracts one ROI frame from each clip.
3. Runs tiled YOLOv8 inference and deduplicates boxes across tile boundaries.
4. Appends predictions to `data/predictions.csv`.

## Pipeline Scripts

- `code/get_clips.py`
  - Downloads clips between sunrise-30 minutes and sunset+30 minutes.
  - Uses the nearest Surfline clip windows and can backfill up to the previous 5 days.
- `code/get_cropped_frame.py`
  - Reads downloaded clips and saves cropped JPG frames.
- `code/detect_surfers.py`
  - Runs YOLO on 4 horizontal overlapping tiles and writes counts.
- `run_pipeline.sh`
  - Runs all three steps in order and powers off the VM when done.

## Schedule

Recommended cadence is every 3 days to reduce compute/storage costs while still backfilling recent data.

Cron expression used by setup:

```cron
0 20 */3 * * /home/surfer/sum-surfers/run_pipeline.sh >> /var/log/surfers.log 2>&1
```

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
bash run_pipeline.sh
```

## Required Environment Variables

Create `.env` from `.env.example` and set:

- `SURFLINE_CAMERA_ID`
- `SURFLINE_ACCESS_TOKEN`

Optional:

- `MODEL_PATH` (override default YOLO weights path)

## Data Locations

- Clips: `data/not_needed_in_repo/surf_clips`
- Crops: `data/j_shore_cam/surf_crops`
- Predictions: `data/predictions.csv`
- Model weights default:
  `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`

## GCP Deployment

Use the guide in `GCP_SETUP.md`.

High-level flow:

1. Create VM.
2. Copy repo, model weights, and `.env`.
3. Run `setup_vm.sh` once.
4. Configure Cloud Scheduler to start the VM every 3 days.

## Notes

- `.env` is ignored by git and should never be committed.
- `run_pipeline.sh` shuts the VM down at the end by default.
- If you want to test without shutdown, comment out `sudo shutdown -h now` in `run_pipeline.sh`.
