# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Automated surfer counting from a Surfline camera. Downloads short clips around
daylight hours, extracts one cropped frame per clip, runs a tiled YOLOv8 model
over each frame, and appends per-frame surfer counts to `data/predictions.csv`.

## Current architecture: hybrid local + VM pipeline

The README describes the original single-machine/VM design, but the pipeline
actually in use (as of the "Hybrid pipeline" commit) splits work between the
laptop and a GCP VM to avoid leaving the VM running:

- **`code/local_pipeline.sh`** — runs on the laptop via cron. Downloads clips,
  extracts crops, checks local clip storage, starts the VM, uploads new crops,
  SSHes in to run detection (`vm_pipeline.sh --no-shutdown`), pulls
  `predictions.csv` back, records a success timestamp
  (`data/.last_local_success`), syncs that timestamp to the VM, then stops the
  VM. This is the primary entry point for scheduled runs — not
  `run_pipeline.sh`.
- **`code/vm_pipeline.sh`** — runs on the VM. Two callers:
  - Local pipeline (`--no-shutdown`): just runs detection, no shutdown (caller
    pulls results and stops the VM itself).
  - Cloud Scheduler startup script (backup path, no flag): checks for new
    crops via `vm_check.py`; if none and it's been ≥
    `REMINDER_THRESHOLD_DAYS` (default 4) since the last successful local
    run, emails a reminder that the laptop pipeline didn't fire; otherwise
    runs detection and shuts down. Also checks total VM data size against
    `VM_DATA_LIMIT_GB` and emails a warning if exceeded.
- **`code/vm_check.py`** — prints `<new_crop_count> <days_since_last_local_success>`
  by diffing crops in `data/j_shore_cam/surf_crops` against filenames already
  present in `predictions.csv`, and reading `data/.last_local_success`.
- **`code/manage_clips.py`** — local clip storage manager. `--check` mode
  (used by cron) emails a warning if `data/not_needed_in_repo/surf_clips`
  exceeds `CLIPS_DIR_LIMIT_GB`; interactive mode offers to delete oldest
  month / oldest 3 months / everything older than 12 months.
- **`code/send_email.py`** — shared Gmail SMTP + App Password sender used by
  the reminder and storage-warning emails.
- **`run_pipeline.sh`** — older, VM-only, self-contained pipeline
  (download → crop → detect → shut down). Still present and functional but
  superseded by the local/VM split above for the live schedule.

### Core detection scripts (used by both pipelines)

- `code/get_clips.py` — downloads Surfline clips between sunrise-30min and
  sunset+30min, snapping to the nearest ~9-minute Surfline clip window;
  backfills up to `CLIP_LOOKBACK_DAYS` (default 5) days. Camera lat/long and
  timezone are hardcoded at the top of the file (currently a Southern
  California / `America/Los_Angeles` location) — update there if the camera
  changes.
- `code/get_cropped_frame.py` — reads each downloaded clip, grabs the frame at
  `FRAME_TIME_SEC` (2.5s in), crops to the hardcoded ROI
  (`ROI_X, ROI_Y, ROI_W, ROI_H`), and writes a JPG. Skips frames whose output
  already exists; logs skips to `surf_crops/skipped_frames.log`.
- `code/detect_surfers.py` — splits each crop into 4 horizontal overlapping
  tiles (matches the training tiling: 376×180px tiles, 20% overlap), runs
  YOLOv8, and de-duplicates boxes across tile boundaries via NMS before
  appending a row to `data/predictions.csv`. `DETECT_MODE` controls scope:
  `recent` (default, last `DETECT_RECENT_DAYS` days), `all`, or
  `start-date` (via `DETECT_START_DATE`) — the `recent` default exists
  specifically so a fresh machine doesn't try to backfill the entire crop
  history.

## Data locations

- Clips (gitignored, not needed in repo): `data/not_needed_in_repo/surf_clips`
- Crops: `data/j_shore_cam/surf_crops`
- Predictions: `data/predictions.csv`
- Model weights (default): `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`
- Success/state markers: `data/.last_local_success`, `data/.last_vm_success`
  (legacy: `data/.last_local_run`)

## Environment variables (`.env`, gitignored)

Required: `SURFLINE_CAMERA_ID`, `SURFLINE_ACCESS_TOKEN`.

Notable optional ones (see `.env.example` for the full list):
`MODEL_PATH`, `SMTP_USER`/`SMTP_APP_PASSWORD`/`EMAIL_TO` (Gmail App Password,
not OAuth — reverted from an earlier attempt at a different email method),
`CLIPS_DIR_LIMIT_GB`, `VM_DATA_LIMIT_GB`, `GCP_PROJECT`/`GCP_ZONE`/`GCP_INSTANCE`,
`DETECT_MODE`/`DETECT_RECENT_DAYS`/`DETECT_START_DATE`,
`REMINDER_THRESHOLD_DAYS`.

## Running locally

```bash
source .venv/bin/activate
bash code/local_pipeline.sh
```

Requires `gcloud` on `PATH` and configured for the target project (the script
adds common install locations to `PATH` itself since cron's `PATH` is
minimal).

## GCP deployment

See `GCP_SETUP.md`. VM setup is via `setup_vm.sh`; Cloud Scheduler is the
backup/reminder trigger, not the primary schedule driver — the laptop cron
job running `local_pipeline.sh` is.
