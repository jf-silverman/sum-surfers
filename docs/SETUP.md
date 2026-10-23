# Setup

How to run the project yourself, from a one-off look at the live camera to the
full nightly pipeline. The [README](../README.md) has the short version.

## Run it yourself

**You can build a dataset like this one yourself, starting today.** Everything
the forecast runs on is publicly reachable: the camera's live stream is open,
and all eight Surfline forecast endpoints this project reads — weather, rating,
tides, surf, swells, wind, energy, consistency — answer anonymous requests
seven days forward at one row per hour (re-verified 2026-09-26), as does the
observed-weather backfill from Open-Meteo.

That means a machine which stays awake can run this entire pipeline end to end:
watch the stream, count the surfers, collect the matching conditions, build the
feature table, fit the model, draw the charts. Nobody can reconstruct a history
for a spot they were not already watching — the camera only reaches back five
days, today included — so a record like this one is not downloaded, it is
accumulated. For scale, this repo's stands at roughly 1,835 counted hours across
138 days as of 2026-10-08, gathered since October 2025.

### Collect live data

`watch_live.py` needs no account, no token, and no environment variables.

While it is running, it records a few seconds off the camera's public live
stream once every few minutes during daylight, cuts three frames from that
clip the same way the scheduled pipeline does, crops each to the region the
detector was trained on, counts surfers with the [trained model weights
committed to this repo](../data/model_out/20260921_fog/train/weights),
and appends the averaged count to its own file — `data/live_watch/live_predictions.csv`,
with the same columns as the main predictions file, kept separate so a demo
run can never mix into the real dataset.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt

python code/watch_live.py --once   # grab one frame and count it
python code/watch_live.py          # keep collecting until you stop it
```

Run it after dark and it will say so and ask whether to wait for first light,
rather than collecting frames it cannot count. Answer no and it exits; add
`--ignore-daylight` to grab a night frame anyway and watch the quality gate
reject it.

It needs [ffmpeg](https://ffmpeg.org/) on your path to read the stream
(`brew install ffmpeg` or `apt install ffmpeg`).

The one real constraint is uptime: a live stream has no rewind, so this collects
only while your computer is awake and the script is running. On an always-on
desktop that costs nothing. On a laptop that travels and gets closed, every hour
the lid is down is an hour missed.

### Try the detector on a demo set

`data/demo/` holds **20 labeled frames chosen to cover the conditions that
matter here**: fog, sun glare with and without surfers in it, winter,
empty water, crowds from 0 to 56 surfers, and light from early morning to
evening, across May to December. None of them was used to train the
detector, so this is a fair test rather than a replay of its training data.

```bash
python code/eval_detector.py --coco-dir data/demo --split test --skip-tile-metrics
```

It runs the real production inference path — tiling, cross-tile
[NMS](HOW_IT_WORKS.md#term-nms), false-positive filtering — on each
frame and compares the surfer count with the number of hand-drawn boxes,
frame by frame and broken out by crowd size. On the demo set it finds
**491 surfers against 498 labeled**, an average error of 0.85 surfers per
frame. That count is the number the forecast actually uses; a detector can
look fine box-by-box while still undercounting crowded frames.

The weights in this repo are the production detector, retrained in September
2026 on labeled fog, glare and pose data. The evaluator will also run on the
older labeled splits in `data/cvat_out_coco/splits/`, but those predate the
retrain and the new model trained on some of their frames, so it skips those
frames automatically (they are listed in
`data/model_out/20260921_fog/train_filenames.txt`). Per-box precision and
recall from Ultralytics' own validator can't skip them, so treat those as
optimistic. The demo set is the clean check.

### Check the forecast model against held-out data

`data/model_release/` holds the fitted forecast model, the **417 rows it was
never trained on** (the most recent days, held out as a block), and the metadata to reproduce the split:

```bash
python code/eval_surf_count_model.py
```

This ships the fitted model rather than only a table of predictions on
purpose. A predictions-versus-actuals file is self-reported — you can
recompute the error from it, but not check that the model was not fit on
those same rows. With the model included you can run it yourself on rows it
never saw; the script re-predicts from the raw predictor values and verifies
it reproduces the shipped numbers before reporting anything. The 1,242
training rows stay unpublished.

It reports MAE, RMSE and bias (broken out by how crowded the day actually
was, which is where the model's weakness shows), and
[prediction-interval](HOW_IT_WORKS.md#term-prediction-interval) coverage
with the two tail-miss rates kept separate.

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

## Environment Variables

**Nothing here is needed to try the project.** Collecting from the live stream,
running the detector on the demo frames and checking the forecast model against
the held-out set all read no environment variables at all — no `.env`, no
account, no token. See [Run it yourself](#run-it-yourself).

Variables only matter once you want more than that. Copy `.env.example` to
`.env` and fill in whichever apply:

**Optional, everywhere:**

- `MODEL_PATH` (override the default YOLO weights path)
- `CLIPS_DIR_LIMIT_GB` (local clip storage warning threshold, default 2.0)
- `DETECT_MODE` / `DETECT_RECENT_DAYS` / `DETECT_START_DATE` (detection scope)
- `SURFLINE_HISTORICAL_TOKEN` (optional override for
  `backfill_historical_predictors.py`, never read by the scheduled pipeline.
  There is no separate historical credential — that script falls back to
  `SURFLINE_ACCESS_TOKEN`, which is the same token, so this only exists to
  point one manual backfill at a different session)

**For the clip-download path only** (`get_clips.py`, and therefore
`local_pipeline.sh`):

- `SURFLINE_CAMERA_ID`
- `SURFLINE_ACCESS_TOKEN`

This pair is what buys the *recent past*. The live stream is public and free, but
it only gives you the moment you are watching — close your laptop and those
hours are gone. Reaching back to pull the clips you missed uses Surfline's
recorded-clip endpoint, and that **needs a paid subscription**. Even then the
window is short: the camera keeps five days including today, so the furthest
back anyone can reach is four days, subscription or not.

Leave both empty and everything else still works; the clip step raises a message
naming what is missing, and only when it actually runs — the check is inside
`clip_api_url()`, not at import, so nothing else in the project is blocked by
their absence.

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
- Model weights default (the September 2026 retrain):
  `data/model_out/20260921_fog/train/weights/best.pt`. The October 2025
  model it replaced is at `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`.

## Schedule

The whole pipeline runs as one job every night after last light
(`code/local_pipeline.sh`): it downloads the day's clips, counts the surfers,
pulls the conditions, rebuilds the training table, refits the forecast where
needed and republishes the README's charts. Its last step,
`code/daily_chart.sh`, builds the charts and commits the README. Both are safe
to run by hand at any time.

