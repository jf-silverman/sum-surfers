#!/usr/bin/env bash
# local_pipeline.sh — Runs on your laptop nightly via cron.
#
# What it does:
#   1. Downloads new Surfline clips locally
#   2. Extracts crop frames locally
#   3. Checks local clips storage (emails warning if > CLIPS_DIR_LIMIT_GB)
#   4. Runs YOLOv8 detection locally, appending to data/predictions/predictions.csv
#   5. Pulls Jack's weather/rating/tide/swell predictors from Surfline
#   6. Backfills real observed weather from Open-Meteo's archive
#   7. Rebuilds data/training_features.csv (the model's training table)
#   8. Records success timestamp (data/.last_local_success)
#
# Runs entirely locally — no GCP VM involved (detection runs on CPU either
# way, so there was no benefit to running it in the cloud).
#
# Run this LATE IN THE EVENING, after dusk, and DAILY. Both matter:
# Surfline's forecast endpoints (Step 5) are forward-looking only — they
# serve today and tomorrow, never a past date without a premium token — so a
# clip only gets its predictors if this runs on the same day the clip was
# recorded. On the previous Tue/Thu schedule, everything captured on the
# other five days aged out before Step 5 ever saw it: 206 quality_ok rows
# across 16 dates have no predictors at all and can't be recovered. Running
# after dusk also means the day's clips are all available in one pass.
#
# Cron entry (nightly at 21:00 local time — adjust path as needed):
#   0 21 * * * /Users/YOUR_USERNAME/Documents/DS/sum-surfers/code/local_pipeline.sh \
#       >> /Users/YOUR_USERNAME/Documents/DS/sum-surfers/data/local_pipeline.log 2>&1
#
# First-time setup:
#   chmod +x code/local_pipeline.sh
#   crontab -e   # paste the line above

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── PATH augmentation for cron (python3 may not be in default PATH) ──────────
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:$PATH"

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE" >&2
    exit 1
fi
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

# ── Python: prefer .venv if present, otherwise use system python3 ─────────────
if [[ -f "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON="python3"
fi

# ── Logging ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Local pipeline starting ==="

cd "$PROJECT_ROOT"

# ── Step 1: Download clips ────────────────────────────────────────────────────
log "Step 1/8 — Downloading Surfline clips..."
"$PYTHON" code/get_clips.py
log "Step 1 done."

# ── Step 2: Extract crop frames ───────────────────────────────────────────────
log "Step 2/8 — Extracting crop frames..."
"$PYTHON" code/get_cropped_frame.py
log "Step 2 done."

# ── Step 3: Check local clips storage ────────────────────────────────────────
# Emails a warning if clips folder exceeds CLIPS_DIR_LIMIT_GB; never fails the pipeline.
log "Step 3/8 — Checking clips storage..."
"$PYTHON" code/manage_clips.py --check || true
log "Step 3 done."

# ── Step 4: Run detection locally ────────────────────────────────────────────
log "Step 4/8 — Running YOLOv8 detection locally..."
"$PYTHON" code/detect_surfers.py
log "Step 4 done."

# ── Step 5: Pull Surfline predictors (weather/rating/tide/swell) for Jack's ──
log "Step 5/8 — Pulling Surfline predictors for Jack's..."
"$PYTHON" code/get_surf_predictors.py
log "Step 5 done."

# ── Step 6: Backfill real observed weather (Open-Meteo archive) ──────────────
# Free, no auth, and it refetches the whole date range in one request, so this
# is a full rewrite rather than an append — safe and idempotent to run daily.
# Was manual-only until 2026-09-09, which let openmeteo_weather.csv (and so
# training_features.csv below, and so the daily chart's model) freeze at
# 2026-08-28 while 161 new quality_ok rows piled up unused. Never fail the
# pipeline over it — the detection data above is the irreplaceable part.
log "Step 6/8 — Backfilling real observed weather (Open-Meteo)..."
"$PYTHON" code/backfill_openmeteo_weather.py || log "WARNING: Open-Meteo backfill failed, continuing."
log "Step 6 done."

# ── Step 7: Rebuild the model's training table ───────────────────────────────
# Joins predictions (target) with all predictor sources (features). Also
# manual-only until 2026-09-09 — see Step 6. Rebuilt from scratch each run.
log "Step 7/8 — Rebuilding training features table..."
"$PYTHON" code/build_training_features.py || log "WARNING: training-features rebuild failed, continuing."
log "Step 7 done."

# ── Step 8: Record success timestamp locally ─────────────────────────────────
LAST_SUCCESS_FILE="$PROJECT_ROOT/data/.last_local_success"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LAST_SUCCESS_FILE"
log "Step 8/8 — Local success timestamp recorded: $(cat "$LAST_SUCCESS_FILE")"

log "=== Local pipeline complete ==="
