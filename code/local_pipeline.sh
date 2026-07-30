#!/usr/bin/env bash
# local_pipeline.sh — Runs on your laptop every 3 days via cron.
#
# What it does:
#   1. Downloads new Surfline clips locally
#   2. Extracts crop frames locally
#   3. Checks local clips storage (emails warning if > CLIPS_DIR_LIMIT_GB)
#   4. Runs YOLOv8 detection locally, appending to data/predictions.csv
#   5. Pulls Jack's weather/rating/tide/swell predictors from Surfline
#   6. Records success timestamp (data/.last_local_success)
#
# Runs entirely locally — no GCP VM involved (detection runs on CPU either
# way, so there was no benefit to running it in the cloud).
#
# Cron entry (every 3 days at 06:00 local time — adjust path/time as needed):
#   0 6 */3 * * /Users/YOUR_USERNAME/Documents/DS/sum-surfers/code/local_pipeline.sh \
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
log "Step 1/6 — Downloading Surfline clips..."
"$PYTHON" code/get_clips.py
log "Step 1 done."

# ── Step 2: Extract crop frames ───────────────────────────────────────────────
log "Step 2/6 — Extracting crop frames..."
"$PYTHON" code/get_cropped_frame.py
log "Step 2 done."

# ── Step 3: Check local clips storage ────────────────────────────────────────
# Emails a warning if clips folder exceeds CLIPS_DIR_LIMIT_GB; never fails the pipeline.
log "Step 3/6 — Checking clips storage..."
"$PYTHON" code/manage_clips.py --check || true
log "Step 3 done."

# ── Step 4: Run detection locally ────────────────────────────────────────────
log "Step 4/6 — Running YOLOv8 detection locally..."
"$PYTHON" code/detect_surfers.py
log "Step 4 done."

# ── Step 5: Pull Surfline predictors (weather/rating/tide/swell) for Jack's ──
log "Step 5/6 — Pulling Surfline predictors for Jack's..."
"$PYTHON" code/get_surf_predictors.py
log "Step 5 done."

# ── Step 6: Record success timestamp locally ─────────────────────────────────
LAST_SUCCESS_FILE="$PROJECT_ROOT/data/.last_local_success"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LAST_SUCCESS_FILE"
log "Step 6/6 — Local success timestamp recorded: $(cat "$LAST_SUCCESS_FILE")"

log "=== Local pipeline complete ==="
