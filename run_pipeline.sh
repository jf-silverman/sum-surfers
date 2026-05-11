#!/usr/bin/env bash
# run_pipeline.sh — Surfer detection pipeline
# Runs on the GCP VM via cron.  Logs to /var/log/surfers.log.
#
# Schedule (cron example — runs at 20:00 local time every 3 days):
#   0 20 */3 * * /home/surfer/sum-surfers/run_pipeline.sh >> /var/log/surfers.log 2>&1

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/activate"
ENV_FILE="$SCRIPT_DIR/.env"

# ── Logging helper ────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Pipeline start ==="

# ── Activate virtual environment ──────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    log "ERROR: venv not found at $VENV"
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV"

# ── Load secrets from .env ────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
else
    log "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# ── Step 1: Download clips ────────────────────────────────────────────────────
log "Step 1/3 — Downloading clips..."
python "$SCRIPT_DIR/code/get_clips.py"
log "Step 1 done."

# ── Step 2: Extract crop frames ───────────────────────────────────────────────
log "Step 2/3 — Extracting crop frames..."
python "$SCRIPT_DIR/code/get_cropped_frame.py"
log "Step 2 done."

# ── Step 3: Run YOLO inference ────────────────────────────────────────────────
log "Step 3/3 — Running surfer detection..."
# Default to recent-only processing so first run on a fresh machine does not
# backfill old historical crops unless explicitly requested.
export DETECT_MODE="${DETECT_MODE:-recent}"
export DETECT_RECENT_DAYS="${DETECT_RECENT_DAYS:-7}"
# Optional manual backfill override:
#   export DETECT_START_DATE=2025-10-01
python "$SCRIPT_DIR/code/detect_surfers.py"
log "Step 3 done."

log "=== Pipeline complete ==="

# ── Shutdown VM after pipeline (cost saving) ─────────────────────────────────
# Comment out the line below if you want the VM to stay running.
sudo shutdown -h now
