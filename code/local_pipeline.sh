#!/usr/bin/env bash
# local_pipeline.sh — Runs on your laptop every 3 days via cron.
#
# What it does:
#   1. Downloads new Surfline clips locally
#   2. Extracts crop frames locally
#   3. Checks local clips storage (emails warning if > CLIPS_DIR_LIMIT_GB)
#   4. Starts the GCP VM
#   5. Uploads new surf crops to the VM
#   6. SSHes into the VM and runs detection (vm_pipeline.sh --no-shutdown)
#   7. Pulls updated predictions.csv back to laptop
#   8. Records success timestamp (data/.last_local_success)
#   9. Syncs success timestamp to VM
#  10. Stops the VM
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

# ── PATH augmentation for cron (gcloud, python3 may not be in default PATH) ──
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/google-cloud-sdk/bin:$HOME/.local/bin:/usr/bin:/bin:$PATH"

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

# ── GCP config (can be overridden in .env) ────────────────────────────────────
GCP_PROJECT="${GCP_PROJECT:-sum-surfers-20260510-a1b2}"
GCP_ZONE="${GCP_ZONE:-us-west2-a}"
GCP_INSTANCE="${GCP_INSTANCE:-surf-detector}"

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
log "Step 1/10 — Downloading Surfline clips..."
"$PYTHON" code/get_clips.py
log "Step 1 done."

# ── Step 2: Extract crop frames ───────────────────────────────────────────────
log "Step 2/10 — Extracting crop frames..."
"$PYTHON" code/get_cropped_frame.py
log "Step 2 done."

# ── Step 3: Check local clips storage ────────────────────────────────────────
# Emails a warning if clips folder exceeds CLIPS_DIR_LIMIT_GB; never fails the pipeline.
log "Step 3/10 — Checking clips storage..."
"$PYTHON" code/manage_clips.py --check || true
log "Step 3 done."

# ── Step 4: Start VM ─────────────────────────────────────────────────────────
log "Step 4/10 — Starting GCP VM ($GCP_INSTANCE)..."
gcloud compute instances start "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet
log "Waiting 45 s for VM to finish booting..."
sleep 45

# ── Step 5: Upload crops to VM ───────────────────────────────────────────────
log "Step 5/10 — Uploading crops to VM..."
CROPS_DIR="$PROJECT_ROOT/data/j_shore_cam/surf_crops"
if [[ -d "$CROPS_DIR" ]]; then
    gcloud compute scp --recurse \
        "$CROPS_DIR" \
        "${GCP_INSTANCE}:~/sum-surfers/data/j_shore_cam/" \
        --zone="$GCP_ZONE" --project="$GCP_PROJECT"
else
    log "  No crops directory found at $CROPS_DIR — skipping upload."
fi
log "Step 5 done."

# ── Step 6: Trigger VM detection (no auto-shutdown; we pull first) ──────────
log "Step 6/10 — Running detection on VM..."
gcloud compute ssh "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" \
    -- "cd ~/sum-surfers && source .venv/bin/activate && bash code/vm_pipeline.sh --no-shutdown"
log "Step 6 done."

# ── Step 7: Pull updated predictions back ────────────────────────────────────
log "Step 7/10 — Pulling predictions.csv from VM..."
gcloud compute scp \
    "${GCP_INSTANCE}:~/sum-surfers/data/predictions.csv" \
    "$PROJECT_ROOT/data/predictions.csv" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT"
log "Step 7 done."

# ── Step 8: Record success timestamp locally ─────────────────────────────────
LAST_SUCCESS_FILE="$PROJECT_ROOT/data/.last_local_success"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LAST_SUCCESS_FILE"
log "Step 8/10 — Local success timestamp recorded: $(cat "$LAST_SUCCESS_FILE")"

# ── Step 9: Sync success timestamp to VM ─────────────────────────────────────
log "Step 9/10 — Syncing success timestamp to VM..."
gcloud compute scp \
    "$LAST_SUCCESS_FILE" \
    "${GCP_INSTANCE}:~/sum-surfers/data/.last_local_success" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT"
log "Step 9 done."

# ── Step 10: Stop VM ─────────────────────────────────────────────────────────
log "Step 10/10 — Stopping VM..."
gcloud compute instances stop "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet

log "=== Local pipeline complete ==="
