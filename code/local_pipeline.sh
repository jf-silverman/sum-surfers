#!/usr/bin/env bash
# local_pipeline.sh — Runs on your laptop every 3 days via cron.
#
# What it does:
#   1. Downloads new Surfline clips locally
#   2. Extracts crop frames locally
#   3. Checks local clips storage (emails warning if > CLIPS_DIR_LIMIT_GB)
#   4. Records a success timestamp (data/.last_local_run)
#   5. Starts the GCP VM
#   6. Uploads new surf crops + timestamp to the VM
#   7. SSHes into the VM and runs detection (vm_pipeline.sh --no-shutdown)
#   8. Pulls updated predictions.csv back to laptop
#   9. Stops the VM
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

# ── Step 4: Record success timestamp ─────────────────────────────────────────
LAST_RUN_FILE="$PROJECT_ROOT/data/.last_local_run"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LAST_RUN_FILE"
log "Step 4/8 — Local timestamp recorded: $(cat "$LAST_RUN_FILE")"

# ── Step 5: Start VM ─────────────────────────────────────────────────────────
log "Step 5/8 — Starting GCP VM ($GCP_INSTANCE)..."
gcloud compute instances start "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet
log "Waiting 45 s for VM to finish booting..."
sleep 45

# ── Step 6: Upload crops + timestamp to VM ────────────────────────────────────
log "Step 6/8 — Uploading crops and timestamp to VM..."
CROPS_DIR="$PROJECT_ROOT/data/j_shore_cam/surf_crops"
if [[ -d "$CROPS_DIR" ]]; then
    gcloud compute scp --recurse \
        "$CROPS_DIR" \
        "${GCP_INSTANCE}:~/sum-surfers/data/j_shore_cam/" \
        --zone="$GCP_ZONE" --project="$GCP_PROJECT"
else
    log "  No crops directory found at $CROPS_DIR — skipping upload."
fi
gcloud compute scp \
    "$LAST_RUN_FILE" \
    "${GCP_INSTANCE}:~/sum-surfers/data/.last_local_run" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT"
log "Step 6 done."

# ── Step 7: Trigger VM detection (no auto-shutdown; we pull first) ────────────
log "Step 7/8 — Running detection on VM..."
gcloud compute ssh "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" \
    -- "cd ~/sum-surfers && source .venv/bin/activate && bash code/vm_pipeline.sh --no-shutdown"
log "Step 7 done."

# ── Step 8: Pull updated predictions back ─────────────────────────────────────
log "Step 8/8 — Pulling predictions.csv from VM..."
gcloud compute scp \
    "${GCP_INSTANCE}:~/sum-surfers/data/predictions.csv" \
    "$PROJECT_ROOT/data/predictions.csv" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT"
log "Step 8 done."

# ── Stop VM ───────────────────────────────────────────────────────────────────
log "Stopping VM..."
gcloud compute instances stop "$GCP_INSTANCE" \
    --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet

log "=== Local pipeline complete ==="
