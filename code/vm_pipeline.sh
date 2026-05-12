#!/usr/bin/env bash
# vm_pipeline.sh — Runs on the GCP VM.
#
# Called in two ways:
#
#   From local_pipeline.sh (via SSH):
#     bash code/vm_pipeline.sh --no-shutdown
#     Runs detection on uploaded crops; does NOT shut down (caller pulls
#     predictions.csv and then stops the VM explicitly).
#
#   From Cloud Scheduler startup script (backup reminder):
#     bash code/vm_pipeline.sh
#     If no new crops are found AND it has been >= REMINDER_THRESHOLD_DAYS since
#     the last local pipeline run, sends a reminder email, then shuts down.
#     If new crops are found, runs detection and shuts down.
#
# Cloud Scheduler startup-script metadata value:
#   cd /home/surfer/sum-surfers && source .venv/bin/activate && bash code/vm_pipeline.sh
#
# Environment variables (from .env on VM):
#   SMTP_USER, SMTP_APP_PASSWORD, EMAIL_TO   — for outgoing email
#   VM_DATA_LIMIT_GB                         — storage warning threshold (default 50)
#   REMINDER_THRESHOLD_DAYS                  — min days before sending a reminder (default 4)
#   DETECT_MODE, DETECT_RECENT_DAYS          — passed through to detect_surfers.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Parse flags ───────────────────────────────────────────────────────────────
SHUTDOWN_AFTER="yes"
for arg in "$@"; do
    [[ "$arg" == "--no-shutdown" ]] && SHUTDOWN_AFTER="no"
done

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python3"
REMINDER_THRESHOLD_DAYS="${REMINDER_THRESHOLD_DAYS:-4}"
VM_DATA_LIMIT_GB="${VM_DATA_LIMIT_GB:-50}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$PROJECT_ROOT"

log "=== VM pipeline starting (shutdown_after=$SHUTDOWN_AFTER) ==="

# ── Check for new crops ────────────────────────────────────────────────────────
CHECK_OUT=$("$PYTHON" code/vm_check.py 2>/dev/null || true)
NEW_COUNT=$(echo "$CHECK_OUT" | awk '{print $1}')
DAYS_SINCE=$(echo "$CHECK_OUT" | awk '{print $2}')
NEW_COUNT="${NEW_COUNT:-0}"
DAYS_SINCE="${DAYS_SINCE:--1}"

log "New unprocessed crops: $NEW_COUNT  |  Days since last local run: $DAYS_SINCE"

# ── Branch: no new crops → maybe send reminder ────────────────────────────────
if [[ "$NEW_COUNT" -eq 0 ]]; then
    # Only send a reminder if enough days have passed (avoids false alarms when
    # Cloud Scheduler fires shortly after a successful local pipeline run).
    if [[ "$DAYS_SINCE" -ge "$REMINDER_THRESHOLD_DAYS" ]] || [[ "$DAYS_SINCE" -eq -1 ]]; then
        DAYS_MSG="unknown (no timestamp found)"
        [[ "$DAYS_SINCE" -ge 0 ]] && DAYS_MSG="$DAYS_SINCE day(s)"

        log "Sending reminder email (days since last run: $DAYS_MSG)..."
        "$PYTHON" code/send_email.py \
            --subject "sum-surfers: no new crops — please run local pipeline" \
            --body "The surf detector VM found no new crops to process.

Days since last local pipeline run: ${DAYS_MSG}.

Your laptop may not have run its scheduled job. Please turn it on and run:

  bash code/local_pipeline.sh

from the sum-surfers project directory, or wait for the next cron cycle." \
        && log "Reminder email sent." \
        || log "WARNING: could not send reminder email."
    else
        log "No new crops but last run was recent (${DAYS_SINCE}d ago) — no reminder needed."
    fi

# ── Branch: new crops found → run detection ───────────────────────────────────
else
    log "Running YOLO detection on $NEW_COUNT new crop(s)..."
    export DETECT_MODE="${DETECT_MODE:-recent}"
    export DETECT_RECENT_DAYS="${DETECT_RECENT_DAYS:-7}"
    "$PYTHON" code/detect_surfers.py
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$PROJECT_ROOT/data/.last_vm_success"
    log "Detection complete. Success timestamp saved."

    # ── VM storage check ──────────────────────────────────────────────────────
    log "Checking VM data storage..."
    "$PYTHON" - <<PYEOF
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "code"))
from send_email import send_email

data_dir = Path.cwd() / "data"
total_gb = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file()) / 1024**3
limit_gb = float(os.environ.get("VM_DATA_LIMIT_GB", "50"))

if total_gb > limit_gb:
    print(f"WARNING: VM data is {total_gb:.2f} GB, exceeding {limit_gb:.1f} GB limit.")
    try:
        send_email(
            "sum-surfers: VM data storage limit exceeded",
            f"The VM data directory is {total_gb:.2f} GB, exceeding the {limit_gb:.1f} GB limit.\n\n"
            f"Directory: {data_dir}\n\n"
            "Please SSH into the VM and review large files:\n"
            "  gcloud compute ssh surf-detector --zone=us-west2-a\n"
            "  du -sh ~/sum-surfers/data/*\n"
        )
        print("Storage warning email sent.")
    except Exception as e:
        print(f"Could not send storage warning email: {e}")
else:
    print(f"VM storage OK: {total_gb:.2f} GB / {limit_gb:.1f} GB")
PYEOF
fi

# ── Shutdown (only when not called with --no-shutdown) ────────────────────────
log "=== VM pipeline complete ==="
if [[ "$SHUTDOWN_AFTER" == "yes" ]]; then
    log "Shutting down VM..."
    sudo shutdown -h now
fi
