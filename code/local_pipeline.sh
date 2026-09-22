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

# ── Failure tracking ──────────────────────────────────────────────────────────
# Every step runs through run_step, which records a failure and carries on
# rather than aborting under `set -e`. Two reasons: a later step is often still
# worth running (predictors do not depend on detection), and a mid-script abort
# used to leave no notification at all — the run simply stopped. Anything that
# failed is emailed once at the end. Added 2026-09-16, when Joel asked to be
# emailed only on a failed pipeline or on storage passing 5 GB.
FAILED_STEPS=()
run_step() {
    local label="$1"; shift
    # `local rc=$?` would capture local's own status, not the command's — it
    # reported "exit 0" for every real failure. Assign on the failure path.
    local rc=0
    "$@" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        log "WARNING: ${label} failed (exit ${rc}) — continuing."
        FAILED_STEPS+=("${label} (exit ${rc})")
    fi
    return 0
}

log "=== Local pipeline starting ==="

cd "$PROJECT_ROOT"

# ── Step 1: Download clips ────────────────────────────────────────────────────
log "Step 1/10 — Downloading Surfline clips..."
run_step "Step 1 (download clips)" "$PYTHON" code/get_clips.py
log "Step 1 done."

# ── Step 2: Extract crop frames ───────────────────────────────────────────────
log "Step 2/10 — Extracting crop frames..."
run_step "Step 2 (extract crops)" "$PYTHON" code/get_cropped_frame.py
log "Step 2 done."

# ── Step 3: Check local clips storage ────────────────────────────────────────
# Emails a warning if clips folder exceeds CLIPS_DIR_LIMIT_GB; never fails the pipeline.
log "Step 3/10 — Checking clips storage..."
"$PYTHON" code/manage_clips.py --check || true
log "Step 3 done."

# ── Step 4: Run detection locally ────────────────────────────────────────────
log "Step 4/10 — Running YOLOv8 detection locally..."
run_step "Step 4 (detection)" "$PYTHON" code/detect_surfers.py
log "Step 4 done."

# ── Step 5: Pull Surfline predictors (weather/rating/tide/swell) for Jack's ──
log "Step 5/10 — Pulling Surfline predictors for Jack's..."
run_step "Step 5 (surf predictors)" "$PYTHON" code/get_surf_predictors.py
log "Step 5 done."

# ── Step 6: Backfill real observed weather (Open-Meteo archive) ──────────────
# Free, no auth, and it refetches the whole date range in one request, so this
# is a full rewrite rather than an append — safe and idempotent to run daily.
# Was manual-only until 2026-09-09, which let openmeteo_weather.csv (and so
# training_features.csv below, and so the daily chart's model) freeze at
# 2026-08-28 while 161 new quality_ok rows piled up unused. Never fail the
# pipeline over it — the detection data above is the irreplaceable part.
log "Step 6/10 — Backfilling real observed weather (Open-Meteo)..."
run_step "Step 6 (Open-Meteo backfill)" "$PYTHON" code/backfill_openmeteo_weather.py
log "Step 6 done."

# ── Step 7: Rebuild the model's training table ───────────────────────────────
# Joins predictions (target) with all predictor sources (features). Also
# manual-only until 2026-09-09 — see Step 6. Rebuilt from scratch each run.
log "Step 7/10 — Rebuilding training features table..."
run_step "Step 7 (training features)" "$PYTHON" code/build_training_features.py
log "Step 7 done."

# ── Step 8: Record success timestamp locally ─────────────────────────────────
LAST_SUCCESS_FILE="$PROJECT_ROOT/data/.last_local_success"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LAST_SUCCESS_FILE"
log "Step 8/10 — Local success timestamp recorded: $(cat "$LAST_SUCCESS_FILE")"

# ── Step 9: Build the daily prediction chart ─────────────────────────────────
# Chained here rather than run from its own cron entry (moved 2026-09-15). As a
# separate 21:15 job it silently missed nights whenever the Mac slept in the gap
# after this pipeline finished — cron cannot wake a sleeping machine and, unlike
# launchd, never catches up a missed run. Real case: 2026-09-14, pipeline ran at
# 20:30, laptop packed up, no chart at all that night and no log line to show it.
# Running it here puts it inside the same `caffeinate -i` wrapper that is already
# holding the machine awake for this job, and means the chart always trains on
# detections that were written minutes earlier rather than last night's.
# Non-fatal: the chart is regenerable, the clip/detection data above is not.
log "Step 9/10 — Building daily prediction chart..."
run_step "Step 9 (daily chart)" bash "$PROJECT_ROOT/code/daily_chart.sh" \
    >> "$PROJECT_ROOT/data/daily_chart.log" 2>&1
log "Step 9 done."

# ── Step 10: Evening forecast-vs-actual email ────────────────────────────────
# Runs last, and deliberately AFTER step 9: step 9 records tomorrow's forecast
# to data/forecasts/, which is what tomorrow evening's email will score. Today's
# email reads the record written on a previous run, so the report always
# compares the day against what was predicted before it happened.
log "Step 10/10 — Emailing the forecast-vs-actual report..."
run_step "Step 10 (evening report)" "$PYTHON" code/email_daily_report.py
log "Step 10 done."

# ── Notify only on failure ───────────────────────────────────────────────────
# The only two things worth an email are a failed pipeline and storage over the
# limit (that one is emailed by manage_clips.py in Step 3). A run that worked
# sends nothing, so an email in the inbox always means something needs doing.
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    log "=== Local pipeline finished with ${#FAILED_STEPS[@]} failed step(s) ==="
    FAILURE_LIST=$(printf '  - %s\n' "${FAILED_STEPS[@]}")
    "$PYTHON" code/send_email.py \
        --subject "sum-surfers pipeline: ${#FAILED_STEPS[@]} step(s) failed $(date '+%Y-%m-%d')" \
        --body "The nightly pipeline finished with failures:

${FAILURE_LIST}
Last 25 lines of data/local_pipeline.log:

$(tail -25 "$PROJECT_ROOT/data/local_pipeline.log")" \
        || log "WARNING: failure email could not be sent."
    exit 1
fi

log "=== Local pipeline complete ==="
