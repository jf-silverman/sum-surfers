#!/usr/bin/env bash
# daily_chart.sh — Runs once a day via cron to generate the surfer-count
# prediction chart for tomorrow (data/charts/surfer_count_YYYY-MM-DD.png) plus
# a detection-review image from today's own ~8am crop.
#
# Independent of local_pipeline.sh's twice-weekly clip-collection cron — this
# only needs the live Surfline forecast + the already-trained model, not new
# clips or detection.
#
# Cron entry (daily at 7pm local time):
#   0 19 * * * /Users/YOUR_USERNAME/Documents/DS/sum-surfers/code/daily_chart.sh \
#       >> /Users/YOUR_USERNAME/Documents/DS/sum-surfers/data/daily_chart.log 2>&1
#
# First-time setup:
#   chmod +x code/daily_chart.sh
#   crontab -e   # paste the line above

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:$PATH"

ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
fi

if [[ -f "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON="python3"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Daily chart generation starting ==="
cd "$PROJECT_ROOT"
if ! "$PYTHON" code/plot_daily_prediction.py; then
    # set -e would otherwise kill the script right here with nothing but a raw
    # traceback in the log and no "complete" marker -- happened for real on
    # 2026-08-31 (an uncaught network error crashed the whole run before it
    # ever reached the git commit/push section below, silently skipping that
    # day's chart/README update with no alert). This explicit marker makes a
    # future failure grep-able instead of requiring a manual timestamp diff
    # across the whole log to even notice a day was skipped.
    log "ERROR: plot_daily_prediction.py failed (see traceback above) — chart NOT generated, README NOT updated this run."
    exit 1
fi

# Auto-commit + push the updated chart/table/README — approved by Joel 2026-08-28
# specifically so the latest chart is visible on GitHub without a manual step.
# Non-fatal: a git/network failure here must never be treated as the whole daily
# chart job failing (the chart itself already generated fine above).
# Added individually, not as one `git add a b c` — a missing pathspec (e.g. no
# detection image yet some days) fails the ENTIRE add and blocks staging the
# others too if done as one command; per-file `|| true` avoids that.
git add data/charts/latest.png 2>&1 || true
git add data/charts/latest_detection.png 2>&1 || true
git add README.md 2>&1 || true
if git diff --cached --quiet; then
    log "No changes to commit (chart/table/README identical to last run)."
else
    if git commit -m "Automated: update daily surfer count chart ($(date '+%Y-%m-%d'))" 2>&1; then
        if git push origin main 2>&1; then
            log "Pushed updated daily chart to origin/main."
        else
            log "WARNING: git push failed — chart committed locally but not pushed. Push manually when convenient."
        fi
    else
        log "WARNING: git commit failed — chart generated but not committed."
    fi
fi

log "=== Daily chart generation complete ==="
