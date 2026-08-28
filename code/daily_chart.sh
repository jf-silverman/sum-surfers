#!/usr/bin/env bash
# daily_chart.sh — Runs once a day via cron to generate the surfer-count
# prediction chart for the current day (data/charts/surfer_count_YYYY-MM-DD.png).
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
"$PYTHON" code/plot_daily_prediction.py
log "=== Daily chart generation complete ==="
