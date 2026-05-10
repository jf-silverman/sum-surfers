#!/usr/bin/env bash
# setup_vm.sh — Run once on a fresh GCP Debian/Ubuntu VM to bootstrap the project.
# Usage:  bash setup_vm.sh
#
# Assumptions:
#   - You have already SSH'd into the VM (gcloud compute ssh ...)
#   - The repo is cloned at ~/sum-surfers  (or adjust PROJECT_DIR below)
#   - Your .env file is already on the VM (copy it up with gcloud scp or Secret Manager)

set -euo pipefail

PROJECT_DIR="${HOME}/sum-surfers"
PYTHON_BIN="python3.11"   # adjust if your VM image ships a different version

echo "=== [1/6] System packages ==="
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
    git \
    python3.11 python3.11-venv python3.11-dev \
    libgl1 libglib2.0-0 \
    ffmpeg

echo "=== [2/6] Create virtual environment ==="
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "=== [3/6] Install Python dependencies ==="
pip install --upgrade pip wheel
pip install \
    requests \
    astral \
    pytz \
    opencv-python-headless \
    ultralytics \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ↑ CPU-only torch is ~500 MB smaller and fast enough for 14 frames/day.
# For GPU inference swap the last two lines with:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/6] Make pipeline script executable ==="
chmod +x "$PROJECT_DIR/run_pipeline.sh"

echo "=== [5/6] Allow pipeline script to shutdown without password ==="
# Adds a passwordless sudo rule for the shutdown command only.
CRON_USER="$(whoami)"
SUDOERS_LINE="$CRON_USER ALL=(ALL) NOPASSWD: /sbin/shutdown"
echo "$SUDOERS_LINE" | sudo tee /etc/sudoers.d/surfer-shutdown > /dev/null
sudo chmod 0440 /etc/sudoers.d/surfer-shutdown

echo "=== [6/6] Install cron job ==="
# Runs at 20:00 VM local time every 3 days.
# The VM timezone is set to America/Los_Angeles so 20:00 is after sunset year-round.
sudo timedatectl set-timezone America/Los_Angeles

CRON_JOB="0 20 */3 * * $PROJECT_DIR/run_pipeline.sh >> /var/log/surfers.log 2>&1"
# Append only if not already present
( crontab -l 2>/dev/null | grep -qF "run_pipeline.sh" ) \
    || ( crontab -l 2>/dev/null; echo "$CRON_JOB" ) | crontab -

echo ""
echo "✅ VM setup complete."
echo ""
echo "Next steps:"
echo "  1. Make sure .env exists at $PROJECT_DIR/.env"
echo "  2. Upload your model weights to:"
echo "       $PROJECT_DIR/data/model_out/20251013/train/runs/detect/train13/weights/best.pt"
echo "  3. Test the pipeline manually:"
echo "       cd $PROJECT_DIR && bash run_pipeline.sh"
echo "  4. Set up GCP Cloud Scheduler to auto-start this VM (see GCP_SETUP.md)"
