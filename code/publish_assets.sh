#!/usr/bin/env bash
# Publishes the nightly images to an orphan `assets` branch instead of main.
#
# Why: these files are rewritten every night, and git keeps every past version
# forever. Committing them to main added ~9 MB/day to the clone size, which by
# 2026-09-26 had reached 94 MB of chart history alone. An orphan branch that is
# force-pushed each night holds exactly one commit, so nothing accumulates --
# the previous version is discarded rather than archived.
#
# Built with plumbing (hash-object / mktree / commit-tree) rather than a
# checkout, so this never touches the working tree or the current branch. Safe
# to run mid-pipeline.
#
# The images stay reproducible from the data in main, which is the reason it is
# acceptable to throw their history away.
set -uo pipefail

BRANCH="assets"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

FILES=(
  "data/charts/published/latest.png"
  "data/charts/published/latest_week.png"
  "data/charts/published/latest_detection.gif"
  "data/charts/published/latest_forecast_vs_actual.png"
)

entries=""
n=0
for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "  publish_assets: skipping missing $f"; continue; }
    blob=$(git hash-object -w "$f") || continue
    entries+=$(printf '100644 blob %s\t%s\n' "$blob" "$(basename "$f")")
    entries+=$'\n'
    n=$((n+1))
done

if [ "$n" -eq 0 ]; then
    echo "  publish_assets: no images to publish"
    exit 0
fi

tree=$(printf '%s' "$entries" | git mktree) || exit 1
# No -p parent: every run makes a fresh root commit, which is what keeps the
# branch from growing.
commit=$(git commit-tree "$tree" -m "Nightly images $(date '+%Y-%m-%d %H:%M')") || exit 1
git update-ref "refs/heads/$BRANCH" "$commit" || exit 1

if git push --force origin "$BRANCH:$BRANCH" 2>&1; then
    echo "  publish_assets: pushed $n image(s) to origin/$BRANCH"
else
    echo "  publish_assets: WARNING push failed; images are local only"
fi
