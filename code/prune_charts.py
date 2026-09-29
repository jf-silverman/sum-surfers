"""
prune_charts.py
---------------
Deletes dated chart and frame files from data/charts/ older than a cutoff.

Why this exists: the nightly run writes four per-day artefacts that nothing ever
cleaned up — the emailed detection frames, the forecast-vs-actual chart, the
detection GIF and the daily prediction chart. All four are gitignored, so they
never touched the repo, but they accumulated on disk at roughly 3 MB a night
(the GIFs alone reached 57 MB across 27 files by 2026-09-29). `manage_clips.py`
prunes the clips directory; nothing watched this one.

What it will NOT touch:
  - the `latest_*` files, which are the current published images
  - latest_detection.json, the sidecar naming the day the animation covers
  - anything whose filename has no parseable YYYY-MM-DD date
  - anything outside data/charts/

Age comes from the DATE IN THE FILENAME, not the modification time. An mtime
changes if a file is copied, restored from a backup or touched by a sync tool,
and that would silently delete the wrong things; the filename date is what the
file is actually about.

Usage:
    python code/prune_charts.py --dry-run      # show what would go
    python code/prune_charts.py                # delete, keeping 30 days
    python code/prune_charts.py --keep-days 90
"""

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = _PROJECT_ROOT / "data" / "charts"
DEFAULT_KEEP_DAYS = 30

# Only these families are pruned. Anything else in the directory is left alone,
# so a new kind of output has to be added here deliberately rather than being
# swept up by a broad glob the day someone adds it.
PATTERNS = [
    re.compile(r"^frame_(\d{4}-\d{2}-\d{2})_.+\.png$"),
    re.compile(r"^forecast_vs_actual_(\d{4}-\d{2}-\d{2})\.png$"),
    re.compile(r"^detection_(\d{4}-\d{2}-\d{2})\.gif$"),
    re.compile(r"^surfer_count_(\d{4}-\d{2}-\d{2})\.png$"),
]


def find_prunable(cutoff):
    """[(path, file_date), ...] for dated files strictly older than `cutoff`."""
    out = []
    for path in sorted(CHARTS_DIR.glob("*")):
        if not path.is_file() or path.name.startswith("latest"):
            continue
        for pat in PATTERNS:
            m = pat.match(path.name)
            if not m:
                continue
            try:
                d = date.fromisoformat(m.group(1))
            except ValueError:
                break          # unparseable date: leave it alone
            if d < cutoff:
                out.append((path, d))
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-days", type=int, default=DEFAULT_KEEP_DAYS,
                   help=f"keep files dated within this many days (default {DEFAULT_KEEP_DAYS})")
    p.add_argument("--dry-run", action="store_true", help="list, do not delete")
    args = p.parse_args()

    if not CHARTS_DIR.exists():
        print(f"  prune_charts: {CHARTS_DIR} does not exist — nothing to do")
        return 0

    cutoff = date.today() - timedelta(days=args.keep_days)
    doomed = find_prunable(cutoff)
    if not doomed:
        print(f"  prune_charts: nothing older than {cutoff} ({args.keep_days}-day window)")
        return 0

    freed = sum(path.stat().st_size for path, _d in doomed)
    verb = "would delete" if args.dry_run else "deleting"
    print(f"  prune_charts: {verb} {len(doomed)} file(s) dated before {cutoff}, "
          f"{freed / 1_048_576:.1f} MB")
    for path, d in doomed:
        print(f"    {d}  {path.name}")
        if not args.dry_run:
            path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
