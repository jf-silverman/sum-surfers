"""
prune_charts.py
---------------
Deletes dated artefacts from data/charts/daily/ once they age out.

Why this exists: the nightly run writes a handful of per-day files that nothing
cleaned up -- the emailed detection frames and forecast-vs-actual chart, the
detection GIF, the daily prediction chart and the week grid. All are gitignored
so they never entered the repo, but they accumulated on disk at ~5 MB a night.

Why it scans a DIRECTORY rather than a list of filename prefixes: the first
version of this script matched four prefixes, and silently missed two whole
families -- `week_*.png`, and an older `detection_*.png` that predated the
animation. Anything written into daily/ is disposable by definition, so a new
chart type cannot quietly escape the prune by not matching a pattern. Files in
data/charts/published/ are never touched: those are what the README and the
assets branch point at.

Retention is per-family because the GIFs are ~3.6 MB each and everything else
is a few hundred KB; a single window either hoards animations or throws away
cheap history for no reason.

Age comes from the DATE IN THE FILENAME, not the modification time. An mtime
changes if a file is copied, restored from a backup or touched by a sync tool,
and that would silently delete the wrong things.

Usage:
    python code/prune_charts.py --dry-run
    python code/prune_charts.py
    python code/prune_charts.py --keep-days 90 --keep-days-gif 30
"""

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = _PROJECT_ROOT / "data" / "charts" / "daily"
PUBLISHED_DIR = _PROJECT_ROOT / "data" / "charts" / "published"

DEFAULT_KEEP_DAYS = 30
DEFAULT_KEEP_DAYS_GIF = 7      # ~3.6 MB each; the published copy is the one in use

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def prunable(cutoff, cutoff_gif):
    """[(path, file_date, is_gif), ...] for dated files in daily/ past their window."""
    out = []
    if not DAILY_DIR.exists():
        return out
    for path in sorted(DAILY_DIR.glob("*")):
        if not path.is_file() or path.name.startswith("latest"):
            continue
        m = DATE_RE.search(path.name)
        if not m:
            continue                      # undated: not ours to judge, leave it
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        is_gif = path.suffix.lower() == ".gif"
        if d < (cutoff_gif if is_gif else cutoff):
            out.append((path, d, is_gif))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-days", type=int, default=DEFAULT_KEEP_DAYS,
                   help=f"window for everything but GIFs (default {DEFAULT_KEEP_DAYS})")
    p.add_argument("--keep-days-gif", type=int, default=DEFAULT_KEEP_DAYS_GIF,
                   help=f"window for .gif files (default {DEFAULT_KEEP_DAYS_GIF})")
    p.add_argument("--dry-run", action="store_true", help="list, do not delete")
    args = p.parse_args()

    if not DAILY_DIR.exists():
        print(f"  prune_charts: {DAILY_DIR} does not exist — nothing to do")
        return 0

    today = date.today()
    cutoff = today - timedelta(days=args.keep_days)
    cutoff_gif = today - timedelta(days=args.keep_days_gif)
    doomed = prunable(cutoff, cutoff_gif)
    if not doomed:
        print(f"  prune_charts: nothing older than {cutoff} "
              f"({cutoff_gif} for GIFs) — nothing to do")
        return 0

    freed = sum(path.stat().st_size for path, _d, _g in doomed)
    gifs = sum(1 for _p, _d, g in doomed if g)
    verb = "would delete" if args.dry_run else "deleting"
    print(f"  prune_charts: {verb} {len(doomed)} file(s) "
          f"({gifs} GIF(s) before {cutoff_gif}, {len(doomed) - gifs} other(s) "
          f"before {cutoff}), {freed / 1_048_576:.1f} MB")
    for path, d, _g in doomed:
        print(f"    {d}  {path.name}")
        if not args.dry_run:
            path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
