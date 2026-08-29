"""One-off script: downloads 12 back-to-back 5-minute clips covering a single
hour (default 8-9am today), to study how much the surfer count varies within
one hour. Not part of the scheduled pipeline — run manually.

Reuses download_clip()/AuthError from get_clips.py rather than reimplementing
the Surfline clip-download API call. Saves into a separate folder from the
production surf_clips/ tree (data/not_needed_in_repo/hourly_variability/) so
it can't be confused with or interfere with the regular pipeline's data.

Usage:
    python analysis/hourly_variability_8to9am/pull_hourly_variability_clips.py [--date YYYY-MM-DD] [--start-hour 8]
"""
import argparse
import sys
import time as time_mod
from datetime import datetime, timedelta
from pathlib import Path

import pytz

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code"))
import get_clips as gc  # noqa: E402

OUT_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "hourly_variability"
CLIP_DURATION_SEC = 300  # 5 minutes
NUM_CLIPS = 12  # 12 x 5min = 1 hour, back-to-back


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default today")
    parser.add_argument("--start-hour", type=int, default=8, help="local hour to start at, default 8 (8am)")
    args = parser.parse_args()

    local_tz = pytz.timezone(gc.LOCATION["timezone"])
    if args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        date = datetime.now(local_tz).date()

    day_dir = OUT_DIR / date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    start_dt = local_tz.localize(datetime.combine(date, datetime.min.time())) + timedelta(hours=args.start_hour)

    auth_failures = 0
    for i in range(NUM_CLIPS):
        clip_start = start_dt + timedelta(seconds=i * CLIP_DURATION_SEC)
        time_str = clip_start.strftime("%H_%M")
        clip_folder = day_dir / time_str
        clip_folder.mkdir(exist_ok=True)
        clip_file = clip_folder / "clip.mp4"

        if clip_file.exists():
            print(f"✅ Clip exists: {date} {time_str}")
            continue

        start_ms = int(clip_start.timestamp() * 1000)
        end_ms = start_ms + CLIP_DURATION_SEC * 1000
        try:
            gc.download_clip(start_ms, end_ms, clip_file)
        except gc.AuthError as e:
            auth_failures += 1
            print(f"⚠️ Failed: {date} {time_str}: {e}")
        except Exception as e:
            print(f"⚠️ Failed: {date} {time_str}: {e}")

        delay = gc.REQUEST_BASE_DELAY_SEC
        time_mod.sleep(delay)

    if auth_failures > 0:
        print(f"\n{auth_failures} clip download(s) failed due to an invalid/expired access token.")

    print(f"\nDone. Clips in {day_dir}")


if __name__ == "__main__":
    main()
