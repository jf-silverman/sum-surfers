"""
Sub-hourly frame sets: two extra one-minute clips per hour, cut into 10
cropped frames each, clip deleted afterwards (added 2026-10-26).

Why. The hour's count comes from one clip near the start of the hour (five
frames one second apart). Counts only decorrelate over about a minute
(PROJECT_HISTORY.md, 2026-10-14), so one clip says little about the rest of
the hour. These sets sample the same hour twice more, about 18 and 36 minutes
after the main clip, to test later whether they improve the hourly estimate.
Nothing downstream reads them yet: the hourly count, the training table and
the forecast are unchanged.

Timing. Main clips are chained about 54 minutes apart on Surfline's 9-minute
schedule (get_clips.py), so their minute drifts through the hour. Placing the
extra clips at fixed offsets from each main clip (SUB_OFFSETS_MIN) splits each
gap into even thirds and never lands on top of a main clip, which fixed clock
minutes would. Surfline serves clips from any start time (checked 2026-10-26:
starts at :19 and :47 returned normally).

Why a minute, not a few seconds. Counts one second apart correlate at about
0.8 and only decorrelate over about a minute, so ten frames spread across a
minute (one every 6 s) sample more of the hour than ten frames in ten seconds.
The download is bigger, but the clip is deleted as soon as the frames are cut.

What is kept. For each extra clip, frames at 3, 9, ..., 57 s are cropped to
the production ROI (get_cropped_frame.py) and saved as
data/j_shore_cam/subhourly_crops/<date>/sub<date>_<HH-MM>_f00.jpg ... _f09.jpg
(JPEG quality 100, like the main crops). The uncropped frame is never written,
and the clip is deleted once all ten crops exist.

Runs over the same five-day window as get_clips.py, after it, so a clip that
fails one night is retried the next. A set whose ten crops already exist is
skipped. Offsets past last light are skipped.

Usage:
    python code/get_subhourly_frames.py            # the full five-day window
    python code/get_subhourly_frames.py --days 1   # today only
"""
from __future__ import annotations

import argparse
import random
import sys
import time as time_mod
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_clips as gc  # noqa: E402
from get_cropped_frame import ROI_H, ROI_W, ROI_X, ROI_Y  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN_CLIPS = ROOT / "data" / "not_needed_in_repo" / "surf_clips"
TMP_CLIPS = ROOT / "data" / "not_needed_in_repo" / "subhourly_clips"
CROPS = ROOT / "data" / "j_shore_cam" / "subhourly_crops"
SUB_OFFSETS_MIN = (18, 36)
SUB_CLIP_SEC = 60
FRAME_TIMES_SEC = [3.0 + 6 * i for i in range(10)]  # 3, 9, ..., 57 s
SETTLE_MIN = 5  # don't request a clip that ended less than this long ago


def crop_names(start_dt):
    stem = f"sub{start_dt.strftime('%Y-%m-%d_%H-%M')}"
    return [f"{stem}_f{i:02d}.jpg" for i in range(len(FRAME_TIMES_SEC))]


def cut_frames(clip, out_dir, names):
    """One forward pass through the clip, cropping at each target time."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise RuntimeError(f"no FPS for {clip}")
    want = {int(t * fps): n for t, n in zip(FRAME_TIMES_SEC, names)}
    last = max(want)
    i, written = 0, 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            if cv2.imwrite(str(out_dir / want[i]), crop, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                written += 1
        i += 1
    cap.release()
    if written != len(names):
        raise RuntimeError(f"{clip.name}: wrote {written} of {len(names)} frames (clip ended at frame {i})")


def main_clip_times(day_folder, date, tz):
    out = []
    if not day_folder.exists():
        return out
    for sub in sorted(day_folder.iterdir()):
        try:
            hh, mm = map(int, sub.name.split("_"))
        except ValueError:
            continue
        out.append(tz.localize(datetime(date.year, date.month, date.day, hh, mm)))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--days", type=int, default=gc.DAY_LOOKBACK_DAYS,
                   help="days back to cover, today included (default: the clip lookback)")
    args = a.parse_args()
    tz = pytz.timezone(gc.LOCATION["timezone"])
    now = datetime.now(tz)
    done = failed = skipped = 0
    for delta in range(min(args.days, gc.DAY_LOOKBACK_DAYS)):
        date = now.date() - timedelta(days=delta)
        _, last_light = gc.get_light_window(date, tz)
        day_crops = CROPS / date.isoformat()
        for main_dt in main_clip_times(MAIN_CLIPS / date.isoformat(), date, tz):
            for off in SUB_OFFSETS_MIN:
                start = main_dt + timedelta(minutes=off)
                if start > last_light or start + timedelta(seconds=SUB_CLIP_SEC, minutes=SETTLE_MIN) > now:
                    continue
                names = crop_names(start)
                if all((day_crops / n).exists() for n in names):
                    skipped += 1
                    continue
                day_crops.mkdir(parents=True, exist_ok=True)
                TMP_CLIPS.mkdir(parents=True, exist_ok=True)
                clip = TMP_CLIPS / f"{names[0][:-8]}.mp4"
                try:
                    if not clip.exists():
                        s = int(start.timestamp() * 1000)
                        gc.download_clip(s, s + SUB_CLIP_SEC * 1000, clip)
                        time_mod.sleep(max(gc.REQUEST_BASE_DELAY_SEC
                                           + random.uniform(0.0, gc.REQUEST_JITTER_SEC), 0.0))
                    cut_frames(clip, day_crops, names)
                    clip.unlink()
                    done += 1
                except gc.AuthError as e:
                    print(f"⚠️ Auth failure, stopping: {e}")
                    sys.exit(1)
                except Exception as e:
                    failed += 1
                    print(f"⚠️ {start:%Y-%m-%d %H:%M}: {e}")
                    clip.unlink(missing_ok=True)  # a bad download is retried next run
    print(f"Sub-hourly frame sets: {done} new, {skipped} already done, {failed} failed")


if __name__ == "__main__":
    main()
