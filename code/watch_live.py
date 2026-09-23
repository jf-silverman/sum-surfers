"""
watch_live.py
-------------
Credential-free, real-time version of the collection pipeline: while this
script is running, it records a few seconds off the beach camera's public
live stream once every few minutes during daylight, cuts three frames from
that clip exactly the way the scheduled pipeline does, crops each to the
region of interest the trained detector expects, counts surfers on all
three, and appends the averaged result to its own CSV.

Why this exists, next to local_pipeline.sh
------------------------------------------
The production pipeline downloads recorded clips through Surfline's
`/cameras/{id}/clip` endpoint, which needs SURFLINE_CAMERA_ID and
SURFLINE_ACCESS_TOKEN — a paid account. That makes the repo impossible for
anyone else to run end to end.

This script needs no account and no environment variables at all. The spot's
live HLS stream URL is served by the public reports endpoint (spotId is a
public identifier, already hardcoded in get_surf_predictors.py) and the
stream itself plays without a token — verified 2026-09-11: a tokenless
request returned a 1280x720 frame, the same resolution the clip pipeline
produces, so the hardcoded ROI crop and the tiled detector apply unchanged.

The tradeoff is that a live stream has no rewind: this collects only while
the computer is awake and this process is running. It cannot backfill. That
is the intended shape — it is the reproducible demo/collection path, not a
replacement for the scheduled pipeline.

Outputs (separate from production by default, so a demo run can never
contaminate the real dataset):
    data/live_watch/frames/live<date>_<time>.jpg          (primary frame)
    data/live_watch/frames/live<date>_<time>_side{1,2}.jpg
    data/live_watch/live_predictions.csv   (same columns as predictions.csv)

The recorded clip itself is deleted as soon as the three crops are cut — at
this cadence, keeping clips would fill a disk quickly.

Usage:
    python code/watch_live.py                  # run until Ctrl-C
    python code/watch_live.py --once           # single collection, then exit
    python code/watch_live.py --interval-min 5
    python code/watch_live.py --ignore-daylight --once   # test after dark

Requires ffmpeg on PATH (brew install ffmpeg / apt install ffmpeg).
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import pytz
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_clips import LOCATION, REQUEST_HEADERS, get_light_window  # noqa: E402
from get_cropped_frame import (  # noqa: E402
    FRAME_TIME_SEC, SIDE_FRAME_OFFSET_SEC, ROI_X, ROI_Y, ROI_W, ROI_H, extract_frame_at,
)
import detect_surfers as ds  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPOT_ID = "5842041f4e65fad6a770880b"  # Jack's / Pleasure Point — public identifier, no token
REPORTS_URL = "https://services.surfline.com/kbyg/spots/reports"

DEFAULT_FRAMES_DIR = _PROJECT_ROOT / "data" / "live_watch" / "frames"
DEFAULT_OUT_CSV = _PROJECT_ROOT / "data" / "live_watch" / "live_predictions.csv"

# 9 minutes matches the production clip cadence (Surfline's clip windows are
# ~9 min apart), so counts collected here are directly comparable to the
# scheduled pipeline's rather than being sampled at a different rate.
DEFAULT_INTERVAL_MIN = 9

CLIP_DURATION_SEC = 6       # one second of margin past the +1.5s side frame
STREAM_URL_TTL_MIN = 60      # re-resolve the stream URL this often; CDN paths can rotate
SLEEP_SLICE_SEC = 20         # wake this often while waiting, so Ctrl-C stays responsive
                             # and a laptop resuming from sleep is noticed promptly
DARK_POLL_MIN = 5            # how often to re-check for first light while it is dark
FFMPEG_TIMEOUT_SEC = 90
WARN_AFTER_CONSECUTIVE_FAILURES = 3


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resolve_stream_url():
    """Look up the camera's live HLS playlist from the public reports endpoint.

    No access token is sent. The browser-like REQUEST_HEADERS are still
    required — services.surfline.com puts a Cloudflare bot check in front of
    plain requests, which 403s regardless of whether a token is present.
    """
    resp = requests.get(REPORTS_URL, params={"spotId": SPOT_ID},
                        headers=REQUEST_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    cameras = (data.get("spot") or {}).get("cameras") or data.get("cameras") or []
    if not cameras:
        raise RuntimeError("Reports endpoint returned no cameras for this spot")

    cam = cameras[0]
    status = cam.get("status") or {}
    if status.get("isDown"):
        raise RuntimeError(f"Camera reported down: {status.get('message') or 'no message'}")
    url = cam.get("streamUrl")
    if not url:
        raise RuntimeError("Camera entry has no streamUrl")
    return url


def record_clip(stream_url, dest_path, duration_sec=CLIP_DURATION_SEC):
    """Record a few seconds off the live stream with ffmpeg, stream-copied.

    A clip rather than a single frame, because counts on the same stretch of
    water swing by several surfers within a couple of seconds (see
    PROJECT_HISTORY.md's 2026-08-25 entry) — the production pipeline averages
    three frames per clip for exactly that reason, and a one-frame live
    version would be noisier than the data it is meant to be comparable to.
    `-c copy` means no re-encode, so this costs about a second.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-headers", "Referer: https://www.surfline.com/\r\nOrigin: https://www.surfline.com\r\n",
        "-user_agent", REQUEST_HEADERS["User-Agent"],
        "-i", stream_url,
        "-t", str(duration_sec), "-c", "copy", "-y", str(dest_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    if proc.returncode != 0 or not dest_path.exists():
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")


def check_clip_dimensions(clip_path):
    """Confirm the stream is still the resolution the hardcoded ROI assumes.

    The detector was trained on 1280x180 strips cut out of 1280x720 camera
    frames. If the stream ever changed resolution, extract_frame_at() would
    keep cropping the same pixel rectangle and silently return the wrong
    patch of water, so this fails loudly instead.
    """
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open recorded clip: {clip_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if (w, h) != (1280, 720):
        raise RuntimeError(
            f"Stream is {w}x{h}, expected 1280x720 — the hardcoded ROI "
            f"({ROI_X},{ROI_Y},{ROI_W},{ROI_H}) would crop the wrong region"
        )


def migrate_csv_header(out_csv):
    """Bring an existing live CSV up to the detector's current column list.

    The detection schema gains a column occasionally (`glare_frac` did, on
    2026-09-19). `ds.append_row()` writes by the CURRENT header, so a file
    created before that change keeps its old header line while new rows carry an
    extra value — every field after the new column silently shifts by one, and a
    reader gets `glare_frac`'s value under `human_count`. Nothing errors; the
    numbers are just wrong, which is worse.

    Rows are re-keyed by name and rewritten under the current header. Width tells
    us which header a row was written with: a row as wide as the current header
    came after the change, one as wide as the file's own header came before.
    The original file is kept alongside as a `.bak` rather than overwritten.
    """
    if not out_csv.exists():
        return
    rows = list(csv.reader(out_csv.open(newline="")))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    if header == ds.CSV_HEADER:
        return

    backup = out_csv.with_name(out_csv.name + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    backup.write_bytes(out_csv.read_bytes())

    migrated, dropped = [], 0
    for r in data:
        if len(r) == len(ds.CSV_HEADER):
            rec = dict(zip(ds.CSV_HEADER, r))
        elif len(r) == len(header):
            rec = dict(zip(header, r))
        else:
            dropped += 1
            continue
        migrated.append({c: rec.get(c, "") for c in ds.CSV_HEADER})

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ds.CSV_HEADER)
        w.writeheader()
        w.writerows(migrated)

    added = [c for c in ds.CSV_HEADER if c not in header]
    log(f"Migrated {out_csv.name} to the current schema "
        f"(added {', '.join(added) or 'nothing'}; {len(migrated)} row(s) kept"
        + (f", {dropped} unparseable row(s) left in {backup.name}" if dropped else "")
        + f"). Previous file saved as {backup.name}.")


def collect_once(model, stream_url, frames_dir, out_csv):
    """One clip -> three crops -> quality gate -> detection. Returns the row written."""
    now = datetime.now(pytz.timezone(LOCATION["timezone"]))
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # "live" prefix, not the pipeline's "crop", so these files can never be
    # mistaken for scheduled-pipeline crops by anything that globs or parses
    # filenames.
    primary_path = frames_dir / f"live{stamp}.jpg"
    side1_path, side2_path = ds.side_frame_paths(primary_path)

    clip_path = frames_dir / f"live{stamp}.mp4"
    try:
        record_clip(stream_url, clip_path)
        check_clip_dimensions(clip_path)
        # Same three offsets the production cropper uses, so per-frame counts
        # here are sampled the same way as every row already in the dataset.
        extract_frame_at(clip_path, FRAME_TIME_SEC, primary_path)
        extract_frame_at(clip_path, FRAME_TIME_SEC - SIDE_FRAME_OFFSET_SEC, side1_path)
        extract_frame_at(clip_path, FRAME_TIME_SEC + SIDE_FRAME_OFFSET_SEC, side2_path)
    finally:
        # The clip is only a means to the three crops; keeping them would fill
        # a disk fast at this cadence.
        clip_path.unlink(missing_ok=True)

    quality_ok, reason, brightness, lap_var = ds.compute_image_quality(primary_path)
    glare_frac = ds.compute_glare_frac(primary_path)
    if quality_ok:
        counts = ds.run_inference_multi(model, primary_path)
    else:
        # Same policy as detect_surfers.py: don't spend inference on a frame the
        # quality gate already rejected, and don't write a count that would be
        # read downstream as a real observation.
        counts = {k: "" for k in ("surfer_count", "confidence_avg", "frame_count_1",
                                  "frame_count_2", "frame_count_3",
                                  "frame_count_mean", "frame_count_stdev")}

    row = {
        "date": now.strftime("%Y-%m-%d"),
        "time_local": now.strftime("%H:%M:%S"),
        "filename": primary_path.name,
        "quality_ok": quality_ok,
        "quality_reason": reason,
        "brightness": round(brightness, 2),
        "lap_var": round(lap_var, 2),
        "glare_frac": glare_frac,
        "human_count": "",
        **{k: counts[k] for k in ("surfer_count", "confidence_avg", "frame_count_1",
                                  "frame_count_2", "frame_count_3",
                                  "frame_count_mean", "frame_count_stdev")},
    }
    ds.append_row(out_csv, row)
    return row


def ask_about_waiting(now, first_light, last_light):
    """At night, ask rather than silently wait. Returns True to wait, False to stop.

    The camera streams around the clock, but a night frame is unusable — the
    quality gate rejects it, so collecting one produces a row with no count.
    Simply sleeping until morning is worse than it sounds for someone trying the
    script out: it looks like a hang, and it quietly assumes they are willing to
    leave a machine awake all night. So state the situation and let them choose.

    Non-interactive callers never see this (there is nobody to answer): they get
    the waiting behaviour, which is what a long-running collector should do.
    """
    if now < first_light:
        when, target = f"before first light ({first_light.strftime('%-I:%M %p')})", first_light
    else:
        target = first_light + timedelta(days=1)
        when = f"after last light ({last_light.strftime('%-I:%M %p')})"

    wait_hours = max((target - now).total_seconds() / 3600.0, 0)
    print()
    print(f"  It is {now.strftime('%-I:%M %p')}, {when}.")
    print("  The camera streams at night, but the image is too dark to count surfers in —")
    print("  the quality gate rejects those frames, so collecting now records nothing useful.")
    print()
    print(f"  Wait {wait_hours:.1f} hours and start collecting at first light "
          f"({target.strftime('%-I:%M %p')})?")
    print("  That means leaving this computer awake and this script running until then.")
    try:
        answer = input("  [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def daylight_window(tz):
    """(first_light, last_light) for today, in local time."""
    return get_light_window(datetime.now(tz).date(), tz)


def sleep_until(target_dt, tz):
    """Sleep in slices until target_dt, re-reading the clock each time.

    Deliberately not a single long sleep(): the wall clock jumps when a laptop
    suspends and resumes, so the remaining time is recomputed from now() on
    every slice rather than counted down.
    """
    while True:
        remaining = (target_dt - datetime.now(tz)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(SLEEP_SLICE_SEC, remaining))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval-min", type=float, default=DEFAULT_INTERVAL_MIN,
                   help=f"Minutes between grabs (default {DEFAULT_INTERVAL_MIN}, matching the clip pipeline)")
    p.add_argument("--once", action="store_true", help="Collect once, then exit")
    p.add_argument("--ignore-daylight", action="store_true",
                   help="Collect regardless of the daylight window (for testing)")
    p.add_argument("--wait", action="store_true",
                   help="At night, wait for first light without asking")
    p.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return p.parse_args()


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it (brew install ffmpeg / apt install ffmpeg) and retry.")

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    tz = pytz.timezone(LOCATION["timezone"])
    log(f"Model: {ds.MODEL_PATH.name} on {ds.DEVICE}")
    model = ds.load_model()

    migrate_csv_header(args.out_csv)

    stream_url = resolve_stream_url()
    stream_resolved_at = datetime.now(tz)
    log(f"Live stream resolved (no access token used): {stream_url}")
    log(f"Writing frames to {args.frames_dir} and rows to {args.out_csv}")

    failures = 0
    asked_about_dark = False
    while True:
        now = datetime.now(tz)

        if not args.ignore_daylight:
            first_light, last_light = daylight_window(tz)
            is_dark = now < first_light or now > last_light
            if is_dark and not asked_about_dark:
                # Ask once per run, not once per loop.
                asked_about_dark = True
                if args.wait or not sys.stdin.isatty():
                    log("Outside the daylight window; waiting for first light.")
                elif not ask_about_waiting(now, first_light, last_light):
                    log("Stopping. Run again after first light, or use --ignore-daylight "
                        "to capture a night frame anyway (it will not be countable).")
                    return
            if now < first_light:
                log(f"Before first light ({first_light.strftime('%H:%M')}); waiting.")
                sleep_until(min(first_light, now + timedelta(minutes=DARK_POLL_MIN)), tz)
                continue
            if now > last_light:
                # Recompute tomorrow's window at first light rather than now — the
                # dates differ, and today's values would be stale by then.
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=30, second=0, microsecond=0)
                log(f"After last light ({last_light.strftime('%H:%M')}); sleeping until tomorrow.")
                sleep_until(tomorrow, tz)
                continue

        if (now - stream_resolved_at) > timedelta(minutes=STREAM_URL_TTL_MIN):
            try:
                stream_url = resolve_stream_url()
                stream_resolved_at = now
            except Exception as e:
                log(f"WARNING: could not refresh stream URL ({e}); reusing the previous one")

        try:
            row = collect_once(model, stream_url, args.frames_dir, args.out_csv)
            failures = 0
            if row["quality_ok"]:
                log(f"{row['filename']}: {row['surfer_count']} surfers "
                    f"(frames {row['frame_count_1']}/{row['frame_count_2']}/{row['frame_count_3']}, "
                    f"sd {row['frame_count_stdev']}, avg conf {row['confidence_avg']})")
            else:
                log(f"{row['filename']}: skipped detection — {row['quality_reason']} "
                    f"(brightness {row['brightness']}, lap_var {row['lap_var']})")
        except Exception as e:
            failures += 1
            log(f"ERROR on this grab ({type(e).__name__}: {e})")
            if failures >= WARN_AFTER_CONSECUTIVE_FAILURES:
                log(f"WARNING: {failures} consecutive failures. The stream URL may have rotated "
                    f"or the camera may be down; still retrying.")
            # One bad grab (a CDN hiccup, a camera restart) must not end a run
            # that is meant to keep collecting for hours.

        if args.once:
            return
        sleep_until(datetime.now(tz) + timedelta(minutes=args.interval_min), tz)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped.")
