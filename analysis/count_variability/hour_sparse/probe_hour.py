"""
What one hour actually looks like when a wave set passes through it.

Background: on 2026-09-29 the 18:17 sample counted 7 surfers against a forecast
of 22.5 -- the largest forecast miss of the day. Frames either side showed 17-20.
A full set had broken across the frame at exactly that moment, hiding the
lineup. The pipeline's own multi-frame averaging did not catch it: its three
frames sit +/-1.5 SECONDS apart, so all three landed inside the same broken set
and reported 8/8/6 with a stdev of 0.94 -- a confident-looking wrong answer.

This samples the same hour properly: 15 clips at 4-minute spacing centred on
18:17 (17:49 to 18:45), and for each clip the same three frames 1.5s apart that
the pipeline uses. 45 images in total, each with its own count, Laplacian
variance and brightness, so within-clip variation (seconds) can be separated
from across-clip variation (minutes).

Writes analysis/count_variability/hour_sparse/hour_probe.csv. Nothing touches
predictions.csv -- these are extra looks at one hour, not pipeline observations.

Usage:
    python analysis/count_variability/hour_sparse/probe_hour.py
    python analysis/count_variability/hour_sparse/probe_hour.py --date 2026-09-29 --centre 18:17
"""
import argparse
import csv
import datetime as dt
import random
import sys
import time
from pathlib import Path

import cv2
import pytz

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "code"))
import get_clips as gc          # noqa: E402
import detect_surfers as ds     # noqa: E402

HERE = Path(__file__).resolve().parent
CLIP_DIR = HERE / "clips"
FRAME_DIR = HERE / "frames"
OUT_CSV = HERE / "hour_probe.csv"

TZ = pytz.timezone("America/Los_Angeles")
N_CLIPS = 15
SPACING_MIN = 4
# The pipeline's own offsets: FRAME_TIME_SEC 2.5 with SIDE_FRAME_OFFSET_SEC 1.5.
FRAME_OFFSETS_SEC = (1.0, 2.5, 4.0)
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 420, 1280, 180


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-09-29")
    p.add_argument("--centre", default="18:17")
    args = p.parse_args()

    y, m, d = map(int, args.date.split("-"))
    hh, mm = map(int, args.centre.split(":"))
    centre = TZ.localize(dt.datetime(y, m, d, hh, mm))
    half = (N_CLIPS - 1) // 2
    offsets = [(i - half) * SPACING_MIN for i in range(N_CLIPS)]

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    model = ds.load_model()
    rows = []

    for k, off in enumerate(offsets):
        when = centre + dt.timedelta(minutes=off)
        mp4 = CLIP_DIR / f"{when:%Y%m%d_%H%M}.mp4"
        if not mp4.exists():
            if k:
                time.sleep(gc.REQUEST_BASE_DELAY_SEC + random.uniform(0, gc.REQUEST_JITTER_SEC))
            start_ms = int(when.timestamp() * 1000)
            try:
                gc.download_clip(start_ms, start_ms + gc.CLIP_DURATION_SEC * 1000, mp4)
            except Exception as e:
                print(f"  {when:%H:%M}  clip failed: {type(e).__name__}: {str(e)[:70]}")
                continue

        cap = cv2.VideoCapture(str(mp4))
        for fi, sec in enumerate(FRAME_OFFSETS_SEC, start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok:
                print(f"  {when:%H:%M}  frame {fi} ({sec}s) unreadable")
                continue
            crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            jpg = FRAME_DIR / f"{when:%H%M}_f{fi}.jpg"
            cv2.imwrite(str(jpg), crop)
            boxes = ds.run_inference_with_boxes(model, jpg)
            okq, reason, bright, lap = ds.compute_image_quality(jpg)
            rows.append(dict(clip_time=f"{when:%H:%M}", offset_min=off, frame=fi,
                             frame_sec=sec, count=len(boxes), lap_var=round(lap, 1),
                             brightness=round(bright, 1), quality_ok=okq, reason=reason))
        cap.release()
        got = [r for r in rows if r["clip_time"] == f"{when:%H:%M}"]
        if got:
            cs = [r["count"] for r in got]
            lv = [r["lap_var"] for r in got]
            print(f"  {when:%H:%M}  {off:+3d}min  counts {cs}  lap_var {lv}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} frame row(s) to {OUT_CSV}")


if __name__ == "__main__":
    main()
