"""
make_conf_band_review.py
------------------------
Builds a review set of just the detections that sit BETWEEN two confidence
thresholds -- the boxes a lower `CONF_THRESH` would newly admit.

Why this exists. Across 220 hand-reviewed frames, 286 of 316 missed surfers had
no box at all, and the question was whether the detector never proposed them or
proposed them below threshold. Re-running inference at a 0.02 floor answered it:
dropping the threshold from 0.195 to 0.10 adds 197 boxes and closes the count
gap almost exactly (3,900 against 3,880 real surfers), and those boxes appear
where the misses were -- frames with more no-box misses gain more boxes,
rho=+0.425, p=4.7e-11.

What it cannot answer is whether those 197 are surfers. Count totals cannot
separate a recovered surfer from a new false positive; that is the same trap
that collapsed the containment-threshold claim once box-level truth existed.
The bounds are wide enough to matter:

    all 197 real        recall    0.919 -> 0.969
    all 197 false       precision 0.987 -> 0.936
    break-even for F1   about 42% must be real

So each box gets looked at. One binary question per box, not a full frame
review -- most frames carry only one or two.

The rendering deliberately shows ONLY the band boxes, numbered, with the
already-accepted boxes drawn faintly for context. A reviewer judging "is this a
surfer" should not have to work out which boxes are the new ones.

Usage:
    python code/make_conf_band_review.py --low 0.10 --high 0.195
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402
from score_reviews import parse_multi_surfer  # noqa: E402

CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
REVIEW = _ROOT / "data" / "reviews" / "review_all.csv"
OUT = _ROOT / "data" / "reviews" / "conf_band_review"

UPSCALE = 2.0
FLOOR = 0.02
BAND_BGR = (90, 227, 157)     # the boxes under judgement
KEPT_BGR = (150, 150, 150)    # already accepted, context only
BAND_ALPHA, KEPT_ALPHA = 0.85, 0.30


def boxes_at_floor(model, img_path):
    tiles, tmp = ds.tile_image_paths(img_path)
    out = []
    for _, xoff, tp in tiles:
        r = model.predict(source=str(tp), conf=FLOOR, device=ds.DEVICE, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
            out.append([x1 + xoff, y1, x2 + xoff, y2, float(b.conf[0].cpu())])
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def survive(boxes, thresh):
    kept = [b for b in boxes if b[4] >= thresh]
    kept = ds.suppress_contained(kept)
    return kept


def _blend(img, draw, alpha):
    overlay = img.copy()
    draw(overlay)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def render(crop, kept, band, out_path, header):
    img = cv2.imread(str(crop))
    if img is None:
        return False
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    def pt(b):
        return ((int(b[0] * UPSCALE), int(b[1] * UPSCALE)),
                (int(b[2] * UPSCALE), int(b[3] * UPSCALE)))

    _blend(img, lambda d: [cv2.rectangle(d, *pt(b), KEPT_BGR, 1) for b in kept], KEPT_ALPHA)

    ordered = sorted(band, key=lambda b: b[0])
    def draw_band(d):
        for i, b in enumerate(ordered, 1):
            p1, p2 = pt(b)
            cv2.rectangle(d, p1, p2, BAND_BGR, 2)
            ly = p1[1] - 5 if p1[1] > 18 else p2[1] + 15
            cv2.putText(d, str(i), (p1[0], ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, BAND_BGR, 1, cv2.LINE_AA)
    _blend(img, draw_band, BAND_ALPHA)

    band_img = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(band_img, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([band_img, img]))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=ds.CONF_THRESH)
    args = ap.parse_args()

    rv = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    rv = rv[rv["date"].str.match(r"^\d{4}-\d{2}-\d{2}$")]
    tp = rv["true_positives"].str.strip()
    s = rv[~tp.str.lower().isin(["na", "n/a"]) & (tp != "")].copy()
    for c in ("true_positives", "missed"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["truth"] = s.true_positives + s.missed
    s["merged"] = [parse_multi_surfer(x)[1] for x in s.multi_surfer_box]
    s["nobox"] = s.missed - s.merged

    OUT.mkdir(parents=True, exist_ok=True)
    model = ds.load_model()
    rows, n_boxes = [], 0
    for i, r in enumerate(s.itertuples(), 1):
        crop = CROPS / r.filename
        if not crop.exists():
            continue
        allb = boxes_at_floor(model, crop)
        kept = survive(allb, args.high)
        lower = survive(allb, args.low)
        keptset = {tuple(np.round(b[:4], 2)) for b in kept}
        band = [b for b in lower if tuple(np.round(b[:4], 2)) not in keptset]
        if not band:
            continue
        name = f"{r.date}_{str(r.time_local).replace(':', '')}_{len(band)}.png"
        header = (f"{r.date} {r.time_local}   {len(band)} box(es) in the "
                  f"{args.low}-{args.high} band   (grey = already accepted)")
        if render(crop, kept, band, OUT / name, header):
            for j, b in enumerate(sorted(band, key=lambda x: x[0]), 1):
                rows.append(dict(image=name, date=r.date, time_local=r.time_local,
                                 box=j, confidence=round(b[4], 3),
                                 is_surfer="", notes="",
                                 filename=r.filename,
                                 nobox_misses_this_frame=int(r.nobox)))
            n_boxes += len(band)
        if i % 40 == 0:
            print(f"  {i}/{len(s)}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "conf_band_review.csv", index=False)
    print(f"\n{n_boxes} box(es) across {len({r['image'] for r in rows})} frame(s) -> {OUT}")
    print("Fill in is_surfer: yes / no  (one row per numbered green box)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
