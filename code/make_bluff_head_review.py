"""
make_bluff_head_review.py
-------------------------
Renders every production detection that the candidate BLUFF-HEAD rule would
drop, so the rule can be judged on more than six examples.

Background (analysis/shore_head_zone/results.md). The ROI is rows 420-600 of a
720-row camera frame, and the bluff path -- fence, bench, people standing and
watching -- sits at roughly y 590-660, below the crop. Only the top of a
standing head reaches the image, so a bluff person's box is ALWAYS clipped by
the ROI's bottom edge. On the 142 labelled frames that signature plus an x
range plus a raised in-zone threshold removes 5 of 6 false positives for zero
lost surfers -- but six false positives is not enough to adopt a production
rule, and the gap between the highest false positive (0.49) and the lowest true
positive (0.53) is 0.04.

This set is the other 24. They come from the 300-frame random sample in
analysis/shore_head_zone/sample_boxes_unlabeled.csv, which is drawn from the
production record rather than the labelled pool, so the verdicts answer the
question the labelled pool cannot: among boxes the rule WOULD fire on, how many
are actually people on the bluff.

Each frame gets two views, because these boxes are 10-15px wide at the very
bottom edge and neither view alone is enough:

  the strip   the full 1280-wide ROI at 2x, every production box drawn, so the
              box under judgement can be compared against the surfers around it
  the zoom    200px of ROI around the box at 8x, so the blob is actually legible

Output:
    data/reviews/bluff_head_review/<date>_<time>_<n>.png
    data/reviews/bluff_head_review/bluff_head_review.csv
    data/reviews/bluff_head_review/bluff_head_guide.md

Usage:
    python code/make_bluff_head_review.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

SAMPLE = _ROOT / "analysis" / "shore_head_zone" / "sample_boxes_unlabeled.csv"
CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
OUT = _ROOT / "data" / "reviews" / "bluff_head_review"

# The candidate rule, from analysis/shore_head_zone/.
ZONE_X1, ZONE_X2 = 400, 680
EDGE_Y = 179.0
RULE_CONF_MAX = 0.50

STRIP_UPSCALE = 2
ZOOM_UPSCALE = 8
# 160 ROI px either side at 8x is exactly the 2560px width of the 2x strip, so
# the two views line up instead of the zoom sitting in a field of black.
ZOOM_HALF_W = 160
# The zoom starts at ROI row 100. Everything the rule can fire on is clipped by
# row 180, and the 100 rows above it are open water that only make the file big.
ZOOM_Y0 = 100

BAND_BGR = (97, 111, 255)   # coral -- the box under judgement
KEPT_BGR = (150, 150, 150)  # other production boxes, context only


def _blend(img, draw, alpha):
    overlay = img.copy()
    draw(overlay)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def render(crop_path, target, others, out_path, header):
    img = cv2.imread(str(crop_path))
    if img is None:
        return False

    strip = cv2.resize(img, None, fx=STRIP_UPSCALE, fy=STRIP_UPSCALE,
                       interpolation=cv2.INTER_CUBIC)

    def pt(b, s, xoff=0, yoff=0):
        return ((int((b[0] - xoff) * s), int((b[1] - yoff) * s)),
                (int((b[2] - xoff) * s), int((b[3] - yoff) * s)))

    _blend(strip, lambda d: [cv2.rectangle(d, *pt(b, STRIP_UPSCALE), KEPT_BGR, 1)
                             for b in others], 0.30)
    _blend(strip, lambda d: cv2.rectangle(d, *pt(target, STRIP_UPSCALE), BAND_BGR, 1), 0.85)

    cx = (target[0] + target[2]) / 2
    x0 = int(max(0, min(img.shape[1] - 2 * ZOOM_HALF_W, cx - ZOOM_HALF_W)))
    x1 = x0 + 2 * ZOOM_HALF_W
    zoom = cv2.resize(img[ZOOM_Y0:, x0:x1], None, fx=ZOOM_UPSCALE, fy=ZOOM_UPSCALE,
                      interpolation=cv2.INTER_NEAREST)
    _blend(zoom, lambda d: [cv2.rectangle(d, *pt(b, ZOOM_UPSCALE, x0, ZOOM_Y0),
                                          KEPT_BGR, 2) for b in others], 0.30)
    _blend(zoom, lambda d: cv2.rectangle(d, *pt(target, ZOOM_UPSCALE, x0, ZOOM_Y0),
                                         BAND_BGR, 2), 0.85)

    width = max(strip.shape[1], zoom.shape[1])

    def pad(a):
        return np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)))

    def label(text, h=30, size=0.6):
        bar = np.zeros((h, width, 3), np.uint8)
        cv2.putText(bar, text, (8, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, size,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return bar

    cv2.imwrite(str(out_path), np.vstack([
        label(header, 34),
        pad(strip),
        label(f"zoom {ZOOM_UPSCALE}x  --  ROI x {x0}-{x1}, rows {ZOOM_Y0}-180   "
            f"(coral = the box in question, grey = other accepted boxes)", 28, 0.55),
        pad(zoom),
    ]))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf-max", type=float, default=RULE_CONF_MAX)
    args = ap.parse_args()

    d = pd.read_csv(SAMPLE)
    d["cx"] = (d.x1 + d.x2) / 2
    fires = ((d.y2 >= EDGE_Y) & (d.cx >= ZONE_X1) & (d.cx <= ZONE_X2)
             & (d.conf < args.conf_max))
    hits = d[fires].copy()
    print(f"{len(hits)} box(es) in {hits.filename.nunique()} frame(s) "
          f"would be dropped by the rule")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn, g in hits.groupby("filename", sort=True):
        m = re.match(r"crop(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-", fn)
        if not m:
            print(f"  skipped (unparsable name) {fn}")
            continue
        date, hh, mm = m.group(1), m.group(2), m.group(3)
        others = [b for b in d[d.filename == fn][["x1", "y1", "x2", "y2", "conf"]].values
                  if not any(abs(b[0] - t) < 0.01 for t in g.x1.values)]
        for j, r in enumerate(sorted(g.itertuples(), key=lambda z: z.x1), 1):
            target = [r.x1, r.y1, r.x2, r.y2, r.conf]
            name = f"{date}_{hh}{mm}_{j}.png"
            header = (f"{date} {hh}:{mm}   conf {r.conf:.3f}   "
                      f"box {r.x2 - r.x1:.0f}x{r.y2 - r.y1:.0f}px at ROI x "
                      f"{int((r.x1 + r.x2) / 2)}, y1 {r.y1:.0f}   "
                      f"(the rule would DROP this box)")
            if not render(CROPS / fn, target, others, OUT / name, header):
                print(f"  missing crop {fn}")
                continue
            rows.append(dict(
                image=name, date=date, time_local=f"{hh}:{mm}", box=j,
                confidence=round(r.conf, 3),
                # The one question. y = a surfer in the water, n = anything
                # else; a person standing on the bluff is the expected n.
                is_surfer="",
                # bluff / shore / wave / flare / other -- what it actually is,
                # so a wrong drop can be told from a right one for the wrong
                # reason.
                what="",
                js_confidence="", notes="",
                x1=round(r.x1, 1), y1=round(r.y1, 1),
                x2=round(r.x2, 1), y2=round(r.y2, 1),
                filename=fn))

    pd.DataFrame(rows).to_csv(OUT / "bluff_head_review.csv", index=False)
    print(f"{len(rows)} row(s) over {len({r['image'] for r in rows})} image(s) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
