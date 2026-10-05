"""
make_review_set.py
------------------
Builds a hand-review set for one whole day: every counted hour rendered with
NUMBERED detection boxes, plus a CSV to fill in.

Why numbered. The review sets already in `data/reviews/` record a count against
the model's count, which cannot separate the two ways a count can be right by
accident: 10 boxes against 10 real surfers might be 10 correct, or 8 correct
plus 2 false boxes and 2 missed people. Numbering each box lets a reviewer say
*which* box is wrong, which is what splits precision from recall.

The three categories, with no fourth: a box on a real surfer is a true positive,
a box on anything else is a false positive (including a second box on one
surfer), and a real surfer with no box is a false negative. There is no true
negative in detection -- no countable set of places the model correctly left
alone -- which is why precision and recall are the measures and accuracy is not.

    precision = TP / (TP + FP)   of the boxes drawn, how many were real
    recall    = TP / (TP + FN)   of the real surfers, how many were found

Pose values match the CVAT `pose` attribute already used on the training set, so
missed-surfer poses here can be compared against what the model trained on:
sitting, prone, standing, SUP, wipeout, unknown.

Usage:
    python code/make_review_set.py --date 2026-10-02
    python code/make_review_set.py --date 2026-10-02 --out-dir data/reviews/my_review
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

PREDS = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
FORECASTS = _PROJECT_ROOT / "data" / "forecasts"
UPSCALE = 2.0                       # the ROI strip is 180px tall; boxes need room to label
BOX_BGR = (90, 227, 157)
# Boxes are drawn translucent so a surfer is never hidden under the line meant
# to point at it -- the whole job here is judging what is under the box. Joel
# asked for 60-70% transparent (2026-10-05). The label digits are kept less
# transparent than the box: they sit above the box rather than over the water,
# and they are how false_positive_box_numbers gets recorded, so they have to
# stay readable on a bright frame.
BOX_ALPHA = 0.35                    # 65% transparent
LABEL_ALPHA = 0.55
POSES = "sitting | prone | standing | SUP | wipeout | unknown"


def _blend(img, draw, alpha):
    """Draw via `draw(overlay)` and composite it onto img at `alpha`.

    Blending the whole frame is safe because the overlay starts as a copy: every
    pixel the draw call does not touch blends with itself and is unchanged.
    """
    overlay = img.copy()
    draw(overlay)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def render(crop_path, boxes, out_path, header, box_alpha=BOX_ALPHA,
           label_alpha=LABEL_ALPHA):
    img = cv2.imread(str(crop_path))
    if img is None:
        return False
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    # Boxes left to right, so the numbering matches how the eye scans the strip.
    ordered = sorted(boxes, key=lambda b: b[0])
    corners = []
    for b in ordered:
        corners.append(((int(b[0] * UPSCALE), int(b[1] * UPSCALE)),
                        (int(b[2] * UPSCALE), int(b[3] * UPSCALE))))

    def draw_boxes(dst):
        for p1, p2 in corners:
            cv2.rectangle(dst, p1, p2, BOX_BGR, 1)

    def draw_labels(dst):
        for i, (p1, p2) in enumerate(corners, start=1):
            # Label above the box, or below when it would fall off the top edge.
            ly = p1[1] - 4 if p1[1] > 16 else p2[1] + 13
            cv2.putText(dst, str(i), (p1[0], ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, BOX_BGR, 1, cv2.LINE_AA)

    _blend(img, draw_boxes, box_alpha)
    _blend(img, draw_labels, label_alpha)
    band = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(band, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([band, img]))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--box-alpha", type=float, default=BOX_ALPHA,
                   help=f"Box opacity 0-1; lower is more transparent (default {BOX_ALPHA})")
    p.add_argument("--label-alpha", type=float, default=LABEL_ALPHA,
                   help=f"Number opacity 0-1 (default {LABEL_ALPHA})")
    p.add_argument("--images-only", action="store_true",
                   help="Re-render the images and leave review.csv alone. Use when "
                        "changing how boxes are drawn for a day already reviewed.")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        _PROJECT_ROOT / "data" / "reviews" / f"full_day_{args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDS, float_precision="round_trip")
    day = df[(df.date == args.date) & (df.quality_ok.astype(str) == "True")
             & df.surfer_count.notna()].sort_values("time_local")
    if day.empty:
        print(f"No quality-passed rows for {args.date}.")
        return 1

    fc_path = FORECASTS / f"forecast_{args.date}.csv"
    fc = {}
    if fc_path.exists():
        f = pd.read_csv(fc_path)
        fc = dict(zip(f.hour_local.astype(str).str[:2].astype(int), f.predicted))
        print(f"Forecast found for {args.date} ({len(fc)} hours).")
    else:
        print(f"NOTE: no recorded forecast for {args.date}; the forecast column will be blank.")

    model = ds.load_model()
    rows = []
    for r in day.itertuples():
        crop = ds.CROPS_DIR / r.filename
        if not crop.exists():
            print(f"  missing crop: {r.filename}")
            continue
        boxes = ds.run_inference_with_boxes(model, crop)
        hour = int(str(r.time_local)[:2])
        pred = fc.get(hour)
        header = (f"{args.date}  {r.time_local}   {len(boxes)} boxes drawn"
                  + (f"   forecast {pred:.1f}" if pred is not None else ""))
        name = f"{str(r.time_local).replace(':', '')}_boxes_{len(boxes):02d}.png"
        if render(crop, boxes, out_dir / name, header,
                  box_alpha=args.box_alpha, label_alpha=args.label_alpha):
            print(f"  {name}")
        rows.append(dict(
            # Column order follows how Joel rearranged it while reviewing
            # 2026-09-23: filename first, and the missed-surfer columns kept
            # together before the false-positive ones, so a reviewer fills in
            # one failure mode at a time rather than jumping between them.
            filename=r.filename, date=args.date, time_local=r.time_local, image=name,
            boxes_drawn=len(boxes), forecast=("" if pred is None else round(pred, 1)),
            # --- to fill in ---
            true_positives="", missed="", missed_poses="", uncertain="",
            false_positives="", false_positive_box_numbers="",
            false_positive_causes="", notes=""))

    csv_path = out_dir / "review.csv"
    # A plain "w" here once stood between a re-render and a day of hand-review:
    # the fill-in columns are written blank, so regenerating images would have
    # erased the answers. Refuse rather than overwrite.
    if args.images_only:
        print(f"\n--images-only: left {csv_path} untouched.")
        print(f"{len(rows)} image(s) re-rendered -> {out_dir}")
        return 0
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        fill_cols = [c for c in ("true_positives", "missed", "missed_poses", "uncertain",
                                 "false_positives", "false_positive_box_numbers",
                                 "false_positive_causes", "notes") if c in existing.columns]
        if fill_cols and int(existing[fill_cols].notna().any(axis=1).sum()):
            n = int(existing[fill_cols].notna().any(axis=1).sum())
            print(f"\nREFUSING to overwrite {csv_path}: {n} row(s) already have review "
                  f"data in them.\nRe-run with --images-only to redraw the images and keep "
                  f"the CSV as it is.")
            return 1
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} hour(s) -> {out_dir}")
    print(f"Fill in {csv_path}:")
    print("  true_positives             boxes that are genuinely on a surfer")
    print("  false_positives            boxes on anything else (incl. a 2nd box on one surfer)")
    print("  missed                     real surfers with no box at all")
    print("  false_positive_box_numbers e.g. 3;7   (the numbers drawn on the image)")
    print("  false_positive_causes      e.g. bird;reflection;beach walker;foam;duplicate")
    print(f"  missed_poses               {POSES}  e.g. prone;prone;sitting")
    print("  uncertain                  of the above, how many calls you could not")
    print("                             confidently make -- a surfer you think is under")
    print("                             whitewater, a shape you cannot resolve. Scored")
    print("                             twice, once counting these as errors and once")
    print("                             excluding them; the gap is the ambiguity budget.")
    print("\n  true_positives + false_positives should equal boxes_drawn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
