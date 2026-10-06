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
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

PREDS = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
REVIEW_CSV = _PROJECT_ROOT / "data" / "reviews" / "review_all.csv"
CSV_COLUMNS = ["date", "time_local", "boxes_drawn", "forecast",
               "true_positives", "missed", "missed_poses", "multi_surfer_box",
               "uncertain_missed", "false_positives", "false_positive_box_numbers",
               "false_positive_causes", "uncertain_box_numbers",
               "standing_box_numbers", "sup_box_numbers", "notes",
               # Identifiers last: nothing is typed into them, so they would
               # otherwise push the fill-in columns off the right of the screen.
               # Every script keys on `filename`, by NAME not position.
               "filename", "image"]
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
            # Doubt is split in two (2026-10-05) because the halves move
            # precision and recall in OPPOSITE directions: an unsure box might
            # be a false positive, an unsure miss might be a phantom. A single
            # "uncertain" count could not be scored, because it did not say
            # which. uncertain_missed sits with the missed columns and
            # uncertain_box_numbers with the false-positive ones, so each
            # failure mode is still filled in one pass.
            true_positives="", missed="", missed_poses="", multi_surfer_box="",
            uncertain_missed="",
            false_positives="", false_positive_box_numbers="",
            false_positive_causes="", uncertain_box_numbers="",
            # The two rare classes, kept apart on Joel's definition (2026-10-05):
            # standing means RIDING A WAVE, SUP means a stand-up paddleboarder.
            # They look nothing alike to a detector -- a rider has spray and a
            # wave face behind them, a SUP is upright on flat water with a
            # paddle -- so merging them would blur the very class being built.
            # The training set has 31 standing boxes and 4 SUP against 617
            # sitting and 512 prone, which is what blocks the multi-class
            # detector. Recording WHICH box in WHICH frame makes every review
            # pass a labeling shortlist: frames on disk, boxes already drawn.
            standing_box_numbers="", sup_box_numbers="", notes=""))

    # One sheet for every day (2026-10-05), rather than a review.csv per folder:
    # Joel fills these in a spreadsheet and one file is less to juggle. Rows are
    # only ever ADDED here -- a row whose filename is already present is left
    # exactly as it is, so re-running for a day that is partly reviewed cannot
    # erase an answer. That replaces the old refuse-to-overwrite guard, which
    # was needed only because the whole file used to be rewritten.
    if args.images_only:
        print(f"\n--images-only: {REVIEW_CSV.name} untouched.")
        print(f"{len(rows)} image(s) re-rendered -> {out_dir}")
        return 0

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if REVIEW_CSV.exists():
        sheet = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
        # Rows 2-3 are the column description and worked example, flagged by a
        # leading "#" in the date. Hold them aside so they are neither matched
        # against real filenames nor dragged into the sort, then put them back
        # on top.
        is_guide = sheet["date"].astype(str).str.startswith("#")
        guide, sheet = sheet[is_guide].copy(), sheet[~is_guide].copy()
        known = set(sheet["filename"])
        fresh = [r for r in rows if r["filename"] not in known]
        kept = len(rows) - len(fresh)
        sheet = pd.concat([sheet, pd.DataFrame(fresh)], ignore_index=True) if fresh else sheet
        # Columns the sheet has gained since: keep every one, blank for new rows.
        for col in CSV_COLUMNS:
            if col not in sheet.columns:
                sheet[col] = ""
        sheet = sheet[CSV_COLUMNS].fillna("")
    else:
        sheet, fresh, kept = pd.DataFrame(rows, columns=CSV_COLUMNS), rows, 0
        guide = pd.DataFrame(columns=CSV_COLUMNS)

    sheet = sheet.sort_values(["date", "time_local"]).reset_index(drop=True)
    if len(guide):
        sheet = pd.concat([guide[CSV_COLUMNS], sheet], ignore_index=True)
    sheet.to_csv(REVIEW_CSV, index=False)
    print(f"\n{REVIEW_CSV}: {len(fresh)} row(s) added, {kept} already present and left alone.")
    data_rows = sheet[~sheet["date"].astype(str).str.startswith("#")]
    print(f"  {len(data_rows)} rows total across {data_rows['date'].nunique()} day(s).")

    print(f"\n{len(rows)} hour(s) -> {out_dir}")
    print(f"Fill in {REVIEW_CSV}:")
    print("  true_positives             boxes that are genuinely on a surfer")
    print("  false_positives            boxes on anything else (incl. a 2nd box on one surfer)")
    print("  missed                     real surfers with no box at all")
    print("  false_positive_box_numbers e.g. 3;7   (the numbers drawn on the image)")
    print("  false_positive_causes      e.g. bird;reflection;beach walker;foam;duplicate")
    print(f"  missed_poses               {POSES}  e.g. prone;prone;sitting")
    print("  standing_box_numbers       boxes whose surfer is STANDING = RIDING A WAVE,")
    print("                             e.g. 9;22. Not a SUP -- that has its own column.")
    print("  sup_box_numbers            boxes holding a stand-up paddleboarder. Upright on")
    print("                             flat water with a paddle; a different shape from a")
    print("                             wave rider, and the rarest class of all (4 boxes")
    print("                             labeled). Both are CVAT shortlists.")
    print("  multi_surfer_box           boxes holding more than one surfer, e.g. 2;18 --")
    print("                             a merged box, not a blind spot. Each one costs a")
    print("                             miss, so count it in `missed` too.")
    print("  uncertain_box_numbers      drawn boxes you would not defend either way,")
    print("                             e.g. 2;7 -- same numbering as the FP column.")
    print("                             These bound PRECISION.")
    print("  uncertain_missed           count of maybe-surfers with no box: a shape under")
    print("                             whitewater you cannot resolve. These bound RECALL.")
    print("                             Both are a SUBSET of the counts above, not extra")
    print("                             categories -- still make your best call. Scoring")
    print("                             runs best-case and worst-case; the gap is the")
    print("                             ambiguity budget.")
    print("\n  true_positives + false_positives should equal boxes_drawn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
