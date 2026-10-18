"""
Where the tree-bough false positives actually are, and what zone catches them.

`TREE_MASK = (1268, 58, 1280, 87)` was derived in August 2026 from a 50-frame
spot-check. The completed confidence-band review found **18 more tree-bough
false positives that escape it** -- the single largest false-positive class in
that review, and the bough moves with the season and the wind, so a zone fitted
to one month's frames was never going to hold.

The band review CSV carries no coordinates, so the boxes are recovered from
`analysis/conf_thresh_recheck/band_boxes_with_coords.csv` (written by
`recheck_threshold.py --dump-band`, which re-detects the review days and dumps
every box in the 0.10-0.195 band) and matched back to Joel's notes by filename
and nearest x1. Matching by rank would be wrong: the review set keeps 22
reviewed rows that the corrected `survive()` no longer produces, so the two
orderings do not line up.

A zone is only acceptable if it clips nothing real. Every candidate is scored
against `analysis/shore_head_zone/production_boxes_on_labeled.csv` -- 3,404
production boxes on 142 COCO-labelled frames, each already matched TP or FP at
IoU 0.5 -- and a candidate that contains a single true positive is rejected.

Usage:
    python analysis/tree_bough_zone/derive_tree_zone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

BAND = _ROOT / "analysis" / "conf_thresh_recheck" / "band_boxes_with_coords.csv"
REVIEW = _ROOT / "data" / "reviews" / "conf_band_review" / "conf_band_review.csv"
LABELLED = _ROOT / "analysis" / "shore_head_zone" / "production_boxes_on_labeled.csv"
MATCH_TOL = 6.0          # px on x1; band boxes within a frame are far apart
BOUGH_NOTE = "bough|branch"


def main():
    if not BAND.exists():
        sys.exit(f"missing {BAND} -- run recheck_threshold.py --dump-band first")
    band = pd.read_csv(BAND)
    rv = pd.read_csv(REVIEW)
    rv["y"] = rv.is_surfer.astype(str).str.strip().str.lower().eq("y")
    rv["note"] = rv.notes.fillna("").str.lower()
    bough = rv[~rv.y & rv.note.str.contains(BOUGH_NOTE)].copy()
    print(f"{len(band)} band boxes re-detected with coordinates; "
          f"{len(bough)} reviewed as a tree bough")

    # Match each reviewed bough row to a re-detected box: same frame, and the
    # box number is the 1-based rank of x1 among that frame's band boxes.
    hits, missed = [], []
    for r in bough.itertuples():
        g = band[band.filename == r.filename].sort_values("x1").reset_index(drop=True)
        if r.box - 1 < len(g):
            cand = g.iloc[r.box - 1]
            if abs(cand.conf - r.confidence) < 0.002:
                hits.append(cand)
                continue
        near = g[np.isclose(g.conf, r.confidence, atol=0.002)]
        (hits.append(near.iloc[0]) if len(near) else missed.append(r.filename))
    b = pd.DataFrame(hits)
    print(f"  matched {len(b)} by confidence; {len(missed)} unmatched "
          f"(reviewed rows the corrected survive() no longer produces)")
    if not len(b):
        sys.exit("nothing matched -- cannot derive a zone")

    b["cx"], b["cy"] = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    print(f"\n--- where they are (n={len(b)}) ---")
    for c in ("x1", "x2", "cx", "y1", "y2", "cy", "conf"):
        print(f"  {c:<5} min {b[c].min():7.1f}  med {b[c].median():7.1f}  "
              f"max {b[c].max():7.1f}")
    cur = ds.TREE_MASK
    inside = ((b.cx >= cur[0]) & (b.cx <= cur[2])
              & (b.cy >= cur[1]) & (b.cy <= cur[3]))
    print(f"\n  current TREE_MASK {cur} already catches {int(inside.sum())} of {len(b)}")

    lab = pd.read_csv(LABELLED)
    lab["cx"], lab["cy"] = (lab.x1 + lab.x2) / 2, (lab.y1 + lab.y2) / 2
    tp = lab[lab.tp == 1]
    print(f"\n--- candidate zones, scored against {len(tp)} true positives "
          f"on {lab.frame.nunique()} labelled frames ---")
    print("  zone (x1,y1,x2,y2)              boughs caught   TPs clipped   verdict")
    cands = [cur]
    x1 = float(np.floor(b.x1.min()))
    for pad_y in (0, 2, 4):
        y1 = float(np.floor(b.cy.min())) - pad_y
        y2 = float(np.ceil(b.cy.max())) + pad_y
        cands.append((x1, max(0.0, y1), 1280.0, min(180.0, y2)))
    cands.append((x1, 0.0, 1280.0, 180.0))
    for z in cands:
        zx1, zy1, zx2, zy2 = z
        got = int((((b.cx >= zx1) & (b.cx <= zx2)
                    & (b.cy >= zy1) & (b.cy <= zy2))).sum())
        clip = int((((tp.cx >= zx1) & (tp.cx <= zx2)
                     & (tp.cy >= zy1) & (tp.cy <= zy2))).sum())
        tag = "REJECT - clips real surfers" if clip else ("ok" if got else "-")
        print(f"  ({zx1:7.1f},{zy1:6.1f},{zx2:7.1f},{zy2:6.1f})  {got:9d}      "
              f"{clip:9d}     {tag}")

    fp = lab[lab.tp == 0]
    print(f"\n--- for reference, the labelled set's own right-edge false positives ---")
    for lo in (1240, 1250, 1260, 1268):
        print(f"  centre x >= {lo}: {int((fp.cx >= lo).sum())} FPs, "
              f"{int((tp.cx >= lo).sum())} TPs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
