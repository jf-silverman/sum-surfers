"""
What the completed bluff-head review says about the zone rule.

24 boxes, every one the zone rule (centre in x 400-680, y 164-180, bottom edge
at or past 179, confidence under 0.50) fires on in a 300-frame production
sample. `measure_shore_zone.py` left the rule unadopted because it rested on
six false positives with a 0.04 gap at the boundary, and named this review as
the thing that would settle it. It did.

SCORING. A suppression rule is judged on what deleting the box does to the
count, which makes a box on a surfer who ALREADY has another box free to
delete -- the count does not change. One of the five boxes Joel marked as a
surfer says exactly that in its note, so it is scored with the 19 rejections
rather than against them. The cost of the rule is the boxes that are the only
box on a real surfer.

Usage:
    python analysis/shore_head_zone/score_bluff_review.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
REVIEW = _ROOT / "data" / "reviews" / "bluff_head_review" / "bluff_head_review.csv"
SAMPLE = Path(__file__).resolve().parent / "sample_boxes_unlabeled.csv"
BAND = _ROOT / "data" / "reviews" / "conf_band_review" / "conf_band_review.csv"

SAMPLE_FRAMES = 300          # frames drawn; 249 of them carried any box at all
DUP_NOTE = "already has a better box"


def main():
    d = pd.read_csv(REVIEW)
    if d.is_surfer.isna().any():
        sys.exit("review incomplete")
    d["y"] = d.is_surfer.str.strip().str.lower().eq("y")
    d["dup"] = d.notes.fillna("").str.contains(DUP_NOTE)
    d["real"] = d.y & ~d.dup          # the only box on a real surfer
    d["cx"] = (d.x1 + d.x2) / 2
    fp, dup, real = int((~d.y).sum()), int(d.dup.sum()), int(d.real.sum())
    print(f"{len(d)} firing boxes reviewed: {fp} shore heads, {dup} duplicate box on an "
          f"already-counted surfer, {real} real surfers")
    print(f"  what the rejections are: {d.loc[~d.y, 'what'].value_counts().to_dict()}")
    print(f"  correctly suppressed {fp + dup}/{len(d)} = {(fp + dup) / len(d):.3f}; "
          f"ratio {(fp + dup) / real:.1f}:1")

    print("\n--- can anything separate the 4 real surfers from the 19 shore heads? ---")
    d["w"], d["h"] = d.x2 - d.x1, d.y2 - d.y1
    for col in ("confidence", "w", "h", "y1", "cx"):
        a, b = d.loc[d.real, col], d.loc[~d.y, col]
        sep = "NO OVERLAP" if (a.min() > b.max() or b.min() > a.max()) else "overlaps"
        print(f"  {col:<11} surfers {a.min():7.2f}-{a.max():7.2f} | "
              f"shore heads {b.min():7.2f}-{b.max():7.2f}   {sep}")
    print("  -> nothing separates them. The rule is a flat trade, not a tunable one.")

    print("\n--- tighter variants, every one fitted to 4 points ---")
    print("  variant            fires  heads  surfers  ratio")
    for lab, m in ([(f"x<={c}", d.cx <= c) for c in (680, 660, 620, 580)]
                   + [(f"conf<{c}", d.confidence < c) for c in (0.45, 0.40, 0.35)]):
        s = d[m]
        f_, r_ = int((~s.y).sum()) + int(s.dup.sum()), int(s.real.sum())
        print(f"  {lab:<18} {len(s):4d}  {int((~s.y).sum()):5d}  {r_:7d}  "
              f"{(f_ / r_) if r_ else float('inf'):5.1f}:1")

    sm = pd.read_csv(SAMPLE)
    cx = (sm.x1 + sm.x2) / 2
    cy = (sm.y1 + sm.y2) / 2
    hit = ((cx >= 400) & (cx <= 680) & (cy >= 164) & (cy <= 180)
           & (sm.conf < 0.50) & (sm.y2 >= 179.0))
    rate = hit.sum() / SAMPLE_FRAMES
    print(f"\n--- what it is worth in production ---")
    print(f"  rule fires on {int(hit.sum())} boxes in {SAMPLE_FRAMES} sampled frames "
          f"= {rate:.3f} boxes/frame")
    print(f"  removes {rate * fp / len(d):.3f} false counts/frame, "
          f"costs {rate * real / len(d):.3f} real surfers/frame")
    print(f"  against a mean hourly count of 14.7 and a ~1.5-surfer per-hour noise floor")

    b = pd.read_csv(BAND)
    b["y"] = b.is_surfer.astype(str).str.strip().str.lower().eq("y")
    sh = b.notes.fillna("").str.lower().str.contains("shore|head of person") & ~b.y
    print(f"\n--- interaction with a lower CONF_THRESH ---")
    print(f"  the band review holds {int(sh.sum())} rejected shore heads over "
          f"{b.image.nunique()} frames, all below the current 0.195")
    for t in (0.15, 0.10):
        n = int((sh & (b.confidence >= t)).sum())
        print(f"    a {t:.2f} floor admits {n} of them = {n / b.image.nunique():.3f}/frame"
              f"  ({n / b.image.nunique() / rate:.0%} of what the zone already catches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
