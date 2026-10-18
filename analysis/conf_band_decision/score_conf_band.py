"""
What the completed confidence-band review says about `CONF_THRESH`.

187 boxes over 113 frames, every one judged by hand. These are the boxes a
threshold below 0.195 would newly admit, so the review is a direct measurement
of what each step down buys and what it costs.

THE SCORING RULE MATTERS MORE THAN ANYTHING ELSE HERE. Joel marked
`is_surfer = n` whenever the surfer under the coral box ALREADY had a grey
(accepted) box -- so `n` means "admitting this box adds a count that should not
be added", whether because the box is on nothing or because it is a second box
on a surfer already counted. That is a stricter rule than the generator's guide
asked for, and it is the right one: production emits a COUNT, and a duplicate
box inflates it exactly as a hallucinated one does. Under the guide's looser
reading ("is a surfer under this box") the band is 61% real; under the rule
actually applied it is 49%.

`boxes_missed` is frame-level and ambiguous in one way that changes the answer,
so both readings are carried through:

  A  surfers with NO box of any colour -- the 92 recovered surfers are NOT in it
  B  surfers still missed after the coral boxes -- the 92 ARE in it

A makes the current undercount 2.96/frame, B makes it 2.14/frame. Which one is
right decides whether 0.10 or 0.15 is the floor; see the RESULT block.

RESULT (2026-10-17, all 187 reviewed)
    band precision          92/187 = 0.492   (0.610 under the guide's looser rule)
    production at 0.195     0.918 on 142 labelled frames
    recommended floor       0.150 -- the deepest step significant under BOTH
                            readings of boxes_missed (A -0.389 p<0.0001;
                            B -0.212 p=0.023), and in the 0.15-0.18 band where
                            admitted-box precision peaks at 0.55-0.58
    0.100 rejected          92 recovered vs 95 spurious; reading B stops
                            improving below 0.14 while precision decays to 0.49
    free win first          28 of the 95 spurious boxes are tree bough (18),
                            shore head (9) and wind sock (1) -- static zones that
                            cost no recall to mask, and lift band precision to 0.579
    pose of the recovered   64 of 92 prone, and prone-SIDE-ON (34) is the biggest
                            group, not end-on (16) as the guide supposed
    NOT ADOPTED             this frame set is biased to undercounting frames;
                            needs the clear-day overcount re-check and Joel's
                            explicit go-ahead. See results.md in this directory.

Usage:
    python analysis/conf_band_decision/score_conf_band.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[2]
REVIEW = _ROOT / "data" / "reviews" / "conf_band_review" / "conf_band_review.csv"
# Production boxes on the 142 COCO-labelled frames, from analysis/shore_head_zone/.
# Used for one number only: the precision of what the detector already emits, as
# the yardstick the band's own precision has to be read against.
LABELLED = _ROOT / "analysis" / "shore_head_zone" / "production_boxes_on_labeled.csv"

PROD_THRESH = 0.195
SWEEP = [0.19, 0.185, 0.18, 0.175, 0.17, 0.16, 0.155, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10]
N_BOOT = 5000
SEED = 7

# Notes keywords. Crude, but the notes are free text and these are the phrases
# Joel actually used; the counts are reported, not acted on.
DUP = ("already|extra box|not needed|3rd box|third box|4th box|5th box|"
       "separate box|tip of board|just the surfboard|both boxes cover|in between")
STATIC = "bough|branch|sock|shore|head of person"


def load():
    d = pd.read_csv(REVIEW)
    blank = d.is_surfer.isna().sum()
    if blank:
        sys.exit(f"{blank} row(s) still unreviewed -- score only a complete set")
    d["y"] = d.is_surfer.astype(str).str.strip().str.lower().eq("y")
    d["note"] = d.notes.fillna("").str.lower()
    d["has_pose"] = d.pose.notna() & d.pose.astype(str).str.strip().ne("")
    return d


def truth_extra(d, frames):
    """Surfers missing from the production count, per frame, under each reading."""
    miss = d.groupby("image").boxes_missed.max().reindex(frames)
    recovered = d.groupby("image").y.sum().reindex(frames)
    return {"A": miss + recovered, "B": miss}


def admitted(d, t, frames):
    s = d[d.confidence >= t]
    g = (s.groupby("image")
          .agg(y=("y", "sum"), n=("y", lambda x: int((~x).sum())))
          .reindex(frames).fillna(0).infer_objects())
    return s, g


def main():
    d = load()
    frames = sorted(d.image.unique())
    rng = np.random.default_rng(SEED)
    print(f"{len(d)} boxes / {len(frames)} frames, all reviewed\n")

    lab = pd.read_csv(LABELLED)
    print(f"yardstick -- production at {PROD_THRESH} on {lab.frame.nunique()} labelled frames: "
          f"{len(lab)} boxes, precision {lab.tp.mean():.3f}")

    print("\n--- the band's own precision ---")
    print(f"  Joel's rule   (adds a surfer the count is missing)  "
          f"{d.y.sum()}/{len(d)} = {d.y.mean():.3f}")
    loose = d.y | (d.note.str.contains(DUP) | d.has_pose)
    print(f"  guide's rule  (a surfer is under the box at all)    "
          f"{loose.sum()}/{len(d)} = {loose.mean():.3f}")
    n = d[~d.y]
    print(f"  of the {len(n)} rejected, {int((n.note.str.contains(DUP) | n.has_pose).sum())} "
          f"are a second box on an already-counted surfer")

    print("\n--- where the rejected boxes are ---")
    for name, pat in [("tree bough, right edge", "bough|branch"),
                      ("shore head, bottom centre", "shore|head of person"),
                      ("wind sock", "sock"),
                      ("sun glare / lens flare", "glare|flare"),
                      ("fog-borderline frame", "foggy")]:
        print(f"  {name:<28} {int((n.note.str.contains(pat)).sum()):3d}")
    static = int(n.note.str.contains(STATIC).sum())
    print(f"  {'in an extendable static zone':<28} {static:3d}"
          f"   -> band precision without them {d.y.sum()}/{len(d) - static} "
          f"= {d.y.sum() / (len(d) - static):.3f}")

    extras = truth_extra(d, frames)
    print("\n--- what each step down admits, and what it does to the count ---")
    print("     T    admit   rec  spur   prec |  "
          "A: d|err|  95% CI          p  |  B: d|err|  95% CI          p")
    for t in SWEEP:
        s, g = admitted(d, t, frames)
        cells = []
        for key in ("A", "B"):
            base = (-extras[key]).abs()
            new = (-extras[key] + g.y + g.n).abs()
            dl = (new - base).to_numpy(float)
            bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean()
                           for _ in range(N_BOOT)])
            p = wilcoxon(dl)[1] if np.any(dl != 0) else 1.0
            cells.append(f"{dl.mean():+.3f}  {np.percentile(bs, 2.5):+.3f} to "
                         f"{np.percentile(bs, 97.5):+.3f}  {p:.4f}")
        print(f"  {t:.3f}   {len(s):4d}  {int(g.y.sum()):4d}  {int(g.n.sum()):4d}  "
              f"{s.y.mean():.3f} |  {cells[0]} |  {cells[1]}")
    for key in ("A", "B"):
        print(f"  reading {key}: current mean |count error| "
              f"{(-extras[key]).abs().mean():.2f}/frame")

    print("\n--- pose of the 92 recovered surfers ---")
    canon = (d.loc[d.y, "pose"].fillna("(blank)").astype(str).str.strip()
             .replace({"s": "prone-s", "e": "prone-e", "a": "prone-a",
                       "s / e": "two in one box", "a & a": "two in one box",
                       "e & a": "two in one box"}))
    for k, v in canon.value_counts().items():
        print(f"  {k:<16} {v:3d}")
    prone = int(canon.str.startswith("prone").sum())
    print(f"  prone of any orientation: {prone} of {int(d.y.sum())}")

    js = pd.to_numeric(d.loc[d.y, "js_confidence"], errors="coerce")
    print(f"\n  reviewer confidence filled on {js.notna().sum()} of {int(d.y.sum())} "
          f"recovered boxes; median {js.median():.2f}, at or below 0.7 on "
          f"{int((js <= 0.7).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
