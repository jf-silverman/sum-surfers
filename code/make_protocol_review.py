"""Builds the two passes that settle which counting protocol is the truth.

THE QUESTION. Three truth protocols disagree, and the disagreement has blocked
three separate detection changes:

  CVAT box annotation   careful, zoomed, one box per surfer
  `reviewdays`          box-by-box review of the 0.195 boxes, truth = TP + missed
  `spotcheck`           a direct `my_count` of the scene

On the frames where they overlap, `spotcheck` reads **3.27 surfers low against
CVAT (14 of 15 frames below, 0 above, sign p = 0.0001)** while `reviewdays` sits
on it (+0.27, 5 below / 5 above, p = 0.74). Correcting `spotcheck` for that one
bias flips it from rejecting the crowd-dependent rules to endorsing them.

What that does NOT settle is which protocol is RIGHT. CVAT and `reviewdays` share
a mode of looking -- slow, magnified, box-assisted -- even though they share
neither frames nor task. So the 11% could be real surfers a glance misses, or
marginal blobs that close looking manufactures. The two readings have opposite
consequences and the existing data cannot separate them, because every frame
with a careful count was counted carefully.

THE DESIGN. One set of fresh frames, counted twice by the same reviewer under the
two modes, unprimed pass first:

  PASS A (this script, --pass a)  clean crops. No boxes, no detector count, no
      frame identifiers that order by crowd. One direct count per frame plus the
      reviewer's own +- uncertainty. This is the `spotcheck` mode.

  PASS B (--pass b, generated only AFTER pass A is returned)  the same frames,
      2x upscaled with the production boxes drawn, numbered, in a SHUFFLED and
      different order. Per-box real/not plus a count of surfers with no box.
      This is the `reviewdays` mode.

Pass B is deliberately not generated up front. If both sheets exist at once the
passes get done together, the second anchors on the first, and the paired
difference -- the entire measurement -- collapses toward zero.

SAMPLING. 30 frames, 6 from each of five crowd strata by production count
(0-2, 3-9, 10-19, 20-34, 35+), drawn from the current model era so pass B's boxes
are the boxes production actually emits. Frames already in `reviewdays`,
`spotcheck` or the CVAT splits are excluded, so neither pass is a re-count of
something the reviewer has seen. Stratifying matters because the one thing the
15-frame estimate could not resolve is whether the gap grows with the crowd
(r = -0.08, p = 0.77, n = 15).

POWER. Paired, two-sided, alpha 0.05: n = 30 detects a mean protocol difference
of 0.53 x SD, which at the observed SD of 2.5 is 1.32 surfers. The effect under
test is about 3.3, so 30 frames is ample and also buys the crowd strata.

Usage:
    python code/make_protocol_review.py --pass a
    python code/make_protocol_review.py --pass b     # only after A is returned
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
PREDS = _ROOT / "data" / "predictions" / "predictions.csv"
SPLITS = _ROOT / "data" / "cvat_out_coco" / "splits_tiled_20260921_fog"
SPOTCHECK = _ROOT / "data" / "reviews" / "model_spotcheck_50" / "review_counts.csv"
REVIEW_ALL = _ROOT / "data" / "reviews" / "review_all.csv"
OUT = _ROOT / "data" / "reviews" / "protocol_review"

ERA_START = "2026-09-21"        # current weights; pass B must show current boxes
STRATA = [(0, 2), (3, 9), (10, 19), (20, 34), (35, 999)]
PER_STRATUM = 6
SEED = 31
UPSCALE = 2.0
BOX_BGR = (97, 111, 255)        # coral, the project's "decide this" colour
BOX_ALPHA = 0.70


def already_seen():
    """Frames the reviewer has already counted, under any protocol."""
    seen = set()
    for sp in ("train", "val", "test"):
        j = json.load(open(SPLITS / f"instances_{sp}.json"))
        for i in j["images"]:
            seen.add(i["file_name"].rsplit("_tile", 1)[0] + ".jpg")
    if SPOTCHECK.exists():
        seen |= set(pd.read_csv(SPOTCHECK).filename.dropna())
    if REVIEW_ALL.exists():
        rv = pd.read_csv(REVIEW_ALL, dtype=str, keep_default_na=False)
        seen |= set(rv.filename[rv.filename != ""])
    return seen


def pick_frames():
    d = pd.read_csv(PREDS)
    d = d[(d.date >= ERA_START) & (d.quality_ok.astype(str) == "True")]
    d = d[d.surfer_count.notna()].copy()
    seen = already_seen()
    d = d[~d.filename.isin(seen)]
    d = d[[(CROPS / f).exists() for f in d.filename]]
    print(f"{len(d)} candidate frames (current era, quality-passed, never reviewed)")
    rng = np.random.default_rng(SEED)
    picked = []
    for lo, hi in STRATA:
        s = d[(d.surfer_count >= lo) & (d.surfer_count <= hi)]
        n = min(PER_STRATUM, len(s))
        if n < PER_STRATUM:
            print(f"  stratum {lo}-{hi}: only {len(s)} available, taking {n}")
        take = s.sample(n, random_state=int(rng.integers(1e6)))
        picked.append(take.assign(stratum=f"{lo}-{hi}"))
    return pd.concat(picked, ignore_index=True)


def render_clean(crop, out_path, label):
    """Pass A: the crop as the detector sees it, upscaled, nothing drawn."""
    img = cv2.imread(str(crop))
    if img is None:
        return False
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([bar, img]))
    return True


def render_boxed(crop, boxes, out_path, label):
    """Pass B: the same crop with the production boxes drawn and numbered."""
    img = cv2.imread(str(crop))
    if img is None:
        return False
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ordered = sorted(boxes, key=lambda b: b[0])
    overlay = img.copy()
    for i, b in enumerate(ordered, 1):
        p1 = (int(b[0] * UPSCALE), int(b[1] * UPSCALE))
        p2 = (int(b[2] * UPSCALE), int(b[3] * UPSCALE))
        cv2.rectangle(overlay, p1, p2, BOX_BGR, 1)
        ly = p1[1] - 5 if p1[1] > 18 else p2[1] + 15
        cv2.putText(overlay, str(i), (p1[0], ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, BOX_BGR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0, dst=img)
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([bar, img]))
    return ordered


def build_pass_a(frames):
    d = OUT / "pass_a_direct_count"
    d.mkdir(parents=True, exist_ok=True)
    # Shuffled, zero-padded names. Nothing in the filename or the header leaks
    # the date, the crowd level or the detector's count -- a reviewer who can see
    # the frames are ordered by crowd is anchored before counting the first one.
    order = frames.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    rows = []
    for i, r in enumerate(order.itertuples(), 1):
        name = f"A{i:02d}.png"
        if not render_clean(CROPS / r.filename, d / name, f"A{i:02d}"):
            continue
        rows.append(dict(image=name, my_count="", my_ci_plusminus="", notes=""))
    pd.DataFrame(rows).to_csv(OUT / "pass_a.csv", index=False)
    # The key stays out of the review folder so it cannot be read mid-pass.
    order.assign(image=[f"A{i:02d}.png" for i in range(1, len(order) + 1)]) \
         .to_csv(OUT / ".pass_a_key.csv", index=False)
    print(f"\npass A: {len(rows)} frames -> {d}")
    print(f"  sheet -> {OUT / 'pass_a.csv'}   (fill my_count and my_ci_plusminus)")
    print(f"  key   -> {OUT / '.pass_a_key.csv'}  (do not open before the pass)")


def build_pass_b():
    key = OUT / ".pass_a_key.csv"
    if not key.exists():
        sys.exit("pass A has not been built; run --pass a first")
    a = OUT / "pass_a.csv"
    filled = pd.read_csv(a)
    if filled.my_count.isna().all():
        sys.exit(f"{a} has no counts yet. Pass B must not be generated until "
                 f"pass A is returned, or the second pass anchors on the first.")
    frames = pd.read_csv(key)
    d = OUT / "pass_b_box_review"
    d.mkdir(parents=True, exist_ok=True)
    model = ds.load_model()
    # A different shuffle from pass A: same frames, different order, so the
    # reviewer is not walking the two sheets in parallel from memory.
    order = frames.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    rows = []
    for i, r in enumerate(order.itertuples(), 1):
        boxes = ds.run_inference_with_boxes(model, CROPS / r.filename)
        name = f"B{i:02d}.png"
        drawn = render_boxed(CROPS / r.filename, boxes, d / name, f"B{i:02d}")
        if drawn is False:
            continue
        for j, b in enumerate(drawn, 1):
            rows.append(dict(image=name, box=j, confidence=round(b[4], 3),
                             is_surfer="", notes=""))
        if not drawn:
            rows.append(dict(image=name, box="", confidence="", is_surfer="", notes=""))
    boxes_df = pd.DataFrame(rows)
    boxes_df.to_csv(OUT / "pass_b_boxes.csv", index=False)
    pd.DataFrame([dict(image=f"B{i:02d}.png", surfers_with_no_box="", notes="")
                  for i in range(1, len(order) + 1)]).to_csv(
        OUT / "pass_b_missed.csv", index=False)
    order.assign(image=[f"B{i:02d}.png" for i in range(1, len(order) + 1)]) \
         .to_csv(OUT / ".pass_b_key.csv", index=False)
    print(f"\npass B: {len(order)} frames, {len(boxes_df)} boxes -> {d}")
    print(f"  per-box sheet -> {OUT / 'pass_b_boxes.csv'}   (is_surfer: y/n)")
    print(f"  per-frame     -> {OUT / 'pass_b_missed.csv'}  (surfers_with_no_box)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="which", choices=["a", "b"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.which == "a":
        build_pass_a(pick_frames())
    else:
        build_pass_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())
