"""
The clear-day overcount re-check that gates a `CONF_THRESH` change.

`analysis/conf_band_decision/` measured a 0.150 floor on 113 frames drawn from
hand-reviewed days -- frames selected because they had misses, so a set biased
toward UNDERcounting, where admitting boxes can only help. The known
counterweight is the other direction: the September 2026 model overcounts by
3-7% in clear ordinary conditions (103% on the spot-check set, 107% on the
60-second set). A lower threshold has to be priced on both.

This runs inference once per frame at a 0.02 floor and then replays the full
production path -- NMS across tile seams, containment suppression, and the
static false-positive zones as they stand TODAY, bluff zone included -- at each
candidate threshold. Two sets:

  spotcheck   data/reviews/model_spotcheck_50/review_counts.csv, `my_count`.
              50 frames sampled across conditions, the overcount set. This is
              the gate: it is independent of the band review.
  reviewdays  data/reviews/review_all.csv, truth = true_positives + missed.
              The 16 hand-reviewed days the band review was drawn from, so NOT
              independent -- carried to show the two directions side by side,
              not as evidence.

With --dump-band it also writes every box in the 0.10-0.195 band on the review
days with its coordinates, which the band review CSV does not carry. That is
what a tree-bough zone has to be derived from.

Usage:
    python analysis/conf_thresh_recheck/recheck_threshold.py
    python analysis/conf_thresh_recheck/recheck_threshold.py --dump-band
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402
from score_reviews import parse_multi_surfer  # noqa: E402

CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
SPOTCHECK = _ROOT / "data" / "reviews" / "model_spotcheck_50" / "review_counts.csv"
REVIEW_ALL = _ROOT / "data" / "reviews" / "review_all.csv"
OUT = Path(__file__).resolve().parent

FLOOR = 0.02
THRESHOLDS = [0.195, 0.18, 0.17, 0.16, 0.15, 0.14, 0.12, 0.10]
BAND_LO, BAND_HI = 0.10, 0.195
N_BOOT = 5000
SEED = 11


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
    """Exactly what ds.run_inference_with_boxes() would emit at this threshold."""
    kept = [b for b in boxes if b[4] >= thresh]
    kept = ds.suppress_contained(ds.nms_across_tiles(kept))
    return ds.filter_false_positive_zones(kept)


def load_spotcheck():
    d = pd.read_csv(SPOTCHECK)
    d["truth"] = pd.to_numeric(d.my_count, errors="coerce")
    d = d.dropna(subset=["truth"])
    return d[["filename", "truth"]].assign(set="spotcheck")


def load_review_days():
    rv = pd.read_csv(REVIEW_ALL, dtype=str, keep_default_na=False)
    rv = rv[rv.date.str.match(r"^\d{4}-\d{2}-\d{2}$")]
    tp = rv.true_positives.str.strip()
    s = rv[~tp.str.lower().isin(["na", "n/a"]) & (tp != "")].copy()
    for c in ("true_positives", "missed"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["true_positives", "missed"])
    s["truth"] = s.true_positives + s.missed
    return s[["filename", "truth"]].assign(set="reviewdays")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-band", action="store_true",
                    help="also write every 0.10-0.195 band box with coordinates")
    args = ap.parse_args()

    frames = pd.concat([load_spotcheck(), load_review_days()], ignore_index=True)
    frames = frames[[(CROPS / f).exists() for f in frames.filename]].reset_index(drop=True)
    print(f"{len(frames)} frames with a hand count and a crop on disk "
          f"({frames.set.value_counts().to_dict()})")

    model = ds.load_model()
    rows, band_rows = [], []
    for i, r in enumerate(frames.itertuples(), 1):
        allb = boxes_at_floor(model, CROPS / r.filename)
        rec = {"filename": r.filename, "set": r.set, "truth": r.truth}
        for t in THRESHOLDS:
            rec[f"n_{t}"] = len(survive(allb, t))
        rows.append(rec)
        if args.dump_band and r.set == "reviewdays":
            hi = {tuple(np.round(b[:4], 2)) for b in survive(allb, BAND_HI)}
            for b in survive(allb, BAND_LO):
                if tuple(np.round(b[:4], 2)) not in hi:
                    band_rows.append({"filename": r.filename, "x1": round(b[0], 1),
                                      "y1": round(b[1], 1), "x2": round(b[2], 1),
                                      "y2": round(b[3], 1), "conf": round(b[4], 4)})
        if i % 25 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "counts_by_threshold.csv", index=False)
    print(f"\n  -> {OUT / 'counts_by_threshold.csv'}")
    if args.dump_band:
        b = pd.DataFrame(band_rows)
        b.to_csv(OUT / "band_boxes_with_coords.csv", index=False)
        print(f"  -> {OUT / 'band_boxes_with_coords.csv'}  ({len(b)} boxes)")

    rng = np.random.default_rng(SEED)
    for name, g in d.groupby("set"):
        base = (g[f"n_{THRESHOLDS[0]}"] - g.truth)
        print(f"\n=== {name}: {len(g)} frames, mean hand count {g.truth.mean():.1f} ===")
        print(f"  at {THRESHOLDS[0]}: mean signed {base.mean():+.2f}, "
              f"|err| {base.abs().mean():.2f}, ratio to truth "
              f"{g[f'n_{THRESHOLDS[0]}'].sum() / g.truth.sum():.3f}")
        print("     T    mean signed   |err|   ratio   d|err| vs 0.195   95% CI           p")
        for t in THRESHOLDS:
            e = (g[f"n_{t}"] - g.truth)
            dl = (e.abs() - base.abs()).to_numpy(float)
            bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean()
                           for _ in range(N_BOOT)])
            p = (np.abs(bs - dl.mean()) >= abs(dl.mean())).mean() if dl.mean() else 1.0
            print(f"  {t:.3f}   {e.mean():+9.2f}   {e.abs().mean():5.2f}   "
                  f"{g[f'n_{t}'].sum() / g.truth.sum():.3f}   {dl.mean():+9.3f}      "
                  f"{np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}  "
                  f"{p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
