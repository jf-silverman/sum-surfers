"""
frame_contrast.py
-----------------
Surfer-sized local-contrast measures per crop, for telling apart frames whose
counts can be trusted from frames whose cannot.

The idea came from Joel (logged in docs/model_and_feature_ideas.md, 2026-09-21)
as a two-tailed measure: a LOW tail for flat, fogged water predicting misses,
and a HIGH tail for saturated sparkle predicting phantoms. Implemented and
validated 2026-10-05. Neither tail behaves as predicted -- the direction is
reversed and the glare half does not exist -- but the measure is useful anyway,
for a different reason. Read the next section before using any of it.

Local contrast is the standard deviation of grey level inside a sliding
surfer-sized window (default 12px; a surfer is a handful of pixels in this
1280px-wide strip), computed over the water band only.

    contrast_p95     95th percentile of those window SDs.
    dead_frac        fraction of windows below an absolute SD floor.
    worst_slab_p95   contrast_p95 of the weakest vertical eighth.

WHAT THESE ACTUALLY MEASURE (validated 2026-10-05 against 42 hand-reviewed
frames with real TP/FP/FN counts). The answer is not what the idea predicted,
in two ways, and both are worth keeping written down.

1. THE SIGN IS REVERSED. The proposal was that LOW contrast (fog, flat water)
   would predict MISSES. The opposite holds: HIGH contrast predicts misses.
   Controlling for the number of real surfers -- necessary, because
   contrast_p95 is itself correlated with crowd size at rho=+0.478, p=0.003 --
   partial Spearman against miss rate:

       dead_frac        rho=-0.520  p=0.0011   more dead water, FEWER misses
       worst_slab_p95   rho=+0.450  p=0.0059   more contrast, MORE misses
       contrast_p95     rho=+0.310  p=0.066    (not significant once controlled)

   The clearest cases: 2026-09-23 08:14 has the flattest water in the set
   (dead_frac 0.72) and 11 surfers with ZERO missed; 2026-10-02 10:11 has
   dead_frac 0.46 and 10 surfers with zero missed. The worst miss rate, 6 of 9,
   is on the HIGHEST-contrast frame in the set (24.60).

   This reads as a chop-and-whitewater measure, not a fog measure. Flat water
   silhouettes a surfer cleanly; chop and foam hide them. It converges with the
   independent foam-pixel result (lineup-band foam vs count, r=-0.536,
   p=0.040) from a completely different method, which is the main reason to
   believe it on n=42.

2. THESE DO NOT GATE USABILITY, and must not be used that way. An 8-frame
   prototype suggested contrast_p95 separated the two frames Joel marked
   uncountable (4.75, 5.88) from countable ones. On all 42 it does not: countable
   frames run down to 2.25, well BELOW both. Calm, empty, perfectly countable
   water is indistinguishable from dead water by this measure. The prototype was
   overfit to its 8 frames. For the uncountable-frame problem see D04 in
   docs/known_bugs.md, which needs a different measure.

Also measured and negative: none of the three predicts the FALSE-POSITIVE rate
(all p > 0.16). Glare does not raise the high tail, because a saturated blob is
FLAT and so lowers local contrast -- the lens-flare frame scored an
unremarkable 12.22. Phantoms need their own measure.

The original low tail (5th percentile of window SDs) was dropped: it does not
discriminate at all, scoring 2.00 on a pure-noise frame against 1.70 on a clean
one, because flat patches exist in every frame. dead_frac against an absolute
floor replaces it.

Usage:
    python code/frame_contrast.py                     # all crops -> CSV
    python code/frame_contrast.py --reviewed          # only hand-reviewed frames
    python code/frame_contrast.py --validate          # score against the review sheet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS = _PROJECT_ROOT / "data" / "j_shore_cam" / "surf_crops"
REVIEW_CSV = _PROJECT_ROOT / "data" / "reviews" / "review_all.csv"
OUT_CSV = _PROJECT_ROOT / "data" / "predictor_vars" / "frame_contrast.csv"

# Water band, in crop rows. The strip is 180px tall and its bottom is beach,
# its top distant water. Surfer box centres sit at p10 67 / median 90 / p90 134
# (measured in analysis/count_variability/hour_sparse/foam_pixels.py), so this
# covers where surfers actually are with margin, and excludes the sand.
WATER_BAND = (55, 145)
WINDOW = 12          # surfer-sized, in px
DEAD_SD = 1.5        # window SDs below this are "dead" -- no texture to detect in
N_SLABS = 8


def contrast_stats(path, window=WINDOW, dead_sd=DEAD_SD, n_slabs=N_SLABS):
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g[WATER_BAND[0]:WATER_BAND[1]]
    # SD per window via box filters: E[x^2] - E[x]^2, clamped (float error can
    # push an exactly-flat window a hair below zero).
    mean = cv2.boxFilter(g, -1, (window, window))
    mean_sq = cv2.boxFilter(g * g, -1, (window, window))
    sd = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    c = window // 2
    sd = sd[c:-c, c:-c]          # drop windows straddling the edge
    if sd.size == 0:
        return None
    slab_p95 = [float(np.percentile(s, 95)) for s in np.array_split(sd, n_slabs, axis=1)]
    return dict(
        contrast_p95=float(np.percentile(sd, 95)),
        contrast_p50=float(np.percentile(sd, 50)),
        dead_frac=float((sd < dead_sd).mean()),
        worst_slab_p95=float(min(slab_p95)),
        best_slab_p95=float(max(slab_p95)),
        mean_brightness=float(g.mean()),
    )


def build(paths):
    rows = []
    for i, p in enumerate(paths, 1):
        st = contrast_stats(p)
        if st is None:
            continue
        rows.append(dict(filename=p.name, **st))
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}")
    return pd.DataFrame(rows)


def validate(df):
    """Score the measures against the hand review -- the only ground truth here."""
    if not REVIEW_CSV.exists():
        sys.exit(f"{REVIEW_CSV} not found.")
    rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
    m = rv.merge(df, on="filename", how="inner")

    tp_raw = m["true_positives"].str.strip()
    na = m[tp_raw.str.lower().isin(["na", "n/a"])]
    scored = m[~tp_raw.str.lower().isin(["na", "n/a"]) & (tp_raw != "")].copy()
    for c in ("true_positives", "false_positives", "missed"):
        scored[c] = pd.to_numeric(scored[c], errors="coerce").fillna(0)

    cols = ["contrast_p95", "dead_frac", "worst_slab_p95"]
    print(f"\nUnusable frames the reviewer marked na (n={len(na)})")
    print("-" * 52)
    for _, r in na.iterrows():
        print(f"  {r['date']} {r['time_local']}  " +
              "  ".join(f"{c}={r[c]:.2f}" for c in cols))
    print(f"\nCountable frames (n={len(scored)})")
    print("-" * 52)
    for c in cols:
        print(f"  {c:16s} min {scored[c].min():6.2f}   p10 {scored[c].quantile(.10):6.2f}"
              f"   median {scored[c].median():6.2f}   max {scored[c].max():6.2f}")
    if len(na):
        print(f"\n  Separation: every na frame's contrast_p95 is "
              f"{'BELOW' if na['contrast_p95'].max() < scored['contrast_p95'].min() else 'NOT below'}"
              f" every countable frame's "
              f"(na max {na['contrast_p95'].max():.2f} vs countable min {scored['contrast_p95'].min():.2f}).")

    # Does low contrast predict misses, as the idea claimed?
    print(f"\nDoes contrast predict detector error? (Spearman, n={len(scored)})")
    print("-" * 52)
    truth = scored.true_positives + scored.missed
    scored["miss_rate"] = np.where(truth > 0, scored.missed / truth, np.nan)
    scored["fp_rate"] = np.where(scored.true_positives + scored.false_positives > 0,
                                 scored.false_positives /
                                 (scored.true_positives + scored.false_positives), np.nan)
    from scipy import stats as st
    for metric, label in (("miss_rate", "miss rate (recall side)"),
                          ("fp_rate", "FP rate (precision side)")):
        sub = scored.dropna(subset=[metric])
        print(f"  vs {label}, n={len(sub)}")
        for c in cols:
            if sub[c].nunique() < 3:
                continue
            r, p = st.spearmanr(sub[c], sub[metric])
            flag = "  <-- significant" if p < 0.05 else ""
            print(f"      {c:16s} rho={r:+.3f}  p={p:.4f}{flag}")
    return scored


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", action="store_true", help="Only frames in the review sheet")
    ap.add_argument("--validate", action="store_true", help="Score against the review sheet")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    if args.reviewed or args.validate:
        rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
        paths = [CROPS / f for f in rv.filename.unique() if (CROPS / f).exists()]
    else:
        paths = sorted(CROPS.glob("crop*.jpg"))
    print(f"{len(paths)} crop(s)")
    df = build(paths)
    if df.empty:
        sys.exit("No frames read.")

    if args.validate:
        validate(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
