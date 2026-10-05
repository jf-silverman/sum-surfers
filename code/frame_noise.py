"""
frame_noise.py
--------------
Multi-scale graininess per crop: how fast fine detail averages away.

Joel's question (2026-10-05): a dawn frame is grainier than a midday one
because the camera runs out of exposure time and raises sensor gain, so is
there an absolute measure of that graininess -- and would comparing several
window sizes show it where a single window size does not?

The physics is right, and the multi-scale part is the key. Averaging an NxN
window of UNCORRELATED noise cuts its standard deviation by N, so on log-log
axes pure sensor grain falls with slope -1. Real image structure is spatially
correlated and does not average down nearly that fast. So the SLOPE separates
grain from texture, while any single window size cannot.

Measured on the water band, sigma of the high-passed block means:

    frame                    w=1   w=2   w=3   w=4   w=6   w=8  w=12  w=16   slope
    06:35 dawn, unusable    3.44  2.22  1.42  1.06  0.96  1.05  1.45  1.90   -0.85
    19:11 dusk              3.23  2.41  1.86  1.64  1.57  1.74  1.98  2.79   -0.50
    10:02 choppy midday    10.29  8.34  6.49  5.61  5.12  5.02  4.90  4.91   -0.44
    12:44 clean midday      5.06  4.58  3.99  4.01  4.23  4.51  4.53  4.71   -0.19
    08:14 flat calm midday  1.78  1.62  1.36  1.38  1.37  1.60  1.87  2.75   -0.20

Two things to read off these. The dawn frame's curve is U-SHAPED: grain
dominates the finest scales and averages away by w=4-6, after which real scene
structure takes over again. The clean midday frame is nearly FLAT -- structure
at every scale, little grain. That shape difference is exactly what Joel
predicted and is invisible at any one window size.

VALIDATED 2026-10-05 against 42 hand-reviewed frames. Two results, one good
and one that cuts against the table above:

  GOOD: the extreme case is cleanly caught. The 06:35 pure-noise frame scores
  -0.85 while all 42 countable frames fall between -0.50 and +0.03. It is far
  outside the countable range, with no overlap -- the one measure so far that
  isolates it. (The other uncountable frame, 07:29, scores -0.19, squarely
  inside the normal range. Correctly so: condensation is a blur problem, not a
  grain problem. Two different failures, see D04.)

  NOT SHOWN: a general dawn/dusk grain gradient. The five-frame table above
  suggests a clean one, but those frames were hand-picked to illustrate and the
  effect does not survive the full sample. Brightness vs slope: rho=-0.249,
  p=0.11, not significant. Dawn/dusk frames (n=9) have a median slope of -0.187
  against midday's -0.261 -- the OPPOSITE of grainier, p=0.057. The likely
  reason is the same chop confound: midday chop is fine-scale and drags midday
  slopes down, while "dawn/dusk by hour" includes well-lit frames. So treat the
  illustrative table as a demonstration that the shapes differ, not as evidence
  that light level predicts the shape. The gain physics is real; this sample of
  42 cannot show it through the chop.

WHAT THIS DOES NOT GIVE: an absolute sigma in DN. Fitting the obvious model
var(w) = sigma_noise^2 / w^2 + c returns 8.77 for the choppy midday frame, the
highest of any frame, because chop is near-pixel-scale too and so is nearly
white at fine scales. The slope is shallower for chop (-0.44) than for grain
(-0.85), but the amplitude is not separable. Three other routes were measured
and failed, recorded here so they are not retried:

  - Immerkaer (1996) 3x3 noise kernel: ranks the choppy frame highest (2.03),
    above the pure-noise frame (1.76). Magnitude cannot separate the two.
  - Temporal differencing across the 3 frames per clip, which should be clean
    since noise is uncorrelated in time: fails because nothing in frame is
    static. Shorebreak washes the sand, so the "static" beach band scored
    14.68 on a clean midday frame, the highest in the set.
  - Near-Nyquist PSD floor: same contamination, choppy ranks 2nd.

JPEG is the hard limit on all of this. On the dawn frame 16.1% of adjacent
pixel pairs are EXACTLY equal and sigma sits at 1-2 DN, the 8-bit quantization
floor -- the compressor removed much of the grain before the file was written.
A clean absolute number needs a region that is both static and textureless, and
this frame has none.

Usage:
    python code/frame_noise.py --validate
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
OUT_CSV = _PROJECT_ROOT / "data" / "predictor_vars" / "frame_noise.csv"

WATER_BAND = (55, 145)
SIZES = (1, 2, 3, 4, 6, 8, 12, 16)
SLOPE_FIT = (1, 2, 3, 4)      # fine scales, where grain lives


def _block_sigma(g, w):
    if w == 1:
        b = g
    else:
        h2, w2 = (g.shape[0] // w) * w, (g.shape[1] // w) * w
        if h2 < w or w2 < w:
            return None
        b = g[:h2, :w2].reshape(h2 // w, w, w2 // w, w).mean(axis=(1, 3))
    # High-pass first, so a large-scale brightness gradient across the frame
    # (sun angle, shoreline) is not counted as structure at every scale.
    return float((b - cv2.GaussianBlur(b, (0, 0), 2.0)).std())


def noise_profile(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)[WATER_BAND[0]:WATER_BAND[1]]
    sig = {w: _block_sigma(g, w) for w in SIZES}
    if any(v is None or v <= 0 for v in sig.values()):
        return None
    xs = np.log(SLOPE_FIT)
    ys = np.log([sig[w] for w in SLOPE_FIT])
    slope = float(np.polyfit(xs, ys, 1)[0])
    vals = [sig[w] for w in SIZES]
    imin = int(np.argmin(vals))
    return dict(
        noise_slope=slope,                     # -1 = pure grain, 0 = pure structure
        sigma_min=float(min(vals)),
        crossover_px=float(SIZES[imin]),       # scale at which grain stops dominating
        u_shape=float(vals[-1] / min(vals)),   # how much structure returns past the dip
        mean_brightness=float(g.mean()),
        **{f"sigma_w{w}": sig[w] for w in SIZES},
    )


def build(paths):
    rows = []
    for i, p in enumerate(paths, 1):
        prof = noise_profile(p)
        if prof:
            rows.append(dict(filename=p.name, **prof))
        if i % 300 == 0:
            print(f"  {i}/{len(paths)}")
    return pd.DataFrame(rows)


def validate(df):
    from scipy import stats as st
    rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
    m = rv.merge(df, on="filename", how="inner")
    tp = m["true_positives"].str.strip()
    na = m[tp.str.lower().isin(["na", "n/a"])]
    ok = m[~tp.str.lower().isin(["na", "n/a"]) & (tp != "")].copy()

    cols = ["noise_slope", "sigma_w1", "crossover_px", "mean_brightness"]
    print(f"\nFrames the reviewer marked UNCOUNTABLE (n={len(na)})")
    print("-" * 64)
    for _, r in na.iterrows():
        print(f"  {r['date']} {r['time_local']}  " + "  ".join(f"{c}={r[c]:7.2f}" for c in cols))
    print(f"\nCountable frames (n={len(ok)})")
    print("-" * 64)
    for c in cols:
        print(f"  {c:16s} min {ok[c].min():7.2f}  p10 {ok[c].quantile(.10):7.2f}"
              f"  median {ok[c].median():7.2f}  max {ok[c].max():7.2f}")

    print("\nDoes the slope track light level, as the gain theory predicts?")
    print("-" * 64)
    r, p = st.spearmanr(ok["mean_brightness"], ok["noise_slope"])
    print(f"  brightness vs noise_slope: rho={r:+.3f}  p={p:.4f}"
          f"   ({'darker frames ARE grainier' if r > 0 and p < .05 else 'not significant' if p >= .05 else 'opposite sign'})")
    ok["hour"] = ok["time_local"].str[:2].astype(int)
    edge = ok[(ok.hour <= 7) | (ok.hour >= 18)]
    mid = ok[(ok.hour > 7) & (ok.hour < 18)]
    if len(edge) >= 3 and len(mid) >= 3:
        u, pu = st.mannwhitneyu(edge["noise_slope"], mid["noise_slope"])
        print(f"  dawn/dusk (n={len(edge)}) median slope {edge['noise_slope'].median():+.3f}"
              f"   vs midday (n={len(mid)}) {mid['noise_slope'].median():+.3f}"
              f"   Mann-Whitney p={pu:.4f}")
    return ok, na


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--all", action="store_true", help="Every crop, not just reviewed ones")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    if args.all:
        paths = sorted(CROPS.glob("crop*.jpg"))
    else:
        rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
        paths = [CROPS / f for f in rv.filename.unique() if (CROPS / f).exists()]
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
