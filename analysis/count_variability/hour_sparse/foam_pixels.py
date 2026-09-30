"""
Counts whitewater pixels directly, rather than inferring turbulence from an
image-wide average.

Joel's point: lap_var averages over the whole strip, so a little foam and a lot
of texture look alike. Foam is distinct in raw pixels -- bright and desaturated
against grey-green water -- so it can just be counted.

The threshold is RELATIVE to each frame's own median brightness, not absolute.
Measured across this hour, the median V barely moves (86 -> 93) while the tail
moves a lot (p99 107 -> 151, max 134 -> 228), so the tail is the signal and a
relative cut survives the light fading toward last light. An absolute threshold
would drift with dusk and manufacture a correlation with the very trend the
analysis is trying to see past.

Calibration on three reference frames, V > median + 30:

    18:17 (a set breaking across the width)   9,805 px
    18:13 (four minutes earlier, calm)           33 px
    18:45 (near last light, calm)                 7 px

A saturation filter was tested and dropped: S < 60 cut the foam count roughly in
half without improving the separation, because the brightness lift already does
the work.

Usage:
    python analysis/count_variability/hour_sparse/foam_pixels.py
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "frames"
PROBE = HERE / "hour_probe.csv"
OUT = HERE / "foam_pixels.csv"

V_LIFT = 30          # brightness above the frame's own median that counts as foam
ROI_PX = 1280 * 180
# The strip splits into three 60px bands. Surfer box centres across all 45
# frames sit at p10 67 / median 90 / p90 134, so the middle band IS the lineup;
# the bottom is shore break washing in and the top is outside the takeoff zone.
BANDS = {"top": (0, 60), "lineup": (60, 120), "inshore": (120, 180)}


def foam_pixels(path):
    """Total foam pixels, and the split by band.

    The split is the finding. Foam inshore is a wave that has already broken and
    is washing to the beach -- it hides nobody. Foam ON the lineup is a set
    breaking over the people being counted. 17:53 carries the most foam of the
    whole hour (12,946 px) with a normal count, because only 445 of it is on the
    lineup; 18:17 has less foam overall but 3,773 px of it on the lineup, and
    that is the clip whose count collapses.
    """
    img = cv2.imread(str(path))
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(int)
    mask = v > np.median(v) + V_LIFT
    out = {"foam_px": int(mask.sum())}
    for name, (y0, y1) in BANDS.items():
        out[f"foam_{name}"] = int(mask[y0:y1].sum())
    return out


def main():
    rows = []
    for p in sorted(FRAMES.glob("*.jpg")):
        m = re.match(r"(\d{2})(\d{2})_f(\d)\.jpg$", p.name)
        if not m:
            continue
        rows.append(dict(clip_time=f"{m.group(1)}:{m.group(2)}", frame=int(m.group(3)),
                         **foam_pixels(p)))
    foam = pd.DataFrame(rows)
    probe = pd.read_csv(PROBE)
    d = probe.merge(foam, on=["clip_time", "frame"])
    d["foam_pct"] = (d.foam_px / ROI_PX * 100).round(3)
    d.to_csv(OUT, index=False)
    print(f"Wrote {len(d)} rows to {OUT}\n")

    print(f"{'time':>6} {'off':>5} {'count':>6} {'foam_px':>9} {'foam%':>7} {'lap_var':>8}")
    for (t, off), g in d.groupby(["clip_time", "offset_min"], sort=False):
        print(f"{t:>6} {off:+5d} {g['count'].mean():6.1f} {g.foam_px.mean():9.0f} "
              f"{g.foam_pct.mean():7.3f} {g.lap_var.mean():8.1f}")

    print("\nDoes foam explain the counts?")
    for label, x, y in [("raw, all 45 frames", d.foam_px, d["count"])]:
        r, p = stats.pearsonr(x, y)
        print(f"  {label:<34} r={r:+.3f} p={p:.4f}  n={len(x)}")

    c = d.groupby("offset_min").agg(count=("count", "mean"), foam=("foam_px", "mean"),
                                    lap=("lap_var", "mean")).reset_index()
    for col in ("count", "foam", "lap"):
        sl, ic, *_ = stats.linregress(c.offset_min, c[col])
        c[col + "_r"] = c[col] - (ic + sl * c.offset_min)
    for label, a, b in [("detrended, across 15 clips", c.count_r, c.foam_r),
                        ("detrended, lap_var for comparison", c.count_r, c.lap_r)]:
        r, p = stats.pearsonr(a, b)
        print(f"  {label:<34} r={r:+.3f} p={p:.4f}  n={len(a)}")

    print("\nAnd WHERE the foam is, which is the real result:")
    for band in BANDS:
        col = f"foam_{band}"
        cb = d.groupby("offset_min")[col].mean().reset_index()
        sl, ic, *_ = stats.linregress(cb.offset_min, cb[col])
        resid = cb[col] - (ic + sl * cb.offset_min)
        r, pv = stats.pearsonr(c.count_r, resid)
        print(f"  {band:<10} r={r:+.3f} p={pv:.4f}   mean {d[col].mean():7.0f} px")

    # A rank test as well: the foam distribution is extremely skewed (a few
    # frames carry almost all of it), which Pearson handles badly.
    r, p = stats.spearmanr(d.foam_px, d["count"])
    print(f"  {'Spearman rank, all 45 frames':<34} rho={r:+.3f} p={p:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
