"""
plot_tide_vs_count.py
---------------------
Distribution of hourly surfer counts in 1-foot tide bands.

A box per band rather than a scatter or a line: the question is how the whole
distribution moves, and a median alone hides that the spread collapses along
with the level. At 1-2 ft the middle half of hours runs 13 to 33; above 5 ft it
is 0 to 1.

Tide is the strongest single-predictor relationship in the data, Spearman
rho = -0.644 over 1,835 counted hours. Two confounds are ruled out: tide is
independent of clock time here (rho = +0.015, p = 0.52, because tide shifts
about 50 minutes a day and 138 days decorrelates it), and the effect holds
within every month with enough data (rho -0.47 to -0.77, all p < 1e-4).

The relationship is NOT monotonic, which is the point of binning rather than
fitting: counts peak in the 1-2 ft band and fall away on both sides.

Usage:
    python analysis/tide_patterns/plot_tide_vs_count.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEATURES = ROOT / "data" / "training_features.csv"

BG, INK, INK_DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA, LIME = "#3ab4c9", "#9de35a"
# Stops at 5 ft: above that the lineup is simply empty (median 0, IQR 0-1
# across 197 hours), so the bands add width without adding information.
EDGES = [-2, -1, 0, 1, 2, 3, 4, 5]
MIN_N = 15


def main():
    t = pd.read_csv(FEATURES).dropna(subset=["tide_ft", "surfer_count"])
    rho, p = st.spearmanr(t.tide_ft, t.surfer_count)
    t["bin"] = pd.cut(t.tide_ft, EDGES)

    groups, labels, ns = [], [], []
    for iv, g in t.groupby("bin", observed=True):
        if len(g) < MIN_N:
            continue
        groups.append(g.surfer_count.values)
        labels.append(f"{int(iv.left)} to {int(iv.right)}")
        ns.append(len(g))

    fig, ax = plt.subplots(figsize=(9.6, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9.5)

    bp = ax.boxplot(groups, patch_artist=True, widths=0.62, showfliers=True,
                    medianprops=dict(color=LIME, linewidth=2.2),
                    whiskerprops=dict(color=INK_DIM, linewidth=1.1),
                    capprops=dict(color=INK_DIM, linewidth=1.1),
                    flierprops=dict(marker="o", markersize=2.6,
                                    markerfacecolor=INK_DIM, markeredgecolor="none",
                                    alpha=0.45))
    for box in bp["boxes"]:
        box.set(facecolor=AQUA, alpha=0.40, edgecolor=AQUA, linewidth=1.2)

    # Scale to the tallest WHISKER, not the tallest outlier, so the boxes are
    # legible. Outliers above the cut are clipped and the caption says so.
    def upper_whisker(g):
        q1, q3 = np.percentile(g, [25, 75])
        return max(g[g <= q3 + 1.5 * (q3 - q1)], default=q3)
    top = max(upper_whisker(g) for g in groups) * 1.04
    n_clipped = int(sum((g > top).sum() for g in groups))
    for i, (g, n) in enumerate(zip(groups, ns), 1):
        ax.text(i, -top * 0.085, f"n={n}", ha="center", color=INK_DIM, fontsize=8.5)
        ax.text(i, np.median(g) + top * 0.022, f"{np.median(g):.0f}", ha="center",
                color=LIME, fontsize=9.5, fontweight="bold")

    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("tide height, feet", color=INK, fontsize=10.5, labelpad=16)
    ax.set_ylabel("surfers counted that hour", color=INK, fontsize=10.5)
    ax.set_ylim(-top * 0.13, top * 1.06)
    ax.set_title("Surfer counts by tide height", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=38)
    sub = (f"{len(t):,} counted hours over {t.filename.str[4:14].nunique()} days  ·  "
           f"Spearman rho = {rho:.3f}")
    ax.text(0, 1.062, sub, transform=ax.transAxes, color=INK_DIM, fontsize=9)
    ax.text(0, 1.022,
            "box is the middle half, line the median, whiskers 1.5x IQR"
            + (f"  ·  {n_clipped} outlier(s) above the axis" if n_clipped else ""),
            transform=ax.transAxes, color=INK_DIM, fontsize=9)

    fig.tight_layout()
    out = HERE / f"tide_vs_count_{t.filename.str[4:14].max()}.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f"saved {out}\n")
    for lab, g, n in zip(labels, groups, ns):
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        print(f"  {lab:>9s} ft  n={n:4d}  median {med:5.1f}  IQR {q1:.0f}-{q3:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
