"""
plot_tide_vs_count.py
---------------------
Hourly surfer count against tide height.

The relationship is the strongest single-predictor effect in the dataset:
Spearman rho = -0.644 over 1,835 counted hours, p = 2.7e-215. Two checks say it
is not an artifact:

  Time of day. Tide shifts about 50 minutes a day, so over 138 days it
  decorrelates from clock time entirely -- tide vs hour rho = +0.015, p = 0.52.
  The daily rhythm of when people surf cannot be producing this.

  Season. The effect holds WITHIN every month with enough data, rho -0.47 to
  -0.77, all p < 1e-4. It is not a summer/winter pattern in disguise.

The shape is not monotonic, which a correlation coefficient hides: counts peak
around 1-2 ft and fall away on BOTH sides. That is why the chart draws a binned
median through the cloud rather than a straight fit.

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
AQUA, LIME, CORAL = "#3ab4c9", "#9de35a", "#ff6f61"


def main():
    t = pd.read_csv(FEATURES).dropna(subset=["tide_ft", "surfer_count"])
    lo, hi = t.tide_ft.quantile([0.05, 0.95])
    rho, p = st.spearmanr(t.tide_ft, t.surfer_count)

    # Median per 0.5ft bin, drawn only where there is enough data to mean it.
    bins = np.arange(np.floor(t.tide_ft.min() * 2) / 2, t.tide_ft.max() + 0.5, 0.5)
    t["bin"] = pd.cut(t.tide_ft, bins)
    agg = t.groupby("bin", observed=True).agg(
        n=("surfer_count", "size"), med=("surfer_count", "median"),
        q1=("surfer_count", lambda s: s.quantile(0.25)),
        q3=("surfer_count", lambda s: s.quantile(0.75)))
    agg["mid"] = [iv.mid for iv in agg.index]
    agg = agg[agg.n >= 15]

    fig, ax = plt.subplots(figsize=(9.2, 6.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9)

    # The band most hours actually fall in, so the eye is not drawn to the tails
    ax.axvspan(lo, hi, color=INK_DIM, alpha=0.07, zorder=0)
    ax.annotate(f"90% of counted hours fall between {lo:.1f} and {hi:.1f} ft",
                xy=((lo + hi) / 2, t.surfer_count.max() * 0.97), ha="center",
                color=INK_DIM, fontsize=9)

    ax.scatter(t.tide_ft, t.surfer_count, s=11, color=AQUA, alpha=0.30,
               edgecolors="none", zorder=2)
    ax.fill_between(agg.mid, agg.q1, agg.q3, color=LIME, alpha=0.16, zorder=3)
    ax.plot(agg.mid, agg.med, color=LIME, linewidth=2.4, zorder=4)

    # Labels sit just above/beside the points they name, with short leaders --
    # a long leader across a 1,800-point cloud is worse than no label.
    peak = agg.loc[agg.med.idxmax()]
    ax.annotate(f"busiest near {peak.mid:.2f} ft (median {peak.med:.0f})",
                xy=(peak.mid, peak.med), xytext=(0, 14),
                textcoords="offset points", ha="center",
                color=LIME, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=LIME, linewidth=0.8,
                                shrinkA=0, shrinkB=3))
    tail = agg[agg.med <= 1]
    if len(tail):
        x0 = tail.mid.iloc[0]
        ax.annotate("above ~4.5 ft\nthe lineup empties", xy=(x0, 1),
                    xytext=(14, 42), textcoords="offset points", ha="left",
                    color=CORAL, fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=CORAL, linewidth=0.8,
                                    shrinkA=0, shrinkB=3))

    ax.set_xlabel("tide height, feet", color=INK, fontsize=10.5)
    ax.set_ylabel("surfers counted that hour", color=INK, fontsize=10.5)
    ax.set_title("Surfer count against tide height", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.015,
            f"{len(t):,} counted hours over {t.filename.str[4:14].nunique()} days  ·  "
            f"Spearman rho = {rho:.3f}  ·  line is the median per 0.5 ft, band is the IQR",
            transform=ax.transAxes, color=INK_DIM, fontsize=9.5)
    ax.set_ylim(-2, t.surfer_count.max() * 1.06)

    fig.tight_layout()
    out = HERE / f"tide_vs_count_{t.filename.str[4:14].max()}.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f"saved {out}")
    print(f"\nrho={rho:.3f} p={p:.1e}  peak median at {peak.mid:.2f} ft")
    print(agg[["n", "med", "q1", "q3"]].round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
