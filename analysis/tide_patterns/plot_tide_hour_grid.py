"""
plot_tide_hour_grid.py
----------------------
Median surfer count over tide height x hour of day.

Why this grid is readable at all: tide and clock time are INDEPENDENT in this
dataset (Spearman rho = +0.015, p = 0.52), because tide shifts about 50 minutes
a day and 138 days is long enough to decorrelate them completely. So each cell
is a near-clean estimate rather than a slice through a confound, and the two
margins can be read separately.

The question it answers, which the scatter cannot: does the tide effect depend
on the time of day? Equivalently -- when the tide is wrong at the hour people
are free to surf, do they go anyway?

Cells with fewer than 8 counted hours are left blank rather than drawn from
one or two observations.

Usage:
    python analysis/tide_patterns/plot_tide_hour_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEATURES = ROOT / "data" / "training_features.csv"

BG, INK, INK_DIM, GRID = "black", "white", "#bbbbbb", "#333333"
MIN_CELL = 8
BANDS = [-3, 1, 2, 3, 4, 5, 10]
LABELS = ["under 1", "1-2", "2-3", "3-4", "4-5", "over 5"]


def main():
    t = pd.read_csv(FEATURES).dropna(subset=["tide_ft", "surfer_count"])
    t["hour"] = t.time_local.str[:2].astype(int)
    t = t[(t.hour >= 6) & (t.hour <= 19)]
    t["band"] = pd.cut(t.tide_ft, BANDS, labels=LABELS)

    med = t.pivot_table(index="band", columns="hour", values="surfer_count",
                        aggfunc="median", observed=True)
    n = t.pivot_table(index="band", columns="hour", values="surfer_count",
                      aggfunc="size", observed=True)
    med = med.where(n >= MIN_CELL)
    med = med.reindex(LABELS)

    fig, ax = plt.subplots(figsize=(11.4, 5.4), facecolor=BG)
    ax.set_facecolor(BG)
    # Sequential, one hue light->dark: this is magnitude, not identity.
    im = ax.imshow(med.values, cmap="YlGnBu_r", aspect="auto",
                   origin="upper", interpolation="nearest")

    ax.set_xticks(range(len(med.columns)))
    ax.set_xticklabels([f"{h%12 or 12}{'am' if h<12 else 'pm'}" for h in med.columns],
                       fontsize=9)
    ax.set_yticks(range(len(med.index)))
    ax.set_yticklabels(med.index, fontsize=9.5)
    ax.tick_params(colors=INK_DIM, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, len(med.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(med.index), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", length=0)

    vmax = np.nanmax(med.values)
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            v = med.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="#555555", fontsize=11)
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="#0b0b0b" if v > vmax * 0.45 else "#e8e8e8")

    ax.set_xlabel("hour of day", color=INK, fontsize=10.5)
    ax.set_ylabel("tide height, feet", color=INK, fontsize=10.5)
    ax.set_title("Median surfer count by tide height and hour",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=34)
    ax.text(0, 1.035,
            f"{len(t):,} counted hours  ·  tide and clock time are independent here "
            f"(rho = +0.015), so rows and columns read separately  ·  "
            f"· = fewer than {MIN_CELL} hours",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)

    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.022)
    cb.set_label("median surfers", color=INK_DIM, fontsize=9)
    cb.ax.tick_params(colors=INK_DIM, labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout()
    out = HERE / f"tide_hour_grid_{t.filename.str[4:14].max()}.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f"saved {out}\n")
    print(med.round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
