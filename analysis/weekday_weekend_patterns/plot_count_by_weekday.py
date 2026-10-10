"""
plot_count_by_weekday.py
------------------------
Distribution of hourly surfer counts by day of the week.

Same box-per-group form as analysis/tide_patterns/plot_tide_vs_count.py, so the
two read as a pair: the question in both is how the whole distribution moves,
not just its centre.

A caution this chart cannot show on its own. Tide is by far the strongest
driver of hourly count (Spearman rho = -0.645), and tide is independent of
clock time but NOT guaranteed to be balanced across weekdays in a 139-day
sample. The script prints each day's mean tide so an apparent weekday effect
can be checked against it before being believed.

Usage:
    python analysis/weekday_weekend_patterns/plot_count_by_weekday.py
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
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main():
    t = pd.read_csv(FEATURES).dropna(subset=["surfer_count"])
    t["date"] = pd.to_datetime(t.filename.str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    t["dow"] = t.date.dt.dayofweek

    groups = [t.loc[t.dow == i, "surfer_count"].values for i in range(7)]
    ns = [len(g) for g in groups]
    tides = [t.loc[t.dow == i, "tide_ft"].mean() for i in range(7)]
    ndays = [t.loc[t.dow == i, "date"].nunique() for i in range(7)]

    def upper_whisker(g):
        q1, q3 = np.percentile(g, [25, 75])
        return max(g[g <= q3 + 1.5 * (q3 - q1)], default=q3)
    top = max(upper_whisker(g) for g in groups) * 1.04
    n_clipped = int(sum((g > top).sum() for g in groups))

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

    for i, (g, n) in enumerate(zip(groups, ns), 1):
        ax.text(i, -top * 0.085, f"n={n}", ha="center", color=INK_DIM, fontsize=8.5)
        ax.text(i, np.median(g) + top * 0.022, f"{np.median(g):.0f}", ha="center",
                color=LIME, fontsize=9.5, fontweight="bold")

    ax.set_xticklabels(DAYS, fontsize=9.5)
    ax.set_xlabel("day of week", color=INK, fontsize=10.5, labelpad=16)
    ax.set_ylabel("surfers counted that hour", color=INK, fontsize=10.5)
    ax.set_ylim(-top * 0.13, top * 1.06)
    ax.set_title("Surfer counts by day of the week", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=38)
    ax.text(0, 1.062,
            f"{len(t):,} counted hours over {t.date.nunique()} days",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)
    ax.text(0, 1.022,
            "box is the middle half, line the median, whiskers 1.5x IQR"
            + (f"  ·  {n_clipped} outlier(s) above the axis" if n_clipped else ""),
            transform=ax.transAxes, color=INK_DIM, fontsize=9)

    fig.tight_layout()
    out = HERE / f"count_by_weekday_{t.date.max():%Y-%m-%d}.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f"saved {out}\n")
    print(f"{'day':>5s}{'days':>6s}{'hours':>7s}{'median':>8s}{'IQR':>10s}{'mean tide':>11s}")
    for i, d in enumerate(DAYS):
        q1, med, q3 = np.percentile(groups[i], [25, 50, 75])
        print(f"{d:>5s}{ndays[i]:6d}{ns[i]:7d}{med:8.1f}{f'{q1:.0f}-{q3:.0f}':>10s}{tides[i]:11.2f}")
    kw = st.kruskal(*groups)
    print(f"\nKruskal-Wallis across the 7 days: H={kw.statistic:.1f} p={kw.pvalue:.4f}")
    r, p = st.spearmanr(t.dow >= 5, t.surfer_count)
    print(f"weekend vs weekday (hourly): rho={r:+.3f} p={p:.2e}")
    print(f"mean tide spread across days: {min(tides):.2f} to {max(tides):.2f} ft "
          f"-- check any weekday effect against this")
    return 0


if __name__ == "__main__":
    sys.exit(main())
