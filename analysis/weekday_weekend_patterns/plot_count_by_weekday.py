"""
plot_count_by_weekday.py
------------------------
Hourly surfer counts by day of week, restricted to tides of 0-3 ft.

WHY THE TIDE FILTER. Tide is far and away the strongest driver of hourly count
(Spearman rho = -0.645) and swamps any weekday effect if left free. Holding it
to 0-3 ft removes that: within the band the response is nearly flat (medians 21
/ 24 / 22 for 0-1, 1-2, 2-3 ft), so the residual half-foot of mean-tide
difference between weekdays is worth about two surfers, not twenty.

WHAT THE FILTER DOES NOT FIX, and the reason an earlier unfiltered version of
this chart was scrapped on 2026-10-09: SEASON. The record covers 139 days
unevenly spread over nine months with four calendar months missing, so each
weekday is sampled from a different month mix. Filtering on tide makes this
WORSE by shrinking the sample -- Wed vs Sun month-mix divergence goes from 32%
to 41%.

So the script prints a within-month check alongside the chart. Read the chart
only as far as that check supports it. The honest summary as of 2026-10-09: the
weekend effect is large and survives the check; the Wednesday dip does not and
should not be believed.

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
TIDE_LO, TIDE_HI = 0.0, 3.0


def main():
    t = pd.read_csv(FEATURES).dropna(subset=["surfer_count", "tide_ft"])
    t["date"] = pd.to_datetime(t.filename.str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    t["dow"] = t.date.dt.dayofweek
    t["m"] = t.date.dt.strftime("%Y-%m")
    f = t[(t.tide_ft >= TIDE_LO) & (t.tide_ft < TIDE_HI)]

    groups = [f.loc[f.dow == i, "surfer_count"].values for i in range(7)]
    ns = [len(g) for g in groups]
    ndays = [f.loc[f.dow == i, "date"].nunique() for i in range(7)]
    tides = [f.loc[f.dow == i, "tide_ft"].mean() for i in range(7)]

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
    ax.set_title(f"Surfer counts by day of the week, tide {TIDE_LO:.0f}-{TIDE_HI:.0f} ft only",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=38)
    ax.text(0, 1.062,
            f"{len(f):,} counted hours over {f.date.nunique()} days  ·  "
            f"tide held to {TIDE_LO:.0f}-{TIDE_HI:.0f} ft so it cannot drive the comparison",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)
    ax.text(0, 1.022,
            "box is the middle half, line the median, whiskers 1.5x IQR"
            + (f"  ·  {n_clipped} outlier(s) above the axis" if n_clipped else ""),
            transform=ax.transAxes, color=INK_DIM, fontsize=9)

    fig.tight_layout()
    out = HERE / f"count_by_weekday_tide0to3_{t.date.max():%Y-%m-%d}.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f"saved {out}\n")

    print(f"{'day':>5s}{'days':>6s}{'hours':>7s}{'median':>8s}{'IQR':>10s}{'mean tide':>11s}")
    for i, d in enumerate(DAYS):
        q1, med, q3 = np.percentile(groups[i], [25, 50, 75])
        print(f"{d:>5s}{ndays[i]:6d}{ns[i]:7d}{med:8.1f}{f'{q1:.0f}-{q3:.0f}':>10s}{tides[i]:11.2f}")

    print("\n--- the season check this chart cannot show ---")
    we = f[f.dow >= 5].surfer_count
    wd = f[f.dow < 5].surfer_count
    u, p = st.mannwhitneyu(we, wd)
    print(f"weekend vs weekday, pooled: {we.median():.0f} vs {wd.median():.0f}, p={p:.2e}")
    print("within each month (weekend minus weekday median):")
    agree = 0
    for m, g in f.groupby("m"):
        a, b = g[g.dow >= 5].surfer_count, g[g.dow < 5].surfer_count
        if len(a) >= 8 and len(b) >= 8:
            agree += (a.median() > b.median())
            print(f"  {m}  weekend {a.median():5.1f}  weekday {b.median():5.1f}"
                  f"  diff {a.median()-b.median():+5.1f}  (n={len(a)}/{len(b)})")
    print(f"  -> weekend higher in {agree} of the months with enough data")
    print("\nWednesday: lowest n of any day; its dip is NOT supported by the")
    print("month-mix check and should not be read as real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
