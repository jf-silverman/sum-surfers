"""
Per-hour surfer count as box plots, for the hours a surfer would actually go.

    surfer_count_box_by_hour_tide_lt3.png
    surfer_count_box_by_hour_weekend_tide_lt3.png   with --split weekend

The companion small-multiple histograms (plot_count_histograms.py) show each
hour's SHAPE; this shows the hours against each other. Fourteen distributions
side by side on one axis is what a box plot is for, and it answers "which hour
is busiest, and by how much" in a way fourteen separate panels cannot.

Why a box plot is defensible here when it is not on the full record. Over all
hours, 15% of counts are exactly zero; a box plot would hide that spike behind
a whisker and imply a continuous variable. Below 3 ft of tide the spike
collapses to 2.6%, the distribution stops being zero-inflated, and the quartiles
become an accurate summary. The individual counts are drawn over the boxes
anyway, so the discreteness and any remaining zeros stay visible -- the box is
the summary, the points are the data.

Tide is Surfline's harmonic PREDICTION (`tide_ft`), not the observed level,
which runs about a foot higher this autumn under the Kelvin wave. The
prediction is what a surfer reads before deciding to go.

Usage:
    python analysis/count_distribution/plot_count_boxplots.py
    python analysis/count_distribution/plot_count_boxplots.py --tide-max 2.5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from plot_count_histograms import load, style, ACCENT, MUTED, TEXT  # noqa: E402

MIN_N = 20          # an hour with fewer frames than this is a rug, not a box
# Splitting by weekend cuts every hour into a 2:5 pair, so the same floor would
# delete most of the chart. 12 is the lowest an hour can go and still have a
# quartile worth drawing; the n is on every box so a thin one is visible.
MIN_N_SPLIT = 12
MEDIAN_BGR = "#9de35a"

# Two series -> two hues, equal in lightness so neither reads as "more". Run
# through the dataviz palette validator against the #111111 surface: CVD
# separation 18.7 dE (protan), normal-vision 25.4, contrast PASS. The lightness
# band check fails at 0.712/0.708, as it does for every chart in this project,
# which keeps the established accent rather than restyling the set.
SERIES = {"weekday": "#3ab4c9", "weekend": "#e8833a"}


def _boxes(ax, series, positions, width, colour):
    bp = ax.boxplot(series, positions=positions, widths=width,
                    whis=(10, 90), showfliers=False, patch_artist=True,
                    medianprops=dict(color=MEDIAN_BGR, linewidth=2))
    for box in bp["boxes"]:
        box.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=1.4)
    for part in ("whiskers", "caps"):
        for a in bp[part]:
            a.set(color=colour, linewidth=1.2)
    return bp


def plot_split(d, tide_max):
    """Weekday against weekend, hour by hour."""
    d = d.copy()
    dt = pd.to_datetime(d["date"])
    d["grp"] = np.where(dt.dt.dayofweek >= 5, "weekend", "weekday")

    hours = sorted(d["hour"].unique().astype(int))
    keep, dropped = [], []
    for h in hours:
        ns = [((d["hour"] == h) & (d["grp"] == g)).sum() for g in SERIES]
        (keep if min(ns) >= MIN_N_SPLIT else dropped).append(h)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="black")
    rng = np.random.default_rng(7)
    off = {"weekday": -0.19, "weekend": 0.19}
    stats = {}
    for g, colour in SERIES.items():
        series = [d.loc[(d["hour"] == h) & (d["grp"] == g), "surfer_count"].to_numpy()
                  for h in keep]
        stats[g] = series
        pos = [i + off[g] for i in range(len(keep))]
        _boxes(ax, series, pos, 0.32, colour)
        for i, s_ in enumerate(series):
            ax.scatter(pos[i] + rng.uniform(-0.1, 0.1, len(s_)), s_, s=6,
                       color=colour, alpha=0.45, linewidths=0, zorder=3)

    style(ax)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([f"{h:02d}\n{len(a)}/{len(b)}"
                        for h, a, b in zip(keep, stats["weekday"], stats["weekend"])])
    ax.set_xlim(-0.7, len(keep) - 0.3)
    ax.set_ylim(bottom=-1.5)

    # Two series, so a legend is required; the medians are also written on the
    # pair with the widest gap, so identity is never carried by colour alone.
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=c, alpha=0.8, label=g) for g, c in SERIES.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              labelcolor=TEXT, fontsize=10)
    gaps = [np.median(b) - np.median(a)
            for a, b in zip(stats["weekday"], stats["weekend"])]
    i = int(np.argmax(np.abs(gaps)))
    meds = {g: np.median(stats[g][i]) for g in SERIES}
    for g in SERIES:
        # the lower median labels downward, so neither label lands inside the
        # box that sits above it
        up = meds[g] >= max(meds.values())
        ax.annotate(f"{meds[g]:.0f}", (i + off[g], meds[g]), color=MEDIAN_BGR,
                    fontsize=10, ha="center", va="bottom" if up else "top",
                    xytext=(0, 9 if up else -9), textcoords="offset points")

    cut = "" if tide_max is None else f", forecast tide below {tide_max:g} ft"
    ax.set_title(f"Surfer count by local hour, weekday against weekend{cut}",
                 color=TEXT, fontsize=14, pad=14)
    ax.set_xlabel("Local hour   (n weekday / n weekend)", color=MUTED)
    ax.set_ylabel("Surfers counted in the hour", color=MUTED)
    note = ("Box = quartiles, whiskers = 10th-90th percentile, dots = the "
            "individual hours.  Quality-gated frames excluded.")
    if dropped:
        note += (f"  Hours with under {MIN_N_SPLIT} frames on either side omitted "
                 f"({', '.join(f'{h:02d}:00' for h in dropped)}).")
    fig.text(0.5, -0.01, note, ha="center", color=MUTED, fontsize=9)

    sfx = "" if tide_max is None else f"_tide_lt{tide_max:g}".replace(".", "p")
    out = _HERE / f"surfer_count_box_by_hour_weekend{sfx}.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    print(f"    {'hour':>5s}{'wd n':>6s}{'wd med':>8s}{'we n':>6s}{'we med':>8s}{'gap':>7s}")
    for h, a, b in zip(keep, stats["weekday"], stats["weekend"]):
        print(f"    {h:02d}:00{len(a):6d}{np.median(a):8.1f}{len(b):6d}"
              f"{np.median(b):8.1f}{np.median(b) - np.median(a):+7.1f}")


def plot(d, tide_max):
    hours = sorted(d["hour"].unique().astype(int))
    keep = [h for h in hours if (d["hour"] == h).sum() >= MIN_N]
    dropped = [h for h in hours if h not in keep]
    series = [d.loc[d["hour"] == h, "surfer_count"].to_numpy() for h in keep]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="black")
    _boxes(ax, series, list(range(len(keep))), 0.62, ACCENT)

    # The counts themselves, over the summary. Jittered on x only -- the y
    # values are integers and must not be moved.
    rng = np.random.default_rng(7)
    for i, s in enumerate(series):
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(s)), s, s=7,
                   color="#cfeef5", alpha=0.35, linewidths=0, zorder=3)

    style(ax)
    ax.set_xticks(range(len(keep)))
    # n goes in the tick label rather than inside the axes: at these counts the
    # zero dots sit on the floor of the plot and an annotation there lands on
    # top of them.
    ax.set_xticklabels([f"{h:02d}\nn={len(s)}" for h, s in zip(keep, series)])
    ax.set_xlim(-0.7, len(keep) - 0.3)
    ax.set_ylim(bottom=-1.5)

    # Direct labels, selectively: the median only on the busiest and the
    # quietest hour, so the eye gets two anchors rather than fourteen numbers
    # to read. n is carried by the tick labels.
    med = [np.median(s) for s in series]
    for i in (int(np.argmax(med)), int(np.argmin(med))):
        ax.annotate(f"median {med[i]:.0f}", (i, med[i]), color=MEDIAN_BGR,
                    fontsize=10, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points")

    cut = "" if tide_max is None else f", forecast tide below {tide_max:g} ft"
    ax.set_title(f"Surfer count by local hour{cut}", color=TEXT, fontsize=14, pad=14)
    ax.set_xlabel("Local hour", color=MUTED)
    ax.set_ylabel("Surfers counted in the hour", color=MUTED)
    n = sum(len(s) for s in series)
    note = (f"Box = quartiles, whiskers = 10th-90th percentile, dots = the "
            f"{n:,} individual hours.  Quality-gated frames excluded.")
    if dropped:
        note += (f"  Hours under {MIN_N} frames omitted "
                 f"({', '.join(f'{h:02d}:00' for h in dropped)}).")
    fig.text(0.5, -0.01, note, ha="center", color=MUTED, fontsize=9)

    sfx = "" if tide_max is None else f"_tide_lt{tide_max:g}".replace(".", "p")
    out = _HERE / f"surfer_count_box_by_hour{sfx}.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    for h, s in zip(keep, series):
        print(f"    {h:02d}:00  n={len(s):3d}  median {np.median(s):5.1f}  "
              f"IQR {np.percentile(s, 25):.0f}-{np.percentile(s, 75):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tide-max", type=float, default=3.0)
    ap.add_argument("--split", choices=("none", "weekend"), default="none")
    args = ap.parse_args()
    d = load(args.tide_max)
    print(f"{len(d):,} counted hours over {d['date'].nunique()} days")
    if args.split == "weekend":
        plot_split(d, args.tide_max)
    else:
        plot(d, args.tide_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
