"""
Per-hour surfer count as box plots, for the hours a surfer would actually go.

    surfer_count_box_by_hour_tide_lt3.png

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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from plot_count_histograms import load, style, ACCENT, MUTED, TEXT  # noqa: E402

MIN_N = 20          # an hour with fewer frames than this is a rug, not a box
MEDIAN_BGR = "#9de35a"


def plot(d, tide_max):
    hours = sorted(d["hour"].unique().astype(int))
    keep = [h for h in hours if (d["hour"] == h).sum() >= MIN_N]
    dropped = [h for h in hours if h not in keep]
    series = [d.loc[d["hour"] == h, "surfer_count"].to_numpy() for h in keep]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="black")
    bp = ax.boxplot(series, positions=range(len(keep)), widths=0.62,
                    whis=(10, 90), showfliers=False, patch_artist=True,
                    medianprops=dict(color=MEDIAN_BGR, linewidth=2))
    for box in bp["boxes"]:
        box.set(facecolor=ACCENT, alpha=0.30, edgecolor=ACCENT, linewidth=1.4)
    for part in ("whiskers", "caps"):
        for a in bp[part]:
            a.set(color=ACCENT, linewidth=1.2)

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
    args = ap.parse_args()
    d = load(args.tide_max)
    print(f"{len(d):,} counted hours over {d['date'].nunique()} days")
    plot(d, args.tide_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
