"""
What the hourly surfer count actually looks like, as a distribution.

Two charts, both from `predictions.csv` (quality_ok rows only):

  surfer_count_hist.png       one bar per integer count, the whole record
  surfer_count_by_hour.png    the same distribution split by daylight hour,
                              as small multiples on shared axes

Why these and not a box plot or a violin. The counts are small integers with a
hard floor at zero and a long right tail, and the zero bar is a feature of the
data rather than an artifact -- empty hours are real and the forecast is scored
on them. A box plot hides the zero spike and implies a continuous variable; a
violin smooths across integers that genuinely differ. One bar per integer shows
the floor, the mode and the tail without interpreting anything.

Small multiples rather than 14 overlaid curves, or a single hour-vs-count
heatmap: overlays at 14 series are unreadable and would need colours no palette
can keep distinguishable, and a heatmap trades the shape of each hour's
distribution for compactness. The question here is the SHAPE per hour, so each
hour gets its own panel on shared axes, which is what makes them comparable.

Theme matches the project's other published charts (dark_background, #111111
axes on black, the same aqua accent) so the README reads as one set.

Usage:
    python analysis/count_distribution/plot_count_histograms.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
PREDS_CSV = _ROOT / "data" / "predictions" / "predictions.csv"
OUT_DIR = Path(__file__).resolve().parent

# The project's chart theme, shared with analysis/weekday_weekend_patterns/.
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#111111",
    "figure.facecolor": "black",
    "savefig.facecolor": "black",
})
ACCENT = "#3ab4c9"      # single series -> one hue; the project's primary accent
MUTED = "#888888"       # recessive grid/axis ink
TEXT = "#dddddd"


def load():
    d = pd.read_csv(PREDS_CSV, float_precision="round_trip")
    d = d[d["quality_ok"].astype(str) == "True"].copy()
    d["surfer_count"] = pd.to_numeric(d["surfer_count"], errors="coerce")
    d = d[d["surfer_count"] >= 0]
    d["hour"] = pd.to_numeric(d["time_local"].astype(str).str.slice(0, 2), errors="coerce")
    return d.dropna(subset=["surfer_count", "hour"])


def style(ax):
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_overall(d):
    c = d["surfer_count"].to_numpy()
    hi = int(c.max())
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="black")
    counts, _ = np.histogram(c, bins=np.arange(-0.5, hi + 1.5))
    ax.bar(np.arange(0, hi + 1), counts, width=0.82, color=ACCENT, linewidth=0)
    style(ax)

    med, mean = np.median(c), c.mean()
    ax.axvline(med, color="#9de35a", linewidth=1.5, linestyle="--")
    # Direct labels, selectively -- never a number on every bar.
    ax.annotate(f"median {med:.0f}", (med, counts.max() * 0.92), color="#9de35a",
                fontsize=10, ha="left", xytext=(6, 0), textcoords="offset points")
    # Dropped well below the bar top: at the bar top this ran straight through
    # the median line and its label.
    ax.annotate(f"{counts[0]} empty hours\n({100 * counts[0] / len(c):.0f}% of all hours)",
                (0, counts[0]), color=TEXT, fontsize=10, ha="left", va="top",
                xytext=(10, -8), textcoords="offset points")

    ax.set_title(f"Hourly surfer count, all {len(c):,} counted hours",
                 color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel("Surfers counted in the hour", color=MUTED)
    ax.set_ylabel("Hours", color=MUTED)
    ax.set_xlim(-1, hi + 1)
    fig.text(0.5, -0.02,
             f"mean {mean:.1f}   median {med:.0f}   90th pct {np.percentile(c, 90):.0f}   "
             f"max {hi}   |  quality-gated frames excluded",
             ha="center", color=MUTED, fontsize=9)
    out = OUT_DIR / "surfer_count_hist.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return med, mean


def plot_by_hour(d):
    hours = sorted(d["hour"].unique().astype(int))
    # Drop hours with too little data to show a shape -- a 3-frame "distribution"
    # is a rug plot pretending to be a histogram.
    keep = [h for h in hours if (d["hour"] == h).sum() >= 20]
    dropped = [h for h in hours if h not in keep]
    hi = int(d["surfer_count"].max())
    ncol = 4
    nrow = int(np.ceil(len(keep) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.5 * nrow),
                             sharex=True, sharey=True, facecolor="black")
    axes = np.atleast_1d(axes).ravel()

    # Width 1, not 2: a width-2 bin pools 0 with 1 and buries the empty-hour
    # spike, which is the one feature of this distribution worth preserving.
    bins = np.arange(-0.5, hi + 1.5, 1)
    for ax, h in zip(axes, keep):
        s = d.loc[d["hour"] == h, "surfer_count"].to_numpy()
        ax.hist(s, bins=bins, color=ACCENT, linewidth=0)
        style(ax)
        ax.set_title(f"{h:02d}:00   n={len(s)}   median {np.median(s):.0f}",
                     color=TEXT, fontsize=10, pad=6)
        ax.axvline(np.median(s), color="#9de35a", linewidth=1.2, linestyle="--")
    for ax in axes[len(keep):]:
        ax.set_visible(False)

    # The bottom-most VISIBLE panel in each column, which is not the last row
    # when the grid is ragged. sharex hides tick labels on all but the true
    # bottom row, so they have to be switched back on explicitly or the label
    # sits under a bare axis.
    for i in range(len(keep) - ncol, len(keep)):
        axes[i].set_xlabel("Surfers in the hour", color=MUTED, fontsize=9)
        axes[i].tick_params(labelbottom=True)
    for i in range(0, len(keep), ncol):
        axes[i].set_ylabel("Hours", color=MUTED, fontsize=9)

    fig.suptitle("Hourly surfer count by local hour, shared axes",
                 color=TEXT, fontsize=14, y=0.995)
    note = (f"Dashed line = that hour's median.  Hours with fewer than 20 counted "
            f"frames omitted ({', '.join(f'{h:02d}:00' for h in dropped)})."
            if dropped else "Dashed line = that hour's median.")
    fig.text(0.5, -0.01, note, ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "surfer_count_by_hour.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main():
    d = load()
    print(f"{len(d):,} counted hours over {d['date'].nunique()} days")
    plot_overall(d)
    plot_by_hour(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
