"""
What the hourly surfer count actually looks like, as a distribution.

Two charts, both from `predictions.csv` (quality_ok rows only):

  surfer_count_hist.png       one bar per integer count, the whole record
  surfer_count_by_hour.png    the same distribution split by daylight hour,
                              as small multiples on shared axes

With --tide-max the same pair is written with a `_tide_lt<N>` suffix, restricted
to hours whose forecast tide is below that height. `tide_ft` is Surfline's
harmonic PREDICTION, which is what a surfer actually reads before deciding to
go -- not the observed water level, which this autumn runs about a foot higher
(see analysis/forecast_tide_residual/). The behavioural cut is the predicted
one, so that is what this filters on.

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
    python analysis/count_distribution/plot_count_histograms.py --tide-max 3.0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
PREDS_CSV = _ROOT / "data" / "predictions" / "predictions.csv"
PRED_VARS_CSV = _ROOT / "data" / "predictor_vars" / "surfline_predictors.csv"
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


def load(tide_max=None):
    d = pd.read_csv(PREDS_CSV, float_precision="round_trip")
    # quality_ok False rows carry no count at all -- dark/foggy/night-noise
    # frames never reach the detector, so they are already out of every number
    # here and no extra gate is needed.
    d = d[d["quality_ok"].astype(str) == "True"].copy()
    d["surfer_count"] = pd.to_numeric(d["surfer_count"], errors="coerce")
    d = d[d["surfer_count"] >= 0]
    d["hour"] = pd.to_numeric(d["time_local"].astype(str).str.slice(0, 2), errors="coerce")
    d = d.dropna(subset=["surfer_count", "hour"])
    if tide_max is None:
        return d
    pv = pd.read_csv(PRED_VARS_CSV, float_precision="round_trip")
    d = d.merge(pv[["filename", "tide_ft"]], on="filename", how="left")
    missing = int(d["tide_ft"].isna().sum())
    if missing:
        print(f"  {missing} counted hour(s) have no tide row -- dropped")
    return d[d["tide_ft"] < tide_max]


def style(ax):
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
    ax.tick_params(colors=MUTED, labelsize=9)


def suffix(tide_max):
    return "" if tide_max is None else f"_tide_lt{tide_max:g}".replace(".", "p")


def caption(tide_max):
    return ("quality-gated frames excluded" if tide_max is None else
            f"quality-gated frames excluded  |  forecast tide below {tide_max:g} ft only")


def plot_overall(d, tide_max=None):
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
    # Label the zero bar in place only when it is the mode -- then the space to
    # its right is empty and the text has somewhere to sit. Under the tide filter
    # the spike collapses from 15% of hours to 2.6%, the mode moves out to ~11,
    # and a label anchored at the zero bar runs straight through the bars beside
    # it; there it goes in the footer instead.
    empty_txt = f"{counts[0]} empty hours ({100 * counts[0] / len(c):.1f}%)"
    zero_is_mode = counts[0] >= counts.max() * 0.9
    if zero_is_mode:
        ax.annotate(empty_txt.replace(" (", "\n("), (0, counts[0]), color=TEXT,
                    fontsize=10, ha="left", va="top",
                    xytext=(10, -8), textcoords="offset points")

    title = (f"Hourly surfer count, all {len(c):,} counted hours" if tide_max is None
             else f"Hourly surfer count, {len(c):,} hours with forecast tide "
                  f"below {tide_max:g} ft")
    ax.set_title(title, color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel("Surfers counted in the hour", color=MUTED)
    ax.set_ylabel("Hours", color=MUTED)
    ax.set_xlim(-1, hi + 1)
    fig.text(0.5, -0.02,
             f"mean {mean:.1f}   median {med:.0f}   90th pct {np.percentile(c, 90):.0f}   "
             f"max {hi}   |  {'' if zero_is_mode else empty_txt + '   |  '}"
             f"{caption(tide_max)}",
             ha="center", color=MUTED, fontsize=9)
    out = OUT_DIR / f"surfer_count_hist{suffix(tide_max)}.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return med, mean


def plot_by_hour(d, tide_max=None):
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

    fig.suptitle("Hourly surfer count by local hour, shared axes"
                 + ("" if tide_max is None else
                    f"  --  forecast tide below {tide_max:g} ft only"),
                 color=TEXT, fontsize=14, y=0.995)
    note = (f"Dashed line = that hour's median.  Hours with fewer than 20 counted "
            f"frames omitted ({', '.join(f'{h:02d}:00' for h in dropped)})."
            if dropped else "Dashed line = that hour's median.")
    fig.text(0.5, -0.01, note, ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"surfer_count_by_hour{suffix(tide_max)}.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tide-max", type=float, default=None,
                    help="keep only hours whose forecast tide_ft is below this")
    args = ap.parse_args()
    d = load(args.tide_max)
    print(f"{len(d):,} counted hours over {d['date'].nunique()} days")
    plot_overall(d, args.tide_max)
    plot_by_hour(d, args.tide_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
