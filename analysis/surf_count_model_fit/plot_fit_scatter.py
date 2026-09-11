"""
plot_fit_scatter.py
-------------------
Predicted vs. actual surfer count for the GBT point model, on the held-out
test rows, across the whole observed count range.

The aggregate error numbers hide the shape of the problem: mean bias on the
held-out set is +0.01 surfers, which reads as an unbiased model. It is not —
it over-predicts quiet hours and badly under-predicts crowded ones, and the
two cancel. This plot is the shape those numbers average away.

Reuses the same loading, split and hyperparameters as
fit_surfer_count_model.py, so these are the same predictions the released
holdout set and the reported metrics come from.

Colors follow the repo's existing chart palette (see plot_daily_prediction.py
and the calibration plot) rather than an independently chosen one: an isolated
chart in different colors is worse than one whose two hues sit slightly outside
the ideal dark-mode lightness band. The pair clears CVD separation (dE 15.9
protan), normal-vision separation (dE 28.7) and 3:1 contrast on this surface.

Usage:
    python analysis/surf_count_model_fit/plot_fit_scatter.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
from fit_surfer_count_model import load_and_prepare, standardize  # noqa: E402

OUT_PNG = Path(__file__).resolve().parent / "fit_scatter.png"

AQUA = "#3ab4c9"
CORAL = "#ff6f61"
BG = "black"
AXES_BG = "#111111"
GRID = "#333333"
TEXT = "white"
MUTED = "#bbbbbb"

POINT_KWARGS = dict(max_iter=300, learning_rate=0.05, max_depth=4,
                    l2_regularization=1.0, random_state=42)

# Bin edges for the trend line. Wider at the top because crowded hours are
# genuinely rare — narrow bins up there would be noise, not signal.
BIN_EDGES = [0, 2, 5, 10, 15, 20, 25, 30, 40, 100]


def binned_means(actual, predicted):
    """Mean prediction within each actual-count bin, for bins with enough rows."""
    centers, means, counts = [], [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (actual >= lo) & (actual < hi)
        n = int(m.sum())
        if n < 5:  # too few rows to read a mean from
            continue
        centers.append(float(actual[m].mean()))
        means.append(float(predicted[m].mean()))
        counts.append(n)
    return np.array(centers), np.array(means), counts


def main():
    X, y, df, numeric_cols = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test = standardize(X_train, X_test, numeric_cols)

    model = HistGradientBoostingRegressor(loss="poisson", **POINT_KWARGS).fit(X_train, y_train)
    actual = y_test.to_numpy(dtype=float)
    predicted = np.clip(model.predict(X_test), 0, None)

    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    print(f"Held-out rows: {len(actual)} | MAE {mae:.2f} | mean bias {bias:+.2f}")

    centers, means, counts = binned_means(actual, predicted)
    for c, m, n in zip(centers, means, counts):
        print(f"  actual ~{c:5.1f} (n={n:3d})  ->  mean predicted {m:5.1f}  ({m - c:+.1f})")

    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": AXES_BG, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "grid.color": GRID,
    })
    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    hi = float(max(actual.max(), predicted.max())) * 1.06
    ax.plot([0, hi], [0, hi], color=MUTED, linestyle="--", linewidth=1.3,
            zorder=1, label="Perfect prediction (predicted = actual)")

    # Semi-transparent so the dense low-count region reads as density rather
    # than a solid blob; 292 points would otherwise be one opaque mass there.
    ax.scatter(actual, predicted, s=26, color=AQUA, alpha=0.45,
               linewidths=0.5, edgecolors=AXES_BG, zorder=2,
               label=f"Held-out hours (n={len(actual)})")

    ax.plot(centers, means, color=CORAL, linewidth=2.0, marker="o", markersize=6,
            markeredgecolor=AXES_BG, markeredgewidth=0.8, zorder=3,
            label="Mean prediction, binned by actual")

    # Direct label on the trend, so identity does not rest on the legend alone.
    ax.annotate("mean prediction", (centers[-1], means[-1]),
                textcoords="offset points", xytext=(-6, 14), fontsize=8,
                color=CORAL, ha="right")

    # The single most important thing on this chart: where the trend crosses
    # the diagonal, the model flips from over- to under-predicting.
    crossing = None
    for i in range(len(centers) - 1):
        if (means[i] - centers[i]) >= 0 > (means[i + 1] - centers[i + 1]):
            crossing = centers[i + 1]
            break
    if crossing is not None:
        ax.axvline(crossing, color=MUTED, linewidth=0.8, alpha=0.45, zorder=0)
        # Anchored to the bottom of the axes, not the top: the top-left is
        # where the legend sits, and these labels collided with it there.
        # The strip just above the x-axis around the crossing is empty —
        # nothing predicts ~0 surfers on an hour that actually had ~20.
        ax.annotate(f"<- over-predicts\nbelow ~{crossing:.0f}", (crossing, hi * 0.015),
                    textcoords="offset points", xytext=(-8, 0), fontsize=7.5,
                    color=MUTED, ha="right", va="bottom")
        ax.annotate(f"under-predicts ->\nabove ~{crossing:.0f}", (crossing, hi * 0.015),
                    textcoords="offset points", xytext=(8, 0), fontsize=7.5,
                    color=MUTED, ha="left", va="bottom")

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")  # so the diagonal really is 45 degrees
    ax.set_xlabel("Actual surfers counted", fontsize=9)
    ax.set_ylabel("Predicted surfers", fontsize=9)
    ax.set_title("GBT forecast vs. reality, held-out hours", color=TEXT, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    legend = ax.legend(loc="upper left", facecolor=AXES_BG, edgecolor=GRID, fontsize=7.5)
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.text(0.5, 0.015,
             f"MAE {mae:.2f} surfers  |  mean bias {bias:+.2f}  |  "
             f"the near-zero bias is two opposite errors cancelling",
             fontsize=7.5, ha="center", color=MUTED)

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
