"""
plot_calibration.py
----------------------
Real calibration/coverage plot for the surf-count GBT quantile-regression
model -- visualizes the finding already stated in prose in README.md and
docs/PROJECT_HISTORY.md ("reported 80% prediction interval is empirically
closer to a 65% interval"): nominal vs. actual empirical coverage on a
real held-out test split, not simulated or eyeballed.

Reuses the exact same setup fit_surfer_count_model.py's fit_quantile_intervals()
already uses (load_and_prepare(), the same train_test_split(test_size=0.2,
random_state=42), standardize(), fit_quantile_model_robust()) so the numbers
here are directly consistent with -- not a separately-computed, possibly
different -- the coverage figures already documented elsewhere.

Quantile levels are limited to 0.10-0.90 in steps of 0.10 (matching
FAN_LEVELS in code/plot_daily_prediction.py) rather than pushing to more
extreme levels (e.g. 0.05/0.95): fit_quantile_intervals()'s own docstring
already found those aren't reliably learnable given how zero-inflated
this dataset is (~11% of rows at surfer_count==0). From those 9 fitted
levels, four real symmetric central intervals are available: 20% (40-60),
40% (30-70), 60% (20-80), 80% (10-90).

Usage:
    python analysis/surf_count_model_calibration/plot_calibration.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from fit_surfer_count_model import load_and_prepare, standardize, fit_quantile_model_robust  # noqa: E402

AQUA = "#3ab4c9"
LIME = "#9de35a"
CORAL = "#ff6f61"
BG = "black"
AXES_BG = "#111111"
GRID = "#333333"
TEXT = "white"

# Same 9 levels as code/plot_daily_prediction.py's FAN_LEVELS -- all
# already-proven learnable on this dataset (see module docstring).
LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
# Symmetric central intervals derivable from those 9 levels.
INTERVALS = [(0.40, 0.60), (0.30, 0.70), (0.20, 0.80), (0.10, 0.90)]


def main():
    X, y, df, numeric_cols = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = standardize(X_train, X_test, numeric_cols)
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    preds = {}
    for level in LEVELS:
        model, min_leaf = fit_quantile_model_robust(X_train, y_train, level)
        pred = np.clip(model.predict(X_test), 0, None)
        preds[level] = pred
        print(f"  fit level={level:.2f}  min_samples_leaf={min_leaf}")

    # Enforce monotonicity across levels per test row (independently fit
    # quantile models have no built-in guarantee against crossing).
    sorted_levels = sorted(preds)
    running = np.zeros(len(X_test))
    for level in sorted_levels:
        preds[level] = np.maximum(preds[level], running)
        running = preds[level]

    y_test_vals = y_test.values
    results = []
    for lower_q, upper_q in INTERVALS:
        nominal = upper_q - lower_q
        lower_pred, upper_pred = preds[lower_q], preds[upper_q]
        covered = (y_test_vals >= lower_pred) & (y_test_vals <= upper_pred)
        actual = covered.mean()
        below = (y_test_vals < lower_pred).mean()
        above = (y_test_vals > upper_pred).mean()
        mean_width = (upper_pred - lower_pred).mean()
        results.append(dict(nominal=nominal, actual=actual, below=below, above=above,
                             lower_q=lower_q, upper_q=upper_q, mean_width=mean_width))
        print(f"\n=== {nominal:.0%} interval ({lower_q:.0%}-{upper_q:.0%}) ===")
        print(f"  Empirical coverage: {actual:.1%} (target {nominal:.0%})")
        print(f"  below-lower miss: {below:.1%} (target {lower_q:.0%})  "
              f"above-upper miss: {above:.1%} (target {1 - upper_q:.0%})")
        print(f"  mean interval width: {mean_width:.1f} surfers")

    # --- Plot: nominal vs actual coverage, with a perfect-calibration diagonal ---
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": AXES_BG, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "grid.color": GRID,
    })
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ax.plot([0, 1], [0, 1], color=LIME, linestyle="--", linewidth=1.5,
             label="Perfect calibration (actual = nominal)", zorder=1)

    nominal_vals = [r["nominal"] for r in results] + [0.0]
    actual_vals = [r["actual"] for r in results] + [0.0]
    order = np.argsort(nominal_vals)
    nominal_vals = np.array(nominal_vals)[order]
    actual_vals = np.array(actual_vals)[order]
    ax.plot(nominal_vals, actual_vals, color=AQUA, linewidth=1.5, marker="o", markersize=6,
             zorder=3, label="Actual (held-out test set)")

    for r in results:
        ax.annotate(f"{r['actual']:.0%}", (r["nominal"], r["actual"]),
                     textcoords="offset points", xytext=(7, -3), fontsize=7.5, color=AQUA)

    ax.fill_between([0, 1], [0, 1], 0, color=CORAL, alpha=0.06, zorder=0)
    ax.text(0.06, 0.80, "Below diagonal = overconfident\n(actual < nominal)", color=CORAL,
             fontsize=8, alpha=0.85)
    # Why the 80% interval sits above the diagonal (unlike the other three)
    # is explained in prose in README.md's "Model Calibration" section
    # rather than annotated on the chart itself -- an in-plot callout for
    # just one point was cramped at this smaller size and covered the
    # trend line.

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal (claimed) coverage", fontsize=9)
    ax.set_ylabel("Actual (empirical) coverage on held-out test set", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("Prediction-interval calibration", color=TEXT, fontsize=11)
    ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.legend(loc="upper left", facecolor=AXES_BG, edgecolor=GRID, fontsize=7.5)
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout()
    out_path = HERE / "calibration_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
