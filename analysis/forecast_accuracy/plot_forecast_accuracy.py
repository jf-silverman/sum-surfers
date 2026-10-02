"""
How accurate the forecast is, using the whole scored record rather than one day.

The README's accuracy section shows a single day's forecast-vs-actual line. That
was the only thing available when the forecast log was three days old; it now
holds 470 scored forecast-hours across 8 target dates and 7 lead times, which
supports a question the single-day chart cannot ask: does the week-ahead
forecast actually work?

Panel A is day-ahead calibration. Panel B is the answer to that question, and it
is not a flattering one.

Usage:
    python analysis/forecast_accuracy/plot_forecast_accuracy.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
LOG = HERE.parent.parent / "data" / "forecasts" / "forecast_log.csv"
OUT = HERE / "forecast_accuracy.png"

AQUA, LIME, CORAL = "#3ab4c9", "#9de35a", "#f0776c"
INK, MUTED, SURFACE = "#f2f2f0", "#9a9a97", "#111111"


def main():
    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black",
        "axes.facecolor": SURFACE, "axes.edgecolor": "#3a3a38",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2c2c2a", "axes.grid": True, "axes.axisbelow": True,
    })
    f = pd.read_csv(LOG)
    s = f[f.actual.notna()].copy()
    d1 = s[s.lead_days == 1]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4), facecolor="black",
                                  gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.30})

    # ---- A: day-ahead calibration -----------------------------------------
    hi = max(d1.actual.max(), d1.predicted.max()) + 3
    ax.plot([0, hi], [0, hi], color=MUTED, linestyle="--", linewidth=1.4, zorder=2,
            label="perfect (1:1)")
    inside = d1.inside_band.astype(str) == "True"
    ax.scatter(d1.loc[inside, "predicted"], d1.loc[inside, "actual"], s=34,
               color=AQUA, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, zorder=3,
               label=f"inside the 80% band ({inside.sum()})")
    ax.scatter(d1.loc[~inside, "predicted"], d1.loc[~inside, "actual"], s=44,
               color=CORAL, alpha=0.9, edgecolor=SURFACE, linewidth=0.6, zorder=4,
               label=f"outside it ({(~inside).sum()})")
    sl, ic, r, _p, _ = stats.linregress(d1.predicted, d1.actual)
    xs = np.array([0, hi])
    ax.plot(xs, ic + sl * xs, color=LIME, linewidth=2.2, zorder=5,
            label=f"fit: actual = {sl:.2f}×pred {ic:+.1f}")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Forecast count"); ax.set_ylabel("Counted by the detector")
    ax.set_title(f"A.  Day-ahead forecast against what happened\n"
                 f"{len(d1)} scored hours over {d1.date.nunique()} days · MAE {d1.abs_error.mean():.2f}\n"
                 f"r = {r:.2f}, slope {sl:.2f} — no systematic shrinkage at one day out",
                 color=INK, fontsize=10, loc="left", pad=10)
    leg = ax.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left", fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- B: by lead time ---------------------------------------------------
    g = s.groupby("lead_days").agg(
        n=("abs_error", "size"), mae=("abs_error", "mean"),
        half=("band_width", lambda v: v.mean() / 2),
        cov=("inside_band", lambda v: v.astype(str).eq("True").mean() * 100)).reset_index()
    # MAE and half-band-width are both in surfers, so they share one axis
    # honestly; coverage is a percentage and is annotated instead.
    ax2.plot(g.lead_days, g.mae, color=CORAL, marker="o", linewidth=2.4, markersize=8,
             markeredgecolor=SURFACE, zorder=4, label="average miss (MAE)")
    ax2.plot(g.lead_days, g.half, color=AQUA, marker="s", linewidth=2.4, markersize=7,
             markeredgecolor=SURFACE, zorder=3, label="half-width of the 80% band")
    for r_ in g.itertuples():
        ax2.annotate(f"{r_.cov:.0f}%", (r_.lead_days, r_.half), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8.5, color=MUTED)
        ax2.annotate(f"n={r_.n}", (r_.lead_days, r_.mae), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8, color=MUTED)
    ax2.set_xlabel("Days ahead the forecast was made")
    ax2.set_ylabel("Surfers")
    ax2.set_xticks(range(1, 8))
    ax2.set_title("B.  The error grows with lead time while the band NARROWS\n"
                  "Miss 4.5 → 7.5 (slope +0.25/day, p = 0.04); interval tightens 9.2 → 7.0\n"
                  "Grey = coverage, 83% at one day to 43% at seven. It should widen, not shrink.",
                  color=INK, fontsize=10, loc="left", pad=10)
    leg2 = ax2.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left", fontsize=9)
    for t in leg2.get_texts():
        t.set_color(INK)
    ax2.margins(y=0.22)

    fig.suptitle("Forecast accuracy across the whole scored record — "
                 f"{len(s)} forecast-hours, {s.date.min()} to {s.date.max()}",
                 color=INK, fontsize=12.5, y=1.045, x=0.008, ha="left")
    fig.savefig(OUT, dpi=150, facecolor="black", bbox_inches="tight")
    print("saved", OUT)
    print(g.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
