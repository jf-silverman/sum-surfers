"""
plot_accuracy_charts.py
-----------------------
The two accuracy charts the README front page carries.

Both are built to the project's existing chart style -- black surface, white
ink, #333333 grid -- so the README reads as one system rather than two.

WHY THESE TWO, AND WHY NOT A PRECISION/RECALL BAR

Precision and recall are two numbers. A bar chart of 0.987 and 0.919 says less
than a two-row table does, and spends the reader's attention on chrome. They go
in a stat table; the chart budget goes where there is structure to see.

  detector_accuracy.png   hand count vs detector count, one point per reviewed
                          frame, with a 1:1 reference and the fitted line. The
                          gap between those two lines IS the undercount. A
                          precision bar hides it completely; this chart lets a
                          reader see the scatter widen at the busy end instead
                          of taking an MAE on trust.

  forecast_accuracy.png   error and interval coverage against lead time, as two
                          panels sharing an x-axis. NOT a dual-axis chart:
                          surfers and percent on one pair of axes would put the
                          crossing point wherever the scaling happened to land.
                          Two panels, one scale each.

The coverage panel is the honest half -- 80% nominal bands hold up at one day
out and fall apart by seven. That caveat is more use to a reader than the
headline MAE.

Usage:
    python code/plot_accuracy_charts.py
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

_ROOT = Path(__file__).resolve().parent.parent
REVIEW = _ROOT / "data" / "reviews" / "review_all.csv"
PREDS = _ROOT / "data" / "predictions" / "predictions.csv"
FLOG = _ROOT / "data" / "forecasts" / "forecast_log.csv"
OUT = _ROOT / "data" / "charts" / "published"

BG, INK, INK_DIM, GRID = "black", "white", "#bbbbbb", "#333333"
# The site palette, shared with plot_daily_prediction.py so the README reads as
# one system rather than two. Validated as a categorical pair on this surface:
# CVD separation dE 22.6 protan / 13.8 tritan against a target of 8, and
# normal-vision dE 23.4 against a floor of 15 -- comfortably apart. The pair
# does fail a lightness-band check (0.712 vs 0.842), i.e. LIME reads brighter
# than AQUA, so LIME is given to the single thin line that has to carry over a
# field of dots and AQUA to the dots themselves, which turns the imbalance into
# the right hierarchy instead of a distraction.
AQUA = "#3ab4c9"    # the data
LIME = "#9de35a"    # the fitted line / the second panel
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9)


def load_review():
    rv = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    rv = rv[rv["date"].str.match(DATE_RE)]
    tp = rv["true_positives"].str.strip()
    s = rv[~tp.str.lower().isin(["na", "n/a"]) & (tp != "")].copy()
    for c in ("true_positives", "missed"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    p = pd.read_csv(PREDS)[["filename", "surfer_count"]]
    s = s.merge(p, on="filename", how="left").dropna(
        subset=["surfer_count", "true_positives", "missed"])
    s["truth"] = s.true_positives + s.missed
    s["det"] = s.surfer_count
    return s


def detector_chart(s):
    sl, ic, r, _, se = st.linregress(s.truth, s.det)
    fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=BG)
    _style(ax)
    hi = max(s.truth.max(), s.det.max()) * 1.06

    ax.plot([0, hi], [0, hi], color=INK_DIM, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    xs = np.array([0, hi])
    ax.plot(xs, sl * xs + ic, color=LIME, linewidth=2.0, zorder=3)
    # 2px surface ring keeps overlapping points readable where the lineup packs in
    ax.scatter(s.truth, s.det, s=42, color=AQUA, alpha=0.78,
               edgecolors=BG, linewidths=1.0, zorder=4)

    # Direct labels placed where the two lines separate, which is the whole
    # point of the chart -- no leader lines to drag across the scatter.
    lx = hi * 0.56
    ax.text(lx, lx + hi * 0.055, "perfect agreement", color=INK_DIM, fontsize=10,
            rotation=38, rotation_mode="anchor", va="bottom", ha="left")
    ax.text(lx, sl * lx + ic - hi * 0.075, f"detector = {sl:.2f} x actual",
            color=LIME, fontsize=10, rotation=36, rotation_mode="anchor",
            va="top", ha="left")

    ax.set_xlim(-2, hi); ax.set_ylim(-2, hi)
    ax.set_xlabel("surfers actually there (counted by hand)", color=INK, fontsize=10.5)
    ax.set_ylabel("surfers the detector counted", color=INK, fontsize=10.5)
    ax.set_title("The detector against a human count", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.015,
            f"{len(s)} frames hand-counted across {s.date.nunique()} days  ·  "
            f"{int(s.truth.sum()):,} real surfers  ·  r = {r:.3f}",
            transform=ax.transAxes, color=INK_DIM, fontsize=9.5)
    fig.tight_layout()
    p = OUT / "detector_accuracy.png"
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    return p, sl, r


def forecast_chart():
    f = pd.read_csv(FLOG)
    s = f[f.actual.notna()].copy()
    s["hit"] = (s.actual >= s.lower_q10) & (s.actual <= s.upper_q90)
    g = s.groupby("lead_days").apply(
        lambda d: pd.Series({"n": len(d),
                             "mae": (d.predicted - d.actual).abs().mean(),
                             "cover": 100 * d.hit.mean()}), include_groups=False)
    g = g.loc[g.n >= 30]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xlabel("days ahead the forecast was made", color=INK, fontsize=10.5)
        ax.set_xticks(list(g.index))

    a1.plot(g.index, g.mae, color=AQUA, linewidth=2.0, marker="o",
            markersize=8, markeredgecolor=BG, markeredgewidth=1.2, zorder=3)
    a1.set_ylim(0, g.mae.max() * 1.30)
    a1.set_ylabel("average error, surfers", color=INK, fontsize=10.5)
    a1.set_title("Error grows with lead time", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=10)
    for x, v in zip(g.index, g.mae):
        a1.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", color=INK_DIM, fontsize=9)

    a2.axhline(80, color=INK_DIM, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    a2.plot(g.index, g.cover, color=LIME, linewidth=2.0, marker="o",
            markersize=8, markeredgecolor=BG, markeredgewidth=1.2, zorder=3)
    a2.set_ylim(50, 100)
    a2.set_ylabel("actual counts inside the 80% range, %", color=INK, fontsize=10.5)
    a2.set_title("...and the prediction range stops holding", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=10)
    # Reference label parked at the left, away from the lead-6/7 markers.
    a2.annotate("80% promised", xy=(g.index[0], 80), xytext=(2, -14),
                textcoords="offset points", ha="left", color=INK_DIM, fontsize=9)
    # Label on the far side of the line from the marker, so neither sits on it.
    for x, v in zip(g.index, g.cover):
        a2.annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                    xytext=(0, 12 if v >= 80 else -17),
                    ha="center", color=INK_DIM, fontsize=9)

    fig.suptitle(f"How the forecast degrades further out  ·  {len(s):,} scored "
                 f"hours, {s.date.min()} to {s.date.max()}",
                 color=INK_DIM, fontsize=10, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "forecast_accuracy.png"
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    return p, g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_review()
    p1, sl, r = detector_chart(s)
    print(f"  {p1.name}  {len(s)} frames, slope {sl:.3f}, r {r:.3f}")
    p2, g = forecast_chart()
    print(f"  {p2.name}  leads {list(g.index)}, MAE {g.mae.round(2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
