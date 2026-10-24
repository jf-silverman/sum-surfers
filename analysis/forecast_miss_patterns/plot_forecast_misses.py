"""
Where the forecast misses: by crowd size, by hour of day, and weekday against
weekend. Reads the scored rows of data/forecasts/forecast_log.csv.

  miss_by_crowd_size.png   mean signed error (forecast minus actual) in crowd
                           bins, cut two ways side by side: by what actually
                           showed up, and by what the forecast said
  miss_by_hour.png         lead-1 mean signed error and share of large misses
                           by hour, weekdays and weekends as separate lines

WHY BOTH CUTS IN THE FIRST CHART. Binned by the actual count, any forecast
that is less than perfect looks too low at the top and too high at the bottom,
calibrated or not: the packed hours are, by selection, the ones where the crowd
came in above what anything could have predicted. That is regression to the
mean, and it shows up even for a forecast with no bias at all. Binned by the
forecast, the question becomes the one a reader of the chart asks -- "when it
says 30, what do I get?" -- and only a bias that survives that cut is a fault
of the model.

The model's own output (`predicted_raw`) is scored, not the published median,
which since 2026-10-21 has the nightly affine correction applied. That keeps
the whole window on one definition.

Usage:
    python analysis/forecast_miss_patterns/plot_forecast_misses.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FLOG = ROOT / "data" / "forecasts" / "forecast_log.csv"
OUT = Path(__file__).resolve().parent

BG, INK, DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA, LIME = "#3ab4c9", "#9de35a"
BINS = [-0.1, 0, 5, 10, 20, 30, 40, 200]
BIN_LABELS = ["0", "1–5", "6–10", "11–20", "21–30", "31–40", "41+"]
BIG_MISS = 14       # about the top decile of |miss| across all leads


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=9)


def load():
    f = pd.read_csv(FLOG)
    s = f[f.actual.notna()].copy()
    s["e"] = s.predicted_raw - s.actual
    s["weekend"] = pd.to_datetime(s.date).dt.dayofweek >= 5
    return s


def _caption(s, extra=""):
    return (f"Made {date.today()}.  Scored forecast hours {s.date.min()} to {s.date.max()} "
            f"({s.date.nunique()} days, {s.weekend.groupby(s.date).first().sum()} of them "
            f"weekend days), from data/forecasts/forecast_log.csv.\n"
            f"Error = model forecast (predicted_raw, before the nightly correction) minus the "
            f"hour's detector count.{extra}")


def crowd_size_chart(s):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=BG, sharey=True)
    out = {}
    for ax, col, title in [(axes[0], "actual", "Cut by what actually showed up"),
                           (axes[1], "predicted_raw", "Cut by what the forecast said")]:
        _style(ax)
        g = s.groupby(pd.cut(s[col], BINS, labels=BIN_LABELS), observed=False).agg(
            n=("e", "size"), bias=("e", "mean"), actual=("actual", "mean"))
        g = g[g.n >= 20]
        out[col] = g
        x = np.arange(len(g))
        ax.bar(x, g.bias, width=0.75, color=[AQUA if b >= 0 else LIME for b in g.bias],
               edgecolor=BG, linewidth=1.5)
        ax.axhline(0, color=DIM, linewidth=1)
        for xi, b, n, a in zip(x, g.bias, g.n, g.actual):
            pct = f"\n{100 * b / a:+.0f}%" if a >= 5 else ""
            ax.text(xi, b + (0.6 if b >= 0 else -0.6), f"{b:+.1f}{pct}",
                    ha="center", va="bottom" if b >= 0 else "top", color=INK, fontsize=8.5)
        ax.set_xticks(x, [f"{l}\nn={n}" for l, n in zip(g.index, g.n)])
        ax.set_xlabel("surfers actually there" if col == "actual" else "surfers forecast",
                      color=INK, fontsize=10.5)
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    lo = min(g.bias.min() for g in out.values())
    hi = max(g.bias.max() for g in out.values())
    axes[0].set_ylim(lo * 1.35, hi * 1.6)
    axes[0].set_ylabel("forecast minus actual, surfers  (above 0 = too high)",
                       color=INK, fontsize=10)
    fig.suptitle("The forecast's misses by crowd size, cut two ways", color=INK, fontsize=13.5,
                 fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.885, "Left: packed hours look badly under-forecast, which any imperfect "
             "forecast shows when cut this way.\nRight: the test of bias -- when the forecast "
             "says N, how far off is it?  % = share of the mean actual count in that bin.",
             color=DIM, fontsize=9, ha="left", linespacing=1.5)
    fig.text(0.012, 0.015, _caption(s, "  All lead times pooled."), color=DIM, fontsize=8.5,
             ha="left", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.87))
    p = OUT / "miss_by_crowd_size.png"
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    return p, out


def hour_chart(s):
    d = s[s.lead_days == 1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.6), facecolor=BG, sharex=True)
    tab = {}
    for wk, color, lab in [(False, AQUA, "weekdays"), (True, LIME, "weekends")]:
        g = d[d.weekend == wk].groupby("hour").agg(
            n=("e", "size"), bias=("e", "mean"), actual=("actual", "mean"),
            big=("e", lambda e: 100 * (e.abs() >= BIG_MISS).mean()))
        g = g[g.n >= 5]
        tab[lab] = g
        for ax, col in [(a1, "bias"), (a2, "big")]:
            ax.plot(g.index, g[col], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=BG, markeredgewidth=1.2, label=f"{lab} ({d[d.weekend == wk].date.nunique()} days)")
    for ax in (a1, a2):
        _style(ax)
    a1.axhline(0, color=DIM, linewidth=1)
    a1.set_ylabel("forecast minus actual,\nsurfers (above 0 = too high)", color=INK, fontsize=10)
    a2.set_ylabel(f"hours missed by\n{BIG_MISS}+ surfers, %", color=INK, fontsize=10)
    a2.set_xlabel("hour of day", color=INK, fontsize=10.5)
    hrs = sorted(d.hour.unique())
    a2.set_xticks(hrs, [f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in hrs])
    a1.legend(frameon=False, labelcolor=INK, fontsize=9.5, loc="lower left")
    a1.set_title("When the next-day forecast misses", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=26)
    a1.text(0, 1.04, "Too high through the morning and midday, too low on weekend afternoons; "
            "the large misses are almost all after 1pm, when the crowds are.",
            transform=a1.transAxes, color=DIM, fontsize=9)
    fig.text(0.012, 0.015, _caption(d, "  Forecasts made the evening before (lead 1) only.\n"
                      f"Each weekday point rests on about 43 hours and each weekend point on about 16 "
                      f"(8 at 7pm), so single weekend points move a lot on one bad day."),
             color=DIM, fontsize=8.5, ha="left", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    p = OUT / "miss_by_hour.png"
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    return p, tab


def main():
    s = load()
    p, out = crowd_size_chart(s)
    print(" ", p.name)
    for k, g in out.items():
        print(f"    by {k}:\n{g.round(2).to_string()}")
    p, tab = hour_chart(s)
    print(" ", p.name)
    for k, g in tab.items():
        print(f"    {k}:\n{g.round(1).T.to_string()}")


if __name__ == "__main__":
    main()
