"""
Year-over-year weekday structure in the surfer counts.

The question this was built to answer was "compare 2025 and 2026 month by month,
weekday and weekend". It cannot be answered as asked: the two years' coverage is
disjoint. 2025 holds Oct/Nov/Dec and 2026 holds Mar/May/Jul/Aug/Sep, so the set
of months present in both years is empty and there is nothing to pair. The
camera pipeline started in Oct 2025 and 2026 has not reached October yet.

What IS comparable is the weekday shape *within* each month, because that
contrast never crosses a month boundary. So this draws two things:

  1. counts by weekday for each year, which mixes in the season difference and
     is labelled as such — it is context, not evidence;
  2. the same weekday shape after subtracting each month's own mean, which
     removes season and year entirely and is the figure to actually read.

Restricted to hours 08-16 so that months with long daylight are not compared
against months with short daylight on frame mix alone.

Usage:
    python analysis/weekday_weekend_patterns/plot_weekday_by_year.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PREDICTIONS = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"

# Same two colours the rest of the project uses for this contrast (and the same
# lime as the detection boxes), kept deliberately: they clear every legibility
# check, and stepping them darker to sit lower on a dark surface drops the
# aqua's chroma below the gray threshold and costs ~3 points of colourblind
# separation. Consistency across the project's charts is worth more.
WEEKDAY_COLOR = "#3ab4c9"
WEEKEND_COLOR = "#9de35a"
YEAR_COLORS = {2025: "#3ab4c9", 2026: "#9de35a"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CORE_HOURS = (8, 16)

INK = "#f2f2f0"
MUTED = "#9a9a97"
SURFACE = "#111111"


def style():
    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black",
        "axes.facecolor": SURFACE, "axes.edgecolor": "#3a3a38",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2c2c2a", "axes.grid": True, "axes.axisbelow": True,
    })


def load_days():
    df = pd.read_csv(PREDICTIONS)
    df = df[(df["quality_ok"].astype(str) == "True") & df["surfer_count"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["time_local"].astype(str).str.slice(0, 2).astype(int)
    df = df[(df["hour"] >= CORE_HOURS[0]) & (df["hour"] <= CORE_HOURS[1])]
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    day = (df.groupby(["ym", "date"])["surfer_count"].mean().reset_index())
    day["year"] = day["date"].dt.year
    day["dow"] = day["date"].dt.dayofweek
    # each month's own mean removed: what is left is weekday shape alone
    day["centered"] = day["surfer_count"] - day.groupby("ym")["surfer_count"].transform("mean")
    return day


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return 0.0
    return 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))


def main():
    style()
    day = load_days()
    years = sorted(day["year"].unique())

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 10.5), facecolor="black")

    # --- Panel A: raw means per weekday, one bar per year -------------------
    width = 0.38
    x = np.arange(7)
    for i, yr in enumerate(years):
        sub = day[day["year"] == yr]
        means = [sub[sub["dow"] == d]["surfer_count"].mean() for d in range(7)]
        errs = [ci95(sub[sub["dow"] == d]["surfer_count"]) for d in range(7)]
        ns = [int((sub["dow"] == d).sum()) for d in range(7)]
        off = (i - (len(years) - 1) / 2) * width
        bars = axA.bar(x + off, means, width * 0.94, label=str(yr),
                       color=YEAR_COLORS[yr], edgecolor=SURFACE, linewidth=2)
        axA.errorbar(x + off, means, yerr=errs, fmt="none",
                     ecolor="#d8d8d5", elinewidth=1.1, capsize=4)
        for b, m, e, n in zip(bars, means, errs, ns):
            axA.text(b.get_x() + b.get_width() / 2, 0.6, f"n={n}",
                     ha="center", va="bottom", fontsize=8, color=SURFACE, weight="bold")
            axA.text(b.get_x() + b.get_width() / 2, m + e + 0.7,
                     f"{m:.0f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    axA.set_xticks(x, DAYS)
    axA.set_ylabel("Mean surfers per frame")
    axA.set_title(
        "A.  Counts by weekday, each year separately — NOT a like-for-like comparison\n"
        "2025 is Oct–Dec only; 2026 is Mar, May, Jul–Sep. No month appears in both years, so the\n"
        "gap between the two bars is season as much as year. Read the shape across days, not the height.",
        color=INK, fontsize=11, loc="left", pad=14)
    leg = axA.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    axA.set_ylim(0, None)

    # --- Panel B: month-centered, the season-free view ----------------------
    means = [day[day["dow"] == d]["centered"].mean() for d in range(7)]
    errs = [ci95(day[day["dow"] == d]["centered"]) for d in range(7)]
    ns = [int((day["dow"] == d).sum()) for d in range(7)]
    colors = [WEEKEND_COLOR if d >= 5 else WEEKDAY_COLOR for d in range(7)]
    bars = axB.bar(x, means, 0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    axB.errorbar(x, means, yerr=errs, fmt="none", ecolor="#d8d8d5",
                 elinewidth=1.1, capsize=4)
    axB.axhline(0, color="#6a6a67", linewidth=1.2)
    for b, m, e in zip(bars, means, errs):
        va = "bottom" if m >= 0 else "top"
        pad = (e + 0.45) if m >= 0 else -(e + 0.45)
        axB.text(b.get_x() + b.get_width() / 2, m + pad, f"{m:+.1f}",
                 ha="center", va=va, fontsize=10, color=INK)
    axB.set_xticks(x, [f"{d}\nn={n}" for d, n in zip(DAYS, ns)])
    axB.margins(y=0.22)
    axB.set_ylabel("Surfers vs that month's average")
    axB.set_title(
        "B.  The same weekday shape with each month's own average subtracted — season and year removed\n"
        "Both years and all eight months pooled. Every weekday sits below its month's average and both\n"
        "weekend days sit above it. Bars are 95% confidence intervals on the mean.",
        color=INK, fontsize=11, loc="left", pad=14)

    handles = [plt.Rectangle((0, 0), 1, 1, color=WEEKDAY_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=WEEKEND_COLOR)]
    leg2 = axB.legend(handles, ["Weekday", "Weekend"], facecolor=SURFACE,
                      edgecolor="#3a3a38", loc="upper left")
    for t in leg2.get_texts():
        t.set_color(INK)

    fig.suptitle("Weekday structure in surfer counts — Jack's / Pleasure Point, hours 08–16, quality-passed frames",
                 color=INK, fontsize=13, y=0.995, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = HERE / "weekday_by_year_2026-09-23.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print("saved", out)

    # ---- the numbers, printed rather than trusted to the picture ----------
    we = day[day["dow"] >= 5]["centered"]
    wd = day[day["dow"] < 5]["centered"]
    print(f"\nweekend - weekday, month-centered: {we.mean() - wd.mean():+.2f} surfers "
          f"(n_weekend_days={len(we)}, n_weekday_days={len(wd)})")
    print("\nper-year day counts:")
    print(day.groupby(["year", "dow"]).size().unstack(fill_value=0)
          .rename(columns={i: d for i, d in enumerate(DAYS)}))


if __name__ == "__main__":
    main()
