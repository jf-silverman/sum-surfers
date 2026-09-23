"""
Per-day counts for 2025 and 2026 on a shared calendar axis.

This is the picture of why a paired-month year-over-year comparison is not
available: laid on a common Jan-Dec axis, the two years do not overlap
anywhere. 2025 runs Oct-Dec, 2026 runs Mar-Sep, and the set of months holding
data in both years is empty.

Each dot is one day's mean surfer count over hours 08-16, quality-passed
frames only. Weekends are drawn as diamonds so the weekly structure is visible
in the raw days rather than only in the aggregate.

Usage:
    python analysis/weekday_weekend_patterns/plot_year_coverage.py
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

YEAR_COLORS = {2025: "#3ab4c9", 2026: "#9de35a"}
CORE_HOURS = (8, 16)
INK, MUTED, SURFACE = "#f2f2f0", "#9a9a97", "#111111"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black",
        "axes.facecolor": SURFACE, "axes.edgecolor": "#3a3a38",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2c2c2a", "axes.grid": True, "axes.axisbelow": True,
    })

    df = pd.read_csv(PREDICTIONS)
    df = df[(df["quality_ok"].astype(str) == "True") & df["surfer_count"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["time_local"].astype(str).str.slice(0, 2).astype(int)
    df = df[(df["hour"] >= CORE_HOURS[0]) & (df["hour"] <= CORE_HOURS[1])]
    day = df.groupby("date")["surfer_count"].mean().reset_index()
    day["year"] = day["date"].dt.year
    day["doy"] = day["date"].dt.dayofyear
    day["weekend"] = day["date"].dt.dayofweek >= 5

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="black")

    # shade each year's covered months, so the empty overlap is the visible point
    for yr in sorted(day["year"].unique()):
        sub = day[day["year"] == yr]
        for m in sorted(sub["date"].dt.month.unique()):
            start = pd.Timestamp(yr, m, 1).dayofyear
            end = (pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(1)).dayofyear
            ax.axvspan(start, end, color=YEAR_COLORS[yr], alpha=0.07, zorder=0)

    for yr in sorted(day["year"].unique()):
        sub = day[day["year"] == yr]
        for is_we, marker, size in [(False, "o", 42), (True, "D", 58)]:
            s = sub[sub["weekend"] == is_we]
            ax.scatter(s["doy"], s["surfer_count"], marker=marker, s=size,
                       color=YEAR_COLORS[yr], edgecolor=SURFACE, linewidth=1.4,
                       zorder=3, label=f"{yr} {'weekend' if is_we else 'weekday'}")

    ax.set_xticks([pd.Timestamp(2026, m, 15).dayofyear for m in range(1, 13)], MONTHS)
    ax.set_xlim(1, 366)
    ax.set_ylabel("Mean surfers per frame (that day, 08–16)")
    ax.set_title(
        "Every day of data, both years, on one calendar axis — the two years never share a month\n"
        "Shaded spans mark the months each year covers. 2025 (aqua) holds Oct–Dec; 2026 (lime) holds Mar, May, Jul–Sep.\n"
        "The overlap is empty, so there is no month pair to compare across years. Diamonds are weekend days.",
        color=INK, fontsize=11.5, loc="left", pad=14)
    leg = ax.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.margins(y=0.12)

    fig.tight_layout()
    out = HERE / "year_coverage_2026-09-23.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print("saved", out)

    cov = day.groupby([day["year"], day["date"].dt.month]).size().unstack(fill_value=0)
    cov.columns = [MONTHS[c - 1] for c in cov.columns]
    print("\ndays of data per year and month:")
    print(cov)
    both = [c for c in cov.columns if (cov[c] > 0).all()]
    print(f"\nmonths with data in BOTH years: {both if both else 'none'}")


if __name__ == "__main__":
    main()
