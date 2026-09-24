"""
The year-coverage scatter, with Joel's hand counts overlaid.

Same calendar axis as plot_year_coverage.py -- every counted day of both years
on one Jan-Dec grid, which is the picture of the two years never sharing a
month. This version adds the frames Joel counted by hand, so it also shows
where the detector has been independently checked and where it has not.

Two things the axis is deliberately honest about:

  * detector points are a DAILY MEAN over hours 08-16, while a hand count is
    ONE frame at one moment. Both are "surfers in a frame", so they share the
    axis, but a hand-count dot is a single observation and a detector dot is an
    average of about ten. A hand count sitting above the detector line for its
    day is therefore not necessarily a disagreement.
  * the 70 frames in data/reviews/count_60sec_var are left out. They are named
    by clip and second (set1_05_50/sec_00.jpg) with no date recorded anywhere,
    so they cannot be placed on a calendar axis at all.

Usage:
    python analysis/weekday_weekend_patterns/plot_year_coverage_hand_counts.py
"""

import os
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
HAND_COLOR = "#d98cf5"     # chosen so it does not become the limiting CVD pair
CORE_HOURS = (8, 16)
INK, MUTED, SURFACE = "#f2f2f0", "#9a9a97", "#111111"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# (path, filename column, hand-count column). Every set that records a date.
HAND_SOURCES = [
    ("data/reviews/model_spotcheck_50/review_counts.csv", "filename", "my_count"),
    ("data/reviews/fog_quality/enhance_batch/enhance_review.csv", "filename", "true_count"),
    ("data/predictions/predictions.csv", "filename", "human_count"),
]


def load_hand_counts():
    frames = []
    for rel, fcol, ccol in HAND_SOURCES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            print(f"  WARNING: {rel} not found - skipped")
            continue
        df = pd.read_csv(p)
        if ccol not in df.columns:
            print(f"  WARNING: {rel} has no '{ccol}' column - skipped")
            continue
        v = pd.to_numeric(df[ccol], errors="coerce")   # drops notes like "lens condensation"
        keep = df.loc[v.notna(), [fcol]].copy()
        keep["hand"] = v[v.notna()].to_numpy()
        keep["src"] = os.path.basename(os.path.dirname(rel))
        frames.append(keep.rename(columns={fcol: "filename"}))
        print(f"  {rel}: {len(keep)} hand counts")
    H = pd.concat(frames, ignore_index=True).drop_duplicates(subset="filename")
    H["date"] = pd.to_datetime(H.filename.str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")
    dropped = H.date.isna().sum()
    if dropped:
        print(f"  {dropped} hand count(s) had no parseable date - not plottable")
    return H.dropna(subset=["date"])


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
    core = df[(df["hour"] >= CORE_HOURS[0]) & (df["hour"] <= CORE_HOURS[1])]
    day = core.groupby("date")["surfer_count"].mean().reset_index()
    day["year"] = day["date"].dt.year
    day["doy"] = day["date"].dt.dayofyear
    day["weekend"] = day["date"].dt.dayofweek >= 5

    print("Hand-count sources:")
    H = load_hand_counts()
    H["doy"] = H["date"].dt.dayofyear
    H["year"] = H["date"].dt.year

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor="black")

    for yr in sorted(day["year"].unique()):
        sub = day[day["year"] == yr]
        for m in sorted(sub["date"].dt.month.unique()):
            start = pd.Timestamp(yr, m, 1).dayofyear
            end = (pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(1)).dayofyear
            ax.axvspan(start, end, color=YEAR_COLORS[yr], alpha=0.07, zorder=0)

    # detector layer, deliberately recessive so the hand counts read on top
    for yr in sorted(day["year"].unique()):
        sub = day[day["year"] == yr]
        for is_we, marker, size in [(False, "o", 34), (True, "D", 46)]:
            s = sub[sub["weekend"] == is_we]
            ax.scatter(s["doy"], s["surfer_count"], marker=marker, s=size,
                       color=YEAR_COLORS[yr], edgecolor=SURFACE, linewidth=1.2,
                       alpha=0.85, zorder=3,
                       label=f"{yr} detector, daily mean ({'weekend' if is_we else 'weekday'})")

    ax.scatter(H["doy"], H["hand"], marker="*", s=210, color=HAND_COLOR,
               edgecolor=SURFACE, linewidth=1.3, zorder=5,
               label=f"Hand count, single frame (n={len(H)})")

    ax.set_xticks([pd.Timestamp(2026, m, 15).dayofyear for m in range(1, 13)], MONTHS)
    ax.set_xlim(1, 366)
    ax.set_ylabel("Surfers in a frame")
    by_year = H.groupby("year").size().to_dict()
    ax.set_title(
        "Every counted day of both years, with hand-counted frames overlaid\n"
        f"Stars are frames Joel counted by eye ({by_year.get(2025, 0)} in 2025, {by_year.get(2026, 0)} in 2026); "
        "circles and diamonds are the detector's daily mean over hours 08–16.\n"
        "A star is one frame at one moment, a dot is an average of about ten, so a star above its day's dot is not by itself a disagreement.",
        color=INK, fontsize=11, loc="left", pad=14)
    leg = ax.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left", ncol=2, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.margins(y=0.12)

    fig.tight_layout()
    out = HERE / "year_coverage_hand_counts_2026-09-24.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print("\nsaved", out)
    print(f"hand counts plotted: {len(H)} across {H.date.dt.strftime('%Y-%m').nunique()} months")
    print(H.groupby(H.date.dt.strftime("%Y-%m")).size().to_string())


if __name__ == "__main__":
    main()
