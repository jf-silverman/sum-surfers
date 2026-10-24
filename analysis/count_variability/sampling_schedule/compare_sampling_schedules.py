"""
Would sampling the camera more than once an hour give a truer hourly count?

The dense hours under ../hour_dense/ are twelve back-to-back 5-minute clips
counted at every second, so each one carries the hour's full count to
compare against. This replays several sampling schedules on them and scores
each against the mean over every quality-ok frame in the hour, which is the
quantity the forecast is trying to predict.

  production   one clip, frames at 1.0 / 2.5 / 4.0 s, mean rounded. The 2.5 s
               frame is taken as the mean of seconds 2 and 3, since the dense
               hours are sampled once a second.
  spread       one clip, three frames at 1/6, 1/2 and 5/6 of its length
  2 clips      two clips 30 minutes apart (clip i and clip i+6), 3 frames each
  4 clips      four clips 15 minutes apart, 3 frames each

Every starting clip in the hour is tried, so production is scored 12 times
per hour, the 2-clip schedules 6 times and the 4-clip one 3 times. With three
dense hours that is a small sample; the result is consistent across all three.

Writes sampling_schedule_error.png next to this script.

Usage:
    python analysis/count_variability/sampling_schedule/compare_sampling_schedules.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DENSE = HERE.parent / "hour_dense"

BG, INK, DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA = "#3ab4c9"
SCHEMES = ["1 clip, 3 frames\n(production)", "1 clip, 3 frames\nspread across it",
           "2 clips 30 min apart,\n3 frames each", "2 clips 30 min apart,\nframes spread",
           "4 clips 15 min apart,\n3 frames each"]


def score_hour(path):
    d = pd.read_csv(path)
    d = d[d.quality_ok]
    truth = d["count"].mean()
    c = {k: g.set_index("second")["count"].sort_index() for k, g in d.groupby("clip_time")}
    clips = sorted(c)
    n = len(clips)

    def prod(k):
        s = c[k]
        return np.mean([s[1], (s[2] + s[3]) / 2, s[4]])

    def spread(k):
        s = c[k]
        return np.mean([s.iloc[len(s) // 6], s.iloc[len(s) // 2], s.iloc[5 * len(s) // 6]])

    est = {k: [] for k in SCHEMES}
    for i in range(n):
        est[SCHEMES[0]].append(prod(clips[i]))
        est[SCHEMES[1]].append(spread(clips[i]))
        if i < n // 2:
            j = i + n // 2
            est[SCHEMES[2]].append((prod(clips[i]) + prod(clips[j])) / 2)
            est[SCHEMES[3]].append((spread(clips[i]) + spread(clips[j])) / 2)
        if i < n // 4:
            est[SCHEMES[4]].append(np.mean([prod(clips[i + m * n // 4]) for m in range(4)]))
    hour = path.stem.split("_")[-1]
    rows = [{"hour": hour, "truth": truth, "scheme": k, "n": len(v),
             "mae": np.abs(np.round(v) - truth).mean(),
             "rmse": np.sqrt(((np.round(v) - truth) ** 2).mean())} for k, v in est.items()]
    return rows


def main():
    rows = []
    for p in sorted(DENSE.glob("hourly_variability_full_*.csv")):
        rows += score_hour(p)
    df = pd.DataFrame(rows)
    pooled = df.groupby("scheme", sort=False)[["mae", "rmse"]].mean()
    print(df.pivot(index="scheme", columns="hour", values="mae").loc[SCHEMES].round(2).to_string())
    print(pooled.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 5.9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=9)
    x = np.arange(len(SCHEMES))
    ax.bar(x, pooled.mae.loc[SCHEMES], width=0.6, color=AQUA, edgecolor=BG, linewidth=2)
    marks = ["o", "s", "^"]
    for (h, g), m in zip(df.groupby("hour"), marks):
        g = g.set_index("scheme").loc[SCHEMES]
        ax.scatter(x, g.mae, marker=m, s=34, color=INK, edgecolors=BG, linewidths=1, zorder=3,
                   label=f"{h}  (hour mean {g.truth.iloc[0]:.1f} surfers)")
    for xi, v in zip(x, pooled.mae.loc[SCHEMES]):
        ax.text(xi + 0.32, v, f"{v:.2f}", va="center", color=INK, fontsize=9.5)
    ax.set_xticks(x, SCHEMES)
    ax.set_ylabel("average miss against the full-hour count, surfers", color=INK, fontsize=10)
    ax.set_title("Sampling twice an hour halves the measurement error", color=INK,
                 fontsize=13.5, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.03, "Bars: mean over the three dense hours.  Markers: each hour on its own.  "
            "Every schedule's count is rounded, as production rounds it.",
            transform=ax.transAxes, color=DIM, fontsize=9.5)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    fig.text(0.012, 0.015,
             f"Made {date.today()}.  Dense hours: twelve consecutive 5-minute clips counted every "
             f"second (analysis/count_variability/hour_dense/hourly_variability_full_*.csv).\n"
             f"Truth = mean count over every quality-ok frame in the hour.  Each schedule is replayed "
             f"from every starting clip in the hour (12, 6 or 3 replays per hour).",
             color=DIM, fontsize=8.5, ha="left", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = HERE / "sampling_schedule_error.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    plt.close(fig)
    print(" ", out.name)


if __name__ == "__main__":
    main()
