"""Charts the 2026-09-29 18:00 hour probe: 45 frames, 15 clips, 4 minutes apart.

Two panels, because the probe answered two different questions and one of them
overturned the hypothesis it started from.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CSV = HERE / "hour_probe.csv"
OUT = HERE / "hour_probe_2026-09-29.png"

AQUA, LIME, CORAL = "#3ab4c9", "#9de35a", "#f0776c"
INK, MUTED, SURFACE = "#f2f2f0", "#9a9a97", "#111111"
FORECAST, LO, HI = 22.52, 6.98, 29.40
CENTRE = "18:17"


def main():
    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black",
        "axes.facecolor": SURFACE, "axes.edgecolor": "#3a3a38",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2c2c2a", "axes.grid": True, "axes.axisbelow": True,
    })
    d = pd.read_csv(CSV)
    clips = d.groupby(["offset_min", "clip_time"])["count"].agg(["mean", "std"]).reset_index()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), facecolor="black",
                                  gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.42})

    # ---- Panel A: the hour -------------------------------------------------
    ax.axhspan(LO, HI, color=AQUA, alpha=0.13, zorder=0)
    ax.axhline(FORECAST, color=AQUA, linewidth=2, zorder=2)
    ax.text(29, FORECAST + 0.5, f"forecast {FORECAST:.1f}", color=AQUA, fontsize=9, ha="right")
    ax.text(29, HI - 1.6, "80% range", color=AQUA, fontsize=8, ha="right", alpha=0.85)

    ax.scatter(d.offset_min, d["count"], s=26, color=MUTED, alpha=0.75, zorder=3,
               label="individual frame (3 per clip, 1.5s apart)")
    ax.scatter(clips.offset_min, clips["mean"], s=95, color=LIME, edgecolor=SURFACE,
               linewidth=1.5, zorder=4, label="clip mean (3 frames)")

    sl, ic, r, p, _ = stats.linregress(d.offset_min, d["count"])
    xs = np.array([d.offset_min.min(), d.offset_min.max()])
    ax.plot(xs, ic + sl * xs, color=CORAL, linewidth=2.2, linestyle="--", zorder=5,
            label=f"trend {sl:+.2f}/min (p<0.0001)")

    c = clips[clips.clip_time == CENTRE]
    ax.scatter(c.offset_min, c["mean"], s=280, facecolor="none", edgecolor="#ffffff",
               linewidth=2.2, zorder=6)
    # Placed right-of-centre and low: the left/bottom corner is where the axis
    # label sits, and the first attempt ran straight through it.
    ax.annotate("18:17 — the one sample\nthe pipeline actually took",
                xy=(0.6, float(c["mean"].iloc[0]) - 0.6), xytext=(11, 3.4),
                color=INK, fontsize=9.5, ha="left",
                arrowprops=dict(color=INK, arrowstyle="->", linewidth=1.3,
                                connectionstyle="arc3,rad=0.15"))
    ax.margins(y=0.10)

    ax.set_xlabel("Minutes from 18:17")
    ax.set_ylabel("Surfers detected")
    ax.set_title(
        "A.  One hour sampled properly — 15 clips 4 minutes apart, 2026-09-29\n"
        "The hour is emptying out at dusk, not holding steady. The single 18:17 sample caught a low point in a real decline,\n"
        "so it reads as a huge forecast miss (−15.5) when the true hour mean of 12.8 misses by −9.8.",
        color=INK, fontsize=11, loc="left", pad=14)
    leg = ax.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- Panel B: the hypothesis that did not survive ----------------------
    rr, pp = stats.pearsonr(d.lap_var, d["count"])
    ax2.scatter(d.lap_var, d["count"], s=40, color=AQUA, edgecolor=SURFACE,
                linewidth=1, zorder=3)
    hot = d[d.clip_time == CENTRE]
    ax2.scatter(hot.lap_var, hot["count"], s=110, color=CORAL, edgecolor=SURFACE,
                linewidth=1.4, zorder=4, label="18:17 frames")
    sl2, ic2, _, _, _ = stats.linregress(d.lap_var, d["count"])
    xs2 = np.array([d.lap_var.min(), d.lap_var.max()])
    ax2.plot(xs2, ic2 + sl2 * xs2, color=MUTED, linewidth=1.6, linestyle=":", zorder=2)
    ax2.set_xlabel("Laplacian variance (higher = more high-frequency detail, i.e. whitewater)")
    ax2.set_ylabel("Surfers detected")
    ax2.set_title(
        f"B.  Laplacian variance does NOT flag hidden surfers — r = {rr:+.3f}, p = {pp:.2f} across all 45 frames\n"
        "The 18:17 frames are high-lap_var AND low-count, which is what suggested the idea; the rest of the hour does not follow.\n"
        "17:53 has the highest lap_var of the probe (245) with 18 surfers visible.",
        color=INK, fontsize=11, loc="left", pad=14)
    leg2 = ax2.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper left", fontsize=9)
    for t in leg2.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="black", bbox_inches="tight")
    print("saved", OUT)
    w = d.groupby("clip_time")["count"].std().mean()
    print(f"within-clip sd {w:.2f} | across-clip sd {clips['mean'].std():.2f} "
          f"| ratio {clips['mean'].std() / w:.1f}x")


if __name__ == "__main__":
    main()
