import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#111111",
    "figure.facecolor": "black",
    "savefig.facecolor": "black",
    "grid.color": "#333333",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

AQUA = "#3ab4c9"
LIME = "#9de35a"

df = pd.read_csv(HERE / "frame_variability_analysis.csv")
df = df[df["quality_ok"]].copy()

# Order clips chronologically and build a readable "h:mm AM/PM" label per clip.
def label_for(clip_time):
    h, m = clip_time.split("_")
    return datetime.strptime(f"{h}:{m}", "%H:%M").strftime("%-I:%M %p")

order = sorted(df["clip_time"].unique(), key=lambda s: tuple(map(int, s.split("_"))))
labels = [label_for(ct) for ct in order]
df["clip_label"] = df["clip_time"].map(dict(zip(order, labels)))

fig, ax = plt.subplots(figsize=(15, 7), facecolor="black")
ax.set_facecolor("#111111")

sns.stripplot(
    data=df, x="clip_label", y="count", order=labels,
    color=AQUA, alpha=0.5, size=4, jitter=0.25, ax=ax, zorder=2,
)

means = df.groupby("clip_label")["count"].mean().reindex(labels)
ax.plot(range(len(labels)), means.values, color=LIME, marker="D", markersize=8,
        linewidth=2, zorder=3, label="Mean count for that clip")

ax.set_xlabel("Clip start time (hour)")
ax.set_ylabel("Detected surfer count (per second)")
ax.set_title("Per-second detected surfer counts across 14 real 60-second clips — 2026-08-27\n"
              "(every dot = one second's detection count; diamonds = per-clip mean)", color="white")
ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="white")
ax.grid(alpha=0.25, color="#333333")
for spine in ax.spines.values():
    spine.set_color("#333333")
plt.xticks(rotation=30, ha="right")

fig.tight_layout()
out_path = HERE / "hourly_60sec_counts_2026-08-27.png"
fig.savefig(out_path, dpi=150, facecolor="black")
print("saved", out_path)
