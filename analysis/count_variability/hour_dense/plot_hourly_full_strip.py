import pandas as pd
import seaborn as sns
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

df = pd.read_csv(HERE / "hourly_variability_full.csv")
df = df[df["quality_ok"]].copy()

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
    color=AQUA, alpha=0.25, size=3, jitter=0.3, ax=ax, zorder=2,
)

grouped = df.groupby("clip_label")["count"]
means = grouped.mean().reindex(labels)
sems = (grouped.std() / grouped.count() ** 0.5).reindex(labels)
ax.errorbar(range(len(labels)), means.values, yerr=sems.values, color=LIME, marker="D", markersize=8,
            linewidth=2, capsize=5, elinewidth=1.5, ecolor=LIME, zorder=3,
            label="Mean count for that 5-min window (± std error)")

ax.set_xlabel("Clip start time")
ax.set_ylabel("Detected surfer count (per second)")
ax.set_title("Per-second detected surfer counts — 12 back-to-back 5-min clips, 8-9 AM, 2026-08-28\n"
              "(every dot = one second's detection count, n≈301/window; diamonds = mean ± std error)", color="white")
ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="white")
ax.grid(alpha=0.25, color="#333333")
for spine in ax.spines.values():
    spine.set_color("#333333")
plt.xticks(rotation=30, ha="right")

fig.tight_layout()
out_path = HERE / "hourly_full_counts_2026-08-28.png"
fig.savefig(out_path, dpi=150, facecolor="black")
print("saved", out_path)
