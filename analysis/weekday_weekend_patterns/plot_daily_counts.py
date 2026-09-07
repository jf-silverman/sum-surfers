import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

plt.style.use("dark_background")
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

df = pd.read_csv(PROJECT_ROOT / "data" / "training_features.csv", parse_dates=["date"])
daily = (
    df.groupby("date")
    .agg(mean_count=("surfer_count", "mean"), is_weekend=("is_weekend", "first"))
    .reset_index()
)
daily["Day type"] = daily["is_weekend"].map({True: "Weekend", False: "Weekday"})
daily["year_month"] = daily["date"].dt.to_period("M")
daily["month"] = daily["date"].dt.strftime("%b %Y")
month_order = [p.strftime("%b %Y") for p in sorted(daily["year_month"].unique())]

WEEKDAY_COLOR = "#3ab4c9"  # aqua blue (brightened for contrast on black)
WEEKEND_COLOR = "#9de35a"  # lime green

wd = daily.loc[~daily["is_weekend"], "mean_count"]
we = daily.loc[daily["is_weekend"], "mean_count"]

OUT_DIR = HERE

# --- Chart 1: monthly bar chart, std-dev error bars (within-month spread) ---
fig1, ax1 = plt.subplots(figsize=(14, 5.5), facecolor="black")
ax1.set_facecolor("#111111")

sns.barplot(
    data=daily, x="month", y="mean_count", hue="Day type", order=month_order,
    palette={"Weekday": WEEKDAY_COLOR, "Weekend": WEEKEND_COLOR},
    errorbar="sd", capsize=0.15, err_kws={"color": "white", "linewidth": 1.2}, ax=ax1,
)
ax1.set_xlabel("Month")
ax1.set_ylabel("Mean surfer count (per day)")
ax1.set_title("Mean surfer count by month — weekday vs weekend — Jack's / 38th St (n=91 days, spans Oct 2025 - Aug 2026)", color="white")
legend1 = ax1.legend(title=None, facecolor="#111111", edgecolor="#444444")
for text in legend1.get_texts():
    text.set_color("white")

fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/weekday_weekend_by_month_2026-08-28.png", dpi=150, facecolor="black", bbox_inches="tight")
print("saved", f"{OUT_DIR}/weekday_weekend_by_month_2026-08-28.png")

# --- Chart 2: KDE distribution ---
fig2, ax2 = plt.subplots(figsize=(14, 5.5), facecolor="black")
ax2.set_facecolor("#111111")

sns.kdeplot(
    data=daily, x="mean_count", hue="Day type",
    palette={"Weekday": WEEKDAY_COLOR, "Weekend": WEEKEND_COLOR},
    fill=False, linewidth=2.5, ax=ax2, common_norm=False,
)
ax2.set_xlabel("Mean surfer count (per day)")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of daily mean surfer counts (KDE, each curve independently normalized)", color="white")
legend2 = ax2.get_legend()
if legend2 is not None:
    legend2.set_title("Day type")
    legend2.get_title().set_color("white")
    legend2.get_frame().set_facecolor("#111111")
    legend2.get_frame().set_edgecolor("#444444")
    for text in legend2.get_texts():
        text.set_color("white")

fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/weekday_weekend_kde_2026-08-28.png", dpi=150, facecolor="black", bbox_inches="tight")
print("saved", f"{OUT_DIR}/weekday_weekend_kde_2026-08-28.png")

# --- Real numbers for the README prose (not baked into the images anymore) ---
print("\n=== Overall stats (for README prose) ===")
print(f"weekday: mean={wd.mean():.1f} std={wd.std():.1f} min={wd.min():.1f} max={wd.max():.1f} n={len(wd)}")
print(f"weekend: mean={we.mean():.1f} std={we.std():.1f} min={we.min():.1f} max={we.max():.1f} n={len(we)}")

print("\n=== Per-month weekend/weekday ratio (for the new first bullet) ===")
g = daily.groupby(["year_month", "is_weekend"])["mean_count"].mean().reset_index()
piv = g.pivot(index="year_month", columns="is_weekend", values="mean_count")
piv.columns = ["weekday_mean", "weekend_mean"]
piv["ratio"] = piv["weekend_mean"] / piv["weekday_mean"]
piv = piv.sort_index()
print(piv.round(2))
print(f"ratio range: {piv['ratio'].min():.2f}x to {piv['ratio'].max():.2f}x")
