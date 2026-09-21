"""
plot_training_metrics.py
---------------------------
Dark-themed remake of Ultralytics' own results.png for the production
YOLOv8s detector's training run, from its real per-epoch log -- no
re-training, no re-estimating, just restyling real numbers.

Points at whichever run is in production (RESULTS_CSV below). Since
2026-09-21 that is the fog retrain; the original October 2025 run
(train13) is the previous production model and remains in git history.

The chart marks the DEPLOYED checkpoint (best.pt, the epoch with the best
mAP@0.5:0.95) rather than quoting the final epoch: quoting epoch 60 for a
best.pt that was really epoch 51 is exactly the reporting error found on
2026-09-11.

Usage:
    python analysis/detector_training_metrics/plot_training_metrics.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS_CSV = PROJECT_ROOT / "data" / "model_out" / "20260921_fog" / "train" / "results.csv"
RUN_LABEL = "2026-09-21 fog retrain"

AQUA = "#3ab4c9"
LIME = "#9de35a"
BG = "black"
AXES_BG = "#111111"
GRID = "#333333"
TEXT = "white"

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.facecolor": AXES_BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": GRID,
})

df = pd.read_csv(RESULTS_CSV)
df.columns = [c.strip() for c in df.columns]

PANELS = [
    ("train/box_loss", "Box loss (train)"),
    ("train/cls_loss", "Class loss (train)"),
    ("train/dfl_loss", "DFL loss (train)"),
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
    ("val/box_loss", "Box loss (val)"),
    ("val/cls_loss", "Class loss (val)"),
    ("val/dfl_loss", "DFL loss (val)"),
    ("metrics/mAP50(B)", "mAP@0.5"),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for ax, (col, title) in zip(axes, PANELS):
    color = LIME if "metrics" in col else AQUA
    ax.plot(df["epoch"], df[col], color=color, linewidth=1.2, alpha=0.55, marker="o", markersize=3)
    smooth = df[col].rolling(5, center=True, min_periods=1).mean()
    ax.plot(df["epoch"], smooth, color="white", linewidth=1.8, linestyle=":")
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(GRID)

n_epochs = int(df["epoch"].max())
best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
best_epoch = int(best["epoch"])
for ax in fig.axes:
    ax.axvline(best_epoch, color=LIME, linestyle="--", linewidth=1.0, alpha=0.7)
fig.suptitle(
    f"YOLOv8s surfer-detector training run ({RUN_LABEL}, {n_epochs} epochs) — real per-epoch log, "
    "not estimated", color=TEXT, fontsize=13, y=0.99,
)
fig.text(
    0.5, 0.935,
    f"Deployed checkpoint (best.pt = epoch {best_epoch}, dashed line): "
    f"precision={best['metrics/precision(B)']:.3f}  "
    f"recall={best['metrics/recall(B)']:.3f}  "
    f"mAP@0.5={best['metrics/mAP50(B)']:.3f}  "
    f"mAP@0.5:0.95={best['metrics/mAP50-95(B)']:.3f}",
    color="#bbbbbb", fontsize=9.5, ha="center",
)

fig.tight_layout(rect=[0, 0, 1, 0.89])
out_path = HERE / "detector_training_metrics.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
print(f"\nDeployed checkpoint (epoch {best_epoch}) real values: "
      f"precision={best['metrics/precision(B)']:.5f} recall={best['metrics/recall(B)']:.5f} "
      f"mAP50={best['metrics/mAP50(B)']:.5f} mAP50-95={best['metrics/mAP50-95(B)']:.5f}")
