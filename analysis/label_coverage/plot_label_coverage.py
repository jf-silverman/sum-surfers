"""
How many surfers are in the frames hand-boxed in CVAT, and how that compares
with what the live camera sees.

Two charts, both written next to this script:

  cvat_counts_per_frame.png   surfers per hand-boxed frame, 5-surfer bins with
                              empty frames as their own (grey) bar
  cvat_vs_live_counts.png     the same bins as a share of frames, side by side
                              with every live frame that passed the quality
                              check in predictions.csv

The labeled set is the six raw CVAT exports, one entry per frame.
`posture_57` is the pose relabel of `reimport_existing_57` (same 57 frames), so
only the newer one is read. The totals match `splits_v2` (172 frames, 3,426
boxes), which is what the production model trained on.

The live counts are the object detector's, which run about 7% low against a
hand count, so the live bars sit slightly left of the truth. At 1-5 surfers
that is a fraction of a surfer, too small to move a frame between bins.

Each chart carries a footer with the date it was made and the date range of
both sets, so a copy that travels without this script still says what it is.

Usage:
    python analysis/label_coverage/plot_label_coverage.py
"""
from __future__ import annotations

import collections
import json
import re
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CVAT = ROOT / "data" / "cvat_out_coco"
PREDS = ROOT / "data" / "predictions" / "predictions.csv"
OUT = Path(__file__).resolve().parent
EXPORTS = ["batch01_73", "fog_19", "glare_12", "glare_more_5", "posture_57", "winter_7"]

BG, INK, DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA, LIME = "#3ab4c9", "#9de35a"


def load_cvat():
    """One row per labeled frame: file name, export, box count."""
    rows = []
    for name in EXPORTS:
        d = json.load(open(CVAT / name / "annotations" / "instances_default.json"))
        n = collections.Counter(a["image_id"] for a in d["annotations"])
        for im in d["images"]:
            rows.append({"file": Path(im["file_name"]).name, "export": name,
                         "count": n.get(im["id"], 0)})
    df = pd.DataFrame(rows)
    assert not df.file.duplicated().any(), "a frame appears in two exports"
    # Two naming schemes: crop2025-10-15_11-50-00.jpg and (the original 57)
    # jacks_20250811_1458.jpg / 20250719_0744.jpg.
    m = df.file.str.extract(r"(\d{4})-?(\d{2})-?(\d{2})[_-]")
    assert m.notna().all().all(), "a frame name carries no date"
    df["date"] = m[0] + "-" + m[1] + "-" + m[2]
    return df


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=9)


def _footer(fig, text):
    fig.text(0.012, 0.015, text, color=DIM, fontsize=8.5, ha="left", va="bottom",
             linespacing=1.5)


def _cvat_note(cv):
    return (f"Labeled set: {len(cv)} frames from {cv.date.min()} to {cv.date.max()}.\n"
            f"CVAT exports {', '.join(EXPORTS)} (data/cvat_out_coco/).")


def counts_chart(cv):
    c = cv["count"].to_numpy()
    nz = c[c > 0]
    edges = np.arange(1, c.max() + 6, 5)
    h, _ = np.histogram(nz, bins=edges)
    ys = [int((c == 0).sum())] + h.tolist()
    xs = np.arange(len(ys))
    labels = ["0"] + [f"{a}–{a+4}" for a in edges[:-1]]

    fig, ax = plt.subplots(figsize=(10, 5.9), facecolor=BG)
    _style(ax)
    ax.bar(xs, ys, width=0.86, color=[DIM] + [AQUA] * len(h), edgecolor=BG, linewidth=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.4, str(y), ha="center", va="bottom", color=DIM, fontsize=9)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, max(ys) * 1.1)
    ax.set_xlabel("surfers boxed in the frame", color=INK, fontsize=10.5)
    ax.set_ylabel("frames", color=INK, fontsize=10.5)
    ax.set_title("Surfers per frame in the hand-boxed CVAT set", color=INK,
                 fontsize=13.5, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.03, f"{len(c)} frames, {c.sum():,} boxes  ·  {int((c == 0).sum())} "
            f"empty frames (grey)  ·  frames with surfers: median {np.median(nz):.0f}, "
            f"mean {nz.mean():.1f}, max {c.max()}",
            transform=ax.transAxes, color=DIM, fontsize=9.5)
    _footer(fig, f"Made {date.today()}.  {_cvat_note(cv)}")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = OUT / "cvat_counts_per_frame.png"
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    return p


def vs_live_chart(cv):
    c = cv["count"].to_numpy()
    p = pd.read_csv(PREDS)
    p = p[p.quality_ok.astype(str) == "True"]
    live = p.surfer_count.astype(int).to_numpy()
    top = max(c.max(), live.max())
    edges = np.r_[0, np.arange(1, 62, 5), top + 1]       # 0 | 1-5 | ... | 56-60 | 61+
    labels = ["0"] + [f"{a}–{a+4}" for a in edges[1:-2]] + ["61+"]
    pl = 100 * np.histogram(live, bins=edges)[0] / len(live)
    pc = 100 * np.histogram(c, bins=edges)[0] / len(c)

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=BG)
    _style(ax)
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, pl, w, color=LIME, edgecolor=BG, linewidth=1.5,
           label=f"live camera frames ({len(live):,}, object detector count)")
    ax.bar(x + w / 2, pc, w, color=AQUA, edgecolor=BG, linewidth=1.5,
           label=f"hand-boxed CVAT frames ({len(c)})")
    fmt = lambda v: f"{v:.1f}" if 0 < v < 1 else f"{v:.0f}"
    for xi, a, b in zip(x, pl, pc):
        ax.text(xi - w / 2, a + 0.4, fmt(a), ha="center", va="bottom", color=DIM, fontsize=8)
        ax.text(xi + w / 2, b + 0.4, fmt(b), ha="center", va="bottom", color=DIM, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(pl.max(), pc.max()) * 1.12)
    ax.set_xlabel("surfers in the frame", color=INK, fontsize=10.5)
    ax.set_ylabel("share of frames, %", color=INK, fontsize=10.5)
    ax.set_title("Hand-boxed frames against the live camera", color=INK,
                 fontsize=13.5, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.03, f"Median surfers per frame: live {np.median(live):.0f}, "
            f"labeled {np.median(c):.1f}  ·  biggest gap at 1–5 surfers "
            f"({pl[1]:.0f}% of live frames, {pc[1]:.0f}% of labeled)",
            transform=ax.transAxes, color=DIM, fontsize=9.5)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9.5, loc="upper right")
    _footer(fig, f"Made {date.today()}.  {_cvat_note(cv)}\n"
            f"Live set: every frame that passed the quality check in predictions.csv, "
            f"{p.date.min()} to {p.date.max()}. The object detector counts about 7% low, "
            f"so live bars sit slightly left of the truth.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = OUT / "cvat_vs_live_counts.png"
    fig.savefig(out, facecolor=BG, dpi=150)
    plt.close(fig)
    return out, labels, pl, pc


def main():
    cv = load_cvat()
    print(" ", counts_chart(cv).name)
    out, labels, pl, pc = vs_live_chart(cv)
    print(" ", out.name)
    for l, a, b in zip(labels, pl, pc):
        print(f"    {l:>6}  live {a:5.1f}%  labeled {b:5.1f}%")


if __name__ == "__main__":
    main()
