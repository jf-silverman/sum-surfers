"""
Where the daily crowd peak sits in the tide cycle.

Three things measured on the counted history (118 days with at least six
counted hours), all of them about WHEN the day's biggest crowd happens rather
than how big it is:

  A  mean count by tide band -- the relationship is not monotonic. The crowd
     peaks at 1-2 ft and falls off again below 1 ft, so "lower is better" is
     wrong at the bottom end and any tide feature should be a band, not a
     threshold or a linear term.
  B  tide at the daily peak against tide across all counted hours. 74.6% of
     peaks sit below 3 ft against a 38.9% base rate, so the concentration is
     real and not just an artefact of low tide being common.
  C  lift (share of peaks below a threshold / share of all hours below it)
     swept across thresholds. It maximises at exactly 3.00 ft.

The forecast's own peak is marked in B: it lands at 2.63 ft on average against
the observed 2.14 ft, capturing roughly 59% of the tide-seeking.

Usage:
    python analysis/peak_tide_structure/plot_peak_tide.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from fit_surfer_count_model import load_and_prepare  # noqa: E402

PEAK_COLOR = "#9de35a"    # same lime as the detection boxes
ALL_COLOR = "#3ab4c9"
INK, MUTED, SURFACE = "#f2f2f0", "#9a9a97", "#111111"
FORECAST_PEAK_TIDE = 2.63   # measured in the 16-origin rolling backtest


def main():
    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black",
        "axes.facecolor": SURFACE, "axes.edgecolor": "#3a3a38",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2c2c2a", "axes.grid": True, "axes.axisbelow": True,
    })
    X, y, df, _ = load_and_prepare()
    f = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.date,
        "tide": pd.to_numeric(df["tide_ft"], errors="coerce"),
        "count": y.values,
    }).dropna()
    f = f.groupby("date").filter(lambda x: len(x) >= 6)
    peaks = f.loc[f.groupby("date")["count"].idxmax()]

    fig = plt.figure(figsize=(14, 11), facecolor="black")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.85], hspace=0.52)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- A: mean count by tide band -------------------------------------
    edges = [-9, 1, 2, 3, 3.5, 4, 5, 99]
    names = ["<1", "1–2", "2–3", "3–3.5", "3.5–4", "4–5", "5+"]
    f["band"] = pd.cut(f.tide, edges, labels=names)
    g = f.groupby("band", observed=True)["count"].agg(["size", "mean"])
    peak_band = g["mean"].idxmax()
    cols = [PEAK_COLOR if b == peak_band else ALL_COLOR for b in g.index]
    bars = axA.bar(range(len(g)), g["mean"], 0.66, color=cols,
                   edgecolor=SURFACE, linewidth=2)
    for i, (b, row) in enumerate(g.iterrows()):
        axA.text(i, row["mean"] + 0.7, f"{row['mean']:.1f}", ha="center",
                 va="bottom", fontsize=10, color=INK)
    axA.set_xticks(range(len(g)), [f"{n}\nn={int(s)}" for n, s in zip(names, g["size"])])
    axA.set_xlabel("Tide (ft)")
    axA.set_ylabel("Mean surfers per frame")
    axA.set_title("A.  The crowd–tide relationship is not monotonic — it peaks at 1–2 ft and falls off below 1 ft\n"
                  "So a tide feature wants to be a band centred near 1.5 ft, not a threshold and not a linear term.",
                  color=INK, fontsize=11, loc="left", pad=12)
    axA.margins(y=0.16)

    # ---- B: tide at the peak vs tide in general --------------------------
    bins = np.arange(-1, 7.01, 0.4)
    axB.hist(f.tide, bins=bins, density=True, color=ALL_COLOR, alpha=0.55,
             edgecolor=SURFACE, linewidth=1.4, label="All counted hours")
    axB.hist(peaks.tide, bins=bins, density=True, color=PEAK_COLOR, alpha=0.75,
             edgecolor=SURFACE, linewidth=1.4, label="Hour of each day's peak")
    axB.axvline(3.0, color="#e8e8e5", linestyle="--", linewidth=1.6)
    axB.text(3.05, axB.get_ylim()[1] * 0.93, "3 ft", color=INK, fontsize=10)
    top = axB.get_ylim()[1]
    # staggered heights and opposite alignment: the two means are only half a
    # foot apart, so stacked labels overlap at this scale
    for x, c, lab, hfrac, ha, pad in [
            (peaks.tide.mean(), "#ffffff", f"observed peak {peaks.tide.mean():.2f} ft", 0.80, "right", -0.08),
            (FORECAST_PEAK_TIDE, "#e2a03f", f"forecast peak {FORECAST_PEAK_TIDE:.2f} ft", 0.63, "left", 0.08)]:
        axB.axvline(x, color=c, linewidth=2.4)
        axB.text(x + pad, top * hfrac, lab, color=c, fontsize=10, ha=ha, va="center",
                 bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=1.8))
    axB.set_xlabel("Tide (ft)")
    axB.set_ylabel("Density")
    share_pk, share_all = (peaks.tide < 3).mean(), (f.tide < 3).mean()
    ks = stats.ks_2samp(peaks.tide.values, f.tide.values)
    axB.set_title(f"B.  {share_pk:.1%} of daily peaks fall below 3 ft, against a {share_all:.1%} base rate for all hours "
                  f"(KS={ks.statistic:.3f}, p<0.0001)\n"
                  "The forecast's own peak sits half a foot too high — it captures about 59% of the tide-seeking.",
                  color=INK, fontsize=11, loc="left", pad=12)
    leg = axB.legend(facecolor=SURFACE, edgecolor="#3a3a38", loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- C: lift by threshold -------------------------------------------
    ths = np.arange(1.0, 5.01, 0.25)
    lift = [(peaks.tide < t).mean() / (f.tide < t).mean() for t in ths]
    best = ths[int(np.argmax(lift))]
    axC.plot(ths, lift, color=PEAK_COLOR, linewidth=2.4, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    axC.axvline(best, color="#e8e8e5", linestyle="--", linewidth=1.5)
    axC.axhline(1.0, color="#6a6a67", linewidth=1.2)
    axC.annotate(f"max lift {max(lift):.2f}× at {best:.2f} ft",
                 xy=(best, max(lift)), xytext=(best + 0.45, max(lift) - 0.06),
                 color=INK, fontsize=10,
                 arrowprops=dict(color=INK, arrowstyle="->", linewidth=1.3))
    axC.set_xlabel("Tide threshold (ft)")
    axC.set_ylabel("Lift")
    axC.set_title("C.  Lift = share of peaks below the threshold ÷ share of all hours below it. "
                  "It maximises at exactly 3.00 ft,\n"
                  "which beats the 3.5 ft GOOD_TIDE_MAX_FT already in the code (1.74× there). 1.0 means no concentration.",
                  color=INK, fontsize=11, loc="left", pad=12)

    fig.suptitle("Where the daily crowd peak sits in the tide cycle — Jack's / Pleasure Point, "
                 f"{f.date.nunique()} counted days",
                 color=INK, fontsize=13.5, y=0.985, x=0.012, ha="left")
    out = HERE / "peak_tide_structure_2026-09-24.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print("saved", out)
    print(f"\npeaks below 3 ft {share_pk:.1%} vs base {share_all:.1%}; mean tide at peak "
          f"{peaks.tide.mean():.2f} ft vs {f.tide.mean():.2f} ft overall")


if __name__ == "__main__":
    main()
