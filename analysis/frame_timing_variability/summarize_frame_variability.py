"""
summarize_frame_variability.py
--------------------------------
One-off analysis script. Reads frame_variability_analysis.csv (built by
analyze_frame_timing.py, same folder) and answers two questions about the
detector's per-second surfer count within a clip:

1. Lag analysis: how much does the count differ between two frames L seconds
   apart, pooled across all 14 clips? A flat/low value at small lags is
   detector noise on an essentially-unchanged scene; a rising trend at larger
   lags reflects real surfers entering/leaving the lineup.

2. Averaging-window analysis: if we average k frames spread evenly across a
   `span`-second window, how much does that reduce the stdev of the count
   estimate, vs. the naive independent-samples expectation sigma_1/sqrt(k)?

Saves frame_variability_analysis_summary.png (same folder) and prints the
numbers used to write findings back to the user -- every number here comes
directly from this run, nothing is estimated.

Usage:
    python analysis/frame_timing_variability/summarize_frame_variability.py
"""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "frame_variability_analysis.csv"
OUT_PNG = HERE / "frame_variability_analysis_summary.png"


def load_clip_series(df):
    """clip_time -> {second: count} for quality_ok rows only."""
    series = defaultdict(dict)
    for row in df.itertuples():
        series[row.clip_time][int(row.second)] = float(row.count)
    return series


def lag_analysis(series, max_lag=58):
    lag_diffs = defaultdict(list)
    for clip_time, s in series.items():
        seconds = sorted(s.keys())
        for lag in range(1, max_lag + 1):
            for t in seconds:
                if t + lag in s:
                    lag_diffs[lag].append(abs(s[t + lag] - s[t]))
    lags = sorted(lag_diffs.keys())
    means = [np.mean(lag_diffs[l]) for l in lags]
    stds = [np.std(lag_diffs[l]) for l in lags]
    ns = [len(lag_diffs[l]) for l in lags]
    return lags, means, stds, ns


def averaging_analysis(series, ks=(1, 2, 3, 4, 5, 6, 8, 10), spans=(2, 5, 10, 20, 30, 58)):
    """For each (k, span), slide a window across every clip's available seconds,
    pick k points as evenly spaced as possible within [start, start+span]
    (snapped to nearest available integer second, each point used once per
    window), require all k distinct and present, average their counts, and
    compare that k-frame mean against the CLIP'S OWN full-minute mean (the
    best available reference we have without human ground truth -- the
    average of all ~63 quality-passing seconds in that clip). Pooling raw
    counts across clips (which range from ~0 to ~45 surfers depending on
    time of day) would swap out real between-clip differences for detector
    noise and make k look useless -- comparing each window to its own
    clip's reference isolates within-clip variability instead.

    Returns:
      results: {(k, span): (stdev_of_residual, n_windows)}
      sigma_within: pooled stdev of (single frame count - its clip's own mean),
        i.e. the naive/independent-samples per-frame noise estimate used for
        the sigma_within/sqrt(k) reference curve.
    """
    clip_means = {ct: np.mean(list(s.values())) for ct, s in series.items()}
    within_clip_devs = [s[t] - clip_means[ct] for ct, s in series.items() for t in s]
    sigma_within = float(np.std(within_clip_devs))

    results = {}
    for k in ks:
        for span in spans:
            if k > 1 and span < k - 1:
                continue  # can't fit k distinct points spread over too-short a span
            residuals = []
            for clip_time, s in series.items():
                available = sorted(s.keys())
                if not available:
                    continue
                ref = clip_means[clip_time]
                max_t = available[-1]
                for start in available:
                    if start + span > max_t:
                        continue
                    if k == 1:
                        targets = [start]
                    else:
                        targets = [start + round(i * span / (k - 1)) for i in range(k)]
                    picked = []
                    used = set()
                    ok = True
                    for target in targets:
                        # nearest available second not already used in this window
                        candidates = sorted(available, key=lambda a: (abs(a - target), a))
                        chosen = None
                        for c in candidates:
                            if c not in used and start <= c <= start + span:
                                chosen = c
                                break
                        if chosen is None:
                            ok = False
                            break
                        used.add(chosen)
                        picked.append(s[chosen])
                    if ok and len(picked) == k:
                        residuals.append(np.mean(picked) - ref)
            if len(residuals) >= 5:  # need enough samples for a meaningful stdev
                results[(k, span)] = (np.std(residuals), len(residuals))
    return results, sigma_within


def main():
    df = pd.read_csv(CSV_PATH)
    df_ok = df[df["quality_ok"] == True].copy()  # noqa: E712
    print(f"Loaded {len(df)} rows total, {len(df_ok)} quality_ok=True")

    counts = df_ok["count"].astype(float).values
    sigma1 = float(np.std(counts))
    mean1 = float(np.mean(counts))
    print(f"Pooled single-frame count: mean={mean1:.2f}  stdev={sigma1:.2f}  n={len(counts)}")

    series = load_clip_series(df_ok)
    for ct, s in series.items():
        print(f"  {ct}: {len(s)} usable seconds (range {min(s) if s else '-'}-{max(s) if s else '-'})")

    lags, lag_means, lag_stds, lag_ns = lag_analysis(series)
    print("\nLag analysis (mean abs diff pooled across clips):")
    for l, m, sd, n in zip(lags, lag_means, lag_stds, lag_ns):
        if l in (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 58):
            print(f"  lag={l:2d}s  mean_abs_diff={m:.3f}  std={sd:.3f}  n={n}")

    avg_results, sigma_within = averaging_analysis(series)
    print(f"\nWithin-clip single-frame stdev (frame count minus its clip's own 63-frame "
          f"mean): sigma_within={sigma_within:.2f}  (pooled sigma1={sigma1:.2f} across all "
          f"clips is dominated by real across-day count differences, not detector noise)")
    print("Averaging-window analysis (empirical stdev of k-frame mean vs. its clip's own "
          "full-minute mean, vs naive sigma_within/sqrt(k)):")
    ks = sorted(set(k for k, _ in avg_results.keys()))
    spans = sorted(set(sp for _, sp in avg_results.keys()))
    for k in ks:
        naive = sigma_within / np.sqrt(k)
        row = []
        for span in spans:
            if (k, span) in avg_results:
                std, n = avg_results[(k, span)]
                row.append(f"span={span:2d}s: std={std:.2f}(n={n})")
        print(f"  k={k:2d}  naive_sigma_within/sqrt(k)={naive:.2f}   " + "  ".join(row))

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(lags, lag_means, marker="o", markersize=3, color="#2E86AB")
    ax.fill_between(lags,
                     [m - sd for m, sd in zip(lag_means, lag_stds)],
                     [m + sd for m, sd in zip(lag_means, lag_stds)],
                     alpha=0.2, color="#2E86AB")
    ax.set_xlabel("Lag between frames (seconds)")
    ax.set_ylabel("Mean |count(t+lag) - count(t)|  (+/- 1 stdev band)")
    ax.set_title("Count drift vs. time lag (pooled, 14 clips)")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(spans)))
    for span, color in zip(spans, colors):
        pts = [(k, avg_results[(k, span)][0]) for k in ks if (k, span) in avg_results]
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, marker="o", markersize=4, color=color, label=f"span={span}s")
    naive_ys = [sigma_within / np.sqrt(k) for k in ks]
    ax2.plot(ks, naive_ys, "--", color="black", label="naive sigma_within/sqrt(k)")
    ax2.set_xlabel("Number of frames averaged (k)")
    ax2.set_ylabel("Stdev of k-frame mean vs. clip's own full-minute mean")
    ax2.set_title("Variance reduction vs. frame count and span")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
