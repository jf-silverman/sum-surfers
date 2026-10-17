"""
Is there a day-of-week effect beyond the weekend/weekday split?

`plot_count_by_weekday.py` draws the seven days and flags a Wednesday dip it
tells you not to believe. This tests it.

THE UNIT OF ANALYSIS IS THE DAY, NOT THE HOUR. Hours inside one day share a
tide, a swell, a wind and a sky, so 738 hours are nowhere near 738 independent
observations. Testing at the hour level would hand back p-values several orders
of magnitude too small -- it is the single easiest way to manufacture a
significant weekday effect out of nothing. Everything here collapses to one
number per date (127 of them) before any test runs.

ADJUSTMENTS, in order:

  hour    each count has its hour's median subtracted, so a day sampled mostly
          at 06:00 is not punished for it
  month   each day's value has its calendar month's median subtracted. The
          record covers nine months unevenly with four missing, so each weekday
          is drawn from a different month mix; without this, season leaks
          straight into the day-of-week comparison

Tide is handled by the filter rather than by adjustment: inside 0-3 ft the
response is nearly flat, so the residual tide difference between weekdays is
worth about two surfers.

TESTS. Kruskal-Wallis across all seven days; Mann-Whitney weekend against
weekday; Kruskal-Wallis across the five weekdays alone, which is the actual
question; pairwise Mann-Whitney among the weekdays with Holm correction; and a
label-shuffling permutation test as a check on the asymptotic p-values at these
sample sizes (about 18 days per weekday).

Usage:
    python analysis/weekday_weekend_patterns/test_day_of_week.py
    python analysis/weekday_weekend_patterns/test_day_of_week.py --tide-max 3.0
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEATURES = ROOT / "data" / "training_features.csv"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_PERM = 20000


def day_level(tide_max):
    t = pd.read_csv(FEATURES).dropna(subset=["surfer_count", "tide_ft"])
    t["date"] = pd.to_datetime(t.filename.str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    t["hour"] = pd.to_numeric(t.filename.str.extract(r"_(\d{2})-")[0], errors="coerce")
    t = t.dropna(subset=["hour"])
    t = t[t.tide_ft < tide_max]

    t["r_hour"] = t.surfer_count - t.groupby("hour").surfer_count.transform("median")
    d = (t.groupby("date")
           .agg(raw=("surfer_count", "mean"), adj=("r_hour", "mean"),
                hours=("surfer_count", "size"), tide=("tide_ft", "mean"))
           .reset_index())
    d["dow"] = d.date.dt.dayofweek
    d["month"] = d.date.dt.strftime("%Y-%m")
    d["adj_m"] = d.adj - d.groupby("month").adj.transform("median")
    return t, d


def kw(groups):
    groups = [g for g in groups if len(g) >= 2]
    return st.kruskal(*groups)


def perm_p(values, labels, observed, rng):
    """How often a reshuffle of the day labels beats the real statistic."""
    hits = 0
    lab = np.array(labels)
    for _ in range(N_PERM):
        rng.shuffle(lab)
        h, _p = kw([values[lab == k] for k in np.unique(lab)])
        hits += h >= observed
    return (hits + 1) / (N_PERM + 1)


def report(d, col, label):
    print(f"\n=== {label} ===")
    print(f"  {'day':>5s}{'days':>6s}{'median':>9s}{'mean':>8s}{'sd':>7s}")
    for i, name in enumerate(DAYS):
        g = d.loc[d.dow == i, col].to_numpy()
        print(f"  {name:>5s}{len(g):6d}{np.median(g):9.1f}{g.mean():8.1f}{g.std(ddof=1):7.1f}")

    seven = [d.loc[d.dow == i, col].to_numpy() for i in range(7)]
    h, p = kw(seven)
    print(f"\n  Kruskal-Wallis, all 7 days:        H={h:6.2f}  p={p:.4f}")

    we = d.loc[d.dow >= 5, col].to_numpy()
    wd = d.loc[d.dow < 5, col].to_numpy()
    u, pw = st.mannwhitneyu(we, wd)
    # rank-biserial, the effect size that goes with Mann-Whitney
    rb = 2 * u / (len(we) * len(wd)) - 1
    print(f"  Mann-Whitney, weekend vs weekday:  p={pw:.5f}  "
          f"rank-biserial={rb:+.2f}  ({np.median(we):.1f} vs {np.median(wd):.1f})")

    five = [d.loc[d.dow == i, col].to_numpy() for i in range(5)]
    h5, p5 = kw(five)
    print(f"  Kruskal-Wallis, Mon-Fri only:      H={h5:6.2f}  p={p5:.4f}"
          f"   <-- the question")

    two = [d.loc[d.dow == i, col].to_numpy() for i in (5, 6)]
    _u, psu = st.mannwhitneyu(*two)
    print(f"  Mann-Whitney, Sat vs Sun:          p={psu:.4f}"
          f"   ({np.median(two[0]):.1f} vs {np.median(two[1]):.1f})")

    rng = np.random.default_rng(3)
    sub = d[d.dow < 5]
    pp = perm_p(sub[col].to_numpy(), sub.dow.to_numpy(), h5, rng)
    print(f"  permutation check on Mon-Fri ({N_PERM:,} shuffles): p={pp:.4f}")

    print("\n  pairwise among weekdays (Holm-corrected):")
    pairs, raw = [], []
    for a, b in itertools.combinations(range(5), 2):
        ga, gb = d.loc[d.dow == a, col], d.loc[d.dow == b, col]
        raw.append(st.mannwhitneyu(ga, gb).pvalue)
        pairs.append((DAYS[a], DAYS[b], np.median(ga) - np.median(gb)))
    order = np.argsort(raw)
    m = len(raw)
    holm, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * raw[idx])
        holm[idx] = min(1.0, running)
    for (a, b, diff), pr, ph in zip(pairs, raw, holm):
        flag = "  *" if ph < 0.05 else ""
        print(f"    {a} vs {b}:  diff {diff:+6.1f}   p={pr:.3f}   Holm p={ph:.3f}{flag}")
    return p5


def power(d, col):
    """What size of weekday-to-weekday difference this record could detect."""
    sd = d.loc[d.dow < 5, col].std(ddof=1)
    n = d[d.dow < 5].groupby("dow").size().min()
    # two-sample t, alpha .05, power .80 -> |d| = 2.8 * sd * sqrt(2/n)
    mde = 2.8 * sd * np.sqrt(2 / n)
    print(f"\n  day-to-day sd {sd:.1f} surfers, smallest weekday n={n} days")
    print(f"  -> the smallest difference this record could detect at 80% power "
          f"is about {mde:.1f} surfers")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tide-max", type=float, default=3.0)
    args = ap.parse_args()

    t, d = day_level(args.tide_max)
    print(f"{len(t):,} counted hours over {len(d)} days, tide below "
          f"{args.tide_max:g} ft")
    print(f"hours per day: median {d.hours.median():.0f}, "
          f"range {d.hours.min()}-{d.hours.max()}")

    print("\n--- month mix, the confound the adjustment is for ---")
    mix = pd.crosstab(d.dow, d.month)
    mix.index = DAYS
    print(mix.to_string())

    report(d, "raw", "unadjusted day means (what the chart shows)")
    report(d, "adj_m", "hour- and month-adjusted (the test)")
    power(d, "adj_m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
