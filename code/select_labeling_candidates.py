"""
select_labeling_candidates.py
--------------------------------
Phase 1 of the training-data-expansion plan (see
/Users/jfs-m3/.claude/plans/dazzling-rolling-lemon.md, or its content
mirrored into docs/PROJECT_HISTORY.md once implemented): produces a
prioritized, data-driven list of candidate images for Joel to review and
manually label in CVAT, combining two real sources -- not an auto-decision,
just a starting shortlist.

1. Error-driven tier: real human-vs-model count mismatches already on
   record in data/reviews/model_spotcheck_50/review_counts.csv, ranked by
   how far off the model was (both by raw percent difference and by a
   count-weighted score, shown side by side rather than silently picking
   one formula).
2. Gap-fill tier: candidates from data/training_features.csv sampled to
   fill in real, verified underrepresentation in the existing training set
   -- by month, tide range, surfer-count range, and weather category --
   using data/cvat_out_coco/splits/instances_*.json to see what's already
   labeled (excluded from candidates) and training_features.csv's real
   distribution to see what's thin.

Every candidate filename is checked against data/j_shore_cam/surf_crops/
before being included -- if a file doesn't exist on disk, it's dropped
with a warning rather than silently included.

Usage:
    python code/select_labeling_candidates.py
    python code/select_labeling_candidates.py --target-n 60 --gap-fill-n 45
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
CROPS_DIR = DATA_DIR / "j_shore_cam" / "surf_crops"
SPOTCHECK_CSV = DATA_DIR / "reviews" / "model_spotcheck_50" / "review_counts.csv"
TRAINING_FEATURES_CSV = DATA_DIR / "training_features.csv"
CVAT_SPLITS_DIR = DATA_DIR / "cvat_out_coco" / "splits"
OUT_CSV = _PROJECT_ROOT / "analysis" / "training_data_expansion" / "labeling_candidates.csv"

TIDE_BUCKETS = [(-2, 0, "<0ft"), (0, 2, "0-2ft"), (2, 4, "2-4ft"), (4, 6, "4-6ft"), (6, 8, ">6ft")]
COUNT_BUCKETS = [(0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"), (30, 40, "30-39"), (40, 1000, "40+")]


def bucket(value, buckets):
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return "unknown"


def existing_cvat_filenames():
    """Filenames already labeled, across all three current splits. The old
    export uses a different naming convention (jacks_YYYYMMDD_HHMM.jpg /
    YYYYMMDD_HHMM.jpg) than production crop*.jpg files, so this is mostly a
    safety net rather than an expected overlap."""
    names = set()
    for split in ["train", "val", "test"]:
        p = CVAT_SPLITS_DIR / f"instances_{split}.json"
        if not p.exists():
            continue
        with open(p) as f:
            coco = json.load(f)
        names.update(img["file_name"] for img in coco["images"])
    return names


def load_error_tier(already_labeled, error_tier_n):
    """Real human-vs-model mismatches from the 50-image spot check, capped to
    the error_tier_n most valuable (by count-weighted score) -- without a
    cap, EVERY usable spot-check row gets included regardless of how small
    the miss is (e.g. a +/-20% gap on a count of 5-6 surfers), which starves
    the gap-fill tier of its share of the total target."""
    df = pd.read_csv(SPOTCHECK_CSV)

    def to_num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    df["my_count_num"] = df["my_count"].apply(to_num)
    df["model_count_num"] = df["model_count"].apply(to_num)
    df = df.dropna(subset=["my_count_num", "model_count_num"])
    df = df[df["my_count_num"] > 0]  # pct_diff undefined at my_count=0
    df["pct_diff"] = (df["model_count_num"] - df["my_count_num"]) / df["my_count_num"] * 100
    df["count_weighted_score"] = df["pct_diff"].abs() * df["my_count_num"]

    rows = []
    for _, r in df.iterrows():
        fname = r["filename"]
        if fname in already_labeled:
            continue
        if not (CROPS_DIR / fname).exists():
            print(f"  WARNING: spot-check candidate {fname} not found on disk, skipping")
            continue
        rows.append(dict(
            filename=fname, source_tier="error", model_count=r["model_count_num"],
            my_count=r["my_count_num"], pct_diff=round(r["pct_diff"], 1),
            count_weighted_score=round(r["count_weighted_score"], 1),
        ))
    rows_by_pct = sorted(rows, key=lambda r: abs(r["pct_diff"]), reverse=True)
    rows_by_weighted = sorted(rows, key=lambda r: r["count_weighted_score"], reverse=True)

    print(f"\n=== Error tier: {len(rows)} usable candidates (of {len(df)} spot-check rows) ===")
    print("Top 10 by raw |pct_diff|:")
    for r in rows_by_pct[:10]:
        print(f"  {r['filename']:32s} model={r['model_count']:.0f} human={r['my_count']:.0f} pct_diff={r['pct_diff']:+.1f}%")
    print("Top 10 by count-weighted score (|pct_diff| x human_count -- favors high-count misses):")
    for r in rows_by_weighted[:10]:
        print(f"  {r['filename']:32s} model={r['model_count']:.0f} human={r['my_count']:.0f} pct_diff={r['pct_diff']:+.1f}%  score={r['count_weighted_score']:.0f}")

    selected = rows_by_weighted[:error_tier_n]
    print(f"Capped to top {len(selected)} of {len(rows)} usable candidates by count-weighted score "
          f"(--error-tier-n={error_tier_n}).")
    return selected  # default ranking; both scores are in the output CSV either way


def load_gap_fill_tier(n_target, exclude_filenames, already_labeled):
    """Samples from training_features.csv to fill in real, verified gaps in
    the current labeled set: month, tide range, count range, weather."""
    df = pd.read_csv(TRAINING_FEATURES_CSV)
    df = df[~df["filename"].isin(exclude_filenames)]
    df = df[~df["filename"].isin(already_labeled)]

    df["tide_bucket"] = df["tide_ft"].apply(lambda v: bucket(v, TIDE_BUCKETS) if pd.notna(v) else "unknown")
    df["count_bucket"] = df["surfer_count"].apply(lambda v: bucket(v, COUNT_BUCKETS) if pd.notna(v) else "unknown")

    months_present = sorted(df["month"].dropna().unique())
    all_calendar_months = set(range(1, 13))
    missing_months = sorted(all_calendar_months - set(months_present))
    if missing_months:
        print(f"\nNOTE: months {missing_months} have ZERO source footage in the whole corpus -- "
              f"this can't be fixed by relabeling, only by capturing new footage those months.")

    # Composite bucket key: (month, tide_bucket, count_bucket, weather_simple).
    df["bucket_key"] = list(zip(df["month"], df["tide_bucket"], df["count_bucket"], df["weather_simple"]))
    bucket_counts = Counter(df["bucket_key"])

    # Sample inversely to frequency: within each bucket, keep at most a small
    # quota, prioritizing rows from the rarest buckets overall until n_target
    # is reached. Oversample explicitly-thin buckets (high count, tide tails,
    # RAIN/FOG) by giving them a higher per-bucket quota.
    def bucket_quota(key):
        _, tide_b, count_b, weather = key
        quota = 1
        if count_b in ("30-39", "40+"):
            quota += 2
        if tide_b in ("<0ft", ">6ft"):
            quota += 1
        if weather in ("RAIN", "FOG"):
            quota += 2
        return quota

    picked = []
    picked_per_bucket = Counter()
    # Iterate rarest buckets first so thin buckets get first pick of their (few) rows.
    df_sorted = df.assign(_bucket_freq=df["bucket_key"].map(bucket_counts)).sort_values("_bucket_freq")
    for _, row in df_sorted.iterrows():
        if len(picked) >= n_target:
            break
        key = row["bucket_key"]
        if picked_per_bucket[key] >= bucket_quota(key):
            continue
        fname = row["filename"]
        if not (CROPS_DIR / fname).exists():
            continue
        picked.append(row)
        picked_per_bucket[key] += 1

    print(f"\n=== Gap-fill tier: picked {len(picked)} of {n_target} target ===")
    return picked, bucket_counts


def print_bucket_coverage(label, rows, dim_name, dim_key):
    counts = Counter(r[dim_key] if isinstance(r, dict) else r[dim_key] for r in rows)
    print(f"  {label} by {dim_name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=str)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-n", type=int, default=57,
                   help="Total candidates to target across both tiers combined (default: 57, "
                        "roughly matching the existing labeled set's size)")
    p.add_argument("--gap-fill-n", type=int, default=None,
                   help="Gap-fill tier target (default: target-n minus however many error-tier "
                        "candidates are found)")
    p.add_argument("--error-tier-n", type=int, default=15,
                   help="Cap on error-tier candidates, ranked by count-weighted score "
                        "(default: 15 -- without a cap, every usable spot-check row would be "
                        "included regardless of how small the miss is)")
    args = p.parse_args()

    already_labeled = existing_cvat_filenames()
    print(f"Existing CVAT export has {len(already_labeled)} labeled images.")

    error_rows = load_error_tier(already_labeled, args.error_tier_n)
    gap_fill_n = args.gap_fill_n if args.gap_fill_n is not None else max(args.target_n - len(error_rows), 0)
    error_filenames = {r["filename"] for r in error_rows}
    gap_rows_raw, bucket_counts = load_gap_fill_tier(gap_fill_n, error_filenames, already_labeled)

    gap_rows = []
    for row in gap_rows_raw:
        gap_rows.append(dict(
            filename=row["filename"], source_tier="gap-fill", model_count=row.get("surfer_count"),
            my_count="", pct_diff="", count_weighted_score="",
            month=row.get("month"), tide_ft=row.get("tide_ft"), tide_bucket=row.get("tide_bucket"),
            count_bucket=row.get("count_bucket"), weather_simple=row.get("weather_simple"),
            hour_local=row.get("hour_local"),
        ))
    # Fill in month/tide/etc. for error-tier rows too, joining back to training_features.csv.
    tf = pd.read_csv(TRAINING_FEATURES_CSV).set_index("filename")
    for r in error_rows:
        meta = tf.loc[r["filename"]] if r["filename"] in tf.index else None
        r["month"] = meta["month"] if meta is not None else ""
        r["tide_ft"] = meta["tide_ft"] if meta is not None else ""
        r["tide_bucket"] = bucket(meta["tide_ft"], TIDE_BUCKETS) if meta is not None and pd.notna(meta["tide_ft"]) else ""
        r["count_bucket"] = bucket(meta["surfer_count"], COUNT_BUCKETS) if meta is not None and pd.notna(meta["surfer_count"]) else ""
        r["weather_simple"] = meta["weather_simple"] if meta is not None else ""
        r["hour_local"] = meta["hour_local"] if meta is not None else ""

    all_rows = error_rows + gap_rows
    cols = ["filename", "source_tier", "model_count", "my_count", "pct_diff", "count_weighted_score",
            "month", "tide_ft", "tide_bucket", "count_bucket", "weather_simple", "hour_local"]
    out_df = pd.DataFrame(all_rows)[cols]

    print(f"\n=== Combined candidate list: {len(out_df)} images ({len(error_rows)} error + {len(gap_rows)} gap-fill) ===")
    print_bucket_coverage("Candidates", all_rows, "month", "month")
    print_bucket_coverage("Candidates", all_rows, "count_bucket", "count_bucket")
    print_bucket_coverage("Candidates", all_rows, "tide_bucket", "tide_bucket")
    print_bucket_coverage("Candidates", all_rows, "weather", "weather_simple")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} candidates to {OUT_CSV}")
    print("Review and prune this list before uploading the images to CVAT for labeling.")


if __name__ == "__main__":
    main()
