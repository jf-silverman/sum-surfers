"""
validate_against_human_counts.py
-----------------------------------
Answers the question the frame-timing-variability study left open:
does averaging more model frames spread over a wider span actually get
CLOSER to the real count (accuracy), or does it just reduce the model's
own variance (precision) without necessarily being more correct?

Uses Joel's real human counts (data/reviews/count_60sec_var/review_counts.csv,
70 usable points across 7 clips -- set1/05_50 excluded, "lens condensation"
throughout, unusable) as ground truth, matched against the model's real
per-second detection counts for the same clips/seconds
(analysis/frame_timing_variability/frame_variability_analysis.csv, every
second 0-62 already computed).

For each human-labeled second, computes error (model - human) for several
real candidate frame-selection strategies, using only real per-second model
counts already on record (no re-running detection, no estimates):
  - Single frame at that exact second (closest to today's ~1 unaveraged frame).
  - k-frame windowed average centered on that second, at a few (k, span)
    combinations spanning the range explored in the original variability
    study (tight/bunched vs wide-spread).

Reports real MAE and bias (mean signed error) per strategy, pooled across
all 7 usable clips -- the actual accuracy comparison, not just stdev.

Usage:
    python analysis/frame_timing_variability/validate_against_human_counts.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_CSV = HERE / "frame_variability_analysis.csv"
HUMAN_CSV = HERE.parent.parent / "data" / "reviews" / "count_60sec_var" / "review_counts.csv"

# (k, half-span in seconds) -- half-span 0 means single frame. Mirrors the
# tight-vs-spread comparison from the original variability study.
STRATEGIES = [
    ("single frame (this second only)", 1, 0),
    ("k=3, tight (~+/-1s, ~2s span)", 3, 1),
    ("k=3, spread (~+/-15s, 30s span)", 3, 15),
    ("k=5, spread (~+/-15s, 30s span)", 5, 15),
    ("k=5, wide (~+/-29s, ~58s span)", 5, 29),
    ("k=10, wide (~+/-29s, ~58s span)", 10, 29),
]


def windowed_mean(model_series, center_second, k, half_span):
    """Picks k seconds evenly spread across [center-half_span, center+half_span]
    (clamped to the clip's real 0-62 range) and averages their real recorded
    model counts. Falls back to whatever seconds actually exist if a requested
    second wasn't recorded (all seconds 0-62 exist in the source CSV here, so
    this is just a safety net, not expected to trigger)."""
    if k == 1:
        wanted = [center_second]
    else:
        lo = max(0, center_second - half_span)
        hi = min(62, center_second + half_span)
        wanted = [int(round(x)) for x in np.linspace(lo, hi, k)]
    vals = [model_series.get(s) for s in wanted if s in model_series.index]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    model = pd.read_csv(MODEL_CSV)
    model = model[model["quality_ok"]]
    human = pd.read_csv(HUMAN_CSV)

    # Only rows with a real numeric human count.
    def to_int(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return None
    human["human_count"] = human["human_count"].apply(to_int)
    human = human.dropna(subset=["human_count"])

    results = {name: [] for name, _, _ in STRATEGIES}
    per_clip_rows = []

    for clip_time, clip_human in human.groupby("clip_time"):
        clip_model = model[model["clip_time"] == clip_time].set_index("second")["count"]
        if clip_model.empty:
            print(f"WARNING: no model data for clip_time={clip_time}, skipping")
            continue
        for _, row in clip_human.iterrows():
            sec = int(row["second"])
            h = row["human_count"]
            for name, k, half_span in STRATEGIES:
                pred = windowed_mean(clip_model, sec, k, half_span)
                if pred is None:
                    continue
                err = pred - h
                results[name].append(err)
                per_clip_rows.append(dict(clip_time=clip_time, second=sec, human=h,
                                           strategy=name, pred=pred, err=err))

    print(f"Usable human-labeled points: {len(human)} across {human['clip_time'].nunique()} clips "
          f"({sorted(human['clip_time'].unique())})\n")

    print(f"{'Strategy':38s} {'n':>4s} {'MAE':>8s} {'Bias (mean err)':>16s} {'RMSE':>8s}")
    for name, _, _ in STRATEGIES:
        errs = np.array(results[name])
        mae = np.mean(np.abs(errs))
        bias = np.mean(errs)
        rmse = np.sqrt(np.mean(errs ** 2))
        print(f"{name:38s} {len(errs):4d} {mae:8.2f} {bias:16.2f} {rmse:8.2f}")

    out_csv = HERE / "human_validation_results.csv"
    pd.DataFrame(per_clip_rows).to_csv(out_csv, index=False)
    print(f"\nWrote per-point results to {out_csv}")


if __name__ == "__main__":
    main()
