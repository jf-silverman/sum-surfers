"""
Recount existing `predictions.csv` rows in place, after a detection change.

Why this and not `DETECT_MODE=all`. That mode only ever ADDS rows -- `main()`
drops any crop already listed in `predictions.csv` -- so it cannot revise
history. Clearing the file first would work and would also throw away
`human_count`, the hand-entered column, which exists nowhere else.

So this rewrites only what detection derives -- `surfer_count`,
`confidence_avg`, `frame_count_1..3`, `frame_count_mean`, `frame_count_stdev`
-- and leaves `human_count`, the quality columns and the row order untouched.
The quality gate is not re-evaluated: rows whose `quality_ok` is not True never
ran detection and keep their empty count, because re-deciding the gate here
would silently mix a second change into the one being made.

It MUST recount through `run_inference_multi`, not `run_inference`. A production
row's `surfer_count` is `round(mean(...))` over three frames -- the primary crop
plus the two side frames either side of it -- and that mean carries a per-frame
stdev averaging 0.69. Recounting the primary frame alone therefore moves rows by
+-1 in both directions purely from frame sampling, which swamps any detection
change being measured: a first attempt at this did exactly that and moved 174 of
382 rows, 71 of them UPWARD, for a change that can only ever subtract.

Always writes a timestamped backup beside the file before touching it.

`--since` exists because of model eras. `predictions.csv` spans two sets of
weights: rows before 2026-09-21 were counted with the October 2025 model, rows
from 2026-09-21 on with `20260921_fog`. Recounting the whole file would apply
today's weights to a year of history, which changes counts in BOTH directions
and silently rewrites the series every calibration and hand review was measured
against. `--since 2026-09-21` recounts only the rows whose weights already match
the loaded model, so the only thing that moves is the detection change being
applied -- and the discontinuity lands on a seam that already exists.

Usage:
    python code/redetect_predictions.py --dry-run                   # count what would change
    python code/redetect_predictions.py --since 2026-09-21 --limit 25
    python code/redetect_predictions.py --since 2026-09-21
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

PREDS = _ROOT / "data" / "predictions" / "predictions.csv"
CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--since", default="",
                    help="only recount rows with date >= this YYYY-MM-DD "
                         "(use the model-era boundary; see the module docstring)")
    args = ap.parse_args()

    with PREDS.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    print(f"{len(rows)} rows in {PREDS.name}")

    todo = [r for r in rows
            if str(r.get("quality_ok", "")).strip() == "True"
            and str(r.get("surfer_count", "")).strip() != ""
            and (CROPS / r["filename"]).exists()
            and r["date"] >= args.since]
    skipped_gate = sum(1 for r in rows if str(r.get("quality_ok", "")).strip() != "True")
    missing = sum(1 for r in rows
                  if str(r.get("quality_ok", "")).strip() == "True"
                  and not (CROPS / r["filename"]).exists())
    print(f"  {len(todo)} to recount; {skipped_gate} quality-gated (left alone); "
          f"{missing} with no crop on disk (left alone)")
    if args.since:
        print(f"  --since {args.since}: earlier rows left alone (different model era)")
    if args.limit:
        todo = todo[:args.limit]
        print(f"  --limit: {len(todo)} this pass")
    if args.dry_run:
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PREDS.with_name(f"predictions.backup_{stamp}.csv")
    shutil.copy2(PREDS, backup)
    print(f"  backup -> {backup}")

    model = ds.load_model()
    index = {r["filename"]: r for r in rows}
    changed, delta, done = 0, 0, 0
    for r in todo:
        res = ds.run_inference_multi(model, CROPS / r["filename"])
        new = res["surfer_count"]
        old = int(float(r["surfer_count"]))
        if new != old:
            changed += 1
            delta += new - old
        tgt = index[r["filename"]]
        tgt["surfer_count"] = str(new)
        tgt["confidence_avg"] = str(res["confidence_avg"])
        for i in (1, 2, 3):
            v = res[f"frame_count_{i}"]
            tgt[f"frame_count_{i}"] = "" if v is None else str(v)
        tgt["frame_count_mean"] = str(res["frame_count_mean"])
        tgt["frame_count_stdev"] = str(res["frame_count_stdev"])
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(todo)}  changed {changed}  net {delta:+d}", flush=True)

    tmp = PREDS.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(PREDS)
    print(f"\n{done} rows recounted; {changed} changed; net {delta:+d} surfers "
          f"({delta / done:+.3f}/frame)")
    print(f"  backup kept at {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
