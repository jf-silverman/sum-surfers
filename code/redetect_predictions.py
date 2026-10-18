"""
Recount existing `predictions.csv` rows in place, after a detection change.

Why this and not `DETECT_MODE=all`. That mode only ever ADDS rows -- `main()`
drops any crop already listed in `predictions.csv` -- so it cannot revise
history. Clearing the file first would work and would also throw away the
hand-entered columns the file accumulates (`human_count`, `frame_count_1..3`,
`frame_count_mean`, `frame_count_stdev`), which exist nowhere else.

So this rewrites exactly one column, `surfer_count`, plus `confidence_avg`,
leaving every other field and the row order untouched. The quality gate is not
re-evaluated: rows whose `quality_ok` is not True never ran detection and keep
their empty count, because re-deciding the gate here would silently mix a
second change into the one being made.

Always writes a timestamped backup beside the file before touching it.

Usage:
    python code/redetect_predictions.py --dry-run       # count what would change
    python code/redetect_predictions.py --limit 25      # small real pass
    python code/redetect_predictions.py
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
    args = ap.parse_args()

    with PREDS.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    print(f"{len(rows)} rows in {PREDS.name}")

    todo = [r for r in rows
            if str(r.get("quality_ok", "")).strip() == "True"
            and str(r.get("surfer_count", "")).strip() != ""
            and (CROPS / r["filename"]).exists()]
    skipped_gate = sum(1 for r in rows if str(r.get("quality_ok", "")).strip() != "True")
    missing = sum(1 for r in rows
                  if str(r.get("quality_ok", "")).strip() == "True"
                  and not (CROPS / r["filename"]).exists())
    print(f"  {len(todo)} to recount; {skipped_gate} quality-gated (left alone); "
          f"{missing} with no crop on disk (left alone)")
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
        boxes = ds.run_inference_with_boxes(model, CROPS / r["filename"])
        new = len(boxes)
        conf = round(sum(b[4] for b in boxes) / new, 4) if new else ""
        old = int(float(r["surfer_count"]))
        if new != old:
            changed += 1
            delta += new - old
        tgt = index[r["filename"]]
        tgt["surfer_count"] = str(new)
        tgt["confidence_avg"] = str(conf)
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
