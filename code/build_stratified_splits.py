"""
build_stratified_splits.py
--------------------------
Phase 4 of the training-data-expansion plan: merges every labeled image —
the existing 57 plus any new CVAT export — into one pool and builds fresh
train/val/test splits stratified by how crowded each frame is.

Why rebuild rather than append
------------------------------
The current split was never stratified by anything, and it shows: all 23
frames above 30 surfers are in train, val tops out at 35, and test tops out
at 24. That means the detector's behaviour on crowded frames — the one
failure mode actually worth measuring, where 2 of 9 human-checked frames
were off by -27 — cannot be measured at all. Appending new images into
CVAT's own split assignment would leave that intact.

Crowding is the stratification axis
-----------------------------------
Crowding comes from the labels themselves (boxes per image), not from
`training_features.csv`. That matters: the original 57 images are
pre-pipeline manual captures from Jul-Aug 2025 and have no row in the
feature table at all, so any scheme keyed on that table could not stratify
them. Box count is ground truth and exists for every labeled image by
definition. Month is reported as a secondary axis but not balanced on — with
~60-120 images, balancing two axes at once leaves buckets of one.

The crowded-frame tension, stated plainly
-----------------------------------------
Crowded frames are scarce (23 of 57) and are wanted in two places at once:
in test, so the failure is measurable, and in train, so the detector has
something to learn it from. `--test-crowded-min` sets that trade explicitly
rather than letting a ratio decide it silently. At the measured ~22%
catastrophic rate, 5 crowded test frames give a ~72% chance of containing at
least one, and 12 give ~95% — but 12 is more than half of every crowded
frame labeled. The default is deliberately modest: a tripwire that catches a
regression, not an estimate of how often it happens. That estimate should
come from the human-count review sets, which are cheaper per frame than
drawing boxes.

Nothing here touches `splits/`; output goes to a new directory, so the
production training path keeps working until a retrain is actually adopted.

Usage:
    python code/build_stratified_splits.py
    python code/build_stratified_splits.py --input-dir data/cvat_out_coco/splits \
                                           --input-dir data/cvat_out_coco/new_export
    python code/build_stratified_splits.py --test-crowded-min 8 --seed 7
    python code/build_stratified_splits.py --force-test crop2026-08-03_10-02-00.jpg
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits_v2"

SPLITS = ["train", "val", "test"]
DEFAULT_RATIOS = {"train": 0.56, "val": 0.26, "test": 0.18}  # matches the current 32/15/10

# Same buckets select_labeling_candidates.py uses, so the two scripts' tables
# can be read against each other.
COUNT_BUCKETS = [(0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"),
                 (30, 40, "30-39"), (40, 10 ** 6, "40+")]
CROWDED_FROM = 30

# The two known catastrophic failures (30 counted as 3; 46 counted as 19).
# Defaulted into test as a permanent regression check on the specific failure
# that motivated this work. They are only labeled if a new export includes
# them, so a miss here is a note, not an error.
DEFAULT_FORCE_TEST = [
    "crop2026-08-03_10-02-00.jpg",
    "crop2026-08-09_07-29-00.jpg",
]

DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),      # crop2026-08-03_10-02-00.jpg
    re.compile(r"(\d{4})(\d{2})(\d{2})"),        # jacks_20250810_0859.jpg / 20250719_1345.jpg
]


def bucket_of(n):
    for lo, hi, label in COUNT_BUCKETS:
        if lo <= n < hi:
            return label
    return COUNT_BUCKETS[-1][2]


def month_of(file_name):
    for pat in DATE_PATTERNS:
        m = pat.search(file_name)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return "unknown"


def find_json_files(input_dir):
    """instances_*.json in the directory, or in an annotations/ subdir.

    Both layouts exist in practice: this repo's splits/ keeps them at the top
    level, while a straight CVAT COCO export nests them under annotations/.
    """
    files = sorted(input_dir.glob("instances_*.json"))
    if not files:
        files = sorted((input_dir / "annotations").glob("instances_*.json"))
    return files


def find_image(input_dir, file_name):
    """Locate an image under an input dir, across the layouts CVAT produces."""
    for candidate in (input_dir / file_name,
                      input_dir / "images" / file_name):
        if candidate.exists():
            return candidate
    for split in SPLITS + ["default"]:
        candidate = input_dir / split / file_name
        if candidate.exists():
            return candidate
    hits = list(input_dir.rglob(file_name))
    return hits[0] if hits else None


def load_pool(input_dirs):
    """Every labeled image across every input, with its boxes. Keyed by filename."""
    pool = {}
    categories = None
    for input_dir in input_dirs:
        json_files = find_json_files(input_dir)
        if not json_files:
            sys.exit(f"No instances_*.json found under {input_dir}")
        for jf in json_files:
            data = json.loads(jf.read_text())
            if categories is None:
                categories = data["categories"]
            elif data["categories"] != categories:
                # Mismatched label sets would silently corrupt category ids in
                # the merged output; a broken export should be fixed, not merged.
                sys.exit(f"categories in {jf} differ from the first export — fix the export first")

            by_image = defaultdict(list)
            for ann in data["annotations"]:
                by_image[ann["image_id"]].append(ann)

            for img in data["images"]:
                name = img["file_name"]
                if name in pool:
                    sys.exit(f"{name} appears in more than one input export — "
                             f"de-duplicate before merging")
                src = find_image(input_dir, name)
                if src is None:
                    print(f"  WARNING: no image file found for {name} under {input_dir}, skipping")
                    continue
                pool[name] = {
                    "image": img,
                    "annotations": by_image.get(img["id"], []),
                    "src_path": src,
                    "n_boxes": len(by_image.get(img["id"], [])),
                    "origin": f"{input_dir.name}/{jf.stem.replace('instances_', '')}",
                }
    for name, rec in pool.items():
        rec["bucket"] = bucket_of(rec["n_boxes"])
        rec["month"] = month_of(name)
    return pool, categories


def allocate(pool, ratios, test_crowded_min, force_test, rng):
    """Assign each image to a split, stratified within each crowding bucket."""
    assignment = {}

    forced = [f for f in force_test if f in pool]
    for f in force_test:
        if f not in pool:
            print(f"  NOTE: --force-test {f} is not in the labeled pool "
                  f"(not labeled yet?) — ignoring")
    for f in forced:
        assignment[f] = "test"

    crowded = [n for n, r in pool.items() if r["n_boxes"] >= CROWDED_FROM]
    crowded_forced = [n for n in forced if n in crowded]
    need = max(0, test_crowded_min - len(crowded_forced))
    if test_crowded_min > len(crowded) / 2:
        print(f"  WARNING: --test-crowded-min {test_crowded_min} takes more than half of the "
              f"{len(crowded)} crowded frames available.\n"
              f"           That buys measurability at the cost of the training signal for the "
              f"very regime being measured.")
    pick_from = [n for n in crowded if n not in assignment]
    rng.shuffle(pick_from)
    for name in pick_from[:need]:
        assignment[name] = "test"
    got = sum(1 for n, s in assignment.items() if s == "test" and n in crowded)
    if got < test_crowded_min:
        print(f"  WARNING: only {got} crowded frames available for test "
              f"(asked for {test_crowded_min}) — the pool does not contain enough")

    # Everything not already pinned, allocated proportionally inside its bucket.
    by_bucket = defaultdict(list)
    for name, rec in pool.items():
        if name not in assignment:
            by_bucket[rec["bucket"]].append(name)

    crowded_buckets = {label for lo, _, label in COUNT_BUCKETS if lo >= CROWDED_FROM}

    for bucket in sorted(by_bucket):
        names = sorted(by_bucket[bucket])
        rng.shuffle(names)
        n = len(names)
        # Crowded buckets get train/val only. The exact number of crowded
        # frames in test is set by --test-crowded-min and already pinned
        # above; letting the proportional split top that up would quietly
        # overrule the flag — asking for 4 produced 7 before this. Every
        # crowded frame not needed for the tripwire is worth more as training
        # signal, since it is the scarcest regime in the set.
        targets = SPLITS if bucket not in crowded_buckets else ["train", "val"]
        weights = {s: ratios[s] for s in targets}
        scale = sum(weights.values())
        # Largest-remainder allocation, so small buckets are not all rounded
        # to zero and the parts always sum back to n.
        exact = {s: n * weights[s] / scale for s in targets}
        base = {s: int(exact[s]) for s in targets}
        left = n - sum(base.values())
        for s in sorted(targets, key=lambda s: exact[s] - base[s], reverse=True)[:left]:
            base[s] += 1
        i = 0
        for s in targets:
            for name in names[i:i + base[s]]:
                assignment[name] = s
            i += base[s]
    return assignment


def write_splits(pool, assignment, categories, out_dir):
    """Write COCO jsons with fresh contiguous ids, and copy the images."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    written = {}
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        names = sorted(n for n, s in assignment.items() if s == split)
        images, annotations = [], []
        for img_id, name in enumerate(names, start=1):
            rec = pool[name]
            img = dict(rec["image"])
            img["id"] = img_id
            img["ann_count"] = rec["n_boxes"]
            images.append(img)
            for ann in rec["annotations"]:
                a = dict(ann)
                a["id"] = len(annotations) + 1
                a["image_id"] = img_id
                annotations.append(a)
            shutil.copy2(rec["src_path"], out_dir / split / name)
        (out_dir / f"instances_{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations,
                        "categories": categories}, indent=1) + "\n"
        )
        written[split] = (len(images), len(annotations))
    return written


def print_table(title, pool, assignment, key):
    print(f"\n{title}")
    keys = sorted({pool[n][key] for n in pool},
                  key=lambda k: [b[2] for b in COUNT_BUCKETS].index(k) if key == "bucket" else k)
    header = f"  {key:<10} {'total':>6}" + "".join(f"{s:>7}" for s in SPLITS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in keys:
        names = [n for n in pool if pool[n][key] == k]
        counts = Counter(assignment[n] for n in names)
        print(f"  {k:<10} {len(names):>6}" + "".join(f"{counts.get(s, 0):>7}" for s in SPLITS))
    total = Counter(assignment.values())
    print(f"  {'TOTAL':<10} {len(pool):>6}" + "".join(f"{total.get(s, 0):>7}" for s in SPLITS))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, action="append",
                   help=f"COCO export directory; repeatable (default: {DEFAULT_INPUT})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--test-crowded-min", type=int, default=5,
                   help="Frames with >=30 boxes to place in test (default 5). "
                        "At the measured ~22%% catastrophic rate: 5 gives ~72%% odds of "
                        "catching one, 12 gives ~95%%.")
    p.add_argument("--force-test", action="append", default=None,
                   help=f"Filename pinned to test; repeatable. Default: {DEFAULT_FORCE_TEST}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Report the split without writing anything")
    return p.parse_args()


def main():
    args = parse_args()
    input_dirs = args.input_dir or [DEFAULT_INPUT]
    force_test = DEFAULT_FORCE_TEST if args.force_test is None else args.force_test

    print(f"Inputs: {', '.join(str(d) for d in input_dirs)}")
    pool, categories = load_pool(input_dirs)
    total_boxes = sum(r["n_boxes"] for r in pool.values())
    print(f"Pool: {len(pool)} labeled images, {total_boxes} boxes, "
          f"classes {[c['name'] for c in categories]}")

    print("\nWhere they come from now:")
    for origin, n in sorted(Counter(r["origin"] for r in pool.values()).items()):
        print(f"  {origin:<24} {n:>4}")

    assignment = allocate(pool, DEFAULT_RATIOS, args.test_crowded_min,
                          force_test, random.Random(args.seed))

    print_table("Crowding (boxes per image) x split — the stratified axis:", pool, assignment, "bucket")
    print_table("Month x split — reported, not balanced on:", pool, assignment, "month")

    crowded_test = sorted(n for n, s in assignment.items()
                          if s == "test" and pool[n]["n_boxes"] >= CROWDED_FROM)
    print(f"\nCrowded (>={CROWDED_FROM} boxes) frames in test: {len(crowded_test)}")
    for n in crowded_test:
        print(f"  {n}  ({pool[n]['n_boxes']} boxes)")
    crowded_train = sum(1 for n, s in assignment.items()
                        if s == "train" and pool[n]["n_boxes"] >= CROWDED_FROM)
    print(f"Crowded frames left in train: {crowded_train}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    written = write_splits(pool, assignment, categories, args.out_dir)
    print(f"\nWrote {args.out_dir}")
    out_images = sum(i for i, _ in written.values())
    out_boxes = sum(a for _, a in written.values())
    for split in SPLITS:
        i, a = written[split]
        print(f"  {split:<6} {i:>4} images, {a:>5} boxes")
    # Nothing may be dropped silently in a merge-and-reindex step.
    if out_images != len(pool) or out_boxes != total_boxes:
        sys.exit(f"CONSERVATION CHECK FAILED: in {len(pool)} images/{total_boxes} boxes, "
                 f"out {out_images}/{out_boxes}")
    print(f"  conservation check OK ({out_images} images, {out_boxes} boxes, none dropped)")
    print(f"\n{DEFAULT_INPUT} is untouched. To train on this instead:\n"
          f"  python code/train_model.py --cvat-coco-dir {args.out_dir} --run-name <date>_stratified")


if __name__ == "__main__":
    main()
