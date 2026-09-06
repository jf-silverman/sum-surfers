"""
prepare_cvat_reimport.py
--------------------------
Part of the training-data-expansion plan (docs/PROJECT_HISTORY.md's
2026-09-05 entries; see /Users/jfs-m3/.claude/plans/dazzling-rolling-lemon.md).

The original CVAT task for the existing 57 labeled images no longer exists
(deleted after export), so adding the new "posture" attribute requires
re-importing them as a fresh CVAT task. This script merges the current
data/cvat_out_coco/splits/{train,val,test} (a clean, non-overlapping
partition of the original 57 images/1451 annotations -- verified: image
ids 1-57 and annotation ids 1-1451 with zero collisions across splits)
back into a single flat COCO 1.0 export, in the exact directory layout
CVAT itself uses for COCO export/import (annotations/instances_default.json
+ images/default/*.jpg) -- matching archive/data/cvat_out_dataset_2025_09_24_coco 1.0.zip's
own layout, confirmed compatible.

The train/val/test split doesn't matter for this step -- Phase 4 of the
plan (build_stratified_splits.py) rebuilds fresh stratified splits from
the combined old+new pool afterward anyway, so there's no reason to
preserve today's split when re-importing for attribute-tagging.

Output: a zip Joel can upload directly via CVAT's Task > Actions >
"Upload annotations" (if creating one task per image set) or, more simply,
create a new CVAT task, add the images, then use "Upload annotations" ->
"COCO 1.0" pointing at this zip to bring in the existing boxes -- OR
create the task directly from this zip's images and import the
annotations.json separately. See the script's printed instructions.

Usage:
    python code/prepare_cvat_reimport.py
"""
import json
import shutil
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = _PROJECT_ROOT / "data" / "cvat_out_coco" / "splits"
OUT_DIR = _PROJECT_ROOT / "data" / "cvat_out_coco" / "reimport_existing_57"
OUT_ZIP = _PROJECT_ROOT / "data" / "cvat_out_coco" / "reimport_existing_57.zip"

SPLITS = ["train", "val", "test"]


def main():
    merged = None
    seen_image_ids, seen_ann_ids = set(), set()

    images_out_dir = OUT_DIR / "images" / "default"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        json_path = SPLITS_DIR / f"instances_{split}.json"
        with open(json_path) as f:
            coco = json.load(f)

        if merged is None:
            merged = {k: coco[k] for k in ["licenses", "info", "categories"] if k in coco}
            merged["images"] = []
            merged["annotations"] = []
        else:
            if coco["categories"] != merged["categories"]:
                raise ValueError(f"{split}'s categories differ from train's -- inconsistent export, fix before merging")

        for img in coco["images"]:
            if img["id"] in seen_image_ids:
                raise ValueError(f"Duplicate image id {img['id']} in {split} -- splits are supposed to be a clean partition")
            seen_image_ids.add(img["id"])
            merged["images"].append(img)
            src = SPLITS_DIR / split / img["file_name"]
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")
            shutil.copy2(src, images_out_dir / img["file_name"])

        for ann in coco["annotations"]:
            if ann["id"] in seen_ann_ids:
                raise ValueError(f"Duplicate annotation id {ann['id']} in {split}")
            seen_ann_ids.add(ann["id"])
            merged["annotations"].append(ann)

        print(f"  merged {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    ann_dir = OUT_DIR / "annotations"
    ann_dir.mkdir(exist_ok=True)
    with open(ann_dir / "instances_default.json", "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged total: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    print(f"Wrote {OUT_DIR}")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT_DIR))
    print(f"Zipped to {OUT_ZIP}")

    print(f"""
=== Next steps in CVAT ===
1. Create a new CVAT task (e.g. "sum-surfers-relabel-2026-09") and upload
   the {len(merged['images'])} images from {images_out_dir} as its data.
2. In the task, add a label attribute to the existing "Surfer" label:
   name "posture", type "select" (radio/dropdown), values:
   standing, sitting, prone, unknown.
3. Import the existing boxes: Task menu -> Actions -> Upload annotations
   -> format "COCO 1.0" -> select {OUT_ZIP.name}. This brings in all
   {len(merged['annotations'])} existing boxes without redrawing them.
4. Go through each image and set the posture attribute on each box
   (boxes already exist -- this is tagging, not drawing).
5. When done, export the task: Actions -> Export task dataset -> COCO 1.0
   -> download the zip. That export is what build_stratified_splits.py
   (Phase 4) will merge with the new-image labeling pass.
""")


if __name__ == "__main__":
    main()
