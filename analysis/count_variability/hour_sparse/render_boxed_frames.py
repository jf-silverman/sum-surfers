"""
Re-renders the 45 probe frames with detection boxes drawn, naming each file by
its count so the set can be flipped through and eyeballed in order.

Joel's observation from the hour_sparse chart: under the dusk trend there looks
to be an oscillation on the 4-minute scale, possibly the swell arriving in sets
-- surfers visible in a lull, hidden while a set washes through. These images
are for checking that by eye, which is the only way to tell "hidden behind
whitewater" from "genuinely not there".

Output: frames_boxed/<HHMM>_f<n>_count_<NN>.jpg, sorted chronologically by name.
Both directories are gitignored; hour_probe.csv holds the measurements.

Usage:
    python analysis/count_variability/hour_sparse/render_boxed_frames.py
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

SRC = HERE / "frames"
OUT = HERE / "frames_boxed"
BOX_BGR = (90, 227, 157)      # the project's lime, in BGR
BOX_ALPHA = 0.9


def main():
    frames = sorted(SRC.glob("*.jpg"))
    if not frames:
        print(f"No frames in {SRC} — run probe_hour.py first.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    model = ds.load_model()

    for path in frames:
        m = re.match(r"(\d{4})_f(\d)\.jpg$", path.name)
        if not m:
            continue
        hhmm, fno = m.group(1), m.group(2)
        img = cv2.imread(str(path))
        boxes = ds.run_inference_with_boxes(model, path)
        _ok, _reason, _bright, lap = ds.compute_image_quality(path)

        overlay = img.copy()
        for b in boxes:
            cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          BOX_BGR, 1)
        img = cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0)

        # Caption band under the strip: the count is in the filename too, but a
        # burnt-in label survives being dragged into a slideshow or a message.
        band = np.zeros((30, img.shape[1], 3), np.uint8)
        cv2.putText(band, f"{hhmm[:2]}:{hhmm[2:]}  frame {fno}/3   "
                          f"{len(boxes)} detected   lap_var {lap:.0f}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1,
                    cv2.LINE_AA)
        out_path = OUT / f"{hhmm}_f{fno}_count_{len(boxes):02d}.jpg"
        cv2.imwrite(str(out_path), np.vstack([np.vstack([band, img])]))
        print(f"  {out_path.name}")

    print(f"\nWrote {len(list(OUT.glob('*.jpg')))} boxed frame(s) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
