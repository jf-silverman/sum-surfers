"""
Cut a 3:2 close-up of one surfer from a labeled frame, for the repo header.

The source is a 1280x180 strip of distant ocean, so "close-up" means enlarging a
very small box: the largest standing-pose surfer in the labeled set is 32x40
pixels. This script picks the crop window around a labeled box, keeps it 3:2,
and enlarges with Lanczos, which is the best of the bad options — no amount of
interpolation invents detail that was never captured.

CONTEXT_MULT controls how much water surrounds the surfer. Tighter is more
dramatic but softer, since the same pixels are stretched further.

Usage:
    python code/make_hero_crop.py --image data/j_shore_cam/surf_crops/crop2025-11-16_12-08-00.jpg \
        --box 592 109 32 40 --out /tmp/hero.png
"""

import argparse
from pathlib import Path

import cv2

ASPECT_W, ASPECT_H = 3, 2
CONTEXT_MULT = 2.6      # crop height as a multiple of the box height
OUT_WIDTH = 1200


def hero_crop(img, box, context_mult=CONTEXT_MULT, out_width=OUT_WIDTH,
              aspect=(ASPECT_W, ASPECT_H), subject_x=0.5, subject_y=0.5):
    """3:2 crop around `box` (x, y, w, h), enlarged to `out_width`.

    `subject_x` / `subject_y` place the subject within the frame: 0.5 centers it,
    and a larger value pushes it right or down, which leaves more open water on
    the other side. Useful when the subject is moving across the frame and
    should have room in front of, or behind, it.
    """
    ih, iw = img.shape[:2]
    bx, by, bw, bh = box
    cx, cy = bx + bw / 2, by + bh / 2

    crop_h = max(bh * context_mult, 24)
    crop_w = crop_h * aspect[0] / aspect[1]

    # A 180px-tall strip usually cannot give the full height a wide crop wants,
    # so clamp to the frame and re-derive the other side to hold the ratio.
    crop_h = min(crop_h, ih)
    crop_w = min(crop_w, iw)
    crop_h = min(crop_h, crop_w * aspect[1] / aspect[0])
    crop_w = crop_h * aspect[0] / aspect[1]

    x0 = int(round(min(max(cx - crop_w * subject_x, 0), iw - crop_w)))
    y0 = int(round(min(max(cy - crop_h * subject_y, 0), ih - crop_h)))
    patch = img[y0:y0 + int(round(crop_h)), x0:x0 + int(round(crop_w))]

    out_h = int(round(out_width * aspect[1] / aspect[0]))
    return cv2.resize(patch, (out_width, out_h), interpolation=cv2.INTER_LANCZOS4)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--box", required=True, nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--context", type=float, default=CONTEXT_MULT)
    p.add_argument("--width", type=int, default=OUT_WIDTH)
    p.add_argument("--subject-x", type=float, default=0.5,
                   help="subject's horizontal place in frame; >0.5 leaves room on the left")
    p.add_argument("--subject-y", type=float, default=0.5)
    args = p.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    out = hero_crop(img, args.box, args.context, args.width,
                    subject_x=args.subject_x, subject_y=args.subject_y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)
    print(f"Wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
