"""
make_fog_gate_review.py
-----------------------
Builds a review set for frames sitting near the FOG GATE, where the quality
gate's decision is least certain and where a wrong call costs a whole hour.

The gate (detect_surfers.compute_image_quality) rejects a frame as
`foggy_or_blurred` when Laplacian variance falls below QUALITY_LAPVAR_THRESH
(12.7). That threshold decides whether an hour gets counted at all, so it has
two failure directions and this script renders both:

  near_pass/   The frame PASSED, sits just above the threshold, is among the
               day's first images, and was counted as ZERO surfers. The question
               is whether that zero is real. A near-foggy frame that the detector
               cannot see through produces a confident zero, and a zero at first
               light is exactly the value a human would not question -- the surf
               often is empty then. If these zeros are wrong, the model trains on
               fog-induced zeros at the hour when crowds start building.

  near_reject/ The frame FAILED for fog but only just (lap_var >= 10 against a
               12.7 gate), so no detection was ever run. The question is the
               mirror image: whether countable surfers were thrown away. 27
               frames have ever been rejected for fog, 18 of them this close to
               the line.

Why first-of-day. Fog here is a morning marine layer that burns off, so the
near-gate frames cluster at first light, and that is also when the count is
least verifiable. Restricting to the day's first few images keeps the set small
enough to review in one sitting without diluting it with midday frames that
happen to be slightly soft.

Both sets are small on purpose -- this is a decision about one threshold, not a
recall study. The whole point is that one verdict per frame is enough to say
whether the gate is set in the right place, and 23 frames is an hour of work.

No boxes are drawn. The near_pass frames are zero-count by construction so
there would be nothing to draw, and the near_reject frames never went through
detection at all. What is being judged is the image, not the detector's output.

Output:
    data/reviews/fog_gate_review/near_pass/<date>_<time>.png
    data/reviews/fog_gate_review/near_reject/<date>_<time>.png
    data/reviews/fog_gate_review/fog_gate_review.csv   one row per frame
    data/reviews/fog_gate_review/fog_gate_guide.md     what to do

Usage:
    python code/make_fog_gate_review.py
    python code/make_fog_gate_review.py --lapvar-max 30 --first-n 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
import detect_surfers as ds  # noqa: E402

PREDS_CSV = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT = _PROJECT_ROOT / "data" / "reviews" / "fog_gate_review"

UPSCALE = 2.0            # surfers here are specks; 1.25 (the GIF value) is too small
BANNER_H = 44
GATE = ds.QUALITY_LAPVAR_THRESH


def render(img_path, caption):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    out = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    h, w = out.shape[:2]
    canvas = cv2.copyMakeBorder(out, 0, BANNER_H, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # ASCII only: cv2's Hershey fonts render an em dash as "???".
    scale, thick = 0.8, 2
    while scale > 0.4:
        (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= w - 20:
            break
        scale -= 0.05
        thick = 2 if scale > 0.6 else 1
    cv2.putText(canvas, caption, (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (235, 235, 235), thick, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lapvar-max", type=float, default=25.0,
                    help="upper edge of the near-pass band (gate is at 12.7)")
    ap.add_argument("--lapvar-min-reject", type=float, default=10.0,
                    help="lower edge of the near-reject band")
    ap.add_argument("--first-n", type=int, default=3,
                    help="how many of the day's earliest images count as first-of-day")
    args = ap.parse_args()

    df = pd.read_csv(PREDS_CSV, float_precision="round_trip")
    for c in ("lap_var", "brightness", "surfer_count", "glare_frac"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "time_local"])

    passed = df[df["quality_ok"].astype(str) == "True"]
    early = passed.groupby("date").head(args.first_n)
    near_pass = early[(early["lap_var"] < args.lapvar_max)
                      & (early["surfer_count"] == 0)].copy()

    near_reject = df[(df["quality_ok"].astype(str) == "False")
                     & (df["quality_reason"] == "foggy_or_blurred")
                     & (df["lap_var"] >= args.lapvar_min_reject)].copy()

    rows = []
    for tier, sel in (("near_pass", near_pass), ("near_reject", near_reject)):
        (OUT / tier).mkdir(parents=True, exist_ok=True)
        for _i, r in sel.iterrows():
            src = ds.CROPS_DIR / r["filename"]
            if not src.exists():
                print(f"  missing crop, skipped: {r['filename']}")
                continue
            verdict = ("PASSED the gate, counted 0" if tier == "near_pass"
                       else "REJECTED for fog, never counted")
            cap = (f"{r['date']} {r['time_local']}  |  lap_var {r['lap_var']:.1f} "
                   f"(gate {GATE})  |  brightness {r['brightness']:.0f}  |  {verdict}")
            img = render(src, cap)
            if img is None:
                print(f"  unreadable, skipped: {r['filename']}")
                continue
            name = f"{r['date']}_{str(r['time_local']).replace(':', '')}.png"
            cv2.imwrite(str(OUT / tier / name), img)
            rows.append({
                "tier": tier, "image": f"{tier}/{name}",
                "date": r["date"], "time_local": r["time_local"],
                "lap_var": r["lap_var"], "brightness": r["brightness"],
                "glare_frac": r["glare_frac"],
                "gate_said": "pass" if tier == "near_pass" else "reject",
                "recorded_count": ("0" if tier == "near_pass" else ""),
                # To fill in:
                "surfers_visible": "",     # yes / no / a count, or `unsure`
                "fog_blocks_view": "",     # yes / no / partly
                "gate_was_right": "",      # yes / no
                "notes": "",
                "filename": r["filename"],
            })

    if not rows:
        print("No frames matched — nothing written.")
        return 1

    out_csv = OUT / "fog_gate_review.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    n_pass = sum(1 for r in rows if r["tier"] == "near_pass")
    n_rej = len(rows) - n_pass
    (OUT / "fog_gate_guide.md").write_text(f"""# Fog-gate review — {len(rows)} frames

The gate rejects a frame as `foggy_or_blurred` below **lap_var {GATE}**. That one
number decides whether an hour is counted at all. These are the frames closest to
it, in both directions.

Open `fog_gate_review.csv`. One row per image. Four columns to fill:

| column | what to put |
|---|---|
| `surfers_visible` | `yes` / `no` / a count if you can count them / `unsure` |
| `fog_blocks_view` | `yes` / `no` / `partly` — is fog the reason, or is the water just empty? |
| `gate_was_right` | `yes` / `no` — should this frame have been counted? |
| `notes` | anything else |

## `near_pass/` — {n_pass} frames

Passed the gate, among the day's first {args.first_n} images, lap_var under
{args.lapvar_max}, and **counted as zero surfers**. The question: is that zero real?

A zero at first light looks unremarkable — the surf often *is* empty then — which
is what makes it the easiest wrong value to miss. If fog is producing these zeros,
the forecast is training on them at the hour when crowds start building.

## `near_reject/` — {n_rej} frames

Failed the gate for fog, but only just (lap_var at or above
{args.lapvar_min_reject}, against the {GATE} gate). **No detection was ever run on
these.** The question: were countable surfers thrown away?

Only 27 frames have ever been rejected for fog, so this is most of that history.

## What the answers decide

`gate_was_right = no` in `near_pass/` argues the threshold is too **low** — fog is
getting through and producing false zeros. The same answer in `near_reject/`
argues it is too **high** — countable frames are being discarded. Both directions
being wrong would mean lap_var is the wrong statistic for this, not that the
number needs moving, and `fog_quality/`'s quadrant work is where that would go
next.

Changing the threshold changes which hours exist in the training data, so it
needs Joel's go-ahead and a re-run of the affected dates, not a quiet edit.
""")

    print(f"Wrote {len(rows)} frames to {OUT}")
    print(f"  near_pass   {n_pass:3d}  (passed, lap_var < {args.lapvar_max}, counted 0)")
    print(f"  near_reject {n_rej:3d}  (rejected for fog, lap_var >= {args.lapvar_min_reject})")
    print(f"  {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
