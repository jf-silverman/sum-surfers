# Tree-bough zone — measured, and `TREE_MASK` stays as it is

**Measured 2026-10-17.** Reproduce with
`python analysis/tree_bough_zone/measure_unfiltered.py` then
`python analysis/tree_bough_zone/derive_tree_zone.py`.

## Why this needed a new measurement

The completed confidence-band review flagged 18 tree-bough false positives
escaping `TREE_MASK = (1268, 58, 1280, 87)` — the largest false-positive class
in that review. Two problems had to be cleared before a zone could be derived.

**The band review carries no coordinates.** Recovering them by re-detecting the
review days and matching back to Joel's notes failed: 4 of 18 matched, and one
of those 4 was a box at x=971, mid-frame, obviously not a bough. The corrected
`survive()` no longer produces 22 of the reviewed rows, so the two orderings do
not line up and confidence is not unique enough within a frame to match on.
That approach is abandoned, not fixed.

**`production_boxes_on_labeled.csv` cannot score a zone.** It is written through
`ds.run_inference_with_boxes()`, which applies the zones, so anything a zone
already drops is absent from it by construction. Scored against it, `TREE_MASK`
returns "0 false positives caught, 0 true positives clipped" — an artifact of
the file. `measure_unfiltered.py` rebuilds the same table with everything up to
the zones and nothing after: **3,443 boxes on 142 labelled frames, 3,125 TP,
318 FP.**

## What the existing zones are actually worth

Scored on that table, at the production threshold, against IoU-0.5 ground truth:

| zone | catches | clips | |
|---|---:|---:|---|
| `TREE_MASK` (1268, 58, 1280, 87) | **39 FP** | **0 TP** | earning its place |
| `FLAG_MASK` (720, 140, 750, 165), conf-gated | 0 FP | 0 TP | inert on this set |
| `BLUFF_MASK` (400, 164, 680, 180), conf+edge-gated | **5 FP** | **0 TP** | holds on frames it was never fitted to |

`TREE_MASK` had never been scored against box-level ground truth before — it was
fitted to a 50-frame spot-check in August 2026. It catches 39 of the 44 false
positives with centre x ≥ 1268 and clips nothing. **The newly adopted
`BLUFF_MASK` independently validates**: 5 FP, 0 TP, on a labelled set that is
not the 300-frame sample it was measured on and not the 24 boxes it was reviewed
on.

`FLAG_MASK` lands on nothing at all here — no false positive and no true
positive falls in it across 142 frames. That is a real 0/0, not the artifact
above. It is not evidence the zone is wrong; it is evidence it earns nothing
measurable, and the band review found exactly one wind-sock false positive,
below threshold.

## No extension is worth taking

Grid search over x1 ∈ [1260, 1276], y1 ∈ [0, 98], y2 ∈ [y1+10, 180], keeping
true positives clipped at zero:

| zone | catches | clips | vs current |
|---|---:|---:|---:|
| (1268, 58, 1280, 87) — current | 39 | 0 | — |
| **(1270, 56, 1280, 180)** — best | **43** | **0** | **+4** |

**+4 false positives over 142 frames, or 0.028/frame.** Note the x edge has to
move *right*, from 1268 to 1270, to buy the extra y range — true positives sit
at centre x 1267.9 and 1268.8 inside the y band. The y1 = 56 edge is pinned by a
true positive at centre y 49.9 with 6 px of margin, and the x1 = 1270 edge by
one at 1268.8 with 1.2 px.

That is the same shape of evidence that held the bluff zone back before its
review — a boundary fitted to individual boxes rather than a margin — for a
seventh of the payoff, and with no review set to resolve it. **`TREE_MASK` stays
as it is.**

The 18 band boughs that prompted this remain excluded by the confidence
threshold alone, which `analysis/conf_thresh_recheck/` has now settled at 0.195.
If a lower threshold is ever revisited, this is the measurement to redo first,
and it should be redone with a labelled set that samples the right edge
deliberately rather than incidentally.
