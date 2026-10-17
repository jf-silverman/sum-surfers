# The on-shore heads: where they come from and what a zone would cost

Joel's proposal, 2026-10-17: put a large box over the bottom centre of the
frame, raise the confidence threshold inside it, and use box size as a second
condition — the people are on the cliff in the **foreground**, so their head
and shoulders should be much bigger than a surfer in the water at the same
image height.

The mechanism is right, the geometry is better than expected, and the size
half of it is backwards. Details below.

## The mechanism, settled

`why_the_zone_exists.png` is a full 1280×720 camera frame with the ROI drawn on
it. The ROI is rows **420–600**. The bluff path — fence, bench, bikes, people
standing and watching — sits at roughly **y 590–660**, which is *below* the
crop. Only the top of a standing head reaches the image at all.

That has a consequence worth more than the zone itself: **a bluff head's box is
always clipped by the bottom edge of the ROI.** It cannot be otherwise. The
body continues below row 600 and the detector never sees it.

So the condition is not "low in the frame" but "cut off by the frame", which is
a far tighter test, and it is free to evaluate.

## What is actually down there

Every one of the 142 labelled frames run through the production pipeline
(`nms_across_tiles` → `suppress_contained` → `filter_false_positive_zones`),
each surviving box matched to ground truth at IoU 0.5 — 3,404 boxes, 3,125 true
positives, 279 false positives.

Boxes clipped by the bottom edge (`y2 ≥ 179`):

| | n | median width | median conf | cx range |
|---|---:|---:|---:|---|
| true positives | 26 | 25.7 | 0.68 | 18 – 994 |
| false positives | 6 | 12.8 | 0.37 | **405 – 656** |

The false positives occupy a narrow slice of x and nothing else. That slice is
exactly where the bluff path is highest in the frame. Visual check of all six:
every one is a dark head-shaped blob sitting on the bottom edge, and in three
of them a second, identical, undetected head sits right beside it.

## Confidence separates. Size does not.

Inside `x 400–680` and clipped by the bottom edge: **11 true positives, 6 false
positives.**

```
false positives   0.26  0.30  0.35  0.40  0.49  0.55
true  positives   0.53  0.55  0.58  0.60  0.66  0.68  0.69  0.73  0.76  0.79  0.79
```

They almost sort. A raised threshold inside the zone, applied only to
edge-clipped boxes:

| in-zone threshold | surfers lost | FPs removed |
|---|---:|---:|
| 0.30 | 0 | 1 |
| 0.40 | 0 | 3 |
| **0.50** | **0** | **5 of 6** |
| 0.55 | 2 | 5 |
| 0.60 | 3 | 6 |

Dropping the edge-clipped condition and using the rectangle alone costs a
surfer at 0.50 and four at 0.60, so the edge test is doing real work.

**The size condition goes the wrong way.** In the zone, false positives have
median width **12.8** and true positives **25.7** — the FPs are the *smaller*
boxes, not the bigger ones. Keyed on size alone:

| rule | surfers lost | FPs removed |
|---|---:|---:|
| w ≥ 12, any conf | 10 | 5 |
| w ≥ 15, any conf | 8 | 2 |
| w ≥ 20, any conf | 7 | 1 |

Unusable in that direction. The reason is the clipping: we see a 10-pixel slice
of the top of a head, while a surfer at the same height is boxed with their
board and comes out twice as wide. The foreground is real — a bluff person's
head is bigger *in the world* — but almost all of it is outside the crop, so
none of that size reaches the detector.

There is a second trap here. The confidence-band review points the opposite
way: there the shore heads are 12.4–19.3 wide and the one real surfer among
them is 10.7, which looks like Joel's hypothesis confirmed. Those boxes are all
between 0.10 and 0.195, and near threshold *both* classes produce small partial
boxes. Width only separates once confidence is high enough for a surfer to be
boxed with their board — which is to say, exactly where the rule does not need
help. A size rule and a low confidence gate cancel each other out.

## What it would be worth

300 random quality-passed production frames, 4,183 boxes (13.94 per frame). The
rule fires on **24 boxes in 23 frames** — 7.7% of frames, **0.08 boxes per
frame**, 0.57% of all detections. Firing confidences: min 0.20, median 0.38,
max 0.49, so these are drawn from the same population as the six measured FPs.

If the labelled sample's 5-in-6 holds, that is roughly **0.08 surfers of
systematic overcount removed per hour**, against a mean hourly count of 14.7
and a measured per-hour noise floor of about 1.5 surfers.

## Where this leaves it

Not adopted. The direction is right and the rule is clean, but it rests on
**six false positives**, and the gap between the highest FP (0.49) and the
lowest true positive (0.53) is 0.04 — fitted to this sample, not a margin.
Nothing about the count justifies the risk at that evidence level: 0.08 per
frame is a twentieth of the noise floor.

What would change that:

1. **A targeted review.** `sample_boxes_unlabeled.csv` already holds the 24
   firing boxes. Rendering those 23 frames and judging each would take the
   evidence from 6 to 30 and either widen the 0.04 gap or kill the rule.
2. **A lower `CONF_THRESH`.** The band review found 8 shore heads in 187 boxes
   across 113 frames — all between 0.10 and 0.195, so all currently excluded by
   the threshold alone. Lowering the threshold to 0.10 would admit them at
   roughly 0.07 per frame, doubling what the zone has to catch and making it
   worth having *before* the threshold moves.

Watch item either way: the x range 400–680 is fitted to six points, and the
bluff carries people further right than that. Any future evidence should
re-derive the range rather than assume it.

## Files

- `measure_shore_zone.py` — the measurement; `--rebuild` re-detects,
  `--rate N` samples N unlabelled production frames
- `production_boxes_on_labeled.csv` — 3,404 production boxes on the 142
  labelled frames, with IoU and TP flag
- `sample_boxes_unlabeled.csv` — 4,183 boxes over 300 random production frames
- `why_the_zone_exists.png` — the full camera frame, the ROI, and the bluff
