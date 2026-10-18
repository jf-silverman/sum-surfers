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

**Reviewed, settled and ADOPTED, 2026-10-17.** Option 1 below was done: all 24
firing boxes were rendered and judged, Joel gave the go-ahead, and the rule now
ships as `BLUFF_MASK` / `CONF_THRESH_BLUFF_MASK` / `BLUFF_EDGE_Y` in
`code/detect_surfers.py`, covered by two tests in
`tests/test_detection_geometry.py`. Scored by
`python analysis/shore_head_zone/score_bluff_review.py`.

| | n |
|---|---:|
| head on shore | 17 |
| head and arms on shore | 1 |
| cowboy hat on shore | 1 |
| box on a surfer who already has a better box | 1 |
| **the only box on a real surfer** | **4** |

Deleting a box that duplicates an already-counted surfer changes no count, so
it scores with the suppressions. **20 of 24 correct, a 5:1 trade**, and the
evidence base goes from 6 points to 30. The 0.04 boundary gap that held the
rule back is gone as a concern — not because it widened, but because it was
the wrong thing to be looking at.

**Nothing separates the 4 real surfers from the 19 shore heads.** Confidence,
width, height, top edge and centre-x all overlap completely:

| | real surfers | shore heads |
|---|---|---|
| confidence | 0.24–0.46 | 0.20–0.49 |
| width | 12.3–20.1 | 9.2–27.3 |
| height | 10.4–16.9 | 7.9–16.3 |
| y1 | 162.7–169.6 | 163.2–171.8 |
| centre x | 534.7–662.6 | 537.9–672.3 |

So the rule is a flat trade, not something to be tuned. Tighter variants look
better on paper — `x<=620` is 8.5:1, `conf<0.40` is 12:1 — but each one is
fitted to which of **four** boxes it happens to exclude, and the ranges above
say there is no real boundary there to find. Take the rule as specified or not
at all.

**What it is worth.** 0.080 boxes/frame firing: **0.063 false counts removed,
0.013 real surfers lost**, net −0.05 counts/frame against a mean hourly count
of 14.7 and a ~1.5-surfer noise floor.

That net is in the wrong direction for the count on its own terms — the
detector undercounts by 2.14 surfers/frame
(`analysis/conf_band_decision/`), so removing boxes, even mostly wrong ones,
moves the count further down. The case for adopting it anyway is that shore
heads are a **systematic** bias, not noise: they fire at the same handful of
bluff positions day after day, so they do not average out across hours the way
a random false positive does, and they concentrate on the clear bright days
when people are on the bluff — the same days the count is already most trusted.
Precision improves; the count cost is a fortieth of the undercount it adds to.

**It is also a precondition for lowering `CONF_THRESH`.** The band review holds
9 rejected shore heads over 113 frames, all currently excluded by the threshold
alone. A 0.15 floor admits 3 of them (0.027/frame, a third of what the zone
catches); a 0.10 floor admits all 9 (0.080/frame, doubling it). The zone should
go in first either way.

**Definition, now settled.** One of the 4 real surfers is noted "surfer standing
on beach w/ board in-hand", which raised the question of whether a person on the
beach is a count. Joel: he is standing in knee-deep water, wading out to surf,
and **a surfer on the beach is a surfer**. The ratio stays 5:1, and the series
counts people who are there to surf, not only people already in the lineup.

**Watch item, unchanged.** The x range 400–680 is still fitted. Two of the four
real surfers sit in its right third (centre x 662 and 668), and so does the
rightmost shore head (672). The bluff carries people further right than 680;
any future evidence should re-derive the range rather than assume it.

**Logged, not acted on.** The first reviewed frame carries the note "blurry w/
condensation over 80% and should be gated out of detection" — it passed the
quality gate. It belongs with the 23 frames already in the fog-gate review.

## Files

- `measure_shore_zone.py` — the measurement; `--rebuild` re-detects,
  `--rate N` samples N unlabelled production frames
- `production_boxes_on_labeled.csv` — 3,404 production boxes on the 142
  labelled frames, with IoU and TP flag
- `sample_boxes_unlabeled.csv` — 4,183 boxes over 300 random production frames
- `why_the_zone_exists.png` — the full camera frame, the ROI, and the bluff
