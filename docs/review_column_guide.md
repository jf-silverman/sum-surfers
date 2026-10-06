# Review CSV — Column Guide

Reference for filling in `data/reviews/review_all.csv` -- one sheet for every
day (2026-10-05; it was a CSV per folder before). Scored by
`code/score_reviews.py`. Column order follows the arrangement Joel settled on
while reviewing 2026-09-23: one failure mode at a time, misses before false
positives, with each kind of doubt sitting next to the calls it qualifies.

## The one rule that must hold

```
true_positives + false_positives = boxes_drawn
```

Every drawn box is either on a surfer or not. `missed` is separate — those
surfers have no box, so they are not part of `boxes_drawn`.

The scorer prints an ARITHMETIC PROBLEMS block naming any row that breaks this.

## Column order

Work columns come first and the two identifiers sit at the far right
(2026-10-06), so the fill-in columns are not pushed off screen by fields
nothing is ever typed into:

```
date, time_local, boxes_drawn, forecast,
true_positives, missed, missed_poses, multi_surfer_box, uncertain_missed,
false_positives, false_positive_box_numbers, false_positive_causes,
uncertain_box_numbers, standing_box_numbers, sup_box_numbers, notes,
filename, image
```

Every script reads columns by NAME, never by position, so reordering them again
is safe. Adding or dropping one is not — see "If your editor drops a column".

## Prefilled — do not edit

| column | meaning |
|---|---|
| `date` | review day |
| `time_local` | clip time, local |
| `boxes_drawn` | how many boxes the detector drew (also in the image filename) |
| `forecast` | what the model predicted for that hour, for context only |
| `filename` | the source crop in `data/j_shore_cam/surf_crops/` |
| `image` | the rendered PNG to look at, in `full_day_<date>/` |

### If your editor drops a column

It has happened once: a grid editor held a copy from before two columns were
added and wrote it back, dropping them. Nothing typed was lost, because the
columns are re-attached by matching on `filename`. If the sheet comes back with
fewer than 18 columns, say so and it can be restored from git the same way --
do not re-type anything.

## To fill in

| column | format | meaning |
|---|---|---|
| `true_positives` | integer | boxes genuinely on a surfer |
| `missed` | integer | real surfers with **no** box at all |
| `missed_poses` | `;`-separated, optional leading count | pose of each missed surfer |
| `multi_surfer_box` | `;`-separated numbers | boxes holding **more than one** surfer |
| `uncertain_missed` | integer | of `missed`, how many you cannot confirm |
| `false_positives` | integer | boxes on anything that is not a surfer |
| `false_positive_box_numbers` | `;`-separated numbers | which drawn boxes those were |
| `false_positive_causes` | free text, `;`-separated | what each actually was |
| `uncertain_box_numbers` | `;`-separated numbers | drawn boxes you would not defend |
| `standing_box_numbers` | `;`-separated numbers | boxes whose surfer is **riding a wave** |
| `sup_box_numbers` | `;`-separated numbers | boxes holding a **stand-up paddleboarder** |
| `notes` | free text | anything else worth remembering |

Pose vocabulary: `sitting | prone | standing | SUP | wipeout | unknown`.

## `multi_surfer_box` — merged boxes

A box drawn around two surfers is a different failure from a surfer the
detector never saw at all, and the two have different fixes: one is the
box-merging logic (NMS and containment suppression), the other is recall. On
2026-09-26 this was **8 of 31 misses**, so it is worth separating.

Each extra surfer inside a box still costs a miss, so count it in `missed` as
well, and list the box number here:

```
missed=2  missed_poses=2 prone  multi_surfer_box=2;18
```

That reads: boxes 2 and 18 each contain two surfers, so two real surfers have
no box of their own. The box itself stays a **true positive** — it is on a real
surfer — so `true_positives` does not change.

The scorer assumes exactly two surfers per listed box, which is what every note
so far describes, and warns if `multi_surfer_box` lists more boxes than
`missed` allows.

## The two uncertainty columns

They are split by **direction**, because they bound different metrics and would
cancel if merged:

- `uncertain_box_numbers` → a drawn box that might not be a surfer → bounds
  **precision**.
- `uncertain_missed` → a surfer you think is there but cannot see → bounds
  **recall**.

Both are a **subset** of the counts beside them, never extra. Still make your
best call in `true_positives` / `false_positives` / `missed`; these only mark
which of those calls are soft.

**When to flag one.** If you had to brighten the image to decide, if you would
plausibly answer differently on a second pass, or if you are calling it from
position and context rather than actually seeing a person.

**When not to.** Effort is not doubt. A call that took a long hard look but
that you are now sure of is not uncertain.

The scorer reports best case (soft boxes are all real, soft misses are all
imagined) against worst case (the reverse). The gap is the ambiguity budget —
how much of the headline number is judgement rather than measurement.

## Worked examples, all real rows from 2026-09-23

**Clean frame.** 11 boxes, all correct, nothing missed:

```
boxes_drawn=11  true_positives=11  missed=0  false_positives=0
```

**One bad box.** 11 drawn, box 1 is a lens flare, and one real surfer has no box:

```
boxes_drawn=11  true_positives=10  missed=1  false_positives=1
false_positive_box_numbers=1
false_positive_causes=lens flare spot; very distinct oversat w/ bright red & blue edges
```

11 = 10 + 1. The `;` here continues one description; it does not mean two
causes. Causes are tallied per frame, not per box, precisely because of this.

**Two false positives sharing one cause.** Boxes 2 and 3 are both heads on the
beach:

```
boxes_drawn=10  true_positives=8  missed=0  false_positives=2
false_positive_box_numbers=2;3
false_positive_causes=heads of people on shore (bottom center)
```

**A merged box.** 36 boxes, all on real surfers, but box 26 holds three people:

```
boxes_drawn=27  true_positives=27  missed=2  missed_poses=1 prone; 1 sitting
multi_surfer_box=26
notes=The 2 false negatives sit inside box 26 (there are 3 surfers total)
```

**Several missed, with poses.** Two surfers missed, both lying down:

```
boxes_drawn=36  true_positives=36  missed=2  missed_poses=2 prone
```

A leading number multiplies, so `2 prone` is two prone surfers. Bare `prone`
means one. Mixed is written out: `1 prone; 1 sitting`. The scorer checks these
add up to `missed` and names any row that does not.

**Hard frame, with doubt.** Dim and hazy. Three boxes drawn; box 2 might be
chop; a fourth shape might be a prone surfer but will not resolve:

```
boxes_drawn=3  true_positives=3  missed=1  missed_poses=prone
uncertain_box_numbers=2  uncertain_missed=1
notes=low light & somewhat hazy; not certain 2 is a surfer.
```

Note the best call is still made — box 2 is counted a true positive and the
miss is counted — and *then* both are flagged soft.

## `standing_box_numbers` and `sup_box_numbers` — the two rare classes

**`standing` means riding a wave. Nothing else.** (Joel's definition,
2026-10-05.) A stand-up paddleboarder goes in `sup_box_numbers` instead, never
in `standing`, even though they are also on their feet.

They are kept apart because they look nothing alike to a detector. A wave rider
is dynamic — spray, a wave face behind them, a leaning posture. A SUP rider is
upright and static on flat water holding a paddle. Merging them would blur the
very class the labeling effort is meant to build.

Record box numbers, e.g. `9;22`. Both are true positives like any other, so no
other column changes.

Why it is worth the extra second: these are the rare poses. The labeled
training set holds **31** standing boxes and **4** SUP, against 617 sitting and
512 prone, and that imbalance is the single thing blocking a multi-class
detector. Recording which box in which frame turns every review pass
into a CVAT shortlist for free — the frames are already on disk and the boxes
are already drawn, so it costs a box number instead of a hunt.

A standing surfer that was MISSED goes in `missed_poses` as `standing`
instead, like any other missed pose. None has ever been missed: across 638 real
surfers, every one of the 49 misses was prone or sitting.

`python code/score_reviews.py` prints the running shortlist.

## Unusable frames

Put **`na`** in `true_positives` for a frame you looked at and could not count —
sensor noise, condensation, flare over the part that matters. Say why in
`notes`. It is excluded from every figure and listed separately in the scorer's
output.

`na` is not the same as leaving the row blank. Blank means not yet reviewed;
`na` is a verdict. The difference matters, because a frame the automated quality
gate passed but a human cannot count is a finding about the gate — see D04 in
`known_bugs.md`, where two such frames had already contributed counts of 1 and
11 to `predictions.csv`.

## Scoring

```bash
python code/score_reviews.py              # everything reviewed so far
python code/score_reviews.py --date 2026-09-23
python code/score_reviews.py --by-day
```

Rows with `true_positives` blank are treated as not yet reviewed and skipped,
so a part-finished day scores fine.

`make_review_set.py` only ever ADDS rows to the sheet. Re-running it for a day
you have already reviewed is a no-op on your answers.

There is no accuracy figure. Object detection has no true negative — the number
of image regions correctly left un-boxed is unbounded — so precision, recall
and count error are the whole picture.
