# Bluff-head review — 24 boxes, 24 images

Each image is one **production detection** that a candidate rule would throw
away. The question for each is simply: **was the detector right?**

## What to do

Open `bluff_head_review.csv`, one row per image.

**`is_surfer`** — `y` or `n`.
`y` = someone in the water surfing, sitting on a board, paddling, wiped out.
`n` = anything else. A person standing on the bluff or walking on the sand is
the expected `n`, and is the whole reason the rule exists.

**`what`** — what it actually is, so a right call can be told from a lucky one:

```
bluff    a person up on the cliff path, head clipped by the frame edge
sand     a person on the beach or in the shallows, not surfing
water    a surfer (pair with is_surfer = y)
wave     foam, shadow, texture
other    anything else -- say what in notes
```

**`js_confidence`** — same convention as the confidence-band review. Leave it
blank when you are sure; write a decimal (`.6`, `.9`) for how certain you are
of the letter you wrote, only when it is below about 0.9.

**`notes`** — anything worth knowing. Particularly: if there are other heads
beside the boxed one that the detector did *not* box, say so.

## What you are looking at

- **top, 2x** — the whole 1280-wide crop, so the box in question can be
  compared against the surfers out in the water on the same frame.
- **bottom, 8x** — ROI rows 100–180 around the box, where the blob is actually
  legible.
- **coral** — the box under judgement. **Grey** — other boxes the detector
  accepted on this frame; context only, ignore them.

The caption gives the box's size and position. All of these are clipped by the
bottom edge of the frame by construction.

## Why

The ROI is rows 420–600 of a 720-row camera frame. The bluff path — fence,
bench, people standing and watching — sits at roughly y 590–660, **below** the
crop, so only the top of a standing head reaches the image and its box is
always cut off by the edge.

On the 142 labelled frames, that signature plus an x range of 400–680 plus a
raised in-zone threshold of 0.50 removes 5 of 6 false positives for **zero**
lost surfers. Six is not enough to change production, and the gap between the
highest false positive (0.49) and the lowest true positive (0.53) is 0.04 —
fitted to the sample rather than a margin.

These 24 come from a 300-frame random sample of the production record, so they
are the boxes the rule would actually fire on in service. Verdicts here take
the evidence from 6 to 30 and either widen that gap or kill the rule.

A high `n` rate adopts it. A few `y`s at the top of the confidence range kills
it. Anything in between probably means the x range or the threshold is wrong
rather than the idea.

Full background: `analysis/shore_head_zone/results.md`.
