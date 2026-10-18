# Confidence-band review, scored — what `CONF_THRESH` should be

**Measured 2026-10-17.** 187 boxes over 113 frames, every one judged by hand.
Reproduce with `python analysis/conf_band_decision/score_conf_band.py`.

## The scoring rule changed the answer

Joel marked `is_surfer = n` whenever the surfer under the coral box **already had
a grey (accepted) box**. That is stricter than the generator's guide asked for,
and it is the rule that matters: production emits a *count*, and a second box on
an already-counted surfer inflates it exactly as a box on open water does.

| reading | band precision |
|---|---:|
| a surfer is under the box at all (the guide's question) | 114/187 = **61.0%** |
| the box adds a surfer the count is missing (the rule applied) | 92/187 = **49.2%** |

22 of the 95 rejected boxes are the gap: real surfers, already counted. Scoring
them as hits would have put the band over the guide's 42% F1 break-even with
room to spare and made the case for 0.10 look much stronger than it is.

For scale, production at 0.195 runs **91.8%** precision on the 142 COCO-labelled
frames. Every box the band admits is roughly twice as likely to be wrong as what
is already in.

## One ambiguity in `boxes_missed` decides how far down to go

`boxes_missed` is frame-level and can be read two ways, and nothing in the CSV
settles it:

- **A** — surfers with *no box of any colour*, so the 92 recovered surfers are
  **not** counted in it. Current undercount **2.96/frame**.
- **B** — surfers still missed *after* the coral boxes, so the 92 **are** in it.
  Current undercount **2.14/frame**.

## The sweep

Per-frame change in |count error| against the current 0.195, paired bootstrap CI
and Wilcoxon over the 113 frames.

| T | admits | recovered | spurious | precision | A: Δ\|err\| | p | B: Δ\|err\| | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.190 | 14 | 7 | 7 | 0.500 | −0.035 | 0.285 | −0.035 | 0.285 |
| 0.180 | 34 | 19 | 15 | 0.559 | −0.124 | 0.029 | −0.071 | 0.202 |
| 0.170 | 44 | 25 | 19 | 0.568 | −0.177 | 0.007 | −0.124 | 0.056 |
| 0.160 | 62 | 36 | 26 | 0.581 | −0.283 | 0.0004 | −0.159 | 0.042 |
| **0.150** | **82** | **45** | **37** | **0.549** | **−0.389** | **<0.0001** | **−0.212** | **0.023** |
| 0.140 | 108 | 59 | 49 | 0.546 | −0.442 | 0.0001 | −0.212 | 0.087 |
| 0.120 | 145 | 74 | 71 | 0.510 | −0.451 | 0.001 | −0.204 | 0.159 |
| 0.100 | 187 | 92 | 95 | 0.492 | −0.558 | 0.0009 | −0.204 | 0.233 |

**The count improves even at 49% precision, and that is error cancellation, not
accuracy.** The detector undercounts by 2 to 3 surfers a frame on this set, so
injecting boxes that are wrong half the time still closes the gap — the spurious
ones land in frames that had room to absorb them. Box-level precision falls the
whole way down. Any claim about these thresholds has to say which of the two is
being optimised.

**0.15 is the recommendation.** It is the deepest step that is significant under
*both* readings of `boxes_missed`, and it sits in the 0.15–0.18 range where the
precision of what gets admitted peaks at 0.55–0.58. Below 0.14 reading B stops
improving entirely (−0.212 → −0.204, p drifts to 0.23) while precision decays to
0.49: the extra 105 boxes from 0.15 down to 0.10 are 47 recovered against 58
spurious, which is worse than a coin flip. Reading A would keep going to 0.10;
reading B says stop at 0.15. 0.15 is the value both agree on.

## Not adopted yet

Two things stand between this and a `CONF_THRESH` change:

1. **This frame set is biased toward undercounting.** All 113 frames have band
   boxes, and they came from hand-reviewed days chosen because they had misses.
   On frames where the detector already *over*counts, admitting 49%-precision
   boxes can only hurt. The clear-day overcount set has to be re-checked before
   anything ships.
2. **Joel's explicit go-ahead**, per the review guide.

## Free wins that do not need a threshold change

**28 of the 95 rejected boxes are in static locations already known about** — 18
tree bough on the right edge, 9 shore head at the bottom centre, 1 wind sock.
Masking those lifts the band's precision from 49.2% to **57.9%** and removes the
same false positives from every threshold in the table, including the current
one. `TREE_MASK` is currently `(1268, 58, 1280, 87)`, which these 18 escape;
`FLAG_MASK` does not cover the wind sock's top. The bluff-head zone measured in
`analysis/shore_head_zone/` covers the 9.

That is the better next move than the threshold: it costs no recall at all.

## The pose question, answered

Of the 92 recovered surfers, **64 are prone** in some orientation:

| pose | n |
|---|---:|
| prone-s (side-on) | 34 |
| prone-e (end-on) | 16 |
| prone-a (angled) | 14 |
| sitting | 10 |
| unknown | 9 |
| wipeout | 3 |
| standing | 3 |
| two surfers in one box | 3 |

The guide's suspicion was end-on, on the reasoning that it is the smallest
silhouette. **Side-on prone is the larger group, by more than two to one.** That
is a labelling target with a name: the surfers the detector nearly misses are
mostly lying flat and broadside, not pointed at the camera.

Reviewer confidence was filled on 42 of the 92, median 0.80, at or below 0.70 on
15 — so the recovered set is not carried by boxes Joel was unsure about.

## Other observations from the notes, not acted on

- **Lens flare** is called out twice as a repeatable artifact in the same place
  ("oversaturated oval, red one end and blue on the other"), plus 4 more
  glare boxes on the left side at midday. A detector for it, or a time-and-place
  conditioned zone, is a separate piece of work.
- **5 boxes sit in frames Joel flagged as too foggy to judge** ("Very foggy;
  borderline image"). These passed the `QUALITY_LAPVAR_THRESH = 12.7` gate. They
  belong with the 23 frames already waiting in the fog-gate review.
- **Size as a discriminator** comes up three times — "much smaller than surfer
  in similar y level of the frame", "probably a bird. Basically a black dot."
  `analysis/shore_head_zone/` found the opposite sign inside the bluff zone
  (false positives median width 12.8, true positives 25.7), so a size rule has
  to be built per-region, not globally.
