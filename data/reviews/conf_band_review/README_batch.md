# Confidence-band review — 197 boxes, 118 frames

## What to do

Open `conf_band_review.csv`. One row per **numbered coral box**.

**`is_surfer`** — `yes` or `no` when you are sure, otherwise **a percent 0–100**:
how likely it is a surfer. `70` means you lean yes but would not defend it.

Writing a percent is better than forcing a guess. Scored as a probability, an
unsure box contributes its fraction rather than being rounded to 0 or 1, which
gives a calibrated expected count as well as a point estimate.

**`pose`** — only for the ones that ARE surfers, and optional:

```
prone-a (angled)  prone-s (side-on)  prone-e (end-on)  prone
sitting  standing  SUP  wipeout  unknown
```

These boxes nearly went undetected, so their pose is the signal. End-on is the
smallest silhouette and the obvious suspect; this is how that gets tested
rather than assumed.

In each image:

- **coral, numbered, thin** — the boxes under judgement. These sit between
  confidence 0.10 and 0.195, so the detector saw them but the current threshold
  throws them away. Drawn translucent so you can see what is underneath, which
  is the whole question.
- **grey, faint** — boxes the detector already accepts. Context only; ignore
  them.

Most frames have one or two coral boxes. The busiest have five.

## Why this is worth the time

Across the 16 hand-reviewed days, **286 of 316 missed surfers had no box at
all**, and the open question was whether the detector never proposed them or
proposed them below threshold.

Re-running inference at a 0.02 floor answered it: dropping the threshold from
0.195 to 0.10 adds exactly these 197 boxes and closes the count gap almost
perfectly — 3,900 boxes against 3,880 real surfers — and the extra boxes appear
where the misses were (frames with more no-box misses gain more boxes,
rho = +0.425, p = 4.7e-11).

So the surfers **were** seen. The threshold is discarding them.

What that evidence cannot settle is whether these 197 are surfers. Counts
cannot tell a recovered surfer from a new false positive, which is exactly the
trap that collapsed an earlier containment-threshold claim once box-level truth
existed. The stakes:

| if the 197 are... | effect |
|---|---|
| all real surfers | recall **0.919 → 0.969** |
| all false positives | precision **0.987 → 0.936** |
| break-even for F1 | about **42%** need to be real |

That spread is why each one gets looked at. It is also why this is worth more
than the whole containment-suppression effort, whose ceiling is +0.008 recall.

## After

Hand it back. The answers give the real precision and recall at a 0.10
threshold, and `CONF_THRESH` gets chosen on box-level ground truth for the
first time — the September sweep used count MAE on 76 frames.

The pose column answers a second question for free: of the surfers the detector
almost missed, which orientations are they? If they are mostly `prone-e`, that
is a labelling target with a name rather than a vague "prone is hard".

A change to `CONF_THRESH` alters production counts, so it needs Joel's explicit
go-ahead and a re-check against the clear-day overcount set before adoption.
