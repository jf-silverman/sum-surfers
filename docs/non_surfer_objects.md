# Non-surfer objects in frame

What else appears in this camera's view, how to tell each one from a surfer, how
the pipeline currently handles it, and which frames to test against. Written
2026-09-22 after finding birds *and a bird's reflection* counted as surfers in a
held-out frame.

The detector has exactly one class, `Surfer`. Everything here is therefore a
false positive when it gets boxed, and the fix is almost always **training data**
(leave the object unboxed so it trains as background) rather than a filter.

> **A note on the file paths below.** Camera frames, review sets and comparison
> images are not committed to this repository — they are working data on the
> machine that runs the pipeline. Frame names are given so the cases are
> identifiable and re-checkable in that working copy, and every claim here is
> reported with the measurement behind it rather than resting on the image alone.
> `data/demo/` holds 20 labeled frames if you want to run the detector yourself.

## Labeling policy

**Leave non-surfer objects unboxed.** Every box exports as class `Surfer`, so
boxing a bird to "mark" it would teach the detector that birds are surfers.
Unlabeled pixels train as background, which is the lesson wanted. Frames known to
contain one are listed in `analysis/training_data_expansion/known_bird_frames.txt`
so a retrain can be re-checked against them.

---

## Birds

The most common false positive, and the one with the clearest visual tell.

**How to tell a bird from a surfer:**

| | bird | surfer |
|---|---|---|
| shape | narrow, often angled; wings break the outline | compact dark blob, usually with a board |
| reflection | **detached** — a separate dark smudge below, with water visible between | continuous with the body, since the person is *in* the water |
| size | can be foreground-sized when close, not only a distant speck | scales with image row |
| position | any row, including above the horizon line of surfable water | within the lineup |

The **detached reflection is the reliable tell**. A bird flying low leaves its
reflection on the water below it with a gap between; a floating surfer has no
such gap. Size alone is not a tell — see the caveat below.

### Confirmed case: `data/reviews/count_60sec_var/set5_09_26/sec_21.jpg`

Two birds flying low, at roughly x=328 and x=356, y≈70 of the 1280×180 strip —
visible in `data/compare_detector_20260921/clearday_scenes/set5_09_26_sec_21.png`.
Both have plainly detached reflections. Measured 2026-09-22:

| model | boxes in that region | what they landed on |
|---|---|---|
| old (Oct 2025) | 2 | one on each bird, conf 0.223 and 0.313 |
| new (Sept 2026 fog) | 2 | **both on the first bird** — the bird at conf 0.719, and its reflection just below at conf 0.316. The second bird is missed. |

Two things worth separating here. First, **birds are counted on frames the model
never trained on** — this frame is from the 60-second review set, not the
training data, which is the confirmation this had been waiting for: earlier
bird sightings were all in frames the model had trained on, so they proved
nothing about unseen data.
Second, and new, **a reflection is being counted as a surfer in its own right**.
The bird box (y 69–75) and the reflection box (y 77–83) are vertically adjacent
rather than nested, so nested-box suppression does not merge them: that filter
compares intersection against the smaller box's area, and these two barely touch.

The human count for that frame is 16; the new model returns 19.

### Caveat: do not filter birds by box size

Birds are not reliably small. `crop2026-08-09_08-23-00.jpg` has a
**foreground-sized** bird at top right — not a distant speck. A naive
"drop small boxes" rule would miss it while deleting genuine distant surfers,
who are legitimately tiny at the top of the frame.

### Birds on the water

Harder, and unresolved. Floating birds have no detached reflection and look much
like a head. Joel's call on `crop2026-09-14_12-35-00` was **0 surfers** with low
confidence: the specks in the far lineup were smaller than a surfer should be for
that row and were probably birds on the water. That frame is why the old model's
9 and the new model's 3 both looked wrong, and it was not recorded as a
`human_count` given the uncertainty.

---

## Fixed camera furniture

Two objects sit in fixed positions and are handled by coordinate zones in
`detect_surfers.py`:

- **Tree bough**, top right (`TREE_MASK = (1268, 58, 1280, 87)`). Detections
  inside are dropped outright — nothing real is ever there.
- **Wind sock / flag**, bottom middle (`FLAG_MASK = (720, 140, 750, 165)`).
  *Not* dropped outright, because real surfers sit in that same spot. Boxes
  there survive only above `CONF_THRESH_FLAG_MASK` (0.5). Shrinking the zone
  instead was tried and could not separate the two.

The flag is visible in the README's daily animation, which points it out to
readers as an example of a misclassification.

---

## People on shore

People walking the beach below the whitewater get boxed as surfers. This is part
of the residual clear-day overcount — the 60-second
clear-day set sits at 102% of the human count after nested-box suppression, and
shore walkers are one of the two known causes, the other being one surfer split
into side-by-side boxes.

The open question is whether the ROI crop should exclude the shore strip
entirely. It has not been changed, because the strip also contains genuine
surfers in the shallows.

---

## Sun glare

Low-angle autumn and winter sun lays a path of specular glitter across the water,
and the old detector read the sparkle as heads — claiming 49, 30, 18, 17, 16 and
15 surfers on water holding 0, 0, 0, 5, 0 and 3.

**Resolved by training, not filtering.** A glare-fraction threshold was tried and
reverted: at the low end the measure cannot tell sun glitter from sunlit
whitewater. One frame at glare 0.0020 is ordinary foam with about 30 plainly
countable surfers, while 0.0032 is a glare path with none. Labeling 21 empty
glare frames and retraining fixed it — held-out glare frames went from 331% of
the real count to 82%. The `glare_frac` column is still recorded so contaminated
rows stay findable.

---

## Testing against these

```bash
# Regenerate the old-vs-new panels, both models through current post-processing
python code/compare_detector_scenes.py

# Run the detector over the labeled frames known to contain birds
cat analysis/training_data_expansion/known_bird_frames.txt
```

When a new non-surfer object turns up, add it here with the frame name, then
decide between the three available responses: leave it unboxed and retrain
(preferred), add a coordinate zone (only for genuinely fixed furniture), or
accept it and record the rate.
