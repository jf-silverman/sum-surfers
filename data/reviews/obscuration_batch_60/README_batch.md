# Obscuration review batch (57 frames)

## What to do

Open `obscuration_review.csv` and fill two columns per frame:

- **`countable`** — `yes` / `partial` / `no`. Could you count the surfers?
  - `yes` — the whole frame is countable
  - `partial` — part is obscured but the rest gives a real count (a floor)
  - `no` — not countable at all
- **`unusable_x_range`** — only when `partial`. Roughly which horizontal span is
  unusable, in eighths left to right, e.g. `1-3` for the left ~40%. This is the
  field that makes a *regional* rule possible instead of a frame-level one.

Images are `01.jpg` … `57.jpg`, 1280x180. The frames are **shuffled**, so their
order tells you nothing — that is deliberate.

## Why there are frames here that look fine

About half this batch is expected to be perfectly countable. Those are controls
and they are not optional. Three separate attempts to build an automatic flag
have now failed the same way: a measure is tuned on a handful of known-bad
frames, looks perfect, and then rejects countable frames in the full corpus.

- The 2026-09-19 glare gate was withdrawn for discarding countable crowded frames.
- A slab-uniformity measure appeared to catch the known bad frame 2026-10-02
  07:29, but inspecting its slab profile showed it scored the *obscured* half
  HIGHER than the clean half — it was firing on the glare, not the obscuration,
  and it rejected a countable 29-surfer frame.
- Measures fitted on 42 hand-reviewed frames rejected two 47-surfer packed
  lineups that are plainly countable.

Without frames that score high on the candidate measure and are *fine*, any
threshold fitted here will only separate this batch from itself.

## How it was chosen

Every `quality_ok` frame was scored on a luminance ramp across the strip
(max/min of the 8 vertical slab means) — the one candidate with a physical
basis (low sun in the lens, or contamination on it) that ranked both known-bad
frames in the top decile. The batch is then stratified by that score:

| ramp decile | frames | role |
|---|---:|---|
| 9 (highest) | 27 | where the bad frames live |
| 8 | 8 | near-miss controls |
| 4-5 | 12 | ordinary controls |
| 0-1 (lowest) | 10 | clean controls |

43 of the 57 hold 5 or more surfers — those are where a wrong verdict actually
costs the dataset, in both directions.

## The one group worth doing first

Several frames come from a 2025-11/12 winter low-sun cluster around 07:47-08:50
with the highest ramp scores in the corpus, nearly all counting 0. If those
turn out to be countable, the luminance-ramp route is dead and roughly ten
labels will have told us. Do those first.

## Then

`python code/score_reviews.py` does not read this batch. Hand it back and the
threshold work happens from these labels plus the corpus scores.
