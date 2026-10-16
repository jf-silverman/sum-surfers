# Fog-gate review — 23 frames

The gate rejects a frame as `foggy_or_blurred` below **lap_var 12.7**. That one
number decides whether an hour is counted at all. These are the frames closest to
it, in both directions.

Open `fog_gate_review.csv`. One row per image. Four columns to fill:

| column | what to put |
|---|---|
| `surfers_visible` | `yes` / `no` / a count if you can count them / `unsure` |
| `fog_blocks_view` | `yes` / `no` / `partly` — is fog the reason, or is the water just empty? |
| `gate_was_right` | `yes` / `no` — should this frame have been counted? |
| `notes` | anything else |

## `near_pass/` — 5 frames

Passed the gate, among the day's first 3 images, lap_var under
25.0, and **counted as zero surfers**. The question: is that zero real?

A zero at first light looks unremarkable — the surf often *is* empty then — which
is what makes it the easiest wrong value to miss. If fog is producing these zeros,
the forecast is training on them at the hour when crowds start building.

## `near_reject/` — 18 frames

Failed the gate for fog, but only just (lap_var at or above
10.0, against the 12.7 gate). **No detection was ever run on
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
