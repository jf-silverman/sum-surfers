# SAHI vs. the hand-rolled 4-tile split

**Measured 2026-09-29. Outcome: no change made — the hand-rolled path is not
beaten, and is faster.**

[SAHI](https://github.com/obss/sahi) (5.5k stars) is a maintained library that
does what `detect_surfers.py` does in about sixty lines: slice a large image,
run a detector on each slice, merge the boxes back. It was worth asking whether
the library is better than the hand-rolled version, and the survey in
`model_and_feature_ideas.md` listed it as worth benchmarking precisely because
either answer is useful.

## Method

Both paths use the **same weights and the same confidence threshold** (0.195), so
the only variable is slicing and merging. SAHI was given the same slice geometry
the project uses: 376 × 180 at 20% width overlap.

Scored on the **32 whole frames in the production detector's test split** — frames
it never trained on, recovered from the tile names in
`splits_tiled_20260921_fog` — against the whole-frame CVAT labels in `splits_v2`
(497 labelled surfers). Whole-frame labels rather than summed tile labels,
because summing across tiles double-counts anything in an overlap.

The metric is **count accuracy**, not box placement. This project answers "how
crowded is it", so a box that lands on the right surfer counts that surfer
whether or not it hugs the outline.

## Result

| path | MAE | bias | total vs 497 | speed |
|---|---|---|---|---|
| **hand-rolled 4 tiles** | **0.94** | −0.44 | 483 (97.2%) | **0.16 s/frame** |
| SAHI sliced | 1.28 | **+0.16** | 502 (101.0%) | 0.20 s/frame |
| SAHI + this project's post-processing | 1.66 | −1.03 | 464 (93.4%) | 0.20 s/frame |

SAHI is worse per frame by **+0.34 MAE** — closer on 4 frames, worse on 11, tied
on 17, paired t-test **p = 0.094**. That is not significant at n = 32, so the
honest reading is *no evidence SAHI is better*, not *SAHI is proven worse*.

## The two things worth keeping from this

**SAHI has the better bias.** It lands at 101.0% of the true total against the
hand-rolled 97.2%, and its bias is +0.16 against −0.44. So it finds slightly
more of the real surfers and the errors cancel better in aggregate — it is worse
per frame but better summed. If this project's question were "how many
surfer-hours this month" rather than "how crowded is it right now", that
ordering could flip.

**Stacking the post-processing backfires.** Adding this project's containment
suppression and false-positive zones to SAHI's output made it clearly worse
(MAE 1.28 → 1.66, total 101.0% → 93.4%). SAHI already merges overlapping boxes
itself, so the containment pass removes genuinely adjacent surfers on top of
that. Two reasonable de-duplication steps compose into over-suppression. Worth
remembering before bolting any second merging stage onto this pipeline.

## Why no change was made

The hand-rolled path is directionally more accurate, 25% faster, has no new
dependency, and carries the false-positive zones and containment suppression
that were each tuned on this camera's own failure cases. SAHI is not in
`docs/requirements.txt` — it was installed only to run this benchmark.

Reproduce with:

```bash
pip install sahi
python analysis/sahi_vs_hand_tiling/benchmark_sahi.py
```
