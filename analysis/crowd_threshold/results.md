# Crowd-dependent threshold and crowd-aware duplicate suppression

**Measured 2026-10-18.** Both rules are **built, tested and OFF by default** in
`code/detect_surfers.py`. Two independent truth protocols say adopt them; a third
says don't; and the reason the three disagree turns out to be the most useful
thing here.

Reproduce:

```bash
python analysis/crowd_threshold/measure_boxes_at_floor.py        # one inference pass, cached
python analysis/crowd_threshold/score_crowd_rules.py             # thresholds + containment, box truth
python analysis/crowd_threshold/score_duplicate_rules.py         # duplicates vs real neighbours
python analysis/crowd_threshold/score_count_schedule.py          # count-indexed schedules
python analysis/crowd_threshold/score_containment_on_counts.py   # containment on hand counts
python analysis/crowd_threshold/score_combined.py                # both rules, all three protocols
```

## The overlapping-box problem is the opposite of what it looks like

Joel's report: too many overlapping boxes when several surfers are close
together. On the 61 held-out labelled frames, production keeps 1,394 boxes of
which 183 are false positives, and the anatomy of those false positives is:

| | boxes |
|---|---:|
| kept boxes stacked on another kept box (IoU ≥ 0.3) | 22 of 1,394 (1.6%) |
| false positives that are stacked | 10 of 183 (5.5%) |
| false positives that are isolated | **173 of 183 (94.5%)** |

**Duplicates are 10 boxes out of 1,394.** And crowding is not where false
positives come from — the FP rate is flat against local density (0.129 at zero
neighbours, 0.108 / 0.149 / 0.124, 0.158 at five or more). What is happening in
the clusters is the reverse: containment suppression is deleting real adjacent
surfers. Scoring `CONTAINMENT_THRESH` against box-level truth is monotone in one
direction on both train and held-out frames:

| containment | precision | recall | mean \|count err\| |
|---|---:|---:|---:|
| off | 0.864 | **0.811** | **2.082** |
| 0.9 | 0.865 | 0.804 | 2.213 |
| **0.7 (production)** | 0.869 | 0.803 | 2.311 |
| 0.5 | 0.874 | 0.799 | 2.377 |
| 0.4 | 0.874 | 0.797 | 2.443 |

This is exactly what Adaptive NMS (Liu et al., CVPR 2019) predicts: in a crowd,
heavy overlap is evidence of two people rather than one found twice, so the
suppression threshold should rise with local density. The density-adaptive
variant confirms the direction — `0.7 + 0.10 × neighbours` within 40px lifts
held-out recall 0.803 → 0.807 and cuts count error 2.311 → 2.148.

The perception was real; the diagnosis wasn't. Suppressing harder was the
instinct, and `-0.05/neighbour` makes everything worse on every set.

## The labelled set cannot test a crowd-dependent threshold

Its emptiest quartile still averages **8.3 surfers** — people annotate the
interesting frames — so the schedules that tighten the sparse end come out
bit-identical to production there. The 263 hand-counted frames do have a sparse
end (emptiest quartile: mean 1.1), and
`analysis/conf_thresh_recheck/counts_by_threshold.csv` already holds each
frame's count at each threshold, so a count-indexed schedule is pure replay.

It works as designed. Mean |count error| by crowd quartile, pooled:

| rule | Q1 (truth 1.1) | Q2 (9.5) | Q3 (21.0) | Q4 (41.0) |
|---|---:|---:|---:|---:|
| flat 0.195 (production) | **0.296** | 1.016 | 1.970 | 2.689 |
| flat 0.150 | 0.352 | 1.000 | 1.836 | 2.443 |
| ramp ≥10→.17 ≥20→.15 ≥30→.12 | **0.296** | 0.984 | 1.896 | 2.361 |

The flat 0.150 cut the earlier re-check rejected costs **+0.056** on the emptiest
quartile. The ramp costs **nothing** there — below a count of 10 it *is* 0.195 —
and still takes 0.33 off the busiest quartile. That was the whole objection to a
lower threshold, and indexing removes it.

## Both rules together, on all three truth protocols

Schedule `[(0,.195),(10,.17),(20,.15),(30,.12)]` with containment slope 0.10:

| truth | n | \|err\| before → after | Δ | 95% CI | p | ratio to truth |
|---|---:|---|---:|---|---:|---|
| box-level CVAT, held out | 61 | 2.31 → **1.85** | −0.459 | −0.738 to −0.213 | **0.002** | 0.924 → 0.953 |
| `reviewdays` hand counts | 220 | 1.50 → **1.30** | −0.191 | −0.336 to −0.050 | **0.010** | 0.926 → 0.964 |
| `spotcheck` hand counts | 43 | 1.23 → 1.58 | **+0.349** | +0.047 to +0.651 | 0.030 | 1.021 → 1.056 |
| emptiest 57 frames (count 0–2) | 57 | 0.228 → 0.246 | +0.018 | — | — | — |

Held-out precision goes 0.869 → 0.852, recall 0.803 → 0.812.

## Why the three protocols disagree — and a correction

`analysis/conf_thresh_recheck/` treated `spotcheck` as the set a change must
clear, because it is the only one independent of the review the 0.150 floor came
from. **That reasoning missed a third protocol.** Box-level CVAT annotation is
independent of both count sets, and at matched crowd levels it agrees with
`reviewdays` band for band while `spotcheck` sits apart. All ratios below are
plain count/truth — no IoU matching enters:

| true count | box-level CVAT | `reviewdays` | `spotcheck` |
|---|---:|---:|---:|
| 6–15 | 0.900 | 0.910 | 1.105 |
| 16–25 | 0.950 | 0.932 | 1.023 |
| 26–40 | 0.903 | 0.901 | 0.962 |
| 41+ | 0.934 | 0.953 | 1.013 |
| all | **0.924** | **0.926** | **1.021** |

Two protocols that share no frames and no method agree that the detector keeps
about 92.5% of the surfers. The third says it keeps 102%.

### The gap is measurable directly, and it is `spotcheck` that moves

**Added 2026-10-18, after the above.** 15 labelled frames are also in
`spotcheck` and 11 in `reviewdays`, so each count protocol can be compared
against CVAT box truth on the *same frames* — no review needed. (Train
contamination is irrelevant here: this compares two human protocols, not the
model.)

| | mean gap vs CVAT | n | below / above | sign p | Wilcoxon p |
|---|---:|---:|---|---:|---:|
| `reviewdays` | **+0.27** | 11 | 5 / 5 | 1.00 | 1.00 |
| `spotcheck` | **−3.27** | 15 | **14 / 0** | **0.0001** | **0.0009** |

The direct count reads **−14.4% relative**, and the bias is flat rather than
crowd-dependent (r = −0.08, p = 0.77). Correcting `spotcheck`'s truth for the
measured 0.888 ratio **reverses its verdict on every rule**:

| `spotcheck` truth | production ratio | both rules: Δ\|err\| | p |
|---|---:|---:|---:|
| as recorded | 1.021 | **+0.349** | 0.029 |
| corrected ×1/0.888 | **0.907** | **−0.502** | **0.000** |

Corrected, its production ratio (0.907) lands with CVAT (0.924) and `reviewdays`
(0.926), and all three protocols endorse the rules. **The entire disagreement
that blocked three rule families is one protocol reading about 11% low.**

This supersedes the two-frame hint below, and reverses its direction: it is not
that the box-by-box protocol reads high, it is that the direct count reads low —
consistent with the known tendency of direct enumeration to undercount dense
scenes.

The earlier re-check also explained the gap as a difference in crowd mix. **That
explanation is wrong** and is corrected here: the two sets' crowd mixes are
nearly identical, and if anything `reviewdays` is the emptier one (23% of its
frames hold ≤2 surfers against `spotcheck`'s 14%; 22% vs 21% hold 30+).

What survives as the live hypothesis is the protocol confound: a box-by-box
review that credits every "missed" surfer reads higher than a direct eyeball
count of the same scene. Only **2 frames appear in both count sets**, and on both
the box-by-box truth is higher — **+2 and +1, mean +1.5**, against a set-level gap
of 1.65. Two frames is a hint, not a finding, but it points the right way and is
the right size, and direct enumeration is known to undercount dense scenes.

## Where this leaves it

- **Both rules are implemented and inert.** `CROWD_THRESH_SCHEDULE = None` and
  `CONTAINMENT_DENSITY_SLOPE = 0.0`. Tonight's run is unchanged. Turning them on
  is a two-line edit; 5 tests pin the off-by-default behaviour, the schedule
  lookup, the density count, and the property that matters most — a box with no
  neighbour is scored at exactly `CONTAINMENT_THRESH`, so the slope cannot touch
  the isolated clear-day false positives containment was adopted for.
- **`detect_surfers.py` now has one detection path.** `run_inference()` wraps
  `run_inference_with_boxes()` instead of duplicating its body. The duplication
  was live: the two copies had to be kept in step by hand, and a re-detect written
  against the wrong one is what produced the 71 phantom upward count changes
  recorded in `docs/PROJECT_HISTORY.md` for 2026-10-18. Verified behaviour-identical
  — recounting 20 already-recounted rows changes nothing.
- **Not adopted, and the blocker is the truth data, not the instrument.** Three
  rule families have now been measured — a flat threshold, a count-indexed
  threshold, and density-adaptive containment — and all three split the same way:
  `reviewdays` and box-level truth gain, `spotcheck` loses. That is no longer a
  property of any one rule.
- **The remaining question is narrower than it was.** The gap is now measured
  (−3.27, p = 0.0001) and all three protocols agree once it is removed. What is
  still open is which protocol is *right*: CVAT and `reviewdays` share a mode of
  looking — slow, magnified, box-assisted — so the 11% is either real surfers a
  glance misses or marginal blobs close looking manufactures. Every frame with a
  careful count was counted carefully, so no existing data separates those.
- **`data/reviews/protocol_review/` settles it**: 30 fresh frames counted twice
  by the same reviewer, direct count first and unprimed, box-by-box second, on
  frames never reviewed before, stratified across five crowd levels. Built by
  `code/make_protocol_review.py`; the protocol is in that folder's
  `counting-protocol-review.md`. n = 30 detects 0.53 × SD ≈ 1.32 surfers against
  an effect of ~3.3.
