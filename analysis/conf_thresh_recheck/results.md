# The clear-day overcount re-check — `CONF_THRESH` stays at 0.195

**Measured 2026-10-17.** 263 frames with a hand count, re-detected at a 0.02
floor and replayed through the full production path (NMS, containment
suppression, and the static zones as they stand today, bluff zone included) at
each candidate threshold. Reproduce with
`python analysis/conf_thresh_recheck/recheck_threshold.py`.

## Verdict: no

`analysis/conf_band_decision/` measured a 0.150 floor on 113 frames and said the
decision waited on this re-check. **The gate is not cleared.** The two sets
disagree in sign at every threshold, and the one that is independent of the band
review moves the wrong way.

| set | n | mean hand count | signed error at 0.195 | ratio to truth |
|---|---:|---:|---:|---:|
| `reviewdays` — the 16 hand-reviewed days, truth = `true_positives + missed` | 220 | 17.6 | −1.30 | 0.926 |
| `spotcheck` — `model_spotcheck_50`, truth = `my_count` | 43 | 16.6 | **+0.35** | **1.021** |

Change in mean |count error| against 0.195:

| T | reviewdays | p | spotcheck | p | pooled | p |
|---|---:|---:|---:|---:|---:|---:|
| 0.180 | −0.055 | 0.039 | +0.000 | 1.000 | −0.046 | 0.051 |
| 0.170 | −0.091 | 0.003 | +0.000 | 1.000 | −0.076 | 0.003 |
| 0.160 | −0.095 | 0.007 | +0.070 | 0.402 | −0.068 | 0.027 |
| **0.150** | **−0.123** | **0.005** | **+0.140** | **0.095** | **−0.080** | **0.038** |
| 0.140 | −0.155 | 0.003 | +0.163 | 0.100 | −0.103 | 0.029 |
| 0.100 | −0.218 | 0.002 | **+0.279** | **0.045** | −0.137 | 0.048 |

The pooled column is significant, but it is 220 frames outvoting 43, and the 43
are the only ones not drawn from the same review the 0.150 floor came from. On
the independent set the move is **+0.140 at 0.150 and +0.279 at 0.100**, with the
overcount ratio going 1.021 → 1.038 → 1.055. A threshold change has to clear the
set it is most likely to damage, and this one does not.

## What the threshold is actually trading

Split the 263 frames by how crowded they truly were, at T = 0.150:

| quartile | n | mean hand count | signed at 0.195 | signed at 0.150 | Δ\|err\| |
|---|---:|---:|---:|---:|---:|
| emptiest | 71 | 1.1 | +0.07 | +0.13 | **+0.056** |
| q2 | 64 | 9.5 | −0.58 | −0.41 | −0.016 |
| q3 | 67 | 21.0 | −1.37 | −0.88 | −0.134 |
| busiest | 61 | 41.0 | −2.43 | −1.89 | **−0.246** |

The same split by direction of the current error: **37 of 263 frames already
overcount at 0.195**, and on those 0.150 adds **+0.270**; on the 106
undercounting frames it takes off **−0.453**.

So the threshold is not a dial on accuracy, it is a dial that **trades
empty-frame precision for crowded-frame recall**. A single global constant is
the wrong instrument for an error whose sign depends on the crowd.

> **Corrected 2026-10-18.** This section originally added that the two sets are
> explained by differing in how much of their mass sits in near-empty frames.
> That is wrong. Their crowd mixes are nearly identical, and `reviewdays` is if
> anything the emptier one (23% of its frames hold ≤2 surfers against
> `spotcheck`'s 14%; 22% vs 21% hold 30+), so crowd mix cannot account for a
> disagreement in sign. The live explanation is the truth-protocol confound named
> below, which `analysis/crowd_threshold/results.md` takes further: a third and
> independent protocol, box-level CVAT annotation, agrees with `reviewdays` band
> for band at matched crowd levels (0.924 vs 0.926 overall) while `spotcheck`
> sits at 1.021. The reasoning in this file that `spotcheck` is "the set a change
> must clear" rests on it being the only independent set, and it is not.

## One confound worth naming

The two sets' truth columns come from different protocols. `reviewdays` truth is
`true_positives + missed` from a box-by-box review; `spotcheck` truth is a direct
`my_count` with its own `my_ci_plusminus` column. A box-by-box protocol that
credits every "missed" surfer will read systematically higher than a direct
count of the scene, which would manufacture part of the −1.30 against +0.35 gap.
Nothing here separates that from a real difference between the sets, so the
−1.30 should not be quoted as the detector's undercount without the caveat.

## What this does and does not settle

- **`CONF_THRESH` stays 0.195.** Not adopted at 0.150 or anywhere below.
- **The band review's measurement stands.** 49.2% of the boxes between 0.10 and
  0.195 are surfers the count is missing. That was never in question; what this
  adds is that admitting them costs more on empty frames than it pays on full
  ones, once a set that contains empty frames is in the sample.
- **The 95 band false positives stay out of production**, excluded by the
  threshold alone — including the 18 tree boughs and 9 shore heads in that set.
  Any future case for a lower threshold has to carry zones for those first.
- **A crowd-dependent threshold is the direction this points at**, not a lower
  constant. **Built and measured 2026-10-18** — see
  `analysis/crowd_threshold/results.md`. It does remove the empty-frame cost
  entirely (the emptiest quartile stays at production's 0.296, against 0.352 for
  a flat 0.150) and still gains on the crowded end. It is implemented and OFF,
  because it splits the truth protocols the same way every other rule does.
