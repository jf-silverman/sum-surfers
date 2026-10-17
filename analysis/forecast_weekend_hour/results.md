# Weekend × hour: real in the counts, worth nothing to the forecast

Run: `python analysis/forecast_weekend_hour/test_weekend_hour.py --origins 101 --seeds 6`
1,943 rows / 146 days, 5-day test blocks.

## Answer: no

| arm | MAE | vs base | p | better | placebo seeds as good |
|---|---:|---:|---:|---:|---:|
| base | 7.044 | — | — | — | — |
| `wx_trig` | 7.054 | +0.010 | 0.78 | 56/101 | 4/6 |
| `wx_raw` | 7.056 | +0.012 | 0.73 | 46/101 | 4/6 |
| `wx_work` | 7.073 | +0.029 | 0.27 | 43/101 | **6/6** |
| `dow_onehot` | 7.076 | +0.032 | 0.31 | 41/101 | **6/6** |

No arm beats the add-a-column null (6 seeds: mean +0.008, sd 0.011), and
`wx_work` does not beat the fake-weekend control either (6/6 as good). Nothing
adopted. `dow_onehot` reproduces the MODEL_CARD's existing rejection on the
same origins.

**The shape is not in doubt** — the weekday/weekend medians this run prints
show it at every hour from 08:00 to 17:00, and the box plot shows it more
sharply still under the tide filter. The learner already has it. Trees
represent interactions natively by splitting on one feature inside a branch of
another, so an explicit product column is a hint, not new information.

## The actual finding is the run order

| | 16 origins | 30 origins | 101 origins |
|---|---:|---:|---:|
| base | 6.887 | 6.693 | 7.044 |
| `wx_trig` | **−0.063** | +0.075 | +0.010 |
| `wx_raw` | **−0.080** | +0.046 | +0.012 |
| `wx_work` | **−0.084** | +0.101 | +0.029 |
| placebo seeds as good as `wx_work` | **0/8** | **12/12** | 6/6 |

At 16 origins every interaction arm improved and cleared both controls
outright. At 30 — the same cut list, more of it — every arm was worse, and
`wx_work` was significantly worse at p = 0.040. Nothing changed but how many
origins were scored.

**The per-origin delta has a standard deviation of 0.27–0.36 MAE.** The
standard error of the mean is that over √n:

```
  16 origins   se ~ 0.090
  30 origins   se ~ 0.065
 101 origins   se ~ 0.030
```

So −0.084 and +0.101 are both inside one standard error of zero, and of each
other. **Sixteen rolling origins cannot resolve an effect of 0.1 MAE** — it
resolves about 0.25. Resolving 0.1 takes roughly 80 origins.

## What this means for the other tests

This project has used 16 origins as its default: the tide-residual arms
(+0.033 to +0.280) and the target-rounding arms (+0.066 to +0.453) were both
decided there.

**No published conclusion changes.** Those runs concluded NO, and a null that
survives a noisy test is still a null — noise cannot manufacture a null, only
destroy a real effect, and both had placebo controls pointing the same way.

**The magnitudes do carry a wider error bar than they were written with**,
about ±0.09 at 16 origins. Where a number was quoted as the size of an effect
rather than as evidence of its absence, read it that way.

**Going forward: 16 origins is not enough for anything claiming under about
0.25 MAE.** Use the full cut list, which is free — this project's entire record
gives 101 origins, and the 101-origin run took about fifteen minutes.

The script now prints the per-origin sd and the standard error of the mean on
every arm, so the resolution limit is visible in the output rather than
something that has to be noticed.
