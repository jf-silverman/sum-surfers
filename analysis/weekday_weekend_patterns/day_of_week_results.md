# Day of week: the weekend split is the whole effect

Run: `python analysis/weekday_weekend_patterns/test_day_of_week.py`
738 counted hours over 127 days, forecast tide below 3 ft.

## Answer

**Weekend vs weekday is real and large. Nothing distinguishes one weekday from
another, and nothing distinguishes Saturday from Sunday.**

| test | p | effect |
|---|---:|---|
| weekend vs weekday | **0.00003** | rank-biserial **+0.46**, median +6.2 surfers/hour |
| all 7 days | 0.0030 | driven entirely by the two weekend days |
| **Mon–Fri alone** | **0.64** | — |
| Sat vs Sun | 0.70 | — |
| Mon–Fri, 20,000 label shuffles | 0.65 | agrees with the asymptotic p |

All ten weekday pairs come back at Holm-corrected **p = 1.000**. The largest
adjusted median gap between any two weekdays is 3.7 surfers (Tue vs Wed),
against a day-to-day standard deviation of 8.2.

Adjusted day medians, in surfers above or below a typical hour:

```
Mon -1.0   Tue +0.6   Wed -3.1   Thu -2.0   Fri  0.0   Sat +7.0   Sun +4.7
```

The Wednesday dip that `plot_count_by_weekday.py` warns about is **−3.1** here
and does not survive correction (uncorrected p = 0.22). That warning was right
and can now be stated as a measurement rather than a suspicion.

## Why the test is built the way it is

**The unit of analysis is the day, not the hour.** Hours inside one day share a
tide, a swell, a wind and a sky, so 738 hours are nowhere near 738 independent
observations. Testing at the hour level is the easiest way to manufacture a
weekday effect out of nothing — it would return p-values orders of magnitude
too small. Everything collapses to one number per date (127 of them) first.

**Two adjustments, in order.** Each count has its hour's median subtracted, so
a day sampled mostly at 06:00 is not punished for it. Then each day's value has
its calendar month's median subtracted — the record covers nine months unevenly
with four missing, and the month mix differs by weekday (Sunday draws 5 of 21
days from 2025-11, Wednesday 1 of 15). Without that, season leaks straight into
the comparison.

Adjustment moves the answer in the expected direction and does not change it:
unadjusted, Mon–Fri gives p = 0.87; adjusted, p = 0.64.

Tide is handled by the filter rather than by adjustment. Inside 0–3 ft the
response is nearly flat, so the residual tide difference between weekdays is
worth about two surfers.

## What this does not rule out

**The smallest weekday-to-weekday difference this record could detect at 80%
power is about 8.4 surfers** — day-to-day sd 8.2, and only 15 Wednesdays. That
is larger than the weekend effect itself (6.2). The weekend effect is
detectable only because it pools 39 days against 88.

So the finding is: **no large day-to-day differences exist among weekdays.** A
genuine 3–4 surfer Friday-evening effect, say, would be invisible here, and
nothing in this test argues against one. Revisit after another winter — at
roughly 30 days per weekday the detectable difference falls to about 6.

## For the forecast

Nothing to change, and the two lines of evidence now agree.

The live model carries only the binary `is_weekend`. A seven-level alternative
exists as a candidate — `build_dow_features()` in `code/forecast_features.py`,
one-hots plus cyclical plus a `dow x hour` interaction — and day-of-week
one-hots are already on the MODEL_CARD's measured-and-rejected list, tested on
rolling origins.

That was a forecasting result: adding the feature did not improve MAE. This is
the reason underneath it. There is nothing for those five extra levels to
encode, so the rejection was not a power problem in the backtest — the signal
is absent in the counts themselves. The `weekend_x_hour` interaction in the
same builder is a separate question this does not test; the per-hour weekday /
weekend chart is where to look for it.
