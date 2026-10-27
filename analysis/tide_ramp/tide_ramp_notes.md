# Does the crowd arrive later after the tide drops below 3 ft this fall?

Run 2026-10-27. Joel saw the actual ramp-up after the tide drops below 3 ft
coming later than forecast in recent weeks, and asked whether the fall Kelvin
wave (actual water about a foot above the tide table) is delaying it, and
whether the peak is higher.

## Method

One event per counted day on which the tide table (NOAA's harmonic prediction
at Monterey, which Surfline's tide tracks at r = 0.985) falls through 3.0 ft
with at least 4 h of daylight left. 72 events. Crowd timing is measured from
the clip counts from 1 h before to 4 h after the crossing:

- **centre**: the count-weighted mean time, the most stable measure;
- **peak lag**: time of the highest count (coarse; one clip an hour);
- **half-rise**: first clip halfway from the pre-crossing count to the peak.

Controls in every regression: the crossing's time of day, weekend/holiday, and
how long the table stays below 3 ft. Water-level data: NOAA 9413450, updated
through 2026-10-27 into a scratch copy (the repo's `tide_residual.csv` stops
at 2026-10-15). Results: `tide_drop_events.csv`.

| period | events | water above table (ft) | real water crosses later by (h, median) | centre (h after table crossing) | peak lag (h) | peak count |
|---|---:|---:|---:|---:|---:|---:|
| fall 2025 (Oct 11-Nov 30) | 16 | 0.25 | 0.67 | 2.06 | 2.20 | 31.0 |
| winter 25-26 | 4 | 0.32 | 0.69 | 1.94 | 1.64 | 17.3 |
| spring 2026 | 6 | 0.22 | 0.69 | 1.94 | 2.26 | 37.2 |
| summer 2026 | 24 | 0.48 | 1.19 | 1.92 | 2.33 | 37.5 |
| fall 2026 (Sep 20-Oct 26) | 22 | 0.95 | 1.04 | 2.08 | 2.35 | 39.8 |

(Means for centre, peak lag and peak count; medians for the rest.)

## Findings

- **Yes, the crowd arrives later when the water runs high, but by minutes,
  not the full delay.** With controls, the crowd's centre moves 0.31 h
  (about 19 min) later per foot of water above the table (p = 0.006, 72
  events). This fall against all other periods: +0.18 h, about 11 min
  (p = 0.03). Fall 2026 against fall 2025 alone: +0.25 h (p = 0.07, 38
  events). Peak lag and half-rise don't show it; with one clip an hour they
  are too coarse.
- **People land between the tide table and the real water.** The real water
  crosses 3 ft about an hour after the table this fall. Measured from the real
  crossing, the crowd now comes *earlier* the higher the water (-0.29 h per
  ft, p = 0.03). So surfers follow the actual water partly: roughly a third
  of the way from the table's time to the real one. That fits the 2026-10-15
  result that swapping in the real water level made the forecast worse.
- **On 6 of the 22 fall days the real water never got below 3 ft**, while the
  table said it did for 1 to 6 hours. In fall 2025 that never happened.
- **The logged forecasts agree.** On the 20 fall events with logged forecasts,
  the actual crowd centre is 2.15 h after the table crossing against the
  forecast's 1.92 h: 14 min later (paired Wilcoxon, p = 0.0002).
- **A higher peak is not shown.** Fall 2026 peaks average 39.8 against 31.0 in
  fall 2025, but with controls the difference is +5.5 (p = 0.15). The "one
  window" idea doesn't hold up either: the peak doesn't rise when the real
  low-tide window is shorter (-0.2 surfers per hour, p = 0.65). The whole day's
  mean does rise with more real low-tide daylight (+0.6 per hour, p = 0.03).
  The peak shortfall in the logged forecasts (40 actual against 28 forecast)
  is the same top-end shortfall seen all year.

## Caveats

- 72 events, 22 of them this fall, and the residual climbs across the record,
  so water level is partly confounded with season and recency.
- Timing is at one clip an hour. The sub-hourly frame sets (from 2026-10-26)
  will give three samples an hour for re-checking this in a few weeks.

## A possible next test (not run)

A partial correction to the tide feature, the table tide plus about a third
of the observed residual. The full correction hurt on 2026-10-15; this result
suggests the partial one is closer to how people behave. The model would need
the residual at forecast time, which NOAA does not forecast; persisting the
last observed value is the obvious stand-in.

## Weekdays against weekends and holidays (2026-10-27)

Joel's idea: weekday surfers follow conditions like the Kelvin wave more
closely, while weekend surfers glance at the tide chart. If so, the delay
should show on weekdays only. 53 weekday events, 19 weekend/holiday; the water
ran about as high on both (0.53 and 0.58 ft above the table on average).

| | events | delay per ft of extra water | 95% CI | p |
|---|---:|---:|---|---:|
| weekdays | 53 | +0.27 h | -0.01 to +0.56 | 0.06 |
| weekends and holidays | 19 | +0.23 h | -0.15 to +0.62 | 0.23 |
| difference (weekend minus weekday) | | +0.11 h | -0.34 to +0.55 | 0.64 |

- **Both groups shift by about the same amount**, about 15 min per foot. The
  weekend estimate is less certain because it rests on fewer days. Nothing
  says weekend surfers ignore the water level.
- Comparing this fall with the rest of the record instead, the delay sits on
  weekends: +0.06 h on weekdays (p = 0.52), +0.44 h more on weekends
  (p = 0.007). That rests on 6 weekend days this fall and runs opposite to
  the idea, so it is a reason not to read too much into either split.
- In the logged forecasts, the actual crowd centre is later than forecast by
  14 min on weekdays (14 days) and 13 min on weekends (6 days).

The weekday-only idea is not supported; with 19 weekend events, a modest
difference either way can't be ruled out.
