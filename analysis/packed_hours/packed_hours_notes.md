# Packed hours: what the forecast misses, and whether local calendars explain it

Run 2026-10-24. Scripts in this folder; data as of the 2026-10-23 nightly run.

## Method

`packed_hours_oof.py` refits the production point model ten times, each
holding out a tenth of the calendar weeks, so every counted hour from
2025-10-11 to 2026-10-23 (1,956 daylight hours, 152 days) has a prediction
from a fit that never saw its week. These are hindcasts on the recorded
conditions, so a miss is something the inputs can't explain even when the
conditions are known.

## What the packed hours look like

- 317 daylight hours had 30+ surfers, on 104 days. The model put them at 27.3
  on average against 39.5 actual; 191 were missed by 10+.
- Missed packed hours are half weekend (51%, against 32% of all daylight
  hours), mostly August to October, and spread across the day from 7am to
  6pm rather than bunched at one hour.
- On the conditions recorded, missed and well-forecast packed hours look
  alike: rating 2.5 against 2.4, swell 4.5 ft against 4.3 ft, period 13.7 s
  against 14.1 s, local temperature 68 against 69 F, tide 2.3 against 2.0 ft.
  Nothing in the inputs separates them.
- **Most of the shortfall is within the day, not the day as a whole.** Only
  31% of the residual variance is whole-day level, and of the 12.2-surfer
  average shortfall on packed hours, 3.6 is the day running high and the
  rest is peak hours spiking above the day's shape. Morning and afternoon
  residuals correlate at 0.34; consecutive days at 0.31.
- The two most under-forecast days in the record are holidays: Friday
  2026-07-03 (Independence Day observed; peak 60 against a forecast peak of
  26) and Labor Day 2026-09-07 (peak 54 against 29). Labor Day weekend's
  Friday and Sunday are also in the top six.

## Local calendar and inland heat (`test_calendar_features.py`)

Four day-level candidates added to the production features, same folds:

| arm | MAE | packed bias | long-weekend MAE | weeks better | p |
|---|---:|---:|---:|---:|---:|
| production features | 6.330 | -12.20 | 10.46 | | |
| + holiday, long weekend | 6.278 | -12.12 | 10.00 | 17/29 | 0.18 |
| + school out (local districts) | 6.358 | -12.14 | 10.62 | 12/29 | 0.33 |
| + inland heat (San Jose max) | 6.383 | -11.91 | 10.62 | 11/29 | 0.21 |
| + all four | 6.320 | -12.18 | 10.08 | 16/29 | 0.44 |

Holidays and long weekends help a little and in the right place (long-weekend
MAE 10.46 to 10.00) but rest on 7 long-weekend days and are not significant.
School breaks add nothing: outside summer, which month already carries, the
record has only 27 school-out weekdays. Inland heat adds nothing. None of them
moves the packed-hour shortfall.

## Sources (all regional, all current as of 2026-10-24)

- **School calendars**: Live Oak School District (elementary) and Santa Cruz
  City Schools (secondary), 2025-26 and 2026-27, the official board-approved
  PDFs linked from the Santa Cruz County Office of Education
  (https://santacruzcoe.org/schools/school-district-calendars/). Transcribed
  into `local_school_calendar.csv`. Live Oak is the district Pleasure Point
  sits in.
- **Holidays**: the `holidays` package (0.103), California subdivision,
  limited to the days most workplaces close (no Columbus Day, Susan B.
  Anthony Day or Diwali).
- **Inland heat**: Open-Meteo archive, daily maximum at San Jose
  (37.34, -121.89), saved as `san_jose_tmax.json`.

## High-school dismissal and Cabrillo's semester (`test_school_hours.py`)

Weekdays only, holidays excluded.

- **Schools and distances** (straight line from the camera, geocoded with
  OpenStreetMap): Cypress High 1.5 km (2039 Merrill St; County Office of
  Education alternative school), Harbor High 4.0 km, Soquel High 4.2 km,
  Delta High 5.3 km (on the Cabrillo campus, about 114 students).
- **Dismissal times.** Harbor and Soquel let out at 3:35 pm Mon/Tue/Thu/Fri
  and 2:10 pm Wednesday (Harbor's schedule is the same in 2024-25 and
  2026-27; Soquel's 2026-27 schedule matches). Cypress ends classes at
  1:25 pm with an optional tutorial to 2:20, and at 1:10 pm Wednesday
  (2026-27). Delta runs two groups: morning students leave at 12:10 pm
  (11:25 am Friday), afternoon students at 3:55 pm (1:25 pm Friday); its
  posted schedule is undated. Cypress's own calendar was not checked; the
  SCCS student days stand in for it. Santa Cruz City Schools' first student days were 2025-08-07
  and 2026-08-06. Cabrillo's fall semester began 2025-08-25 and 2026-08-24,
  and spring 2026 ran Jan 26 to May 23 (break Mar 23-28). No official
  Cabrillo meeting-time grid was found; sample Aptos sections run MW/TTh
  1:00-2:20 pm, with some to 3:05 or 5:35 pm.
- **The school-day shape is real and the model already has it.** On
  school weekdays the crowd builds through the afternoon (about 9 at 9-11am
  to 16-17 from 2-4pm); on non-school weekdays it peaks mid-morning (22 at
  10am). The residuals by hour are near zero for both, because month and hour
  together already encode it.
- **School start, 2026-08-06** (7 weekdays before, 8 after): 8-11am fell from
  22.5 to 18.2, 12-3pm from 15.4 to 9.9, and 4-6pm rose from 13.0 to 16.3.
  The model's 4-6pm shortfall grew slightly (+1.2 to +2.6). Few days, and the
  morning tide shifted a little, so suggestive only. Cabrillo's start
  (2026-08-24) shows no clear change.
- **Wednesday early release.** In school weeks the 3pm hour on Wednesday runs
  3.3 surfers above the other weekdays (residual +5.1, 14 Wednesdays, p =
  0.15). That is the right direction for a 2:10 dismissal; there is only one
  non-school Wednesday to compare against.
- **Cypress's early release.** On school weekdays the mean count jumps
  from 11.7 at 1pm to 15.9 at 2pm, the hour after Cypress lets out, but the
  model's residual there is +0.2, so the hour already carries it. On
  Wednesdays, when Cypress lets out at 1:10, the 1pm hour runs 1.5 above the
  other weekdays (residual +2.3, p = 0.33, 17 Wednesdays): not distinguishable
  from noise.
- **As model features** none helps (week-grouped folds, workday 12pm+
  MAE in brackets): production 6.330 (5.91); high school 6.367 (5.98);
  Cypress out 6.393 (6.01); Delta morning group out 6.410 (5.94); Cabrillo
  6.380 (5.98); all five 6.375 (5.98).

Sources: Harbor High bell schedules (https://www.hh.sccs.net/schedule),
Soquel High 2026-27 bell schedule (linked from
https://www.soquel.sccs.net/bell-schedule), Cypress High 2026-27 bell
schedule (https://cypress.santacruzcoe.org/bell-schedule-2021), Delta High
class period schedule (linked from
http://www.deltaschool.org/calendar--bell-schedule.html),
Cabrillo Dates and Deadlines for fall 2025, spring 2026 and fall 2026
(https://www.cabrillo.edu/admissions-and-records/dates-and-deadlines/).

## Local events from the papers and the visitor bureau (`fetch_visit_sc_events.py`)

Checked Lookout Santa Cruz (weekend roundups and the BOLO calendar), Good
Times, the Santa Cruz Sentinel, Patch, Eventbrite and County Parks. The papers
publish listings as articles, with no feed and no back catalogue to backtest
against. Visit Santa Cruz County (santacruz.org) does have one: its public
events API returns every listing from 2025-07 onward (3,616 events through
2026-12, saved as `visit_sc_events.csv`, one row per event-day).

- **Events near the break, and the day's mean residual** (actual minus
  hindcast): Wharf to Wharf 2026-07-26 −6.1; Capitola Beach Festival
  2026-09-26 +6.2; Capitola Art & Wine 2026-09-12 +1.5; Aloha Outrigger
  2026-09-19 +0.2. The Pleasure Point Street Fair (2026-06-27), the Women on
  Waves contest at Capitola (2025-10-18) and Surfin' Santa (2025-11-29) fall
  on days with no counts.
- **As a count of one-off festivals, special and sporting events** in Santa
  Cruz, Capitola, Aptos, Soquel and Live Oak (titles listed on more than five
  dates dropped, as recurring): weekend days with an event miss by +1.5,
  without by +2.2 (Spearman −0.08, p = 0.59, 49 weekend days); weekdays
  −0.7 against +0.3 (p = 0.53). No sign that events add surfers beyond what
  the model expects, so no model test was run.
- **In-water events at Pleasure Point need a County Parks surf contest
  permit**, capped at two non-consecutive days a month. That permit calendar
  is the authoritative list of contest days at the camera, but it is not
  published; it would need a request to County Parks reservations. Contests
  likely lower the free-surf count (heats clear the lineup), so they matter
  more as a data-quality flag than as a crowd driver.

## 2026-10-10 and Columbus Day

Columbus Day / Indigenous Peoples' Day fell on Monday 2026-10-12. It is a
federal holiday only: the `holidays` package does not list it for California,
Santa Cruz City Schools were in session that Monday (their October days off
were Oct 15-16), and Live Oak's were Oct 19-20. So the long weekend reaches
federal workers and banks, not most families.

- **Saturday 2026-10-10 is a tide day.** Counts were 17 at 7am, then zero from
  9:26 to 13:02 with the tide at 4.7-5.5 ft (the frames show a glassy, empty
  lineup, so the zeros are real), then 38, 49, 64, 47 and 43 from 2:50 to
  6:30pm as the tide dropped to 0.1 ft. The hindcast had the shape (2-3 at
  midday, about 37 in the afternoon) and missed the size of the afternoon peak:
  +26 at 4pm, a mean of +12 from 2 to 6pm. That ranks 3rd of 22 Saturday
  afternoons; the two larger misses (2025-10-25, 2026-09-26) had no
  holiday.
- **The holiday weekend shows no extra crowd.** Day-mean residuals for Sat,
  Sun and Mon were +2.8, +1.3 and +1.0 in 2026, against +1.9, +1.7 and +1.3
  on the average Saturday, Sunday and Monday. The 2025 Columbus weekend
  (Oct 11-13) ran +5.7, +3.6, +3.7, but on mean counts of 6-10.
- The same feed lists nothing near Pleasure Point that Saturday (T-Birds on
  the Santa Cruz Wharf, the Seacliff street fair in Aptos, Open Studios). The
  Capitola Wharf fireworks were Sunday evening.

Sources: Visit Santa Cruz County events API
(https://www.santacruz.org/wp-json/tribe/events/v1/events), County Parks surf
contest permits (https://parks.santacruzcountyca.gov/Home/PermitsReservations/Surfing.aspx),
Lookout Santa Cruz (https://lookout.co/arts-entertainment).

## What explains the packed-hour gap? Three candidates tested (2026-10-25)

Baseline for these tests is 6.401: the nightly run on 2026-10-25 added a day
to the training table, so it is not comparable with the 6.33-6.34 above.

**Binned by the hindcast, the model is calibrated.** When it says 30-40 the
hours average 32.3; at 40+ they average 42.3. The packed-hour shortfall comes
from selecting hours by their outcome: packed hours mostly arrive on hours the
model put at 20-30. About 1.7 of the 12.2 is measurement noise (one frame
against the hour's mean, sd 3.66, from the three per-second hours) favoured by
that selection.

- **Conditions improving since yesterday** (`test_conditions_change.py`):
  rating change 6.338 (15/29 weeks, p = 0.31; beats 6 of 8 placebos, −1.0 sd);
  swell and period change, Surfline surf height change and a new-swell flag
  (period up 2 s and height up; 9 days) no better or worse. The early
  correlation (Spearman +0.21 on day means) does not hold up. Not a cause.
- **Nearby breaks** (`fetch_nearby_ratings.py`, `test_nearby_ratings.py`):
  Surfline's hourly ratings for Steamer Lane, Pleasure Point, The Hook,
  Capitola and Manresa, 380 days each. The convergence idea predicts busier
  hours when the alternatives are worse; the data say the opposite. The
  residual rises with Steamer Lane's rating (Spearman +0.09, p < 0.001), so the
  Lane is acting as a second reading of how good the swell is in town, not as
  a competitor. Jack's-minus-best-alternative carries nothing (−0.03). Missed
  and caught packed hours have the same nearby ratings (best alternative 3.39
  against 3.33). As inputs: Steamer Lane alone 6.278 (−0.12; 15/29 weeks,
  p = 0.09; beats all 8 placebos, −4.6 sd); all four columns 6.290 (21/29
  weeks, p = 0.035, the first weekly-significant input in these tests). The
  packed shortfall barely moves (−12.0 to −11.7 to −11.9).

## Max of the three frames instead of their mean (`max_frame_results.csv`, 2026-10-26)

The three frames per hour come from one clip, 1.5 s apart (1.0, 2.5 and
4.0 s), so a surfer hidden behind a wave in one frame is often visible in the
next. The mean counts that surfer as a fraction of a person; the max counts
them once.

**Against Joel's hand counts** (218 primary frames, truth = true positives +
missed), the max is the closest of the three at every crowd size above 5:

| Truth | Primary frame | Mean of 3 | Max of 3 |
|---|---|---|---|
| 0-5 (67) | +0.03 | +0.12 | +0.53 |
| 6-15 (48) | −0.96 | −1.00 | −0.28 |
| 16-30 (59) | −1.90 | −1.79 | −0.71 |
| 31+ (46) | −2.70 | −2.76 | −0.98 |
| Overall MAE | 1.49 | 1.59 | 1.41 |

The hand count is of one frame, so it also misses anyone hidden in that frame.
The true undercount from averaging is, if anything, larger.

**As the training target** (production GBT, same folds), the max-trained
forecast is a scaled copy of the mean-trained one (correlation 0.993; daylight
mean 15.81 against 14.93). Against hand counts it is closer: MAE 6.43 against
6.69, bias −3.07 against −4.12, and −11.7 against −13.4 on hand counts of 31+.
On its own target the packed shortfall barely moves: −11.5 (−28.9%) against
−12.0 (−30.3%), with the spread ratio unchanged at 0.80. Averaging hides about
2 surfers on a busy hour; it is not what leaves the packed hours 12 short.

## First fair day after poorer days (`first_fair_day_results.csv`, 2026-10-26)

Do people wait for "fair" and turn out the first day it appears? A day's
rating is the mean of Jack's hourly Surfline rating from 7am to 6pm, taken from
its full 381-day rating history so every day has a yesterday. "Fair day" means
that mean is 2.5 or more (mostly fair or better); a looser "any fair hour"
version is also shown. Daily mean residual (actual minus the production
hindcast) and raw daily mean count:

| Definition | Days | First fair (n) | Repeat fair (n) | Residual p | Count, first vs repeat |
|---|---|---|---|---|---|
| Mostly fair | all | +2.0 (18) | +1.2 (42) | 0.56 | 15.6 vs 16.4 |
| Mostly fair | weekday | +1.3 (16) | +0.1 (27) | 0.43 | 15.1 vs 13.7 |
| Any fair hour | all | +2.1 (16) | +1.0 (75) | 0.46 | 14.9 vs 16.2 |

First fair days run about 1 surfer further above the forecast than repeat ones,
the direction the idea predicts, but well inside noise. Only 2 first fair days
fell on a weekend, so the weekend case, where it would matter most, cannot be
tested yet. A first_fair_day flag in the GBT changes nothing (6.408 and 6.396
against 6.401; 15/29 weeks, p = 0.95 and 0.90).

The scripts for this section and the two above it (`test_conditions_change.py`,
`fetch_nearby_ratings.py`, `test_nearby_ratings.py`, `test_max_frame.py`,
`test_first_fair_day.py`) and the fetched ratings are kept local; only their
results are committed.

## Five frames one second apart (`five_frames_results.csv`, 2026-10-26)

Four more frames per clip (0.5, 1.5, 3.5 and 4.5 s, with the existing 2.5 s
count) were counted with the production detector for the 2,073 hours whose
clips are still on disk (`extra_frame_counts.csv`); re-extracting the 2.5 s
frame reproduced production's count in 8 of 8 clips checked. 2,054 hours have
all five frames. Every arm below uses the same hours.

Against hand counts of the 2.5 s frame (215 frames), bias by crowd size:

| Truth | Mean of 3 | Max of 3 | Mean of 5 | Max of 5 |
|---|---|---|---|---|
| 0-5 | +0.08 | +0.53 | +0.11 | +0.77 |
| 6-15 | −0.82 | −0.16 | −1.04 | −0.02 |
| 16-30 | −1.88 | −0.76 | −1.76 | −0.12 |
| 31+ | −2.83 | −0.98 | −2.50 | −0.30 |
| Overall MAE | 1.54 | 1.39 | 1.53 | 1.49 |

The max of five removes the busy-frame undercount almost entirely, at the cost
of overcounting near-empty water (+0.77), where a stray false positive in any
of five frames sets the max.

The GBT refitted on each target:

| Target | Daylight mean | Shortfall, own target 30+ | MAE vs hand counts | Bias vs hand counts, 31+ |
|---|---|---|---|---|
| Mean of 3 (production) | 14.97 | −11.5 (−29.0%) | 6.50 | −13.2 |
| Max of 3 | 15.85 | −11.2 (−28.1%) | 6.27 | −12.1 |
| Mean of 5 | 15.00 | −11.7 (−29.4%) | 6.55 | −13.8 |
| Max of 5 | 16.17 | −11.2 (−27.9%) | 6.37 | −11.9 |

All four forecasts are scaled copies of each other (mean of 3 against max of 5,
correlation 0.992), and the spread ratio stays at 0.80. More frames make the
count more accurate; they do not change which hours the model expects to pack.

**Median and hybrids (added the same day).** Median of 3 and median of 5 behave
like the means (−2.96 and −2.48 at 31+). Hybrids read the crowd level from the
median of 5, keep the median below a cut and use the max of 5 at or above it.
Bias against hand counts:

| Truth | Max of 5 | Median <6, max ≥6 | Median <16, max ≥16 | Median <31, max ≥31 |
|---|---|---|---|---|
| 0-5 | +0.77 | +0.06 | +0.05 | +0.05 |
| 6-15 | −0.02 | −0.07 | −0.96 | −0.98 |
| 16-30 | −0.12 | −0.12 | −0.47 | −1.78 |
| 31+ | −0.30 | −0.30 | −0.30 | −0.74 |
| Overall MAE | 1.49 | **1.32** | 1.46 | 1.51 |

The cut at 6 is the best count tested: the median removes the one-frame false
positive that inflates near-empty water, and the max recovers hidden surfers
from 6 up, where the undercount starts (it is proportional, not confined to
crowds). A cut at 31 leaves 6-30 undercounted. The cut was picked on the same
215 frames it is scored on. As training targets the hybrids change little:
cut 6 gives a 30+ shortfall of −11.0 (−27.3%) and MAE vs hand counts 6.27,
against −11.5 (−29.0%) and 6.50 for production.

## Spring 2026 breaks, and the frame-pair review (2026-10-26)

`frame_pairs_review.md` shows crowded hours the hindcast got right next to ones
it forecast 10+ low: five pairs from consecutive hours of one day, and five
from different days matched on the forecast, hour, weekday/weekend and
conditions. Holidays (Labor Day, Independence Day and the like) and their long
weekends appear only in the same-day pairs, so a holiday is only compared with
itself; the tables carry a "Holiday / long weekend" column. School and college
breaks are not treated as holidays: Joel left the 2026 spring breaks
undetermined.

Spring 2026 breaks, from official calendars except where noted:

| School | Dates | Source |
|---|---|---|
| Santa Cruz City Schools (Harbor, Soquel, Cypress) | spring break Apr 6-10; non-student day Fri Mar 13 | district calendar via the County Office of Education (`local_school_calendar.csv`) |
| Live Oak School District | spring break Apr 6-10; non-school day Mon Mar 16 | same |
| Delta Charter (Cabrillo campus) | not found | |
| Cabrillo College | spring break Mar 23-28; campus closed Mar 31 | cabrillo.edu, Spring 2026 dates and deadlines |
| UCSC | instruction ended Mar 13; finals Mar 16-20; spring quarter began Mar 30; César Chávez Day Mar 27 | registrar.ucsc.edu, 2025-26 calendar PDF |
| UC Berkeley | spring recess Mar 23-26, holiday Mar 27 | registrar.berkeley.edu, 2025-26 calendar |
| Stanford | winter quarter ended Mar 20, spring began Mar 30 | approaching.stanford.edu |
| SJSU | spring recess Mar 30-Apr 3 | sjsu.edu registrar |
| Palo Alto USD | Apr 6-10 | aggregators only |
| Scotts Valley USD | Apr 6-10 | aggregator only |
| Fremont Union HSD | Apr 12-16 per two aggregators, Mar 16-20 per one; unresolved | aggregators only |

Mon 2026-03-16, the same-day pair with the biggest miss (48 counted against a
16 hindcast at 12:44), was a Live Oak non-school day and the first day of UCSC
finals, three days after a Santa Cruz City Schools non-student Friday.
Cabrillo held regular classes that week (semester; finals May 18-23, no
entries for Mar 16-20 on its Spring 2026 dates page). The
record has no counted days between 2026-03-17 and 2026-04-20, so the college
and high-school spring breaks themselves cannot be tested on this year's data.

## Lunchtime dip (`lunch_dip_results.csv`, `lunch_dip_profile.csv`, 2026-10-26)

Joel saw peaks around 10am and 2pm in the hourly averages with tides over 3 ft
removed. On weekdays at 3 ft or under the dip is there: 26.0 at 10am, 22.2 at
noon, 20.2 at 1pm, back to 23.6 at 2pm. The hindcast already follows it
(24.0, 22.7, 22.0, 23.7), and its residuals at noon and 1pm are −0.5 and −1.8,
so the model, if anything, puts slightly too many surfers at lunch. The model's
clock is hour_sin/hour_cos, and the GBT splits on both to carve out the dip.
Weekend low-tide hours from 9am to 1pm have under 15 rows each, so the weekend
shape cannot be read at this tide.

As inputs (MAE 6.401 production): a lunch flag for clip times 12:00-13:29
6.410 (13/29 weeks, p = 0.52); hour as an integer 6.396 (p = 0.41); minutes
after midnight 6.346 (17/29, p = 0.26). Lunch hours' bias is +0.18 in
production. Not a cause of the misses; not adopted.

## Swell period and the swell mix (`swell_mix_results.csv`, 2026-10-26)

**Period is already in the model** (`primary_swell_period_s`, the most
powerful train's period). Daylight crowds rise with it: 12.2 at 9 s or under,
12.5 at 9-11 s, 14.5 at 11-13 s, 15.1 at 13-15 s, 19.8 above 15 s. The
residual creeps up with period (−0.1 to +1.2; Spearman +0.04, p = 0.09), so
the model takes most of it. Dropping period leaves MAE 6.423 against 6.401
(worse in 20/29 weeks, p = 0.09); dropping Surfline's consistency input gives
6.347 (better, p = 0.64), and dropping the rating 6.392 (p = 0.77): the rating
overlaps the swell inputs.

**The rest of the mix.** Production keeps one train per hour. Every train
Jack's reports was fetched for 382 days (`fetch_swell_mix.py`, kept local
with `swell_mix.csv`; 1-7 trains an hour, mostly 3-5). Per hour: trains
carrying 10%+ of the power, the top train's share of the power, the
power-weighted spread of directions, the share of power under 10 s, and the
second train's period and height.

None of the clash measures tracks the residual (|Spearman| ≤ 0.04). As inputs,
direction spread is slightly worse (6.418) and the clash trio is no better
(6.405). The one gain is the top train's share of the power, a measure of how
clean the swell is: 6.322 (19/29 weeks, p = 0.18; beats all 8 placebos,
−3.7 sd). That is the same size as Steamer Lane's rating and the low-tide
window. None of these moves the packed-hour shortfall (−11.8 to −12.2). Not
adopted.

## Training only on hours whose frames agree (`frame_agreement_results.csv`, 2026-10-26)

Joel's idea: train on the half of hours with the lowest spread across the
three frames, as cleaner labels. Every arm is scored on all daylight hours of
the held-out weeks, and on the held-out hours in the low-relative-spread half.

| Trained on | MAE, all hours | MAE, clean hours | Bias at actual 30+ |
|---|---|---|---|
| All hours (production) | 6.401 | 6.030 | −12.0 |
| Lowest 50% raw sd | 6.821 (worse, p = 0.002) | 6.521 (worse) | −15.5 |
| Lowest 50% sd / √(mean+1) | 6.675 (p = 0.08) | 6.119 | −12.7 |
| All, weight 1/(1+sd²) | 6.455 | 6.061 | −12.6 |
| All, weight 1/(1+relative sd²) | 6.388 (p = 0.92) | 5.997 (p = 0.57) | −12.1 |

Raw spread grows with the crowd, so the low-sd half keeps a mean of 9.2
surfers and 77 of the 329 packed hours; the model trained on it under-forecasts
crowds more. Scaled by crowd size the half is representative (mean 14.5, 169
packed hours), but halving the data costs more than the cleaner labels gain,
even on the clean hours. Weighting is a wash. The three frames are 1.5 s apart
and correlate at about 0.8, so their spread mostly reflects detection
difficulty (occlusion, chop), not how well the frames sample the hour, and
that part of the label noise is small next to the forecast's error. Not
adopted.

## What the biggest crowded-hour misses share (`profile_big_misses.py`, 2026-10-27)

Hindcasts refreshed on the median/max counts (2,079 hours, 156 days). Crowded =
30+ surfers: 375 hours on 114 days. "Big miss" = the quarter of those with the
largest shortfall (18.9+ surfers under): 94 hours on 44 days. Tests are run on
days (each crowded day's mean shortfall, Spearman), with Benjamini-Hochberg
across 34 traits. **No trait survives the correction (best q = 0.20)**, so the
leads below are patterns to check, not findings. Full table:
`big_miss_profile.csv`; days ranked by total shortfall: `big_miss_days.csv`.

Leads, strongest first:

- **Long weekends.** 25% of big-miss hours fall on a long weekend against 7% of
  the other crowded hours (p = 0.04). Four of the 13 worst days are Labor Day
  weekend (Fri 9/4 to Mon 9/7); 7/3 and Thanksgiving Sunday 11/30 are also in the
  top six.
- **Misses come in runs.** 28 of the 44 big-miss days sit next to another one,
  against 21 if they were scattered at random (p = 0.05). A day's shortfall
  tracks the day before's (rho 0.22, p = 0.03). Something lasts several days
  that the inputs don't carry.
- **Friday and Monday.** The share of crowded hours that are big misses is 35%
  on Friday and 29% on Monday, against 16 to 21% Tuesday to Thursday. The model
  only knows weekend or not.
- **The day's peak.** 45% of big-miss hours are the day's busiest hour, against
  27% of the rest. The forecast flattens the peak, as found above.
- **Late morning and higher tide.** 35% of 9am-12pm crowded hours are big misses,
  against 19-20% after 12pm. Mean tide is 2.5 ft against 2.0 ft (p = 0.01), but on
  busy days the model's drop from 2-3 ft to 3-4 ft (28 to 20) matches the actual
  drop (33 to 23), so tide is probably not the cause.

Not shared (the misses look like the other crowded hours): Jack's rating,
swell height, period and direction, wind, local and inland temperature, cloud,
change in rating or swell from the day before, days since a fair day, nearby
spots' ratings, Visit Santa Cruz events, school days off, and the spread of the
five frames. Because the frame spread is the same, the misses are not a
counting artifact.

**The logged forecasts (Sep 23 to Oct 26, 61 crowded hours) agree.** The 15
worst under-forecasts are all between 2pm and 6pm in October. Eight of them fall
on Friday 10/9 to Monday 10/12, the Indigenous Peoples' Day weekend; that
Monday is not in the holiday list. Two more fall on Thursday 10/15 and Friday
10/16, the two Santa Cruz City Schools non-student days: peaks of 59 and 46
against 27 and 21.
