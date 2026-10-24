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
