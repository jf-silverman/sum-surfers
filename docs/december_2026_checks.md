# Early December 2026 Checks

Written 2026-11-01. Everything parked with "recheck later" in the project
documents that comes due around the start of December, with the baseline to
compare against and how to rerun it. Scripts marked *(local)* are kept out of
git.

## Before December

- **Surfline access token.** The last one lasted from about 2026-09-30 to
  10-30, and the new one went in on 10-30, so expect a 401 around **11-29**.
  The pipeline emails an auth-failure warning when it happens. Refresh it then
  and rerun the pipeline the same night: the camera keeps five days.
- **Cloudflare blocks.** Every Surfline request is now logged
  (`data/logs/surfline_requests.csv`) and the evening email flags any 403s.
  When the next block happens, line it up against that night's request
  sequence (step, endpoint, rate) to find the trigger.
- **Thanksgiving (11-26) and the long weekend.** Nothing to do on the day; it
  adds holiday rows for the holiday check below.

## Forecast changes made in October, up for review

### 1. Kelvin-wave tide correction (0.75 ft gate), now in shadow

**Moved to shadow on 2026-11-02**, after its first full day doubled the miss
(11.5 vs 5.9 average miss). The published forecast uses Surfline's tide; the
corrected forecast is recorded beside it as `shadow_tide_corr_predicted` in
`forecast_<date>.csv` and `week_<date>.csv` whenever the gate is on. In
December, score the two against the same actuals on every gated day; the
details below were written before the switch.


Adopted 2026-11-01 for a recheck at the start of December. When the previous
day's mean non-tidal residual is above 0.75 ft, the model's tide input is the
Surfline tide plus that residual (`tide_correction_ft`).

- **Baseline:** −0.038 MAE overall (p 0.80), −0.152 on the four October 2026
  windows (better in 2 of 4).
- **Risk to look for:** it cuts forecasts sharply while on (11-02: 11 am 16
  against 25 without it). If surfers follow the Surfline tide chart, as the
  2026-10-15 test suggested, gated days will under-forecast.
- **How:** rerun `analysis/forecast_tide_residual/test_october_tide.py`
  *(local)*, adding November. Also score the gated days in `forecast_log.csv`
  against a rerun of the same days without the correction, and check whether
  the gate was on at all in November (`tide_correction_ft > 0` in
  `data/training_features.csv`). The 2026-11-02 forecast was redone after a
  cache defect (see the 2026-11-02 history entries) and can be included; its
  wind and swell directions came from the 11-02 morning fetch.

### 2. Detrended residual feature (`tide_resid_dev_prior_ft`)

Adopted 2026-10-30 despite a worse backtest; `CLAUDE.md` gives about 12-30,
but it shares data with item 1, so take an interim look then.

- **Baseline:** +0.188 MAE (p 0.52, better in 9 of 16). Almost all of it is
  the 10-09 to 10-13 window, where the feature went past its training range
  (+4.30 there; −0.086 across the other 15).
- **How:** `analysis/forecast_tide_residual/test_tide_residual.py` and
  `residual_windows.py`. Look at whether a November pulse repeated the
  10-09 pattern (forecasts halved while the feature was above about 0.5 ft).

### 3. Three-look hour count in the evening email

Adopted for the email only on 2026-11-01; training, `forecast_log.csv` and
the README still score the hourly clip.

- **Baseline (9 days):** 37% above the hourly count, +8.5 surfers at 15+; a
  model retrained on it scored 6.46 against 6.47 for the current model times a
  flat ratio, so so far it is a level shift.
- **How:** rerun `analysis/hour_max_view/test_three_look_target.py` *(local)*
  on about five weeks of between-hours sets. Deciding whether it should become
  the training target needs hand counts over whole busy hours, which do not
  exist yet.

### 3b. Conditions at the clip's minute (done 2026-11-03)

Tested early: tide at the clip's minute is worse (+0.13), the nearest hour
worse still (+0.38), and the best fit is the tide about 30 minutes before the
clip (−0.10, not significant), which lands near the start of the hour the
model already uses. Nothing to do; see the 2026-11-03 history entry.

### 3c. Two ideas from the 2026-11-02 miss (Joel)

- **The good-tide line moves with swell size.** It broke above Surfline's 3 ft
  tide on a day with nearshore energy 639 kJ (2–5 times the week before),
  17 s period and 4 ft surf. Test a tide feature measured against an
  energy-dependent threshold (or tide x energy terms) on the 16 origins. Big
  days are rare in the record (3 days at that energy), so pre-register the
  form and expect a noisy answer.
- **Pent-up demand.** After a weekend of low energy at high tide, a good
  Monday drew 34 at peak. Test days since the last good session (rated fair
  or better with a sub-3 ft daylight window), or the previous two days'
  best rating, as day-level features. Earlier count lags were rejected; this
  one lags conditions, not counts, so it can serve the 7-day outlook.

### 3d. A rain correction (from the 2026-11-03 miss)

The model ignores rating, wind and rain (too few examples; monotonic
constraints did not help). In the live record, rain hours came in at 0.35 of
the forecast, but on 7 hours over 3 days. Once winter adds rain hours, fit a
multiplier for hours whose evening-before weather is rain or showers (the
forecast inputs had showers for 11-03's 10 am–1 pm), forward in time and
against the 16-origin baseline; adopt only with a clear gain. Leave general
"bad conditions" alone unless the record turns: before 11-03, POOR, rain or
10+ mph wind hours were under-forecast (ratio 1.15).

**Add strong wind (2026-11-13).** 11-12 was empty all day (0–1 surfers, POOR,
14–17 mph wind every afternoon hour, no rain) while every version of the
forecast said about 20–30 from 2 pm; frames were clear and confirmed empty by
eye. Before 11-02 the record had no hour at 15+ mph; now 11-03 and 11-12 both
supply empty ones. Test a correction for POOR rating with wind of about 14+
mph alongside the rain one, same method, and check it would not have cut the
busy 10+ mph afternoons (those ran at 1.07 of forecast).

### 3e. Daylight remaining (Joel, 2026-11-09)

After the switch to standard time the crowd peak settled around 3-4 pm. It
follows the low tide, but last light caps it: in late October, with lows at
5-6 pm and last light near 6:40, peaks came 1-2 hours before the low; from
11-05 to 11-09, lows at 2-4 pm and last light at 5:31, the peak sat on the
low-tide hour. The forecast's peak ran early on the weekend (1 pm vs 2:14 on
11-07, 2 pm vs 3:08 on 11-08). The model has clock time and tide but nothing
saying how much light is left.

- **Test:** add "minutes of daylight remaining at this hour" (last light minus
  the hour; `get_clips.get_light_window`), and optionally "minutes from the
  day's low tide to last light", on the 16 origins with a placebo. Score the
  hourly MAE and, as in known_bugs L11, peak-hour timing, since an earlier
  within-day test helped timing without moving MAE.
- **School:** UCSC and Cabrillo dates are now in
  `data/predictor_vars/academic_calendar.csv`; this week's weekday peaks fell
  after school hours anyway, so school cannot be separated from tide yet.

## Pre-set statistical looks

### 4. Top-end forecast correction (hinge above 20)

Joel's rule: no adoption until the per-date Wilcoxon against the production
straight line reaches p < 0.05, with one pre-set look at 90 scored dates.

- **Interim at 60 dates (about 11-29):** counts only if **p < 0.01**.
- **Main look at 90 dates:** about **12-29**. Do not rerun nightly and stop at
  the first p < 0.05.
- **Baseline (25 dates):** hinge better on 16 of 25, p = 0.18.
- **Caution:** counts moved onto the median/max scale on 2026-10-26. Restrict
  to dates after that, or recompute old actuals on one scale first.
- **How:** `analysis/forecast_recalibration/test_top_end_correction.py`.

### 5. Year as a predictor

Inert until October 2026 because month already stood in for year. Its first
real use is forecasting November and December, where every same-month
training row is from 2025.

- **Baseline:** −0.002 MAE; identical forecasts in 15 of 16 windows.
- **How:** rerun `analysis/forecast_year_term/test_year_term.py` *(local)*
  once November is in. Joel planned to look toward the end of the year; early
  December is the first point with four weeks of November.

## Data that should have built up by December

### 6. Prediction-band coverage by lead time

`calibrate_lead_bands.py` fits its widening factors partly on bands whose
lower edge sat at 0 (44% of rows, 97% coverage for free). The note says rerun
once a season of two-sided bands exists. By early December the hurdle-fitted
bands (2026-10-20) will have about six weeks scored.

- **How:** rerun `code/calibrate_lead_bands.py`; report coverage by lead
  against the band's nominal level, split by whether the lower edge is above 0.
- **Paired idea, only if coverage has recovered:** shrinking far-lead point
  forecasts toward a (month, hour, weekend) climatology
  (`docs/model_and_feature_ideas.md`, "the condition for revisiting").

### 7. Holiday and long-weekend features

The wider holiday list (adding Indigenous Peoples' Day) was the only lead to
reach p < 0.05 (MAE 6.737 to 6.662, p = 0.026), on two counted Mondays, chosen
after seeing the misses. Veterans Day (11-11) and Thanksgiving weekend give it
fresh, unchosen days.

- **How:** the packed-hours holiday test (`analysis/packed_hours/`), scored
  only on the new holiday days so the result is not the same days that
  suggested it.

## Labelling and data work (Joel)

### 8. Hand counts as the scoring reference

Paused on Joel scoring nine generated review days (127 frames, 2,386 boxes),
six of them seasonal (2025-10-25, 2025-11-04, 2025-12-02, 2026-03-12,
2026-07-25, 2026-08-16).

- **Then:** rerun the paired t-test (forecast vs detector counts against
  forecast vs hand counts; last p = 0.097) and refit the detector undercount
  slope (0.865 at 138 frames). Correcting counts before they enter training is
  a production change and needs Joel's go-ahead.
- Hand counts over whole busy hours would also settle item 3.

### 9. Standing-pose set

Batches 05 and 06 (10 riding clips) are in CVAT. Target for a usable
train/test set: about 350–400 standing boxes from 30–40 distinct clips over
10+ days, with 80–100 standing in a held-out test set.

- **How:** once 05 and 06 are exported, count standing boxes per clip to
  replace the 8–15 per clip estimate, then set how many more clips to collect.

### 10. Non-surfer label

Joel is boxing occasional non-surfers (a flying bird on 2026-11-01). Use a
separate CVAT label (`non_surfer`, with a `kind` attribute), not a pose.
Before the next split build, `build_stratified_splits.py` needs a step that
handles it: today it refuses exports whose label sets differ, and would
otherwise turn the detector into a two-class model.

## Not yet: wait past December

- **Year-over-year crowd trend (known_bugs L10).** Needs October to December of
  both years paired; revisit in January.
- **Day-level swell features (ideas doc).** Promising, unproven at 16 origins;
  retest the swell terms only, pre-registered, when the record is longer.
- **School-week hints.** The Wednesday 3 pm early-release bump (+3.3, p = 0.15,
  14 Wednesdays) and the midday dip after school starts; recheck as the record
  grows.
- **Wave and foam detection.** Parked until Joel is ready
  (`analysis/wave_foam_detection/wave_foam_detection_research.md`).
