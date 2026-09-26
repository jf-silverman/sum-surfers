# Works Cited

Academic literature consulted for this project, with what each one actually
contributes to it. Gathered 2026-09-26 while looking for prior work on
predicting how crowded a surf break will be.

**The short version: no published work predicts surfer counts at a break from
conditions.** The surf-crowding literature is sociological and economic rather
than predictive, and the closest quantitative analogue comes from general beach
attendance rather than surfing. That gap is the reason this project had to
derive its own predictors by measurement rather than adopt an established set.

---

## Directly analogous: forecasting attendance from camera counts

**Machine Learning Beach Attendance Forecast Modelling from Automatic
Video-Derived Counting.** *Journal of Marine Science and Engineering*, 13(6),
1181 (2025). https://doi.org/10.3390/jmse13061181

The nearest thing in print to what this project does: beach attendance at
Biscarrosse Beach counted automatically from video through summer 2023, then
forecast with XGBoost from weather, wave, tide and time inputs. Reported R²
0.97 and RMSE 70.4 users against counts ranging 0–2031, with hour-of-day and
daily mean air temperature the two most influential variables — those two alone
reaching R² 0.90.

Two findings bear on this project:

1. **Daily-mean weather outperformed instantaneous values**, the authors'
   explanation being that beach users plan around forecasts rather than
   reacting to the conditions of the moment. This project's model feeds the
   forecast *the current hour's* temperature and wind, and carries almost no
   day-level aggregates beyond `good_tide_hours` / `_frac` / `_hours_left`.
2. **Air temperature ranking that highly** is consistent with what was measured
   here independently on 2026-09-25: within Oct–Mar mornings, emptiness tracks
   temperature at r = −0.252 (p = 0.005), monotonically from 62.5% empty below
   48 °F to 12.5% above 60 °F.

*Caveat on this entry:* the publisher and the HAL mirror both refused automated
access, so the summary above rests on the abstract and indexed excerpts, not a
full-text read. Worth reading properly before leaning on it.

## Surf crowding: why it is studied, but not predicted

**The institutional foundations of surf break governance in Atlantic Europe.**
*Public Choice* (2021). https://link.springer.com/article/10.1007/s11127-021-00929-3

Over seven hundred surf spots on the European Atlantic coast. Treats crowding
as a commons problem managed by informal norms rather than formal institutions.
Relevant here for one transferable idea: surfers **substitute across breaks** —
a spot draws fewer people when better conditions are available elsewhere. This
project models Jack's in complete isolation, with no relative or competing-spot
predictor of any kind.

**Camaraderie, common pool congestion, and the optimal size of surf gangs.**
*Economics of Governance* (2018).
https://link.springer.com/article/10.1007/s10101-018-0211-6

Models congestion at a break against the social benefits of a local group.
Establishes that break quality attracts surfers and that crowding is
self-limiting past some point — a mechanism that would show up in count data as
saturation, which is worth remembering when reading the upper tail of this
project's counts.

**Managing Stoke: Crowding, Conflicts, and Coping Among Virginia Beach
Surfers.** *Journal of Park and Recreation Administration*.
https://js.sagamorepub.com/index.php/jpra/article/view/7596

Perceived crowding and how surfers respond to it. Descriptive rather than
predictive; included as evidence of what the surfing literature does and does
not cover.

## Recreation demand and seasonality

**Climate Influences on Day and Overnight Use at California State Beaches and
Coastal Parks.** *Land*, 14(2), 324. https://www.mdpi.com/2073-445X/14/2/324

California-specific, which matters since the beach-attendance work above is
French Atlantic. Weather-to-visitation relationships at state beaches and
coastal parks.

**Nonlinear dynamics of surfing participation, flow, and enjoyment.**
*Frontiers in Sports and Active Living* (2026).
https://pmc.ncbi.nlm.nih.gov/articles/PMC13318669/

Participation frequency and its relationship to experience, from self-report
rather than observation. Notes seasonal variation in how often surfers go out.
A reminder that the population generating these counts is not fixed across the
year.

---

Tool and library references — Ultralytics, CVAT — are in
[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md#main-resources) rather than here.
