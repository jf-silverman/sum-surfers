# Project History

A chronological record of how this project was built and how the model's
real-world performance has been investigated. See `README.md` for how to
run things and `PROJECT_FILES.md` for what every file is.

## Chronological summary

### Initial build (Aug – Oct 2025)
- Repo started, early one-off test captures pulled manually from the camera
  (`jacks_20250719`, `jacks_20250807-0811`, `jacks_20250810` — later
  archived, see below).
- CVAT used to annotate surfers on those captures; exported as both COCO
  and YOLO-format datasets, tiled to match the eventual 4-tile inference
  split (`data/cvat_out_coco`, `data/cvat_out_yolo_rebuilt`).
- YOLOv8s trained on that data (`2025-10-13`, "YoloV8s model completed.
  with 86 precision & 82 recall" — weights at
  `data/model_out/20251013/train/runs/detect/train13/weights/best.pt`,
  still the production model as of this writing).
- Automated clip-pull, frame-crop, and ROI-crop pipeline added
  (`code/get_clips.py`, `code/get_cropped_frame.py`), backfilling 5 days
  of crops on first run.

### Cloud VM era (May – Jul 2026)
- Pipeline moved to a hybrid design: clip download stayed local, detection
  ran on a GCP VM, with email alerts and VM storage management added.
- Several fixes along the way: VM dependency install index, output
  directory creation on a fresh VM, clip-request pacing/logging, reminder
  logic simplified to "last successful laptop run," email alerts reverted
  to Gmail SMTP App Password after an issue with the original method.
- Browser-like request headers added to the clip API call to get past
  Cloudflare's bot check on `services.surfline.com` — plain requests were
  getting a 403 challenge regardless of token validity. `CLAUDE.md` added
  as a local-only (gitignored) architecture reference for future sessions.

### Drop GCP, fully local (2026-07-24)
- Detection ran on CPU on the VM anyway, so the cloud hop added ~$5/month
  of cost with no benefit. GCP VM and disk deleted; `local_pipeline.sh`
  rewritten as the sole entry point, running clip download → crop
  extraction → clip-storage check → detection → success timestamp, all on
  the laptop via cron (Tue/Thu, offset 3 minutes after another project's
  wake-the-Mac trigger — later moved to Mon-Fri 18:25 wake time, pipeline
  now at 18:28 Tue/Thu).

### Surfline predictors (2026-07-30)
- Investigated Surfline's public forecast API
  (`services.surfline.com/kbyg/spots/forecasts/*`) via Chrome DevTools;
  found working weather, rating, tide, and wave/swell endpoints for Jack's
  (spotId `5842041f4e65fad6a770880b`) — no accesstoken needed (passing one
  actually 403s the `rating` endpoint).
- `code/get_surf_predictors.py` built and wired into `local_pipeline.sh` as
  step 5/6; writes `data/surfline_predictors.csv`, matched to
  `predictions.csv` rows by filename.
- Investigated the site's "Historical" date-picker for backfilling old
  rows — confirmed it renders real historical data, but not through any
  visible `fetch`/`XHR` call (checked by monkey-patching both in the live
  page). No lightweight historical endpoint found. Deferred rather than
  keep digging; backlog left for holidays and local Santa Cruz events as
  future predictors, and for the user to bring back findings if they
  reverse-engineer the historical mechanism themselves.
- A one-off NOAA-based tide backfill script (`code/backfill_tide.py`,
  cosine interpolation between hi/lo predictions) predates this and is now
  superseded by the Surfline tide endpoint for anything going forward;
  kept on disk for reference, not part of the scheduled pipeline.

### Repo cleanup (2026-08-08)
- Archived the pre-pipeline one-off captures (`jacks_20250719`,
  `jacks_20250807-0811`, `jacks_20250810` — crops, frames, and raw clips)
  and two superseded CVAT export zips into `archive/`, mirroring their
  original paths. Kept the unzipped CVAT training data and the model
  weights in place (same commit date as the model itself).
- `PROJECT_FILES.md` added as a reference for what every file/folder is.

### Model performance review (2026-08-20 onward, current)
Prompted by wanting to know whether the model was "still performing as
well as it was at first."

- **50-image manual spot check** (`data/model_review_50/review_counts.csv`):
  50 crops spread across 50 days, annotated with detection boxes, manually
  counted by the reviewer against the model's count. Found MAE 2.74,
  correlation 0.91, model undercounting on average — and a strong,
  quantified fog effect (MAE 7.0 on foggy-but-usable frames vs. 1.45 on
  clean frames).
- **False-positive investigation**: reviewer's notes repeatedly flagged a
  static tree-bough artifact (far-right edge) and a flag/wind-sock artifact
  (bottom-middle) as false positives. Coordinates pinpointed by re-running
  inference on flagged images; validated against the **original CVAT test
  set** (10 held-out images with real annotated ground truth, recovered
  from `archive/` + `cvat_out_yolo_rebuilt/labels/test`) to confirm the
  fix wouldn't remove real detections.
  - Tree-bough zone: tightened to `x=[1268,1280], y=[58,87]` — clean win,
    test-set MAE 1.90 → 1.80, zero real detections lost.
  - Flag zone: geometry alone wasn't enough to separate the flag from real
    surfers sitting in the same spot; combined with a confidence threshold
    (`conf < 0.5`) to avoid masking real (higher-confidence) detections.
  - Both implemented as a post-NMS filter in `code/detect_surfers.py`
    (`TREE_MASK`, `FLAG_MASK`, `filter_false_positive_zones`).
- **Original-test-set vs. recent-eyeball comparison**: on the same
  pipeline/thresholds, test-set MAE=1.90/corr=0.96 vs. spot-check
  MAE=2.74/corr=0.91 — real but modest degradation, consistent undercounting
  bias both times, small sample size caveat noted.
- **Confidence-threshold experiment**: lowering `CONF_THRESH` barely moved
  counts on genuinely foggy (low-detail) images — the model isn't
  generating candidate boxes there at all, so there's nothing to recover.
  But on the **clear** test-set images, lowering to 0.10 nearly halved MAE
  (1.80 → 0.90) with no overshoot — a separate finding from the fog
  question, not yet acted on.
- **Image-quality metrics**: `brightness`, `contrast`, and `lap_var`
  (Laplacian variance — a standard blur/sharpness proxy) computed per
  image. `lap_var < 30` turned out to predict "unreliable count" (fog,
  condensation, or just smooth/low-texture water) with strong precision —
  zero false flags on good images in the 50-image set.
- **Bright-fog batch** (`data/fog_review/bright_fog/`, 50 images, low
  lap_var): reviewer categorized each as good (>80%)/`<80%`/`na`. Confirmed
  `lap_var` separation cleanly. Also flagged the reviewer's contrast idea
  directly: "we should be able to get [more surfers] if we turn up
  contrast but can't do this when there is water texture."
- **Dark/night batch** (`data/fog_review/dark_night/`, 50 images, low
  brightness): generated, **review in progress** as of this writing.
- **Original test-set/model-benchmark comparison, round 2**: re-ran the
  mask + threshold work against the CVAT test set repeatedly throughout to
  keep validating each change against real ground truth rather than only
  the eyeballed batches.
- **Water-texture metric investigation**: tried a global Sobel-gradient
  mean first — separated a confounded smooth/textured label set cleanly,
  but confounded (all "smooth" examples were also foggy; all "textured"
  examples were also clear). Moved to a **local windowed variance** metric
  (box-filter trick, water-region only, detections masked out) at several
  window sizes (9/15/21/29px) per reviewer's own suggestion that a global
  average would miss localized texture — ranked the one known overshoot
  case correctly, but n=1 was too thin to trust a threshold.
- **CLAHE contrast + sharpening experiment** (`data/fog_review/enhance_batch/`,
  50 low-lap_var images, before/after sheets + CSV): reviewer's full
  results showed a strong, count-weighted win —
  - Original: 108 detected / 426 true surfers across 26 scoreable images,
    **74.9% weighted miss rate**.
  - CLAHE+sharpen: 244 detected, **43.4%** miss rate.
  - CLAHE+sharpen @ conf=0.10: 272 detected, **38.5%** miss rate, small
    overshoot cost (10 extra out of 426 true).
  - The water-texture-predicts-overshoot hypothesis did **not** hold up at
    this larger sample size (the 3 overshoot cases had *low* texture
    scores; the one high-texture case undershot) — dropped as a driver of
    overshoot; the four open leads above look more likely causes.
  - Night images (11 of the 50): confirmed unusable (`true_count=0`
    always), and importantly, **enhancement actively hurts them** —
    hallucinated 23 false detections out of 11 pure-noise frames (41 at
    the low confidence threshold). Confirms these need to be filtered out
    *before* enhancement runs, not fed through it.
  - Label rendering bug caught and fixed mid-review: 4 of the original 50
    sheets had a detection box hidden under the count/date label
    (opaque background box). Fixed by switching to bright-yellow text at
    0.3 alpha with no background, and all 50 sheets regenerated.

- **Dark/night batch, round 1** (`data/fog_review/dark_night/`, 50 images):
  reviewer found the batch's own selection logic (low brightness alone)
  was unreliable — most flagged images turned out to actually be daytime
  ("not night" on 45 of 50). Reviewer identified two better signals for a
  follow-up batch: (1) **bookend timing** — first/last crop of the day are
  the primary suspects, since the pipeline only targets daylight hours;
  (2) **the inverse rule** — a high detection count at high confidence is
  itself proof an image is *not* night, so those should be excluded from
  candidate selection up front.
- **Bookend batch, round 1** (`data/fog_review/bookend_batch/`, first
  version, 20 images): built by sampling first/last-of-day crops directly.
  Reviewer's inverse-rule feedback led to deleting this round entirely and
  rebuilding.
- **Bookend batch, round 2** (same folder, rebuilt): pre-filtered
  candidates using the model's own (masked) detection output — excluded
  any bookend crop with `count>=3 and avg_conf>=0.4` before sampling,
  cutting the candidate pool from 150 to 93. All 20 selected images came
  back `count=0` as expected. Reviewer labeled 6 usable / 14 night; filled
  in directly by request ("y"/"ok visibility" already marked → remaining
  rows set to "no"/"night"). Brightness alone didn't cleanly separate the
  6 from the 14 (heavily overlapping ranges); `rgb_spread` (max-min of
  R/G/B channel means, i.e. color presence) looked like a stronger
  candidate signal but wasn't tested at scale yet at this point.
- **"Shape content" and row/column-profile investigation**: reviewer's own
  hypothesis — night images are unstructured noise, day images (even dim
  ones) show real shapes like wave bands — tested via Canny edge density,
  contour count, denoised contour size, and row/column-collapsed
  brightness-profile "roughness" (Durbin-Watson-style statistic). **None
  of these cleanly separated usable/night at n=20**: Canny edge density
  responds to sensor noise as much as real structure (one night image had
  710 contours, more than any usable image); denoising to filter noise
  was too aggressive and killed real signal in some usable frames too.
- **Literature research**: searched current CV practice for day/night and
  fog/haze detection. Standard fog metric is **FADE** (Fog Aware Density
  Evaluator), which combines variance, sharpness, contrast, entropy, dark
  channel, saturation, and colorfulness into one composite score — not a
  single hand-picked threshold — because no individual feature is reliable
  alone. This matched what was being found empirically and motivated
  combining all labeled batches into one fitted rule rather than continuing
  to search for a single clean metric.
- **Combined dataset + rule fitting**: merged all 4 review batches (169
  unique labeled images: 132 usable / 37 unusable) with brightness,
  saturation, `rgb_spread`, and `lap_var` computed consistently for all.
  - A full 4-feature logistic regression (no `sklearn` — hand-rolled
    gradient descent) got 89.9% train / 87.6% 5-fold CV accuracy, but its
    errors revealed a structural issue: saturation and `rgb_spread` are
    highly correlated (r=0.89, redundant), and brightness's *single*
    linear weight can't correctly represent both failure modes at once,
    since night pushes brightness down while fog pushes it up.
  - A **two-branch rule** — `unusable = (brightness < 75.4) OR (lap_var <
    12.7)`, thresholds grid-searched — matched the full model's accuracy
    (89.9% train / 85.8% 5-fold CV) with two interpretable thresholds
    instead of an opaque weighted sum, and directly encodes the two
    distinct failure modes. **Adopted as the production rule.**
  - Remaining errors are concentrated in a genuinely continuous "moderate
    fog" zone (`lap_var` ~13-28): 10 of 37 truly-unusable images still
    pass (73% recall on "unusable"), mostly `bright_fog` frames just above
    the cutoff. Matches the literature's point that fog severity is
    graded, not binary.
- **Quadrant fog-scoring experiment**: tested whether splitting a day-fog
  image into 4 equal vertical strips and using the worst strip's `lap_var`
  (rather than the whole image) would better match the reviewer's
  usable/drop labels, per the reviewer's hypothesis that partial fog
  should still force a drop. Result: worst-quadrant scoring performed
  *worse* (85.9%/78.8% CV vs. whole-image's 89.9%/82.8% CV on the 99-image
  day-fog subset) — but the reason mattered: several `bright_fog` rows had
  explicit reviewer notes like *"Thick fog on 20% of image; count in
  visible area is good"* rated `>80%` (usable). The reviewer's own past
  judgment was more lenient than the "any bad quadrant = drop" policy they
  actually want for automation (a human can reason "the fog is only over
  open water"; the pipeline can't). Those 5 rows were corrected to `na` in
  `bright_fog_review.csv` to reflect the intended policy. Per-quadrant
  scores for all 99 day-fog images saved to
  `data/fog_review/quadrant_fog_scores.csv` for future use regardless of
  the immediate result, per reviewer's explicit request. A follow-up
  folder (`data/fog_review/quadrant_mismatch/`, 13 images) was built
  showing frames with 1-3 markedly-foggier-than-the-rest quadrants
  (semi-transparent quadrant dividers, red/green per-quadrant score
  labels) for the reviewer to make individual calls on.

### Image-quality gate implementation (2026-08-25)

- The two-branch rule above was implemented as `compute_image_quality()`
  in `code/detect_surfers.py`, run **before** detection (per the
  "exclude first" ordering decided earlier — quality metrics are cheap,
  YOLO inference is not, and skipping avoids ever writing a count for a
  frame that shouldn't have one).
- `data/predictions.csv` migrated to a 9-column schema (added
  `quality_ok`, `quality_reason`, `brightness`, `lap_var`) — all 1238
  pre-existing rows preserved, new columns left genuinely blank for them
  (not backfilled/guessed) since quality wasn't evaluated for those runs.
- Failed frames are still recorded (date/time/filename + quality fields)
  with `surfer_count`/`confidence_avg` left blank — kept for future
  tuning, per the "keep but exclude from scoring" plan, rather than
  deleted or estimated.

**State as of the image-quality-gate implementation:** the tree/flag
exclusion masks and the image-quality gate are both implemented in
`code/detect_surfers.py`. The CLAHE+sharpen enhancement pipeline, the
duplicate-box/small-object/flag-leakage findings, and the two-step
candidate-then-verify idea are investigated but not yet implemented (see
`bugs.md`/`model_and_feature_ideas.md`). The per-quadrant partial-fog policy has data
and a reviewer-confirmed direction but isn't coded yet (see
`model_and_feature_ideas.md`).

### `predictions.csv` retroactive quality backfill + human_count column (2026-08-25)

- Backfilled `quality_ok`/`quality_reason`/`brightness`/`lap_var` for all
  1238 pre-existing rows (all crop files were still on disk, so this only
  needed the cheap brightness/`lap_var` computation, no model
  re-inference). Result: **1085 quality_ok=True, 153 quality_ok=False**
  (129 `dark_or_night`, 24 `foggy_or_blurred`). Existing `surfer_count`
  values were left untouched even on flagged rows — non-destructive, just
  makes the flag available for filtering.
- Added a `human_count` column (blank, for the user to fill in manually
  over time as a running accuracy check) to both `predictions.csv` and
  `detect_surfers.py`'s `CSV_HEADER`.

### Naive baseline check: "predict surfer count at 7:10am tomorrow" (2026-08-25)

Asked directly by the user. Answer: **we don't have a real predictive
model** — everything built so far is detection *on an existing image*,
not a forecast from time/weather/tide alone. As a naive baseline, pulled
102 historical quality-ok rows within 40 minutes of 07:10 local time:
median 3.5, mean 6.9, **stdev 8.3** (nearly as large as the mean) — even
restricting to the same weekday only 8 points, ranging 0-14. Conclusion:
time-of-day alone is a weak predictor; real forecasting would need
swell/tide/weather features, which motivated scoping a real prediction
model next.

### Surfer-count prediction model — scoping (2026-08-25)

Proposed a 4-phase plan (feature set from `surfline_predictors.csv` +
Poisson/negative-binomial regression given surfer counts are non-negative
and skewed, k-fold CV, then a script pulling tomorrow's forecast through
the fitted model). **Immediately blocked**: `surfline_predictors.csv` had
only 59 rows across 4 dates at the time (2026-08-04 to 2026-08-18) —
`get_surf_predictors.py` is forward-looking only and doesn't backfill, so
each scheduled pipeline run just adds whatever "today" happened to be,
patchily. Decided Phase 0 (historical predictor backfill) had to come
first — with it, ~1085 quality-ok counted frames become usable for
training instead of ~54.

### Historical predictor backfill — endpoint discovered via user-provided HAR (2026-08-25)

Earlier automated attempts (fetch/XHR monkey-patching on the live page)
found nothing — the historical view's data fetch wasn't visible that way.
The user captured a HAR file (full network trace, not just fetch/XHR)
while using the site's own "Historical" toggle, which resolved it:

- **Mechanism**: the exact same `services.surfline.com/kbyg/spots/forecasts/*`
  endpoints already in use accept a `start=YYYY-MM-DD` query parameter.
  Anonymous requests are rejected for any `start` before yesterday
  ("Parameters out of bounds" / "start parameter limited to today or
  yesterday for non-premium users"). With an `x-auth-accesstoken` header
  from the user's logged-in, premium browser session, historical dates
  (tested back to 2025-10-15, the start of the whole dataset) return real
  data. Confirmed via a live `requests` call using the token extracted
  from the HAR.
- Also discovered from the same HAR: three previously-unused endpoints —
  `wind` (speed/direction/directionType/gust), `energy`
  (offshore/nearshore wave energy), and `consistency` (nested under
  `consistency.waveCount`, confirmed by direct testing since it didn't
  refire in the HAR capture itself). All three folded into the predictor
  schema per user request.
- **Security handling**: the token is the user's real account credential,
  not a public key. It was read from the user-provided HAR file (which
  the user sent specifically for this purpose), used for one verification
  call, and the temp file holding it was deleted immediately after — never
  printed, logged, or persisted by Claude. `*.har` and `data/external/`
  added to `.gitignore` as a safety net. The two raw HAR files remain on
  disk at `data/external/` (containing the real token) — flagged to the
  user as something to review/delete once no longer needed.

### `code/get_surf_predictors.py` refactor + `code/backfill_historical_predictors.py` (2026-08-25)

- Refactored `get_surf_predictors.py` so the field-extraction logic
  (`merge_into_by_hour`) is a pure function separate from the network
  fetch, shared by both the live (forward-looking, no-auth) script and
  the new historical backfill script — avoids duplicating the
  per-endpoint parsing logic.
- Extended `CSV_HEADER` from 14 to 21 columns: added `wind_speed_mph`,
  `wind_direction_deg`, `wind_direction_type`, `wind_gust_mph`,
  `energy_offshore_kj`, `energy_nearshore_kj`, `consistency_wave_count`.
  `data/surfline_predictors.csv` migrated (59 existing rows preserved,
  new columns blank for them, same additive-migration approach as the
  `predictions.csv` quality-gate migration).
- New `code/backfill_historical_predictors.py` (explicitly **not** part
  of the scheduled pipeline, same convention as `backfill_tide.py`):
  CLI-driven (`--start`/`--end` date range, `--token`/env var/interactive
  prompt, `--min-pause`/`--max-pause` jitter bounds defaulting to 3-35s
  per the user's explicit bot-avoidance request, `--dry-run`). Chunks the
  requested range into ≤14-day windows (margin below the ~16-17 days
  observed working in the site's own historical requests) to minimize
  total request count, matches only against `predictions.csv` rows not
  already covered in the output CSV (safe to run repeatedly/incrementally
  — "a couple dates at a time," per the user's stated usage pattern), and
  writes into the same `data/surfline_predictors.csv` the live script
  uses. Verified end-to-end via `--dry-run` (correct chunking, correct
  target-row matching, correct already-covered skipping, correct
  future-date rejection) — the live-request path is unverified pending
  the user running it with a real token.

### Historical predictor backfill — first live runs (2026-08-25)

User ran `backfill_historical_predictors.py` for real with a token pasted
into `.env` (via `source .env` before running). Two runs: 2025-10-15 to
2025-10-16 (28 rows, single chunk, 7 requests, all matched, 0 unmatched),
then 2025-10-11 to 2025-10-19 (70 rows, single chunk covering the gap plus
the already-done dates, which were correctly skipped). `surfline_predictors.csv`
grew from 59 → 88 → 158 rows. 74 of 85 total dates still need backfilling;
user's stated plan is ~5-10 dates per day until it's fully covered.

### Multi-frame count averaging (2026-08-25)

User proposed averaging 3 counts from frames 1-3 seconds apart within the
same clip, to reduce per-frame noise (occlusion, mid-paddle surfers). Idea
was validated empirically **before** building anything: extracted 3 frames
(1.5s/3.0s/4.5s) from 6 real high-count clips and ran detection on each —
confirmed real, meaningful variance (stdev 0.8-3.3 per clip, one clip
swinging from 58 to 65 across just 3 seconds). This matches the general
project preference (validate against real data before committing to a
pipeline change) and the standing CV-research-literature-check preference,
though here the validation was empirical rather than literature-based.

**Scope discovery mid-build**: user pointed out `data/not_needed_in_repo/surf_clips`
still held far more raw clips than assumed — 1322 `clip.mp4` files across
90 distinct dates (1.8GB), covering **57 of the 85** `predictions.csv`
dates (not just the last few weeks). This meant multi-frame averaging
could be backfilled onto most of the existing dataset, not just applied
going forward — a much bigger scope than initially planned. Asked the user
directly how the backfilled multi-frame data should relate to the
already-published `surfer_count` values (since those had been referenced
by the manual review batches): chose **additive columns, original
`surfer_count` left untouched** — non-destructive, so prior review work
stays valid.

**Implementation**:
- `code/get_cropped_frame.py` — extracts 2 additional "side" frames per
  clip (`FRAME_TIME_SEC ± SIDE_FRAME_OFFSET_SEC` = 1.0s and 4.0s, alongside
  the unchanged 2.5s primary), named `<base>_side1.jpg`/`_side2.jpg`.
  Primary-frame filename/timing is untouched — every quality-gate
  threshold, review batch, and visualization built around that single
  image all keep working without modification.
- `code/detect_surfers.py` — new `side_frame_paths()`/`run_inference_multi()`:
  runs detection on all 3 frames (chronological order: side1, primary,
  side2 → `frame_count_1/2/3`), computes mean/stdev, and sets
  `surfer_count`/`confidence_avg` from the mean going forward. Falls back
  gracefully to single-frame behavior when side frames aren't on disk
  (verified identical output to the old `run_inference()` in that case).
  `CSV_HEADER` extended with `frame_count_1/2/3`, `frame_count_mean`,
  `frame_count_stdev`; `predictions.csv` migrated (1238 rows preserved,
  new columns blank). Also fixed a latent bug caught during this change:
  the crop-file glob (`crop*.jpg`) would have matched the new `_side*.jpg`
  files and tried to process them as independent primary crops — excluded
  explicitly.
- New `code/backfill_multiframe_counts.py` (not part of the scheduled
  pipeline): for rows with `quality_ok=True`, no existing `frame_count_mean`,
  and a clip still on disk, extracts any missing side frames and runs
  `run_inference_multi()`, writing only the new columns — `surfer_count`/
  `confidence_avg` never touched. Writes progress incrementally (every 25
  rows) so a long run can be interrupted safely.
- **Verification**: tested the primary/side naming convention matches
  between the two scripts (exact match confirmed); tested graceful
  single-frame fallback (identical to old behavior); ran a real 16-row
  backfill for 2026-08-16 and confirmed zero rows' `surfer_count`/
  `confidence_avg` changed; ran `get_cropped_frame.py` for real, which
  extracted side frames for essentially all 1322 clips still on disk in
  one pass (2634 side-frame files); ran the real `detect_surfers.py`
  `main()` end-to-end on 84 previously-unprocessed crops from
  2026-07-03 onward — quality gate, multi-frame detection, and CSV writing
  all confirmed working together, with real per-frame variance visible in
  the output (e.g. `frames=58,56,61` on one image) and zero pre-existing
  rows altered.

**State**: multi-frame averaging is live for all new detection going
forward. The 57-date historical backfill (extraction is already
essentially done pipeline-wide; only the detection/aggregation step
remains) has not been run in bulk yet — `backfill_multiframe_counts.py`
is ready and verified on a small sample, but the full run across all
covered dates is still pending.

### Historical predictor backfill — completed via HAR clicking + token script (2026-08-26 to 2026-08-27)

The 85-date historical predictor gap (weather/rating/tide/swell/wind/
energy/consistency) identified earlier is now fully closed —
`data/surfline_predictors.csv` has a matching row for every
`predictions.csv` row, zero remaining gap, verified by direct diff.

- New `code/backfill_predictors_from_har.py`: an alternative to the
  token-based script — parses HAR files exported from clicking through
  the site's own Historical view instead of making live requests itself.
  Key design points, discovered/decided this session:
  - The site's Historical view calls a **different endpoint set** than
    the live forecast script: `rating`, `surf`, `swells`, `wind`,
    `energy`, `tides` (plus unused `sunlight`/`spectra`/`regions-
    conditions`) — never `weather` or `consistency`. `surf`+`swells`
    together carry the same fields the live `wave` endpoint returns in
    one response, so they're combined by timestamp into a synthetic
    `wave` record before reuse `get_surf_predictors.py`'s
    `merge_into_by_hour()`.
  - Each request the site makes returns a multi-day window (~16 days)
    even for a single-day view, but Joel's insight was that only the
    day matching the request's own `start=` param is genuine day-of
    data — later days in that window are effectively a several-days-out
    forecast despite the whole response being labeled "historical". The
    script filters every response down to just that one day
    (`filter_to_offset(..., offset_days=0)`), meaning one HAR click
    covers exactly one date, not sixteen.
  - Added a narrow, explicit fallback: if a target date has no day-of
    data in any HAR, the script falls back to that date's 1-day-out
    record from an adjacent request (`offset_days=1`) rather than being
    skipped — used for exactly one date (2025-11-21, clicked as 11-20)
    where the exact date wasn't directly clickable. Every fallback-
    sourced row is reported by name, both in `--dry-run` and the real
    write, so it's never silent.
  - Caught and fixed a real bug during use: `combine_surf_and_swells`
    initially only iterated `surf` records, silently dropping valid
    `swells` data whenever a request's `surf` call was missing (observed
    for the 2025-11-21 request) — fixed to union timestamps from both
    lists.
  - **Known permanent gap of this method**: `weather_condition`/
    `temperature_f`/`pressure_mb`/`consistency_wave_count` are
    structurally unavailable — the Historical view just never fetches
    them, no matter what's clicked.
- Workflow hiccups worked through live: an empty 0-entry HAR export
  (log cleared before capture), a HAR missing most of its intended date
  range (DevTools "Preserve log" was off, so page reloads during retry
  wiped earlier captures — fixed by turning it on), and real 429
  rate-limit responses from Surfline's own site under repeated
  clicking (confirmed via HAR status-code inspection, not guessed) —
  Preserve Log being on meant the retries survived in the file anyway.
- **Filling the weather/consistency gap for HAR-sourced dates**: rather
  than accept the permanent gap, removed the 389 partial rows (29
  dates) that had come from HAR captures, then re-ran
  `backfill_historical_predictors.py` — which uses the personal-token
  API and does return weather/consistency — in 9 chunks, each ≤10 days
  (one full day-of request per date, `--chunk-days 1`), covering
  2025-10-20 to 2025-10-30, 2025-11-21 to 2025-11-25, 2025-11-30 to
  2025-12-07, and 5 scattered single-day gaps in 2026-07/08. All 9
  chunks completed with zero denials.
- `code/backfill_historical_predictors.py` also hardened this session:
  a 400 or 429 response now raises `RequestDenied` and aborts the
  entire run immediately (writing whatever was fetched before the
  denial), rather than the old behavior of backing off 60s and quietly
  continuing — per Joel's explicit instruction to stop on denial rather
  than keep hammering a rate-limited endpoint.
- Also fixed: `code/get_clips.py` clip-download failures were silently
  accumulating 401s for days before being noticed (the `SURFLINE_
  ACCESS_TOKEN` had gone stale). Added a distinct `AuthError` exception
  (raised only for HTTP 401, separate from other/transient failures)
  and `send_auth_failure_email()`, matching `manage_clips.py`'s
  existing storage-warning email pattern — `local_pipeline.sh` will now
  email automatically any time a scheduled run hits an invalid/expired
  token, with no separate monitoring needed. Verified live (16 real
  401s triggered a real email send).
- Also added a self-reverting one-off clip-duration override
  (`data/.clip_duration_override` in `get_clips.py`'s `resolve_clip_
  duration()`): if present, its contents (an integer, seconds) are used
  for that run only and the file is deleted immediately on read — so
  even a crash mid-run can't leave the override silently active for a
  future run. Used once for a real one-off: a 60-second-clip
  experiment (vs. the standard 5s) to investigate whether longer clips
  would help with frame-edge/wave-occlusion count variability Joel
  observed; confirmed both 60s and 30s clip requests work against
  Surfline's API (actual returned duration ~2s longer than requested,
  same overshoot pattern as the standard 5s clips) before setting the
  override for a real scheduled cron run.

### Surfer-count prediction model — Phase 1 (feature table) and Phase 2 (model fitting) built (2026-08-27)

Continuing the Phase 0-3 modeling plan scoped 2026-08-25 (Phase 0,
predictor backfill, is now done — see above).

- **Phase 1** — new `code/build_training_features.py`: joins
  `predictions.csv` (target `surfer_count`) with `surfer_predictors.csv`
  on `filename`, restricted to `quality_ok=True` rows, adds
  `hour_local`/`day_of_week`/`is_weekend`/`month` derived from the
  timestamp. Writes `data/training_features.csv` (not lossy — every
  matched row is kept even with some blank predictor fields; dropping/
  imputing is left as a Phase 2 decision). After the predictor backfill
  above closed the gap, this produced 1160 rows with only 15 (1.3%,
  all 2026-07-23) still missing `energy_offshore/nearshore_kj`.
- **Phase 2** — new `code/fit_surfer_count_model.py`, comparing three
  model families on an 80/20 held-out split (916 train / 229 test):
  - Poisson GLM: confirmed badly overdispersed (Pearson chi2/df_resid
    = 8.65, should be ~1.0), AIC 11175.
  - Negative binomial GLM (statsmodels' MLE-based `NegativeBinomial`,
    not the GLM family's fixed-alpha=1.0 default, which a convergence
    warning flagged as not actually estimating the dispersion
    parameter — refit properly with a Poisson-warm-started BFGS
    optimizer after standardizing the raw-scale numeric features, which
    fixed a real non-convergence): alpha=0.747, AIC 6574 (much better
    than Poisson, confirming overdispersion was real) — but held-out
    MAE (~9) barely moved vs. Poisson. Cyclical features (hour, month,
    wind/swell direction) are sin/cos-encoded rather than raw degrees.
  - Gradient-boosted trees (`sklearn.HistGradientBoostingRegressor`,
    Poisson loss): clear winner on held-out accuracy — MAE 6.34 vs ~9
    for both GLMs (~30% better), RMSE 9.07 vs ~12. Permutation
    importance: `tide_ft` dominates, followed by time-of-day, wave
    energy (which wasn't significant in the GLM), `is_weekend`.
  - Honest takeaway across all three: MAE 6-9 against a mean count of
    ~15 is a real, usable baseline but not precise — useful for
    directional/relative predictions, not an exact headcount.
- **Uncertainty bands** (in lieu of a full Bayesian refit, after
  discussing the tradeoff with Joel): 80% prediction intervals via GBT
  quantile regression (separate 10th/50th/90th percentile models).
  **Caught and fixed a real bug during this**: the 10th-percentile
  model, with the same hyperparameters as the point model
  (`l2_regularization=1.0`), collapsed to predicting a constant 0 for
  *every single row* on both train and test data — because ~11% of all
  rows are `surfer_count==0`, "always predict 0" already nearly
  minimizes the pinball loss at that quantile, and L2 regularization on
  leaf values pushed the fit the rest of the way into that trivial
  optimum instead of learning real feature-dependent splits. This
  degenerate model still *looked* well-calibrated in aggregate (~82%
  coverage against an 80% target) because a lower bound that's always 0
  can never exclude anything from below — Joel caught this by noticing
  all 5 rows in a demo table had a lower bound of exactly 0, which
  prompted the investigation. Fix: `l2_regularization=0.0` and
  shallower trees (`max_depth=3`) for the quantile models specifically
  (point/median/upper models weren't affected). After the fix, true
  empirical coverage is honestly lower — 66.8% for a nominal 80%
  interval (both tails individually miss more than their 10% target) —
  and pushing to a wider 90% interval (5th/95th) doesn't help: the
  5th-percentile model collapses to constant 0 again regardless of
  tuning, meaning there just isn't enough learnable signal below the
  10th percentile given how zero-inflated this ~1145-row dataset is.
  **`[0.1, 0.9]` with the corrected hyperparameters is the honest
  ceiling for this dataset's size**, not the originally-reported 80%.
- New `code/demo_predictions.py`: shows N random held-out predictions
  (point + 80% range) alongside the actual count and main predictor
  conditions, for eyeballing rather than only trusting aggregate
  metrics — this is what surfaced the quantile-collapse bug above.
- **Caveat found, then corrected same day**: initially reported (based on
  a stale assumption, not verified) that only 91 of 1160 rows (7.8%) had
  multi-frame data available, because "most older clips have since been
  deleted by storage cleanup." Joel caught this was wrong by pointing
  directly at `data/not_needed_in_repo/surf_clips` — clips going back to
  2025-10-11 (and 2025-11-02 onward) were still on disk. Running
  `backfill_multiframe_counts.py --start 2025-10-11 --end 2026-08-27` for
  real (rather than assuming the earlier small-sample run was
  representative) backfilled `frame_count_*` for 756 more rows in well
  under a minute (verified zero non-`frame_count_*` columns changed via
  before/after diff), bringing multi-frame availability to 847/1160
  (73%).
- **Second bug found in the same investigation**: even after that
  backfill, refitting produced byte-identical results to before — because
  `backfill_multiframe_counts.py` deliberately never overwrites
  `surfer_count` itself (the earlier non-destructive design decision),
  so `build_training_features.py` was still reading the stale
  single-frame `surfer_count` as the modeling target for all 1160 rows,
  including the 756 just backfilled. Confirmed 471 rows where
  `surfer_count` disagreed with `round(frame_count_mean)`. Fixed
  `build_training_features.py`'s `build_row()` to prefer
  `round(frame_count_mean)` over `surfer_count` wherever multi-frame data
  exists (added `resolve_target_count()` + a new `used_multiframe` output
  column so it's visible per-row which convention was used), matching
  the same rounding convention `detect_surfers.py`'s live
  `run_inference_multi()` uses.
- **Refit on the actually-corrected target**: GBT RMSE improved 9.07 →
  8.85 (~2.4%), MAE 6.34 → 6.21 — a modest, genuine improvement,
  consistent with the earlier (2026-08-26) small-sample validation
  finding that multi-frame averaging gives a real but not dramatic
  accuracy gain. The GLMs barely moved (Poisson/NegBin MAE ~9 either
  way). Quantile-interval coverage similarly unchanged (~65-67%,
  consistent with the honest-ceiling finding above). Remaining 313 rows
  (27%) genuinely have no clip left on disk (or hit the rare unreadable-
  frame decoder issue) and still use the single-frame count.

**State**: Phase 0-2 of the modeling plan are done. Phase 3 (a small
script that pulls tomorrow's forecast and outputs a live prediction) is
covered further down this file, after two data-integrity bugs found and
fixed the same day (`extract_frame_at()`'s seek bug and the `side2`
mispositioning it caused).

### `extract_frame_at()` seek bug — root-caused and fixed; median-of-5 tested (2026-08-27)

Joel proposed extracting 5 frames per clip and taking the median instead
of averaging 3, as a way to be less swayed by single-frame outliers
(occlusion, false positives). Testing this on the same 43-clip
`model_review_50` review batch used earlier surfaced a much bigger,
previously-misdiagnosed bug in `code/get_cropped_frame.py`'s
`extract_frame_at()`:

- **Not clip corruption, not "near the tail"** (as earlier entries in
  this file assumed for isolated failures like `2026-07-27_19-56` and
  `2026-08-16_18-17`) — it's that `cv2.VideoCapture.set(CAP_PROP_POS_
  FRAMES, ...)` is fundamentally unreliable on these clips when called
  on a freshly-opened capture. 41 of 43 clips failed outright trying to
  seek to a mid-clip frame (t=4.5s) that should have been well within
  a normal ~6.2s/155-frame clip.
- First fix attempt (add a single warm-up `.read()` before seeking)
  fixed seeking to *later* frames but broke seeking to *earlier* ones —
  confirmed via `cap.get(CAP_PROP_POS_FRAMES)` reporting a nonsensical
  `-5.0`/`-4.0` position and silently re-returning the warm-up frame
  for low frame numbers, verified by diffing extracted images
  (`t=0.5`/`1.5`/`2.5` came back byte-for-byte identical).
- **Actual fix**: stopped using `.set()` to seek at all. Sequential
  `.read()` calls were confirmed 100% reliable for every frame in every
  clip tested, so `extract_frame_at()` now reads forward from frame 0 to
  the target frame index rather than seeking. Verified: 0 failures
  across all 43 clips × 5 frame positions (previously 41/43 failed at
  just one position), and consecutive extracted frames show real,
  consistent pixel differences (~14/255 mean abs diff) instead of the
  duplicate-frame artifact the broken warm-up fix produced. Performance
  cost is negligible (~0.2s for 3 frames per clip).
- **The actual median-of-5 vs mean-of-3 question, answered on corrected
  data**: no meaningful difference. MAE 1.235 (mean-of-3) vs 1.244
  (median-of-5) vs 1.239 (mean-of-5), excluding the same 2 known
  gross-miscount outliers from the earlier multi-frame validation;
  per-row, median-of-5 improved 18/41, worsened 16/41, tied 7/41 vs.
  mean-of-3 — essentially a coin flip. **Not adopted** — no accuracy
  case for the added extraction/inference cost (5 frames vs 3 per
  clip). The real value of this investigation was finding and fixing
  the seek bug, which affects every frame extraction in the pipeline
  going forward, not just this specific idea.

### `side2` frames were universally mispositioned — root-caused, fixed, models refit (2026-08-27)

Joel asked a direct follow-up to the seek-bug discovery above: "does that
mean a lot of our 3 shots per clip were closer than 1.5 seconds apart?"
Checking properly (rather than assuming) revealed something worse than
"closer" — the **original** `extract_frame_at()` (before any fix today,
plain `cap.set(CAP_PROP_POS_FRAMES, N)` on a fresh capture, no warm-up)
had been silently mispositioning `side2` (intended t=4.0s, frame~100) for
its entire history, not failing loudly.

- Verified via full-clip pixel-diff scan (same technique used earlier in
  this session to try to anchor the CVAT test set): every sampled
  `_side2.jpg` actually matched frame 150 (t=6.00s) instead of the
  intended frame 100 — **30/30 in a random sample, 0/30 landed anywhere
  near the intended position.** Universal, not intermittent. `primary`
  and `side1` were unaffected (0 mismatches in the same sample) — the
  bug is specific to seek targets past whatever internal threshold this
  clip encoding has, which frame 100 apparently crosses and frame ~62
  (primary) doesn't.
- This meant `frame_count_3`, `frame_count_mean`, `frame_count_stdev`,
  and (via `build_training_features.py`'s `resolve_target_count()`) the
  "multi-frame-corrected" `surfer_count` used as the modeling target for
  all 847 multi-frame rows were computed from a wrong side2 value for
  the entire history of the multi-frame system (since 2026-08-25) — this
  includes the original small-sample validation, the full 756-row
  backfill, and today's earlier "corrected" model refit.
- **Fix**: re-extract `side2` for all 847 affected rows using the
  already-fixed sequential-read `extract_frame_at()`, re-run
  `run_inference_multi()`, and overwrite `frame_count_1/2/3/mean/stdev`
  (a genuine correction, unlike `backfill_multiframe_counts.py`'s
  fill-blanks-only behavior) — `surfer_count`/`confidence_avg` still
  never touched. Ran as a temporary one-off script, deleted after use.
  Verified: 847/847 fixed, 0 errors; re-checked the same two previously-
  confirmed-corrupted files, both now land at frame 100 (t=4.00s)
  exactly; confirmed zero non-`frame_count_*` columns changed via
  before/after diff.
- **Rebuilt `training_features.csv` and refit all models on the
  genuinely-correct data.** The honestly surprising result: barely
  moved. GBT MAE 6.21→6.15, RMSE 8.85→8.79; GLM MAE ~9 either way;
  quantile coverage 65.5%→65.1%. A systematically-wrong third frame
  value gets diluted by averaging into `frame_count_mean` (1 of 3
  values) and further diluted across 916 training rows — so the bug was
  real and worth fixing for data integrity (the raw `_side2.jpg` images
  and per-frame values were genuinely wrong, and would keep corrupting
  any future row-level analysis), but it did not meaningfully change any
  of the model-comparison conclusions reported earlier today.

### Median-of-5 re-tested on corrected data, then Phase 3 (live prediction script) built (2026-08-27)

- Re-ran the median-of-5 vs. mean-of-3 comparison a third time, now that
  `side2` is genuinely fixed everywhere (the first median-of-5 test used
  the seek-bug-broken extraction; the second used the sequential-read
  fix but unknowingly reused already-on-disk, still-corrupted `side2`
  files as its `mean3` baseline via `run_inference_multi()`'s
  file-exists-skip logic). Same conclusion as before, now on trustworthy
  data both ways: MAE 1.244 identical for mean-of-3 and median-of-5
  (mean-of-5 marginally best at 1.239, negligible); per-row 14
  improved/13 worsened/14 tied vs. mean-of-3 — a coin flip. **Confirmed
  not adopted.**
- **Phase 3**: new `code/predict_surf_count.py` — pulls live
  forward-looking predictors via `get_surf_predictors.py`'s
  `build_predictor_map()` (today+tomorrow, no auth token needed), trains
  fresh "production" GBT point + 10th/90th-quantile models on *all* of
  `training_features.csv` (not the 80/20 split used for validation —
  that already confirmed generalization; production wants every
  available row), builds a matching feature row for each requested
  time (same cyclical encoding, same weather-category collapsing —
  falls back to `OTHER` for a live weather condition not seen in
  training, e.g. `LIGHT_RAIN` — same train-set standardization), and
  prints a prediction with an 80% range plus the underlying conditions
  for transparency. Verified live: correctly fetched tomorrow's
  forecast, produced sane predictions for the default 4 daylight hours,
  correctly snapped `--hours 07:10` to the 07:00 forecast bucket
  (answering the original "what would we predict at 7:10am tomorrow"
  question from 2026-08-25), and gracefully reported "no forecast data
  available" for a date outside the live 2-day window rather than
  erroring.

**State**: Phases 0-3 of the modeling plan are all done.

### Simplified weather categories; found and fixed a second quantile-model collapse (2026-08-28)

Joel noticed the model's raw `weather_condition` (21 distinct Surfline
values) was being collapsed almost entirely into a generic `OTHER`
bucket for lack of per-category sample size — including every single
rain-related value, meaning the model had no way to learn a rain effect
at all. Asked for a day/night boolean pulled out separately, plus 4
merged categories: CLEAR (clear+mostly clear), CLOUDY_OVERCAST
(cloudy+mostly cloudy+overcast), RAIN (all shower/rain/drizzle variants),
FOG (fog+mist, kept distinct per Joel's request despite staying sparse,
~4 rows even merged).

- New `simplify_weather_condition()` in `build_training_features.py`
  (shared/importable) strips the `NIGHT_` prefix into `is_night` and
  maps the base condition via `WEATHER_SIMPLE_MAP`; unmapped/blank
  values fall back to `OTHER` rather than erroring. New `weather_simple`/
  `is_night` output columns. Result on the current dataset: CLEAR 1007,
  CLOUDY_OVERCAST 92, **RAIN 57** (well past any reasonable threshold —
  previously silently destroyed), FOG 4.
- `fit_surfer_count_model.py`'s `load_and_prepare()` updated to use
  `weather_simple`/`is_night` directly instead of the old ad hoc
  rare-category-to-OTHER collapse; dead `MIN_CATEGORY_COUNT` constant
  removed. `predict_surf_count.py` updated to apply the same mapping to
  live forecast data.
- Real payoff: `wx_RAIN` is now a **statistically significant** predictor
  (IRR≈0.64, p≈0.01 — rain associated with ~36% fewer surfers, exactly
  the intuitive direction) and `is_night` is highly significant
  (IRR≈0.24, p<0.001) — signal that was previously invisible to the
  model, now usable.
- **Second occurrence of the quantile-model collapse bug** (first found
  2026-08-27): refitting after this change, the lower-quantile (10th
  percentile) GBT collapsed to a constant-0 prediction again — same
  failure mode, but the previously-"fixed" `l2_regularization=0.0`/
  `max_depth=3` combo no longer prevented it on the updated dataset. A
  broader sweep showed `min_samples_leaf` is the more reliable lever
  than l2/depth, but critically **the collapse boundary is not
  monotonic in it either** (`min_leaf=50` can work while `75` collapses
  again, then `100` works) — meaning no single fixed hyperparameter
  combo can be trusted to stay safe as the dataset keeps growing.
  **Real fix this time**: `fit_quantile_model_robust()` in
  `fit_surfer_count_model.py` — fits with escalating
  `min_samples_leaf` candidates and keeps the first one whose
  *training*-set predictions actually vary (std > 0.5), raising a clear
  `RuntimeError` instead of silently returning a trivial model if every
  candidate collapses. `demo_predictions.py` and `predict_surf_count.py`
  both switched to this shared function instead of their own local
  fixed-hyperparameter quantile fits, closing the same fragility risk
  everywhere it existed. On the current dataset this picked
  `min_samples_leaf=150` for the lower model and produced the
  best-calibrated lower tail seen all session: 10.3% below-lower miss
  rate against a 10% target (previously 15-42% off in every prior
  attempt).

### Point estimate switched from mean to median — fixes it sometimes falling outside its own range (2026-08-28)

Built an ad hoc chart (in the scratchpad, not committed) showing today's
hourly predictions with 33%/66% CI bands, requested by Joel to visualize
`predict_surf_count.py`'s output. He noticed the point estimate sometimes
fell *outside* the 33% band and asked why.

Root cause: the point estimate came from a separately-trained Poisson-loss
GBT (which targets something close to the conditional *mean*), while the
bands came from independently-trained quantile-loss GBTs. Nothing ties a
mean-focused model to a median-centered band — for a right-skewed target
like surfer counts (heavy overdispersion, confirmed early in Phase 2), the
mean sits above the median, so it's expected (not a bug) for it to drift
outside a narrow band like 33% at some hours. Roughly half the hours in
the demo chart showed this.

Fix, applied to both the chart and the actual `predict_surf_count.py`
script (`train_production_models()`): use the **median** quantile model as
the point estimate instead of the separate mean-based model. Guarantees
the point estimate is always inside any range that contains it, at a
small accuracy cost (MAE ~6.96 vs ~6.15). The mean-based Poisson point
model and its `HistGradientBoostingRegressor`/`POINT_KWARGS` import are
now dead code in `predict_surf_count.py`, removed.

Joel then asked to keep the mean-based estimate around rather than drop
it entirely, specifically so mean-vs-median divergence stays visible as
a forward-looking diagnostic while more data accumulates and models keep
improving. Re-added: `predict_surf_count.py` now trains both models
(median as the primary, range-consistent point estimate; mean as a
second value shown alongside it) and auto-flags any hour where they
diverge by more than 25% ("check for growing skew").

### Camera coordinates were wrong; clip window used sunset+30min instead of real dusk (2026-08-28)

Extending the demo chart's x-axis to show a night hour on both ends
(previous entry) surfaced a much bigger issue when Joel questioned it:
he pointed out the camera runs 24/7 (the sunrise/sunset window is a
pipeline *download* decision, not a camera limitation) and that "last
light" isn't reliably 30 minutes after sunset — Surfline provides real
first/last light times we weren't using.

- **`get_clips.py`'s hardcoded camera coordinates were wrong**: 33.790,
  -118.486 (Southern California) instead of the real spot, Pleasure
  Point in Santa Cruz. Confirmed via web search: Pleasure Point is
  36.9577°N, -121.9688°W — about 340 miles off. This had been silently
  wrong the whole project (masked because the timezone,
  America/Los_Angeles, happened to be correct either way, and the
  resulting sunrise/sunset times were still plausible-looking, just
  off).
- **Verified against Surfline's own live data**: fetched the `sunlight`
  forecast endpoint (dawn/rise/set/dusk, confirmed free/no-token for
  live near-term dates) for this spot at summer solstice — real sunset
  is 8:31pm, vs. 8:07pm from the old (wrong-coordinate) astral
  calculation, a 24-minute error. Real dusk (true last light) is 9:02pm
  — 25 minutes past the old sunset+30min cutoff (8:37pm) even before
  accounting for the coordinate error. Combined, the pipeline had been
  missing up to ~49 minutes of genuinely usable evening light near
  solstice, worse than either problem alone.
- **Fix**: corrected `LOCATION` to Pleasure Point's real coordinates,
  and replaced the sunrise-30min/sunset+30min heuristic with actual
  dawn/dusk (civil twilight — the standard "usable light" boundary).
  New `get_light_window()`: for *today* specifically, tries Surfline's
  live `sunlight` fetch first (same spot Surfline itself forecasts for,
  so it can't drift from the real location) and falls back to astral
  (now using the corrected coordinates) on any failure; for backfill/
  lookback days (which are in the past, so the live anonymous endpoint
  can't reach them without the premium historical token — see
  `backfill_historical_predictors.py`) it goes straight to astral. Per
  Joel's explicit direction: astral first if it provides real first/
  last light, web search for the real coordinates rather than guessing,
  Surfline for the live/forward case specifically.
  - Verified astral's dawn/dusk (civil twilight, using the corrected
    coordinates) matches Surfline's own real dawn/dusk within 1-2
    minutes at solstice — good enough to trust for backfill without
    needing the historical token at all for this purpose.
- **Verified live**: ran `get_clips.py` for real (today only, mid-morning
  so only early-morning clips had actually happened yet — later-hour
  404s were expected "not recorded yet", not new failures). Clips
  correctly started at 6:08am (matching the real ~6:10am dawn) and
  extended through 7:38pm, well past the old (wrong) ~7:54pm cutoff and
  reaching toward the corrected 8:11pm dusk.
- **Model impact**: the model's previously-observed 5am-8pm trained hour
  ceiling was an artifact of this bug, not a true camera/location limit
  — real dusk-session data has just never been collected. The 9pm
  extrapolation-flagged chart point from the previous entry reflects a
  genuinely missing collection window, not an permanently unreachable
  one; it should start getting real training support as new data
  accumulates under the corrected window.

**Not yet done**: whether Surfline's clip archive can still serve the
previously-missed evening/morning minutes for *already-collected* past
dates (a retroactive backfill) hasn't been tested — clip retention
limits are unknown. Going forward, collection is fixed; recovering the
historical gap is a separate, unconfirmed question.

### Surfline clip-archive retention tested; real weather history added; daily chart automated (2026-08-28)

- **Surfline's clip archive retention confirmed**: tested real downloads
  at ages 6-10 days back — a clean boundary at exactly 6 days (6 days
  ago succeeds, 7+ consistently 404s). Of the last 25 collected days,
  only 5 had a large-enough dawn/dusk-window gap to matter, and only 3
  of those were within the 6-day retention window — recovered and
  integrated those 3 real clips (crop+detect run for real; 1 passed the
  quality gate, 2 correctly rejected as too dark despite technically
  being within civil twilight). This is a small, one-time recovery, not
  a systematic fix — separately, confirmed the **local** clip archive
  (`data/not_needed_in_repo/surf_clips`) has zero unprocessed clips (all
  1408 already have a matching `predictions.csv` row) — there was no
  hidden local data to mine.
- **New `code/backfill_openmeteo_weather.py`**: adds real observed
  historical weather (Open-Meteo's free, no-auth archive API — ERA5
  reanalysis, not a forecast) for every `predictions.csv` row, as a
  separate `data/openmeteo_weather.csv` (non-destructive, doesn't touch
  Surfline's forecast-based fields). Validated `real_humidity_pct`
  against the 24 real `foggy_or_blurred` quality-gate rows before
  committing to it: 87.5% mean humidity on foggy rows vs 69.8% on clear
  rows from the same dates — a real, meaningful separation. `weather_code`
  doesn't reliably flag fog at all (several known-foggy rows show
  clear/mostly-clear codes — consistent with the literature on
  coarse-grid reanalysis struggling with localized coastal marine-layer
  fog). `build_training_features.py`/`fit_surfer_count_model.py` updated
  to join and use the new columns. Refit result: GBT MAE 6.15→6.02,
  RMSE 8.79→8.51 — modest, real improvement. Interesting nuance:
  `real_cloud_cover_pct` and `real_pressure_mb` turned out to be the
  statistically significant new features in the NegBin fit, not
  `real_humidity_pct` despite its strong bivariate fog signal — likely
  because humidity and cloud cover are correlated and cloud cover
  captures more of the shared variance once both are in the model.
- **New `code/plot_daily_prediction.py`** + **`code/daily_chart.sh`**:
  generates the daily prediction chart (median + 33%/66% bands, tide,
  wave-energy bars, weather-coded markers, night shading, model/detector
  info footer) to `data/charts/surfer_count_YYYY-MM-DD.png`. Cron
  install blocked by a sandbox permission restriction — Joel will add
  `0 19 * * * .../code/daily_chart.sh >> .../data/daily_chart.log 2>&1`
  himself.
  - **Caught a real bug while listing an hourly table for Joel**: an
    hour-range filter hardcoded to start at 5am showed a confident-
    looking prediction for 5am on a day whose real dawn was 6:10am —
    the model doesn't know a given day's specific dawn time, only the
    coarse `is_night` flag (which was correctly True at 5am, but not
    enough alone to suppress the extrapolated prediction) and cyclical
    hour features, and hour=5 training examples mostly come from
    summer days where 5am was already light. Fixed both the ad hoc
    query and `plot_daily_prediction.py` to compute each day's *real*
    dawn/dusk via `get_light_window()` and filter to that, instead of a
    fixed hour range — the fixed `TRAINED_HOUR_MIN/MAX=5,20` constant is
    kept, but now only for the separate "has the model ever trained on
    this hour at all" extrapolation flag, not for deciding which hours
    to show.

### Daily chart posted to GitHub README (2026-08-28)

Joel asked for the daily chart to be visible somewhere prominent on
GitHub, plus an hourly range table (33% range, no median). Since
`data/` has been untracked-by-convention all along, making the chart
show up on GitHub's rendered README required actually committing it —
asked Joel to confirm the automation model before building it (auto-
commit-and-push daily via cron vs. generate-locally-push-manually); he
chose full automation.

- `plot_daily_prediction.py` now also writes `data/charts/latest.png`
  (a single, git-tracked, daily-overwritten file — not a new file
  every day, to avoid unbounded repo growth from daily binary commits)
  and `data/charts/latest_table.md` (hour → 33% range only, per Joel's
  "no median needed"), then rewrites the marked section of `README.md`
  between `<!-- DAILY_CHART_START -->`/`<!-- DAILY_CHART_END -->`
  (idempotent, safe to run daily) with the embedded image + table +
  a timestamp.
- `daily_chart.sh` now stages exactly those 3 files after generation,
  commits only if something actually changed (`git diff --cached
  --quiet` check — avoids empty commits when the numbers happen to
  round to the same table twice), and pushes to `origin/main`. Git/
  network failures are logged but explicitly non-fatal — a failed
  push must never be treated as the whole daily chart job failing,
  since the chart itself already generated successfully by that point.
  Commit messages are prefixed `Automated:` to distinguish unattended
  cron commits from real interactive session work in the git log.
- Verified live: ran the real commit+push once to confirm the whole
  chain works before trusting it to the unattended 7pm cron job.

### Chart+table combined side by side; daily detection-review image added (2026-08-28)

Two more asks on the same daily-chart feature: render the chart and
33%-range table side by side (table baked into the chart image itself,
simpler than a separate markdown table Joel confirmed was fine to
fold in), and add a detection-review image above it — the day's ~8am
crop with real bounding boxes/confidence labels and the model's
predicted range/median for that hour overlaid.

- Chart figure restructured to a 2-panel `GridSpec` (chart : table =
  3.2 : 1) instead of a separate `.md` table file — `write_range_table()`
  removed, the table renders directly via `ax_table.table(...)` in the
  same image.
- New `detect_surfers.run_inference_with_boxes()`: identical tiling/
  NMS/false-positive-filter pipeline as the production `run_inference()`,
  just returns the surviving boxes instead of discarding them after
  counting — reused, not reimplemented, so the drawn boxes are exactly
  what the real counting pipeline sees, not a simplified stand-in.
- New `generate_detection_image()` in `plot_daily_prediction.py`: finds
  the day's `predictions.csv` row closest to 8am (real clip data, not
  synthetic), draws real boxes + confidence labels via `cv2`, and
  overlays both the actual detected count and the model's predicted
  range/median for that same hour (pulled from the already-computed
  daily chart data, not a separately-sourced number) in a white banner
  strip below the image so text never overlaps real image content.
  Skips cleanly (not an error) if today's ~8am clip hasn't been
  captured yet when the chart runs. Saved to `data/charts/
  latest_detection.png` (git-tracked, same daily-overwrite convention
  as `latest.png`).
- `update_readme()` now embeds the detection image above the chart
  within the same marked section.
- Fixed a real bug caught before it could break the unattended cron job:
  `daily_chart.sh` originally did `git add fileA fileB fileC` in one
  command — confirmed via test that if ANY one pathspec doesn't exist
  (e.g. no detection image yet some days), git fails the ENTIRE add and
  stages nothing, not just the missing file. Changed to add each file
  individually with its own non-fatal error handling.

### Frame-timing variability study; hourly-window variability check (2026-08-27 to 2026-08-28)

- Downloaded 14 real ~60-second clips (2026-08-27, one per daylight hour,
  via a one-off `CLIP_DURATION_OVERRIDE_FILE` run) specifically to study how
  much the detector's per-frame count varies over time within a clip.
  `code/analyze_frame_timing.py` extracts one ROI-cropped frame per second
  (882 frames total, all passed the production quality gate), runs the real
  production detector on each, and writes `data/frame_variability_analysis.csv`.
- **Lag finding**: count drift grows steadily with time separation — no
  sharp noise floor. Mean absolute count difference: 0.91 at 1s apart, 1.40
  at 10s, 2.10 at 30s, plateauing around 2.3-2.4 by 40-58s. Adjacent-second
  frames are highly correlated (redundant).
- **Averaging finding**: spacing matters far more than frame count. Today's
  production default (3 frames, ~2-3s span) gives stdev≈1.53 vs. a clip's
  own full-minute reference mean — barely better than a single frame
  (1.74) and worse than the naive independent-samples bound (1.00) because
  adjacent frames are too correlated. Spreading 5 frames across the full
  ~58s span instead gets stdev≈0.42 (~3.6x tighter); diminishing returns
  past k≈6 at wide span. See `data/frame_variability_analysis_summary.png`.
  **Caveat**: this measures detector *stability*, not *accuracy* — pending
  human validation (`data/count_review/`, 8 sets x 10 images spanning the
  activity range, `human_count` column to fill in) before changing
  production clip duration/frame spacing. See `model_and_feature_ideas.md`.
- Separately, pulled 12 real back-to-back 5-minute clips covering the full
  8-9am hour on 2026-08-28 (`code/pull_hourly_variability_clips.py`,
  `code/analyze_hourly_variability.py`) to check within-hour variability
  directly, using 5 frames spread across each 5-minute window (per the
  spacing finding above). Real result: mean count rose from 20.6 (8:00) to
  a peak of 29.4 (8:35), tapering to 20.6 by 8:55 — stdev 2.71 across the
  hour, a swing of ~9 surfers (~35% of the mean). Given the ~0.4 noise
  floor from the lag study, this is real crowd turnover within the hour,
  not detector noise — a partial explanation for why the daily prediction
  model's intervals are wide: count varies substantially within an hour,
  not just day to day.

### Surf-condition rating vs. surfer count — checked, found not predictive (2026-08-28)

Joel asked whether `rating_value` (Surfline's condition rating, already a
model feature since Phase 1) tracks surfer count, and whether it might be
masked by multicollinearity with wave-size/energy variables instead.

- `rating_value` has near-zero correlation with `surfer_count` (Pearson
  r=0.043, p=0.148; Spearman ρ=0.025, p=0.386) and doesn't rank in the
  GBT's top-15 permutation feature importances. Its own range is narrow
  in this dataset (1-4, mean 2.4, std 0.54) — mostly POOR-to-FAIR.
- Checked the multicollinearity hypothesis directly: `rating_value` **is**
  strongly correlated with the wave/energy variables it's derived from
  (`energy_nearshore_kj` r=0.738, `surf_max_ft` r=0.633, `surf_min_ft`
  r=0.632, `energy_offshore_kj` r=0.651, all p<0.0001) — but those same
  wave variables show weak, and in several cases *negative*, correlation
  with surfer count themselves (`primary_swell_height_ft` r=-0.074 p=0.011,
  `primary_swell_period_s` r=-0.067 p=0.022, `consistency_wave_count`
  r=-0.111 p=0.0001, `energy_offshore_kj` r=-0.091 p=0.002;
  `energy_nearshore_kj`/`surf_min_ft`/`surf_max_ft` not significant). So
  there's no stronger wave-based predictor hiding behind `rating_value` —
  the underlying wave signal itself doesn't drive crowd size positively
  in this data. All effect sizes are small (r² well under 2%); statistical
  significance here is mostly a large-n (1160) effect, not a strong
  practical driver. Logged as an open question in `model_and_feature_ideas.md`
  (possible confounders: weather, season, day-of-week not yet controlled
  for) rather than treated as a settled causal finding.

### Fog-forecast research: Open-Meteo forecast API vs. archive API (2026-08-28)

Joel asked whether we have a working image-based fog classifier to use as
a predictor, and suggested a non-Surfline fog forecast paired with it,
noting Surfline itself rarely calls fog — confirmed: `weather_simple` has
only 4 `FOG` rows out of 1160 (0.3%).

- No image-derived fog predictor exists — the image-quality gate's
  `foggy_or_blurred` flag only skips detection, it was never stored as a
  model feature (and can't be: a skipped frame has no `surfer_count` to
  predict). `real_humidity_pct` (Open-Meteo archive/ERA5) remains the only
  fog-related model feature, as a continuous proxy.
- Tested `api.open-meteo.com/v1/forecast` (forward-looking) directly
  against Pleasure Point's coordinates and compared it field-by-field with
  `archive-api.open-meteo.com/v1/archive` (already used for historical
  backfill). `relative_humidity_2m`/`cloud_cover`/`weather_code`/
  `surface_pressure` are continuous and present with real values on
  **both** endpoints (confirmed via live requests to both) — exactly the
  fields `real_humidity_pct` etc. are already built from, so per Joel's
  requirement (continuous, and available on both the historic and
  forecast side) no new boolean/class feature is needed here.
- `visibility` (meters) also appeared in the forecast response and reads
  as a more direct fog signal (~12,200m on foggy-looking morning hours vs.
  45,000+m on a clear afternoon) — but a live archive-API request
  confirmed `visibility` comes back `null` for every hour tested on that
  endpoint (ERA5 reanalysis doesn't diagnose it). Fails the
  both-sides-continuous requirement, so not pursued via this source.
- **Found a real bug in the process** (now in `bugs.md`): `predict_surf_count.py`
  never actually populates the `real_*` Open-Meteo features at live-
  prediction time — `get_surf_predictors.py`'s forward-looking pull is
  Surfline-only, so `build_feature_row()` silently leaves them `NaN` on
  every real prediction (no crash, since `HistGradientBoostingRegressor`
  handles NaN natively, but the validated fog/weather signal from training
  goes unused at inference time). Wiring in the forecast endpoint fixes
  this at the same time as adding genuine forward-looking fog signal — see
  `model_and_feature_ideas.md`.

### Live Open-Meteo forecast wired into predict_surf_count.py (2026-08-28)

Implemented the fix from the fog-forecast research above. New
`fetch_openmeteo_forecast()` + `merge_openmeteo_forecast()` in
`code/get_surf_predictors.py`: fetches `api.open-meteo.com/v1/forecast`
(same 5 fields — `temperature_2m`/`relative_humidity_2m`/`cloud_cover`/
`weather_code`/`surface_pressure` — as `backfill_openmeteo_weather.py`'s
archive pull) and merges the resulting `real_*` values into the same
`{local_hour: {field: value}}` dict `build_predictor_map()` already builds
from Surfline's endpoints, keyed identically (Open-Meteo already returns
local wall-clock time strings when `timezone` is passed, so no UTC-offset
math is needed the way `local_hour_key()` needs it for Surfline's raw UTC
epoch timestamps). Coordinates imported from `get_clips.LOCATION` (not a
third hardcoded copy, given the project's history with a coordinate bug).
Fetch is non-fatal on failure — same as every other endpoint in this file.

`build_predictor_map()` is shared by both `get_surf_predictors.main()`
(writes `surfline_predictors.csv`) and `predict_surf_count.py` (feature
rows) — verified the extra keys don't leak into `surfline_predictors.csv`
(its `CSV_HEADER`/`row_from_predictors()` only pull the original Surfline
fields, unaffected) and that `predict_surf_count.py`'s standardized
feature row now gets real, non-NaN values for all 4 `real_*` columns
(previously silently `NaN` on every live prediction).

### Fog/humidity checked as a predictor, found weak; tide and weekend/weekday confirmed as the strongest predictors (2026-08-28)

Followed up on the fog-forecast work above by directly checking how
explanatory `real_humidity_pct`/`real_cloud_cover_pct` actually are for
`surfer_count`, using `data/training_features.csv` (n=1175, 1160 after
dropping rows with missing numeric features):

- Pearson correlation with `surfer_count`: `real_humidity_pct` r=-0.075
  (p=0.0105), `real_cloud_cover_pct` r=-0.062 (p=0.0338) — statistically
  significant given the sample size but small (r² well under 1%).
  `real_temperature_f` (r=0.182) and `real_pressure_mb` (r=0.132) are
  noticeably stronger.
- Mean `surfer_count` by `weather_simple`: CLEAR=15.4 (n=1022),
  CLOUDY_OVERCAST=12.1 (n=92), RAIN=8.3 (n=57), FOG=5.0 (n=4 — too few
  rows to treat as reliable).
- GBT (`HistGradientBoostingRegressor`) permutation importance on a held-out
  test split: `real_humidity_pct` ranks 16th of 31 features,
  `real_cloud_cover_pct` ranks 15th. **`tide_ft` (0.603) and `is_weekend`
  (0.162) are the two strongest predictors by a wide margin** — the next
  closest is `hour_cos` (0.104), then `energy_nearshore_kj` (0.087). Fog/
  humidity/cloud cover is a real but minor signal in comparison; tide and
  day-of-week are the actual primary drivers of surfer count at this spot
  (Jack's, 38th St, Pleasure Point).

### Daily chart now forecasts tomorrow by default; detection image decoupled from forecast date (2026-08-28)

`plot_daily_prediction.py` previously defaulted `target_date` to *today*,
so the 7pm cron run just re-showed today's own forecast for its remaining
hours instead of anything forward-looking. Changed the no-`--date` default
to `today + 1 day` for the forecast chart/table, while the detection image
still always uses today's own ~8am crop (`detection_date`, independent of
the forecast date) — the two were previously the same `target_date`
parameter, which meant shifting the forecast date to tomorrow would have
broken the detection image (no crop exists yet for a future date).
`--date` still pins both to the same explicit date, for backfill/testing.

Refactored the per-hour quantile-model prediction into a shared
`predict_for_hour()`/`predict_nearest_hour()` closure in `main()` so the
detection image's "Predicted: ..." text is computed directly for its own
hour via `by_hour` (which covers today+tomorrow, `DAYS=2`) rather than
matched from the forecast chart's day-scoped DataFrame — the old matching
logic would have silently used the wrong day's prediction once the two
dates diverged. Verified live: forecast chart correctly shows tomorrow
(Aug 29) with dawn/dusk computed for that date, detection image still
shows today's (Aug 28) 7:56am crop with a "Predicted" figure now sourced
from today's own hour instead of tomorrow's.

Also changed detection-box/label color from red to lime green (`#9de35a`,
matching the dark theme's tide-line/wave-energy-bar color), same 0.75
alpha-blended transparency as before.

### `data/` reorganized into predictions/predictor_vars/reviews; new top-level analysis/ folder (2026-08-29)

Cleaned up `data/`, which had accumulated flat CSVs and inconsistently-named
review folders directly at its root as the project grew. New structure:

- `data/predictions/predictions.csv` (was `data/predictions.csv`).
- `data/predictor_vars/surfline_predictors.csv` and
  `data/predictor_vars/openmeteo_weather.csv` (were both directly under
  `data/`).
- `data/reviews/<name>/` for every human-review dataset, one subfolder per
  dataset with a name distinguishing it from the others (previously each
  had its own top-level `data/` folder with an inconsistent naming
  convention): `count_review/` → `reviews/count_60sec_var/` (renamed to
  reflect what it's actually reviewing — the frame-timing-variability
  60-second-clip study), `fog_review/` → `reviews/fog_quality/`,
  `model_review_50/` → `reviews/model_spotcheck_50/`.
- `data/training_features.csv` stays directly under `data/` — it's a
  derived join of the predictions/predictor_vars data, not raw source
  data itself.
- New top-level `analysis/` folder: each one-off analysis (not part of
  the scheduled pipeline) gets its own subfolder holding whatever mix of
  CSVs, charts, and the script(s) that produced them, instead of the
  previous scatter (CSVs/PNGs under `data/`, scripts under `code/`).
  Three so far: `analysis/frame_timing_variability/` (the 2026-08-27
  14-clip lag/averaging-window study), `analysis/hourly_variability_8to9am/`
  (the 2026-08-28 12-clip within-hour study, both the 5-frame sparse
  sample and the full 1-frame/second version), and
  `analysis/weekday_weekend_patterns/` (the weekday/weekend charts
  embedded in `README.md`).

Updated every literal path in `code/` that pointed at a moved file
(`detect_surfers.py`, `get_surf_predictors.py`, `backfill_openmeteo_weather.py`,
`backfill_tide.py`, `build_training_features.py` — 6 path constants total)
and fixed the moved analysis scripts' `sys.path`/`_PROJECT_ROOT`
calculations, since they're now one directory deeper (`analysis/<name>/`
instead of `code/`) — verified each of the four moved scripts still
imports `detect_surfers`/`get_cropped_frame`/`get_clips` from `code/`
correctly, and re-ran all four chart-producing scripts in place to confirm
they regenerate identical output at their new paths. `.gitignore` updated
to the new paths (kept as explicit per-file entries rather than
whole-folder ignores, since the `analysis/` and `data/reviews/`
subfolders also hold real, git-tracked `.py` scripts alongside their
gitignored data outputs). Nothing here was previously git-tracked except
the two `weekday_weekend_*.png` files (moved with `git mv` to preserve
history) and `README.md`'s references to them, which were updated to the
new `analysis/weekday_weekend_patterns/` path.

### New code/train_model.py replaces detect_surfers_v2.ipynb as the model-retraining path (2026-08-29)

While investigating what's needed to rebuild the detector (see the
`model_out/yolov8s.pt` path-mismatch finding above), decided the training
notebook was better as a proper script — matches how the rest of the
pipeline works, and makes it a real one-command entry point for whenever
there's a new or larger CVAT annotation export to train on, rather than
something to click through cell-by-cell.

`code/train_model.py` reimplements the notebook's three real stages
(COCO tiling, COCO→YOLO conversion, YOLO fine-tune + validation) as
functions with a CLI. Two things it does differently from the notebook,
both deliberate:

- Tile dimensions (`TILE_W`/`TILE_H`/`STEP_X`/`NUM_TILES`) are imported
  directly from `detect_surfers.py` instead of being re-hardcoded, so
  training tiling can never silently drift out of sync with production
  inference tiling the way two independent copies could.
- Each run writes to fresh, dated output folders
  (`cvat_out_coco/splits_tiled_<run-name>/`, `cvat_out_yolo_rebuilt_<run-name>/`,
  `model_out/<run-name>/`) instead of overwriting the current production
  training data/weights in place — refuses to proceed if a run-name's
  output folders already exist, unless `--force` is passed. Nothing
  production-facing changes until `MODEL_PATH` is manually pointed at the
  new weights, after reviewing the printed validation metrics.

Verified the tiling + COCO→YOLO conversion stages are correct by running
them against the current `data/cvat_out_coco/splits/` export into a
throwaway run-name and diffing the output against what's on disk today:
tiled image/annotation counts matched exactly (128/60/40 images,
1268/356/188 annotations, same as the notebook's original run), and a
sample of generated YOLO label `.txt` files were byte-identical to the
existing `cvat_out_yolo_rebuilt/labels/`. One real discrepancy found and
fixed: the notebook's source cell wrote `data.yaml`'s class name as
`'Surfer'`, but the actual file on disk today reads `'surfer'` (lowercase)
— matched the script to the current on-disk convention rather than the
stale notebook source (no functional effect either way, single-class
model, purely a label string). Did not run the full training stage
(would take real compute time for no additional verification value beyond
what `ultralytics`'s own training loop already does).

Also attempted to fix the `yolov8s.pt` path directly in the notebook's
training cell via a notebook-editing tool — that stripped the cell's
stored output, which was the *actual* historical training log (real
per-epoch losses, model summary, final metrics — the source
`plot_daily_prediction.py`'s `DETECTOR_PRECISION`/`DETECTOR_RECALL`
constants cite as "actual training log"). Reverted immediately via
`git checkout`; the notebook is untouched except its `docs/PROJECT_FILES.md`
description being corrected to point future readers at `train_model.py`
instead.

`data/cvat_out_coco/splits/`, `cvat_out_coco/splits_tiled/`,
`cvat_out_yolo_rebuilt/`, and `model_out/20251013/` are all currently
git-tracked (864 files, ~184MB for `model_out/20251013/` alone) — an
existing convention `train_model.py`'s future dated output folders would
also follow if/when a retrained model is adopted, but nothing about
committing new run folders is automatic; that stays a deliberate decision
each time.

### Fixed: README's detection image disappeared entirely between local_pipeline.sh runs (2026-08-29)

Reported by Joel: the detection image was missing from the GitHub README.
Root cause was the previous day's fix (forecast defaults to tomorrow,
detection stays on today) — `find_nearest_hour_crop()` only ever searched
`detection_date` itself (usually "today"). `local_pipeline.sh` runs twice
a week, not daily, so on any day it hasn't run yet, no crop exists for
"today" and the whole detection section vanished from README even though
`data/charts/latest_detection.png` from a previous day was still sitting
on disk, valid and useful.

Fixed `find_nearest_hour_crop()` to search backward through up to 7 days
before falling back to nothing, so it now always shows the most recent
available detection image instead of disappearing between pipeline runs.
Verified live: with `local_pipeline.sh` last run 2026-08-28, today's
(2026-08-29) `plot_daily_prediction.py` run correctly fell back to and
found the 8/28 crop.

Caught and fixed two knock-on bugs while making this change:
- The dated output filename used the search-anchor date
  (`detection_date`) instead of the crop's actual date, so a
  fallback would save as e.g. `detection_2026-08-29.png` while
  containing an 8/28 image. Now named from the row's real `date`.
- The "Predicted: ..." text's live-forecast lookup
  (`predict_nearest_hour`) was also keyed on the search-anchor date
  rather than the crop's actual date — on a fallback to a past day, it
  would have silently shown *today's* live forecast prediction next to
  an *older* day's detection image, a real mismatch. Now keyed on the
  crop's actual date; correctly prints "not available for this hour"
  when that date has no live forecast entry (as any past date won't),
  rather than showing a wrong number.

### Real human count vs. detector: 49 actual vs. 27/28 detected on a foggy frame (2026-08-29)

Joel was in the water at ~7:30am on 2026-08-29 and asked for a detection
image to compare against his own count. Pulled the nearest real clip
(7:29am), ran it through the actual production crop + detect pipeline
(not a simplified version), and got 28 surfers on the primary frame /
27 on the production 3-frame average (`crop2026-08-29_07-29-00.jpg`).
Joel counted 49 in the same (somewhat foggy) image. Recorded 49 as
`human_count` on that row in `predictions.csv`.

Zoomed into the raw crop vs. the boxed version to see where the gap came
from: surfers sitting further out in flatter water were mostly boxed
correctly, but a whole line of small dark figures sitting in or right
against the whitewater/breaking-wave foam had almost no boxes on them —
a visually clear, consistent pattern across the width of the frame, not
scattered/random misses. Logged as a new open item in `bugs.md`, linked
to the existing "two-step candidate-then-verify detection" feature idea
in `model_and_feature_ideas.md` (lower confidence threshold specifically
in candidate-blob regions) — this is a concrete real example of exactly
the failure mode that idea was meant to address. Single data point so
far; worth checking whether the pattern holds on other foggy/whitewater-
heavy frames before investing in a fix.

### Refined the 2026-08-29 undercount breakdown; started a two-agent blob-detection experiment (2026-08-30)

Joel refined the miss breakdown on `crop2026-08-29_07-29-00.jpg`: of the
gap between 27/28 detected and 49 actual, 7 were missed in the
whitewater band and 9 were missed in flatter water above it — all 9
prone (lying flat, paddling), a distinct failure mode from the
whitewater-contrast one. Logged the posture-classification idea
separately in `model_and_feature_ideas.md` (filed under Feature Ideas,
not Model Ideas, since it's about the detector — Model Ideas in that doc
is specifically the count-*prediction* model).

Started two background agents to evaluate the "two-step candidate-then-
verify detection" idea from `model_and_feature_ideas.md` — Agent A
establishes the naive-baseline cost of just globally lowering
`CONF_THRESH` (real precision/recall/F1 at 0.10/0.12/0.15 vs the current
0.195, on the real tiled val set at `data/cvat_out_coco/splits_tiled/val/`);
Agent B implements and evaluates the actual candidate-blob approach
(classical CV blob detection for dark-blob-on-smooth-background
candidates, then a targeted lower-threshold YOLO rescan only in those
regions), same real val set, same metrics, so the two can be compared
directly. Both explicitly instructed not to modify the repo, not to
retrain, not to commit, and to report only real computed numbers — no
fabricated stats. Results pending as of this entry; will be logged here
once both report back.

### Two-step candidate-blob experiment results: dead end (2026-08-30)

Both agents reported back with real numbers against the tiled val set
(`data/cvat_out_coco/splits_tiled/val/`, 60 tiled images, real GT boxes).

**Agent A (naive global threshold lower)**: precision/recall/F1 at
several thresholds vs the production 0.195:

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.195 (production) | 0.863 | 0.802 | 0.831 | 38 | 59 |
| 0.15 | 0.859 | 0.815 | 0.837 | 40 | 55 |
| 0.12 | 0.844 | 0.819 | 0.831 | 45 | 54 |
| 0.10 | 0.839 | 0.822 | 0.831 | 47 | 53 |

Lowering the threshold to 0.10 buys back ~2 points of recall at the cost
of a ~24% increase in false positives (38→47). On the real motivating
frame (`crop2026-08-29_07-29-00.jpg`), box count barely moved (28→29 at
conf=0.10) and the extra box did not land in the whitewater band — the
model isn't suppressing candidates there even at conf=0.10, it's simply
not proposing them.

**Agent B (actual candidate-blob two-step)**: local-background-
subtraction blob detection (median-blur background estimate, size-
filtered 3-48px × 6-30px, local-smoothness-gated), YOLO rescan at
conf=0.10 in flagged candidate windows, merged via the same production
NMS. Real result: F1 0.819→0.762 (worse than baseline), recovering only
**+1 true positive out of 356 real GT boxes** while adding 55 more false
positives (62→117) — confirmed not a tuning artifact via a 5-point
parameter sweep (best alternate config: F1=0.784, still below baseline).
On the real frame: 4 extra boxes, all 2-7px (well under real surfer box
size), none corresponding to an actual surfer on visual inspection —
3 of 4 clustered near the known tree-bough false-positive zone.

**Root cause, structural not tunable**: the candidate detector's premise
— a missed surfer sits against a locally *smooth* background — directly
contradicts the whitewater-band failure mode it was meant to fix.
Whitewater is high-variance foam texture by definition, so the same
smoothness gate that's supposed to exclude foam-texture false positives
also excludes real surfers sitting in the foam. This is a structural
mismatch, not a parameter-tuning problem — confirmed by the sweep finding
no configuration came close to baseline F1.

**Conclusion**: closing this angle for the whitewater-contrast problem.
Both approaches (naive threshold, candidate-blob) cost meaningfully more
false positives than the recall they buy back, and neither one recovers
detections in the actual region that motivated the experiment. Logged
the untried alternative (motion/optical-flow-based candidate detection,
since whitewater moves coherently with the wave while a surfer's head
doesn't) in `model_and_feature_ideas.md` for future consideration — not
attempted this round.

### Round 2: tested (and disproved) the smooth-water hypothesis for the two-step approach (2026-08-30)

Joel's hypothesis after round 1: the two-step method failed specifically
because it was tested against whitewater (a structural conflict with the
smoothness gate) — it should work on undercounts in smooth, non-
whitewater water instead. Resumed both agents against real data to test
this directly: the earlier 50-image human spot-check
(`data/reviews/model_spotcheck_50/review_counts.csv`) has 12 rows where
`my_count` (human) exceeds `model_count`, two of them large and
explicitly noted as smooth water — `crop2026-08-03_10-02-00.jpg`
(model=3, actual=30) and `crop2026-08-09_07-29-00.jpg` (model=19,
actual=46) — plus several explicitly-textured/prone-surfer cases used as
negative controls.

Agent B (two-step) found and fixed a real methodological bug in its own
evaluation first: its earlier predictions ran YOLO on in-memory arrays,
but production always predicts from a written temp JPEG file, and the
two give meaningfully different results on some tiles (one real example:
7 boxes from an array vs 12 from the file). Fixed by routing every
prediction through the same file round-trip production uses; re-verified
baseline now reproduces `review_counts.csv`'s real `model_count` almost
exactly, and the round-1 val-set conclusion held (F1 0.820 baseline vs.
0.769 two-step, consistent with the original finding — not affected by
the bug).

**Result on the two flagship smooth-water cases: zero new boxes from
two-step, on both.** An ablation with the smoothness gate fully disabled
also produced zero rescued boxes — ruling out gating as the bottleneck.
Raw inference at conf=0.05 (vs production 0.195) barely moved box counts
(3→4, 23→25 pre-filter): the trained model simply never proposes a
candidate at these locations, at any confidence, confirming this isn't a
threshold or gating problem at all — in smooth water OR whitewater.
Across the other 10 spot-check images, two-step did add boxes, but
inspection showed the large majority (53 of ~57 new boxes) were
duplicate sub-boxes landing inside an already-detected surfer's box, not
real recoveries — a real risk of count inflation, not just a null
result. Naive threshold-lowering (Agent A) showed the identical pattern
on the same two flagship images (3→4 and 19→20 boxes against 30/46
actual) — confirming this isn't specific to the candidate-blob method
either.

**Conclusion: the smooth-water hypothesis is disproved by direct test.**
Water texture (whitewater vs. smooth) isn't the real differentiator —
fog/low-contrast conditions appear to suppress the model's features on
certain surfers regardless of background, a representation-level gap no
inference-time threshold or candidate-region trick can fix. This raises
the bar for the whole two-step research angle: a real fix likely needs
better/more training data for foggy, low-contrast conditions (see
`code/train_model.py`, the retraining path), not further inference-time
engineering. Updated `model_and_feature_ideas.md`'s "Two-step
candidate-then-verify detection" entry with the full round-2 numbers and
verdict.

### Found and fixed why GitHub stopped updating: an uncaught network exception silently killed a whole daily_chart.sh run (2026-09-03)

Joel noticed GitHub hadn't been updating. `git log` showed local and
origin were actually in sync as of the 2026-09-02 auto-commit — so it
wasn't a push failure. Diffing `data/daily_chart.log`'s
"starting"/"complete" markers across recent days found the real gap: the
2026-08-31 19:00 run crashed with an uncaught
`requests.exceptions.ConnectionError` (DNS resolution failure —
`services.surfline.com` unreachable, machine likely offline/asleep at
that exact moment) and never got anywhere near its git commit/push step,
because `get_surf_predictors.py`'s `build_predictor_map()` only caught
`requests.exceptions.HTTPError` around each Surfline endpoint fetch —
and `ConnectionError` is a sibling exception, not a subclass of
`HTTPError` (verified directly: `issubclass(ConnectionError, HTTPError)`
is `False`; `issubclass(ConnectionError, RequestException)` is `True`).
So a transient network hiccup propagated all the way up through
`main()` uncaught, and `daily_chart.sh`'s `set -e` killed the whole
script right there — no chart, no README update, no log marker beyond a
raw traceback, and no alert of any kind (unlike `get_clips.py`'s
auth-failure email pattern). Runs on 09-01 and 09-02 succeeded normally
afterward, so this was a single silently-skipped day, not an ongoing
failure — but a real gap with no visibility until someone happened to
check the log.

Fixed `build_predictor_map()`'s except clause to catch the broader
`requests.exceptions.RequestException` (matches the pattern
`fetch_openmeteo_forecast()`'s error handling already used, a few lines
below — that one was already correct). Also hardened `daily_chart.sh`
itself: wrapped the python invocation so a future failure of any kind
logs an explicit grep-able `ERROR: ... chart NOT generated` line instead
of silently dying via `set -e` with nothing but a bare traceback —
directly informed by how much manual log-diffing it took to find this
one. Verified the fix is real (not just plausible) by checking the
exception hierarchy directly, then ran `daily_chart.sh` manually to
confirm it completes normally and catches GitHub up immediately rather
than waiting for tonight's cron — pushed successfully
(`e83fd5a`, forecast for 2026-09-04). Detection image still shows
2026-09-01 (last local_pipeline.sh success) — expected, not a bug; the
twice-weekly clip cron was due to run later that same evening.

### Added retry-with-backoff so a short home-internet outage recovers same-evening (2026-09-03)

Joel confirmed the 2026-08-31 outage was his home internet being down,
and asked for resilience to that specific case rather than only relying
on the next day's cron. Two more fixes, same root cause as the one
above:

- `get_surf_predictors.py`'s `fetch()` already had a 3-attempt retry
  loop, but it had the identical narrow-exception bug —
  `except requests.exceptions.HTTPError` doesn't catch the
  `ConnectionError`/`Timeout` a real connectivity failure raises, so the
  retries never actually fired; the first attempt would propagate
  immediately. Broadened to `RequestException`, verified the same way
  (`issubclass(ConnectionError, RequestException)` is `True`).
- `daily_chart.sh` now retries the whole `plot_daily_prediction.py`
  invocation up to 4 times with increasing delays (0, 5, 15, 30 min —
  ~50 min total window) before giving up for the day. A short-to-medium
  home-internet blip (the common case) now recovers within the same
  evening instead of waiting a full 24h for the next scheduled run;
  longer outages still fall back to that next-day retry as before, no
  regression there. Verified the bash retry/break logic in isolation
  with a simulated fail-fail-succeed sequence before trusting it, and
  reran `plot_daily_prediction.py` directly to confirm the broadened
  `fetch()` exception handling doesn't break the normal (network-up)
  path.

### Human validation of frame-spacing accuracy (2026-09-04)

Joel filled in `data/reviews/count_60sec_var/review_counts.csv` with real
counts (70 usable points across 7 clips; the 8th, `05_50`, was fogged
throughout with lens condensation, unusable) and added a `notes` column.
This was the pending piece from the 2026-08-28 frame-timing variability
study — that study found wider-spread frame averaging cut the detector's
own noise, but only measured self-consistency (stdev), not real accuracy.

Wrote `analysis/frame_timing_variability/validate_against_human_counts.py`,
matching Joel's human counts against the model's already-computed
per-second counts (`frame_variability_analysis.csv` has every second
0-62 for all 14 clips) for several real frame-selection strategies —
real numbers, no re-running detection needed:

| Strategy | MAE | Bias | RMSE | std(err) |
|---|---|---|---|---|
| single frame | 1.24 | -0.36 | 1.74 | 1.72 |
| k=3, tight (~2s, today's production) | 1.18 | -0.45 | 1.54 | 1.49 |
| k=3, spread (30s) | 1.14 | -0.35 | 1.54 | 1.51 |
| k=5, spread (30s) | 1.12 | -0.40 | 1.51 | 1.47 |
| k=5, wide (58s) | 1.30 | -0.39 | 1.60 | 1.56 |
| k=10, wide (58s) | 1.20 | -0.38 | 1.49 | 1.45 |

**Real finding: every strategy undercounts by roughly the same amount
(~0.35-0.45 mean bias) regardless of averaging window — wider/more-frame
averaging mainly reduces variance (std(err) 1.72→1.45-1.56), not the
systematic bias.** This is the expected statistical behavior (averaging
narrows spread around a mean, it doesn't move the mean toward truth) but
is a real, direct answer to the open question: the accuracy gain from
wider spread is real but modest (best MAE 1.12 vs. single-frame 1.24,
~10%), and going wider than ~30s stops helping (k=5@58s is worse than
single-frame on MAE, though still lower std(err)). Logged full numbers
and the practical takeaway (worth a modest default change, but doesn't
fix the underlying undercount bias — that needs the model-side fixes
already logged, not frame-averaging tuning) in
`model_and_feature_ideas.md`. n=70 across 7 clips — real but modest
sample size, numbers are directional.
