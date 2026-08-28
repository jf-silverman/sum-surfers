# Project History

A chronological record of how this project was built and how the model's
real-world performance has been investigated. See `README.md` for how to
run things and `PROJECT_FILES.md` for what every file is.

## Open engineering leads

Findings from the 2026-08 model review that aren't implemented yet:

1. **Possible duplicate/overlapping boxes.** Across several enhancement-batch
   images, the reviewer (Joel) saw fewer visible boxes on screen than the
   reported count implied (e.g. `08-07_06-26`: "only see two boxes" but
   count=5; similar on `08-10_06-35` and `08-10_07-29`). Suggests near-identical
   boxes on the same surfer aren't fully merging. Worth checking whether the
   NMS IoU threshold (`IOU_NMS=0.45` in `code/detect_surfers.py`) needs
   tightening.
2. **Small non-surfer objects getting boxed.** A bird (`08-09_08-23`) and a
   likely bird/otter (`08-10_09-17`) were detected — both smaller than a
   typical surfer box. Surfer boxes have run roughly 10-45px wide, 10-30px
   tall all session; a min-size (and/or aspect-ratio) filter could screen
   these out as a cheap post-processing step, same idea as the tree-bough/flag
   exclusion zones but based on size instead of location.
3. **Occasional residual flag leakage.** One enhancement-batch image
   (`08-07_06-26`) had a detection on "the flag top, which is black." The
   flag mask is coordinate-based so contrast enhancement shouldn't move it,
   but worth a spot check that the mask still holds on enhanced frames.
4. **Two-step candidate-then-verify idea (bigger scope).** Reviewer's own
   proposal: find dark, surfer-sized blobs against a smooth gray background
   first (classical blob detection), then selectively apply a lower YOLO
   confidence threshold only in those candidate regions, rather than
   lowering the threshold globally. Not started — a genuine sub-project if
   pursued.
5. **Fog/night image flagging — implemented 2026-08-25.** See
   "Image-quality gate implementation" below; the two-branch rule
   (`brightness < 75.4 OR lap_var < 12.7`) is live in
   `code/detect_surfers.py`, computed before detection runs. Known
   remaining weak spot: "moderate fog" frames near the `lap_var` boundary
   (roughly 13-28) are a genuinely continuous zone, not a clean cutoff —
   see the two-branch rule's error analysis below.
6. **Partial-fog / mixed-quadrant frames — data collected, policy decided,
   not yet acted on algorithmically.** Splitting a frame into 4 vertical
   quadrants and scoring each separately confirmed some "usable" frames
   have 1-3 clearly-foggy quadrants alongside clearly-clear ones (see
   `data/fog_review/quadrant_fog_scores.csv` and
   `data/fog_review/quadrant_mismatch/`). Reviewer decided: **for
   automated detection, any frame with a markedly foggy quadrant should be
   dropped**, even if the reviewer's own manual judgment would have kept
   it (their past labels reflect "is the visible part still countable,"
   which the automated pipeline can't reason about the way a human can).
   The current gate uses whole-image `lap_var` only, which matched the
   reviewer's *historical* labels better than worst-quadrant scoring did
   — implementing the stricter per-quadrant policy is a possible future
   refinement, not yet built into `detect_surfers.py`.

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
`code/detect_surfers.py`. The CLAHE+sharpen enhancement pipeline and
engineering leads #1-4 above are investigated but not yet implemented.
Lead #6 (per-quadrant partial-fog policy) has data and a
reviewer-confirmed direction but isn't coded yet.

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
