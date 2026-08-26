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
