# How the Object Detection Model Works

A plain-language walkthrough of how a raw camera frame becomes a surfer
count, plus a glossary for the terminology used throughout this repo and
in `PROJECT_HISTORY.md`.

## The pipeline, end to end

1. **Capture**: `code/get_clips.py` downloads short video clips from the
   Surfline camera.
2. **Frame + crop**: `code/get_cropped_frame.py` grabs a primary frame at
   2.5s in, two "side" frames 1.5s before and after it, and four more at
   0.5, 1.5, 3.5 and 4.5s (so 0.5-4.5s gives five frames one second apart),
   and crops each down to a fixed region of interest (ROI):
   a wide, short strip of the ocean, `1280×180` pixels. Three frames exist
   because counts on the *same* clip vary meaningfully second to second
   (occlusion by whitewater, a surfer paddling behind another) — see
   "Multi-frame count averaging" below.
3. **Image-quality gate**: before running the (expensive) model at all,
   the primary crop is checked for whether it's even usable — see
   "Image-quality gate" below. Frames that fail skip detection entirely
   (for all three frames); `surfer_count` is left blank for them in
   `data/predictions/predictions.csv` rather than guessed.
4. **Tiling**: that strip is an unusual shape for an object-detection model
   (very wide, very short), and surfers are small relative to the whole
   frame. Rather than feed the model the whole strip at once, each of the
   three frames is split into **4 overlapping horizontal tiles** (376×180px
   each, 20% overlap), matching how the training data was tiled. Smaller,
   more "normal-shaped" tiles make small, distant surfers easier for the
   model to find.
5. **Detection**: each tile is run through the trained **YOLOv8** model
   separately, producing a list of candidate boxes, each with a
   **confidence score**.
6. **Merge + de-duplicate**: because tiles overlap, a surfer near a tile
   boundary can get detected twice (once in each tile). All the boxes from
   all 4 tiles are translated back into the full-frame's coordinate system,
   then **Non-Maximum Suppression (NMS)** removes the duplicates. NMS alone
   misses one common case: a small box sitting entirely *inside* a larger
   one. Their IoU is low — the union is nearly the whole large box — so both
   survive, and one surfer gets counted twice. A second pass therefore drops
   the lower-confidence box of any pair whose overlap exceeds 70% of the
   *smaller* box's area. This cut clear-day counts from 107% of the human
   count to 102%.
7. **False-positive filtering**: a couple of known static objects in the
   scene (a tree bough on the right edge, a flag/wind sock at the bottom)
   reliably trigger false detections. A small coordinate-based filter drops
   boxes that land in those specific zones — see `code/detect_surfers.py`
   and `PROJECT_HISTORY.md` for exactly how those zones were derived.
8. **Count**: steps 4-7 run independently on every frame. The row's
   `surfer_count` is the median of the five one-second frames when that
   median is under 6, and their max from 6 up (`frame_count_hybrid`; see
   "Median/max count" below). Rows without all five frames fall back to the
   mean of the three 1.5s-apart frames, the count used before 2026-10-26.
   `confidence_avg` is the mean of the three frames' average confidences.
   Everything is appended to `data/predictions/predictions.csv` with the raw
   per-frame counts.

## Sub-hourly frame sets (collected from 2026-10-26)

Each night, `code/get_subhourly_frames.py` also takes two extra one-minute
clips per hour, 18 and 36 minutes after that hour's main clip, crops ten frames
from each (3 to 57 s, one every 6 seconds, same ROI) into
`data/j_shore_cam/subhourly_crops/<date>/`, and deletes the clip. The main
clips are about 54 minutes apart, so the three clips split each gap into even
thirds. Nothing reads these frames yet. They are being collected to test
whether sampling the hour three times, rather than once, improves the hourly
estimate: counts only decorrelate over about a minute, so one clip says little
about the rest of the hour. They take about 38 MB a day.

## Multi-frame count averaging

A single instant of video is a noisy sample of "how many surfers are out
there" — a surfer can be briefly hidden behind whitewater, mid-paddle
behind another surfer, or just outside a tile boundary in one specific
frame. Testing on real clips confirmed this: counts on the *same* ~6-second
clip varied by up to 7 surfers across frames just 1.5-3 seconds apart (see
`PROJECT_HISTORY.md`, 2026-08-25 entry, for the actual numbers).

To reduce that noise, `code/get_cropped_frame.py` extracts three frames per
clip instead of one (primary + two side frames), and `code/detect_surfers.py`
runs detection on all three (`run_inference_multi()`), averaging the
results. `predictions.csv` stores the full detail, not just the average:

- `frame_count_1` / `frame_count_2` / `frame_count_3` — the three raw
  per-frame counts, in chronological order (frame 2 is always the original
  2.5s primary frame, so this stays comparable to how counts were computed
  before this change).
- `frame_count_mean` / `frame_count_stdev` — the average (what
  `surfer_count` is rounded from) and how much the three frames disagreed.
  A high stdev is itself a signal: it flags a scene that's ambiguous or
  changing fast (overlapping surfers, choppy conditions), similar in spirit
  to how `lap_var` flags an unreliable *image* rather than an unreliable
  *count*.

**How much the averaging is actually worth** (measured 2026-10-14, three hours
sampled once per second — 10,862 frames, 36 clips; see
`analysis/count_variability/noise_floor/`). Against a clip's true per-second
level, one frame is off by **2.00** surfers and the rounded 3-frame mean by
**1.59** — a 20% reduction, where three *independent* samples would give 42%.
The three frames sit 1.5s apart and counts 1–3s apart correlate at r = 0.83 to
0.77, so they are close to the same observation. Counts only decorrelate over
about **60 seconds**: r = 0.58 at 10s, 0.03 at 60s.

Spreading the three frames across the whole clip instead would cut the error to
**1.17** on the 2026-08-28 hour, a 43% reduction rather than 20%. That is a real
and cheap improvement, but it needs a re-extraction and re-detection pass over
the archive to apply retroactively, so it is logged rather than adopted.

## Median/max count (2026-10-26)

The five frames one second apart are close to the same moment, so the
question is less noise than occlusion: a surfer behind a wave face in one
frame is visible in the next. Against 215 hand-counted frames, the mean of
three runs short in proportion to the crowd (−0.8 at 6-15 surfers, −2.8 at
31+). The max of five recovers those surfers but over-counts near-empty water
(+0.8), where one stray false positive in any frame sets it. The median keeps
the low end clean and the max fixes the rest:

| Hand count | Mean of 3 | Max of 5 | Median <6, max ≥6 |
|---|---|---|---|
| 0-5 | +0.08 | +0.77 | +0.06 |
| 6-15 | −0.82 | −0.02 | −0.07 |
| 16-30 | −1.88 | −0.12 | −0.12 |
| 31+ | −2.83 | −0.30 | −0.30 |

The crowd level is read from the median itself, since there is no truth to
choose by live. The cut at 6 was picked on the same 215 frames, so it should
be rechecked against new hand counts. New columns: `frame_count_t05`,
`frame_count_t15`, `frame_count_t35`, `frame_count_t45` (the 2.5s count is
`frame_count_2`) and `frame_count_hybrid`. Past hours whose clips were on
disk were counted the same way and migrated by `code/migrate_hybrid_counts.py`.
Hours forecast on the new count run about 6% higher; the share of the
packed-hour forecast gap it closes is small (−11.5 to −11.0 surfers).

**Historical backfill**: for existing `predictions.csv` rows where the raw
clip is still on disk, `code/backfill_multiframe_counts.py` fills in the
`frame_count_*` columns without touching the original `surfer_count`/
`confidence_avg` — those were already referenced by prior manual review
work, so they're left as-is; the new columns are purely additive. Rows
whose clip has since been deleted (per the clip-storage cleanup policy)
simply never get `frame_count_*` populated — `run_inference_multi()` falls
back to single-frame behavior automatically when side frames aren't
available, so nothing breaks, it just can't be improved retroactively.

## Image-quality gate

Fog, lens condensation, low light, and night frames all degrade detection
in ways that don't just lower confidence — the model often doesn't
generate any candidate boxes there at all, so there's nothing to recover
by tuning thresholds after the fact (see `PROJECT_HISTORY.md`'s
confidence-threshold experiment). Rather than count those frames
unreliably, `code/detect_surfers.py`'s `compute_image_quality()` checks
two cheap image statistics before detection ever runs:

- **Brightness** (mean grayscale pixel value) — too low means the frame is
  effectively night, even accounting for ambient light like streetlight
  reflections in fog.
- **Laplacian variance** (`lap_var`, a standard blur/detail proxy — see
  glossary) — too low means the frame is too foggy, blurred, or lens-
  condensation-affected to trust, independent of brightness.
A third statistic, **glare fraction**, is recorded per frame but does *not*
gate anything. It is the share of pixels brighter than `QUALITY_GLARE_LEVEL`
(240), written to the `glare_frac` column by `compute_glare_frac()`.
Low-angle autumn and winter sun lays a path of specular glitter across the
water and the detector reads the sparkle as heads — on labeled frames it
reported 49, 30, 18, 17, 16 and 15 surfers on water holding 0, 0, 0, 5, 0
and 3 — and the two checks above pass those frames comfortably, since
glittery water is both bright and sharp.

It is recorded rather than gated because the measure cannot separate glare
from sunlit whitewater at the low end: one frame at 0.0020 is ordinary foam
with about 30 plainly countable surfers, while 0.0032 is a glare path with
none. Neither small-blob share nor local-context brightness told them apart.
Gating on it would discard good crowded frames to catch bad ones, so the
column exists to make contaminated rows findable, and the real fix is the 21
labeled empty glare frames now in the training pool.

A frame fails the gate if brightness is below `QUALITY_BRIGHTNESS_THRESH`
**or** `lap_var` is below `QUALITY_LAPVAR_THRESH` (both in
`code/detect_surfers.py`). Two separate branches are used instead of one
combined score because night and fog push brightness in *opposite*
directions — night is dark, fog is artificially bright — so a single
brightness term can't represent both failure modes correctly at once.
Thresholds were fit on 169 hand-labeled images across four review
batches; see `PROJECT_HISTORY.md` for the full derivation, accuracy
numbers, and known limitations (this is a first-pass estimate, not a
solved problem).

A third rule (added 2026-09-22) catches a failure the first two let
through. Frames recorded **before first light or after last light** can be
both bright enough and sharp enough to pass, because the camera raises gain
in the dark and the amplified noise looks like detail. The tell is
counter-intuitive: *high* `lap_var`, since sensor noise is higher-frequency
than real water. So an out-of-window frame with `lap_var` above 180 is
rejected as `night_sensor_noise`.

The scope matters as much as the threshold. This rule was fit on 46
out-of-window frames counted by hand (28 unusable, 18 countable), where it
sorts 38 of the 46 correctly. Applied to *every* frame it would be actively
harmful — `brightness < 90 and lap_var > 180` rejects 29% of all frames that
currently pass, mostly bright mid-afternoon ones, including 272 frames
holding five or more surfers. Overcast daylight is dim and real water texture
is sharp, so the two conditions fire together constantly on good frames.
A clock-only rule fails too: one reviewed frame 14 minutes *before* first
light is perfectly countable, while frames 0-4 minutes out are not. Usable
light does not track the almanac, which is why the image statistic does the
work and the clock only decides where to apply it.

Failed frames are still recorded in `predictions.csv` — with
`quality_ok=False`, a `quality_reason`, and the measured `brightness`/
`lap_var` — so the data is kept for tuning, but `surfer_count` is left
blank rather than treated as zero or estimated.

## The model: YOLOv8

**YOLO** ("You Only Look Once") is a family of *single-pass* object
detectors — unlike older approaches that scan an image in multiple stages,
YOLO looks at the whole image once and directly predicts all the bounding
boxes, their classes, and their confidence scores in one forward pass.
That makes it fast enough to run on a laptop CPU, which matters here since
this pipeline has no GPU.

This project uses **YOLOv8s** (the "small" size variant) via the
[Ultralytics](https://docs.ultralytics.com/) library, trained on a single
class: `surfer`.

## The training data: CVAT

The training images (raw camera crops from the early manual test captures,
now archived — see `PROJECT_FILES.md`) were hand-annotated using
[CVAT](https://www.cvat.ai/) (Computer Vision Annotation Tool) — an
open-source browser tool for drawing bounding boxes on images. Each
surfer in each training image got a box drawn around it; CVAT then
exports that as a dataset the model can train on, in either **COCO** or
**YOLO** annotation format (see glossary below) — both formats were
exported and are kept in `data/cvat_out_coco/` and
`data/cvat_out_yolo_rebuilt/`.

The exported images were tiled to match the same 4-tile split described
above, so the model trains on exactly the shape of input it'll see in
production.

## Confidence threshold and NMS, concretely

Two numbers in `code/detect_surfers.py` control what actually gets
counted:

- **`CONF_THRESH` (0.195)**: the model assigns every candidate box a
  confidence score from 0-1. Anything below this cutoff is discarded
  before it's even considered a detection. Lower this number and the
  model reports more boxes (recovers weaker, less-certain detections, but
  risks more false positives).

  This was re-swept in September 2026 after the detector was retrained, on
  76 held-out frames holding 1,419 labeled surfers, and 0.195 survived: the
  whole-frame count error bottoms out at 0.200, essentially where it already
  sat, and the F1 score is nearly flat anywhere between 0.125 and 0.425.
  A tempting refinement — a *lower* threshold near the bottom of the frame,
  where whitewater makes surfers hardest to see — turned out not to pay:
  recall there rises only 78% to 80% and the change in count error sits
  inside the noise. The bottom line is that this number is no longer where the
  remaining accuracy is.
- **`IOU_NMS` (0.45)**: when two boxes overlap by more than this fraction
  (their **IoU** — see glossary), NMS treats them as duplicates of the
  same object and keeps only the higher-confidence one.
- **`CONTAINMENT_THRESH` (0.7)**: the nested-box rule from step 6 above —
  if two boxes overlap by more than 70% of the *smaller* box's area, the
  lower-confidence one is dropped. This catches the duplicates that IoU
  cannot see.

The daily detection animation (`data/charts/latest_detection.gif`,
generated by `code/plot_daily_prediction.py` and embedded at the top of
the README) draws this same `CONF_THRESH` cutoff directly on the image: a
box only appears if the model's confidence for it was 0.195 or higher, and
each box's label is that confidence score. The image's own caption repeats
the threshold so it's readable without cross-referencing this doc.

## Small-object detection: what the detector does about it

A surfer here is **6 to 11 pixels tall**, and the smallest decile covers about
61 px² — a few dozen water-coloured pixels. That is the central difficulty, and
several standard techniques bear on it. What this project uses, and what it
does not:

**Tiling (in use).** The 1280×180 strip is cut into 4 overlapping tiles of
376×180 and each is run separately, so the model sees a surfer at a far larger
fraction of the frame. This is the single biggest reason detection works at
all. It is the same idea as the published **SAHI** (Slicing Aided Hyper
Inference) method; this project implements its own slicing and cross-tile
de-duplication rather than using that library. Benchmarking against it remains
open.

**Feature pyramids (inherited).** YOLOv8 carries an FPN/PANet neck and three
detection heads at strides 8, 16 and 32, so features are combined across
scales. Nothing here configures that; it comes with the architecture. The
stride-8 head is the one doing essentially all the work at this object size —
the stride-32 head covers objects far larger than any surfer in frame.

**Input resolution (in use).** Tiles are run at `imgsz 640`, so a 376×180 tile
is upscaled. That gives the stride-8 head roughly 47×22 cells across a tile,
which is what makes an 8-pixel object addressable at all.

**Photometric augmentation (in use).** Training applies `hsv_h 0.015`,
`hsv_s 0.7`, `hsv_v 0.4`, horizontal flips, ±10° rotation and mosaic. The
value jitter at 0.4 is why global brightness or contrast normalisation at
inference is not expected to help — the model is already largely invariant to
a global tone curve.

**Geometric scale augmentation (barely).** `scale` is 0.1, a ±10% zoom range.
This is deliberate rather than neglected: measured across 742 boxes, apparent
size tracks image row at rho = +0.60, but the near-to-far ratio is only
**2.1× in area**. The ROI is a narrow horizontal slice of the perspective
field, so the scale range the model must cover is small by the standards of
street-scene detection, where 20–50× is routine.

**Perspective normalisation (not used, and not worth it).** Warping to a ground
plane so objects are size-constant is a standard fixed-camera technique. At a
2.1× range it would buy very little and cost interpolation blur.

**Not used, and candidates:** a P2 detection head at stride 4 (the standard
YOLO modification for objects below ~16 px), small-object copy-paste
augmentation (`copy_paste` is currently 0.0), and size-weighted loss. These are
logged in the private ideas notes rather than attempted.

The real limit: at 61 px² the information is close to what the sensor and the
camera's H.264 encoder preserved. No architecture recovers detail that was
never captured.

## Detector accuracy: the deployed checkpoint

The production detector is the September 2026 retrain, which added labeled fog,
glare and pose data and ran 60 epochs. The checkpoint actually deployed is
`best.pt` — **epoch 46**, the epoch with the best mAP@0.5:0.95, which is what
the training run saves as "best" — not the last epoch. Validation DFL loss
creeps back up after about epoch 40 while training DFL loss keeps falling, a
mild sign of overfitting, and that is why the run does not simply keep the end
of training.

On the validation set that checkpoint scores **precision 88.5%, recall 81.3%,
[mAP](#term-map)@0.5 86.8%, mAP@0.5:0.95 38.5%**. These are the numbers quoted
in the daily chart's caption. They sit on a harder validation set than the
previous model's, which now includes hazy fog frames, so they do not compare
directly with its 85.6% / 82.0%.

mAP@0.5:0.95 (38.5%) being far below mAP@0.5 (86.8%) is expected rather than
alarming: it averages over much stricter box-overlap requirements, up to
near-perfect box placement, which is not what this project needs — a box that
lands on the right surfer counts that surfer correctly whether or not it hugs
the outline.

![YOLOv8s detector training metrics — loss, precision, recall, mAP over 60 epochs](../analysis/detector_training_metrics/detector_training_metrics.png)

The per-epoch training log above is real, not reconstructed after the fact. The
white dotted line on each chart is a 5-epoch rolling average. The top and bottom
rows' first three charts are training and validation [loss](#term-loss) (box,
class, DFL); the validation row is the meaningful one, since it is measured on
images the model never trains on. The last two charts in each row are
[precision](#term-precision) / [recall](#term-recall) and mAP, all computed on
validation data each epoch. Precision and recall climb from noisy starting
values to roughly 88% and 81%, flattening by about epoch 40 — training had
largely converged well before epoch 60.

**What the retrain was worth, measured on whole-frame counts rather than
validation boxes**, on frames neither model trained on: on hazy frames the new
model finds **98%** of the real surfers against **59%** for the old one, and
across all held-out frames the average error per frame fell from 5.16 surfers to
0.94. Against independent hand counts its average error per frame halved, 2.61 →
1.28. The one regression was ordinary clear-day frames, counted about 3–7% high;
most of that was one surfer boxed twice, a small box nested inside a larger one,
which survives overlap-based de-duplication because the two boxes share little
area relative to the larger one. Suppressing nested boxes brought clear-day
counts to **102%** of the human count and cut average error there by 38%, with
hand-counted frames landing at exactly 100%.

### Against a human count

The validation figures above score boxes against labelled boxes. The question
the forecast depends on is simpler: is each box on a surfer, and is every surfer
boxed? That was checked by hand on production frames — **220 frames across 16
days spanning a year, 3,880 real surfers**:

| | |
|---|---|
| precision — of the boxes it drew, how many were on a surfer | **98.7%** |
| recall — of the surfers really there, how many it found | **91.9%** |
| average count error | **1.5 surfers** per frame |

These run higher than the validation set's 88.5% / 81.3% because they ask a
more forgiving question: a box on the right surfer counts even if it does not
hug the outline closely enough to pass an overlap test. Neither is wrong; they
measure different things, and both are kept.

![Detector count against a hand count, one point per reviewed frame](https://raw.githubusercontent.com/jf-silverman/sum-surfers/assets/detector_accuracy.png)

In a single frame the detector counts about 7% low (fitted slope 0.94), and
the gap widens in a crowded lineup. The chart above plots the hourly count
instead, which since 2026-10-26 takes the median of five frames a second apart
on near-empty water and their max from 6 surfers up; that recovers most of the
surfers a single frame misses, and the slope is 0.99 (see "Median/max count"). **It misses far more than it invents.** Of 316
missed surfers, 286 had no box at all and 30 were inside a box drawn around
several people; of 46 false positives, 35 were on something that is not a
surfer — a bird, a wind sock, a head on the beach — and 11 were a second box on
a surfer already counted. Two in three misses were lying on their boards, and
only 10 were standing: a rider is an unmistakable shape, a prone surfer a dark
smudge a few pixels across. Hand counting is itself imperfect, and 8 boxes and
45 possible misses were flagged as judgement calls the reviewer would not
defend either way. The raw record is
[`data/reviews/review_all.csv`](../data/reviews/review_all.csv).

### Across every confidence threshold: precision-recall and FROC

The figures above are at the production confidence threshold (0.195). To see
the whole trade-off, the detector's boxes on the 61 labelled frames it never
trained on (1,509 surfers) were replayed through the production path at every
threshold from 0.02 to 0.95 (`analysis/detector_pr_curve/`).

The usual summary of a yes/no classifier, an ROC curve, does not apply here: it
plots true positives against the *rate* of false positives, which needs a count
of true negatives, and a detector has none to count, since every patch of empty
water is a "correct rejection". Detection uses two substitutes:

- **Precision-recall**: precision against recall as the threshold moves. Its
  area is the average precision ([AP](#term-map)), the detection counterpart of
  the area under an ROC curve.
- **[FROC](#term-froc) (free-response ROC)**: the share of real objects found
  against the number of false marks per image. It comes from medical imaging,
  where a radiologist marks suspected lesions anywhere on a scan rather than
  calling the whole image normal or abnormal (Bunch et al., 1978), which is the
  same problem as boxing surfers anywhere in a frame. It remains a standard way
  to evaluate lesion-detection systems.

![Precision-recall and FROC curves across confidence thresholds](../analysis/detector_pr_curve/pr_froc.png)

| Box counts as found if it overlaps the label by | AP | At 0.195: precision | recall | false boxes per frame |
|---|---|---|---|---|
| IoU 0.5 (the usual standard) | 0.778 | 0.87 | 0.80 | 3.0 |
| IoU 0.3 | 0.890 | 0.95 | 0.88 | 1.1 |

Surfer boxes are 10-20 px, so a box a few pixels off can fail IoU 0.5 while
sitting on the right surfer; the IoU 0.3 curve is the one comparable with the
hand review above. The production threshold sits at the knee of both curves:
lowering it toward 0.02 adds about 2 points of recall while roughly doubling
the false boxes, and raising it costs recall faster than it gains precision.
Recall levels off near 0.82 and 0.90 even at the lowest threshold, because the
remaining misses are never proposed at any confidence; only better training
data or a better model would find them.

## Predictors: weather, tide, swell, wind, energy, consistency

Separate from detection, `code/get_surf_predictors.py` pulls conditions
data for Jack's from Surfline's public forecast API
(`services.surfline.com/kbyg/spots/forecasts/*`) and writes it to
`data/predictor_vars/surfline_predictors.csv`, matched to `predictions.csv` rows by
filename/nearest hour. This is forward-looking only (today + tomorrow) —
it runs on every scheduled pipeline execution and just accumulates
whatever "today" happens to be each time.

For **historical** dates, the same endpoints accept a `start=YYYY-MM-DD`
parameter, but require an authenticated, premium Surfline session (an
`x-auth-accesstoken` header) — anonymous requests are capped at
yesterday. `code/backfill_historical_predictors.py` is a separate,
manually-run script (not part of the scheduled pipeline) that uses this
to backfill predictors for existing `predictions.csv` rows. See that
script's docstring for usage, and `PROJECT_HISTORY.md` for how the
mechanism was discovered.

## Forecast accuracy and calibration

The forecast model is off by about **7.6 surfers** ([MAE](#term-mae)) on a
typical count of around 16 when asked to predict days it has never seen. Treat
its outputs as directional estimates, not precise counts.

**That number went up on 2026-09-23, and the model did not get worse.** Until
then accuracy was measured with a random 80/20 split of *hours*. At roughly 13
hours per day that puts hours from the same day on both sides of the split: the
model learned a day's crowd level from that day's own hours and was then scored
on the rest of it, which is not a forecast but a fill-in. Splitting by whole day
costs about 1.2 MAE, and holding out the most recent days rather than random ones
costs about another 0.8, because the model then also has to cope with a genuine
seasonal level shift. The published figure is now the forward-in-time one: fit on
the past, scored on days that had not happened yet, which is the only version
that matches how the forecast is used.

### The live record

Held-out rows test the model; the live record tests the forecast as published.
Every forecast is written to `data/forecasts/forecast_log.csv` before its day
and scored against the detector's count afterwards, so the record cannot be
revised after the fact. From 2026-09-23 to 2026-10-21, **2,555 scored hours**:
the forecast made the evening before missed by **5.4 surfers** on average (367
hours) and ran about 1.2 high; at seven days out the average miss was 6.4. The
80% range held at every lead, catching 80–85% of actual counts. In September it
had fallen to 63% at seven days out, before lead-dependent widening was added.

![Forecast error and prediction-range coverage against lead time](https://raw.githubusercontent.com/jf-silverman/sum-surfers/assets/forecast_accuracy.png)

The record so far covers autumn only. Neither the hurdle-fitted band edges nor
the nightly recalibration of the median, both adopted 2026-10-20/21 and
described below, is in these figures yet.

Empirical coverage of the GBT quantile [prediction intervals](#term-prediction-interval)
is checked directly against a held-out split by
`analysis/surf_count_model_calibration/plot_calibration.py`, rather than trusted
from the nominal target:

![Prediction-interval calibration for the surf-count model](../analysis/surf_count_model_calibration/calibration_plot.png)

Re-measured 2026-10-20 on 417 held-out rows (2026-09-20 → 2026-10-19), holding
out the most recent days as a block:

| nominal | 20% | 40% | 60% | **75%** | 80% |
|---|---:|---:|---:|---:|---:|
| empirical coverage | 16.8% | 32.4% | 49.2% | **59.5%** | 74.8% |

Every band is overconfident — real counts fall outside them more often than the
nominal level says — and the narrow bands badly so. The 75% band is the one
production draws; it misses **33.6% low against a 15% target** and 7.0% high
against 10%.

**The low edge is now the weak side, and that is a reversal.** The 2026-09-23
measurement had the 80% band missing 9.4% low and 18.8% high, with the high edge
losing. The direction flipped with the window, and the cause is visible in the
error-by-count breakdown in [`MODEL_CARD.md`](../MODEL_CARD.md): the model runs
+4.01 surfers on hours with 0–5 actual, so the lower bound sits above actuals
that are at or near zero.

**An awkward detail worth stating.** On this same split the old q0.10 lower edge
had not collapsed — it still fitted (at `min_samples_leaf` 200, the last
candidate tried) and was a varying model, not a floor. Measured on these rows the
old 80% band covers 74.8% of its target and the new 75% band only 59.5% of its.
The collapse is real on the full record the chart fits, where every candidate
from 20 to 200 returns a training standard deviation of 0.0000, so the chart had
no alternative — but the band is not better calibrated for having moved, only
more accurately labelled. The fix for the coverage itself is the hurdle model
below, or recalibration, not another edge level — and the hurdle has since been
adopted for the edges, taking 59.5% to 70.5% on the same rows.

**A rule for reading any coverage number here**, learned twice the hard way:
a coverage figure near its target is only good news once you have checked that
both edges can actually be missed. Before the night-frame review this check read
82% coverage while missing low only 4.6% of the time — the band's bottom edge sat
under 1 surfer for 72% of hours, so there was almost nothing left to undershoot,
and the apparent calibration came from a floor rather than from the model.
Resolving 46 frames recorded outside the light window (28 genuinely unusable, and
entering the record as empty hours or phantom counts) un-pinned that lower bound,
which now sits under 1 surfer for 22% of hours instead of 72%. Coverage fell from
82% to 75.5% *because* the interval became informative. The same thing happened on
2026-09-08, when correcting the swell data moved the lower bound off zero and
dropped coverage from 82.8% to 72%.

**It happened a third time, and the fix was different.** From the 2026-10-12
refit the lower edge pinned at zero on every hour again — but this time the data
was right and the level was wrong. The empirical 10th percentile of hourly
surfer count is now exactly 0.00, because the share of empty hours is seasonal
and rising into winter (7.6% in 2026-08, 19.5% in 2026-10, against 33.8% and
38.9% in 2025-11 and 2025-12), so a flat zero is the correct 10th percentile
rather than a failed fit. On 2026-10-20 the band's lower edge moved to **q0.15,
making the interval 75% central rather than 80%** — the first level that still
varies with the predictors. The numbers quoted above, and the 80% figures
elsewhere in this file, describe the band as it was before that date; grouping
`forecast_log.csv` by `band_lower_q` separates the two eras.

### Zero-inflation, and why the band is 75%

The reason the lower edge keeps drifting onto the floor is that the count
distribution is **zero-inflated and seasonally so**. Empty hours by month:

| | 2026-07 | 2026-08 | 2026-09 | 2026-10 | 2025-11 | 2025-12 |
|---|---:|---:|---:|---:|---:|---:|
| hours with zero surfers | 8.0% | 7.6% | 13.0% | 19.5% | 33.8% | 38.9% |

Past roughly a tenth, the true 10th percentile of the count is zero, so a model
fit to that level is *correct* to predict zero everywhere — "always predict 0"
already nearly minimises the pinball loss. This is not a hyperparameter that can
be tuned around: on the 2026-10-20 record, every `min_samples_leaf` candidate
from 20 to 200 collapsed to a flat zero with a training standard deviation of
0.0000. The 15th percentile is the first level that still varies with the
predictors, which is where the band's lower edge now sits.

### Two automatic checks, and the model that replaces this one if they trip

The October 2026 collapse ran for eight days before anyone noticed, because
nothing failed — the band was computed, drawn and recorded as designed, and a
band with its floor on zero looks like a band. Two checks now run with every
scored day:

- **The lower edges of the newest forecast.** Reported when the edge does not
  vary at all across the forecast hours, or sits at zero on at least 80% of
  them. Until 2026-10-20 any zero was reported, which caught an earlier
  four-day partial episode; once the edges came from a hurdle fit (below), a
  zero on a dawn or dusk hour became an informative answer rather than a
  symptom, and that rule would have fired every night.
- **The share of empty hours**, cumulative and trailing 30 days, against a **30%
  threshold**. The leading indicator, and because it moves seasonally it warns
  before a band collapses rather than after.

Both are two small table reads, about 11 ms, with no refitting: the check only
decides whether the question needs looking at.

Lowering the edge again was never going to be the answer indefinitely, because
past some point the problem is the model's shape. One quantile model was being
asked to describe two things at once — whether anyone is out at all, and how
many are out given somebody is. A **hurdle model** separates them: a classifier
for `P(count > 0)`, quantile models fit on non-empty hours only, composed back
so that a target quantile q predicts zero when `P(empty)` exceeds q and the
conditional quantile at `(q - p_zero) / (1 - p_zero)` otherwise.

**The band's two edges have been fitted this way since 2026-10-20; the median
has not.** The first measurement of this concluded the opposite — better pinball
on 5 of 5, worse coverage on 3 of 5 — and was wrong, because it split rows at
random and so leaked each test day's own crowd level into training (the same
flaw as B21). Re-measured on a forward split by whole days, and on four
successive ~15-day rolling windows, by `code/eval_hurdle_model.py`:

| on 417 held-out hours | one model per level | hurdle |
|---|---:|---:|
| mean pinball loss | 2.2616 | **2.2512** (CI [−0.050, +0.030]) |
| 75% band coverage | 59.0% | **70.5%** |
| missed below the band (target 15%) | 33.3% | **22.5%** |
| 80% band coverage | 72.4% | **82.3%** |
| median bias, hours with 0–5 surfers | +4.67 | +4.49 |

Smaller below-band miss rate in **4 of 4** rolling windows; lower pinball in
1 of 4. So: adopted for the edges, where it fixes the lower tail repeatably at
no cost on the proper scoring rule, and deliberately **not** for the median,
whose +4.5 bias on near-empty hours it leaves where it was. That is a
point-prediction problem for recalibration, not something a hurdle addresses —
which is worth stating plainly, because the hurdle was originally proposed here
as the fix for exactly that bias and it is not.

One consequence for reading a forecast: a hurdle lower edge of 0.00 is now an
answer rather than a failure. It means `P(no surfers)` exceeded the band's lower
quantile for that hour — a dawn hour in late October, correctly — and on the
2026-10-21 outlook that is 35 of 90 hours. The nightly check was changed to
match: it fires when the lower edge does not **vary** across the day, or sits at
zero on 80% or more of its hours, rather than on any zero at all.

The limit now is the classifier: AUC 0.826 but it under-predicts emptiness
(11.7% against an actual 16.5%), which accounts for most of the distance between
22.5% and the 15% target. Isotonic calibration lifts AUC to 0.856 and Brier to
0.1042 while moving that mean only to 12.5%, and the week-by-week shortfall
tracks the rising autumn empty share — forward distribution shift rather than a
calibration-method choice.

**The median is corrected outside the model, nightly, since 2026-10-21.**
`fit_point_recalibration` in `plot_daily_prediction.py` fits
`actual ≈ a + b × predicted_raw` by least squares on every scored row of
`forecast_log.csv`; `apply_point_recalibration` maps each hour's median through
it and clamps the result into `[lower edge, upper edge]` and at zero. The first
fit was a = −3.86, b = 1.147, on 2,451 scored rows over 28 target dates.

It is a correction rather than a model change because in-model ideas had all
come back flat, and the problem is one trees cannot fix. When the crowd level
drifts below the training range, a tree's leaves cannot follow it. A line fitted
on the recent record can. It chases a moving level, so it is refit every run
rather than frozen.

Measured forward in time on the live record, each target date corrected only
from dates scored before it: lead-1 MAE 5.695 → 5.142, all-lead MAE 6.193 →
5.510, bias +1.93 → +0.15, and better on 23 of 26 dates (sign test p = 0.0001).
Clamping costs 0.008 MAE and guarantees lower ≤ median ≤ upper.

Three design points matter more than the numbers:

- **It fits on `predicted_raw`, never on `predicted`.** From 2026-10-21 on,
  `predicted` is the corrected value. If the fit read that column, each night
  would be fitting a correction to the previous night's correction, and it would
  settle at "no change" whatever the model's real error was. Every row written
  before that date was uncorrected, so `build_forecast_log.py` fills
  `predicted_raw` from `predicted` for those rows. That is a fact about those
  rows, not a guess.
- **It turns itself off rather than failing.** Below 400 scored rows or 7
  target dates, or if the fitted slope leaves 0.5–1.5, the map is the identity
  and the chart says so in its output. A missing log never stops the chart.
- **It is separable from the hurdle band.** MAE depends only on the median and
  coverage only on the edges, so the two 2026-10-20/21 changes can be told apart
  in the live record.

It does not fix everything. By actual count, 0–5 goes +4.54 → +2.98 but 30+
goes −13.27 → −14.54. A single line cannot correct both the over-forecast quiet
hours and the under-forecast packed ones, because the packed hours are mostly
forecast in the middle of the range. A median of 0 now appears on quiet hours
and is earned: forward in time, 71.3% of the hours sent to 0 were empty, and
their MAE fell from 0.89 to 0.39.

Full investigation in `PROJECT_HISTORY.md`.

## Known limitations (short version)

Fog, lens condensation, and low light all degrade the model's ability to
generate any candidate boxes at all — not just lower their confidence.
The image-quality gate above (implemented 2026-08-25) catches the clearest
cases of this; the boundary between "moderately foggy but still countable"
and "too foggy to trust" is inherently continuous, not a hard line, so
some frames will still be misclassified either direction. Retraining on
labeled fog and glare frames in September 2026 did more for this than any
filter: on held-out hazy frames the detector went from finding 59% of the
real surfers to 98%.

Things other than surfers still get boxed — birds (and, in one measured
case, a bird's *reflection*), people walking the beach, and one surfer
occasionally split into two side-by-side boxes.
[What isn't a surfer](non_surfer_objects.md) catalogs each one with its
visual tell and current handling.

The forecast's prediction range is **75%, not 80%** — the lower edge moved from
the 10th to the 15th percentile on 2026-10-20, because a rising share of empty
hours made the 10th percentile exactly zero and a band floored at zero cannot be
missed from below. Coverage figures measured before that date describe the older
band; grouping `data/forecasts/forecast_log.csv` by `band_lower_q` separates the
two. See "Zero-inflation, and why the band is 75%" above.

Full investigation, metrics, and known error rates are in
`PROJECT_HISTORY.md`.

---

## Glossary

- <a id="term-bounding-box"></a>**Bounding box** — a rectangle (x1, y1, x2, y2 coordinates) drawn around
  a detected object.
- <a id="term-confidence-score"></a>**Confidence score** — the model's own estimate (0-1) of how sure it is
  that a given box actually contains the object class it's predicting.
- <a id="term-cvat"></a>**CVAT** — Computer Vision Annotation Tool, the software used to
  hand-label surfers in the training images. [cvat.ai](https://www.cvat.ai/)
- <a id="term-coco-format"></a>**COCO format** — a common JSON-based annotation format for object
  detection datasets (named after the "Common Objects in Context" dataset
  that popularized it).
- <a id="term-epoch"></a>**Epoch** — one full pass through the entire training dataset during
  training. This model trained for 60 epochs; the weights actually deployed
  are the best-scoring checkpoint along the way (epoch 46 for the current
  detector), not the last one, since later epochs can overfit.
- <a id="term-gbt"></a>**GBT (Gradient-Boosted Trees)** — the model type used for the
  surfer-count forecast (a separate model from the YOLOv8 detector above,
  see `PROJECT_HISTORY.md`'s "Surfer Count Prediction Model" section).
  Builds many small decision trees one after another, each one correcting
  the errors of the trees before it.
- <a id="term-glm"></a>**GLM (Generalized Linear Model)** — a family of statistical models
  (e.g. Poisson, negative-binomial) tried as an alternative to GBT for
  the surfer-count forecast; GBT ended up more accurate, see
  `PROJECT_HISTORY.md`.
- <a id="term-ground-truth"></a>**Ground truth** — the "correct answer" — in this project, the
  hand-drawn CVAT boxes (for the training/test data) or a human's manual
  count (for the later review batches), used to measure how well the
  model is actually doing.
- <a id="term-image-quality-gate"></a>**Image-quality gate** — the pre-detection check (see above) that skips
  running the model on frames too dark or too low-detail to reliably
  count, based on brightness and `lap_var`, plus a narrower rule that
  rejects frames outside the first-light/last-light window whose high
  `lap_var` is amplified sensor noise rather than detail.
- <a id="term-iou"></a>**IoU (Intersection over Union)** — a measure of how much two boxes
  overlap: the area they share divided by the total area they cover
  together. 1.0 = identical boxes, 0.0 = no overlap at all. Used both to
  decide if two predicted boxes are "the same" detection (NMS) and to
  decide if a predicted box "matches" a ground-truth box when scoring
  accuracy.
- <a id="term-loss"></a>**Loss** — a single number a model tries to minimize during
  training: how wrong its current predictions are, measured against
  ground truth. Each epoch adjusts the model to make this number a
  little smaller. YOLOv8 tracks three separate loss components: **box
  loss** (how far off a predicted box's edges are from the real box),
  **class loss** (how confident/correct the predicted class label
  was — with only one class, "surfer," here, this mostly measures how
  well the model recognizes "surfer-ness" itself), and **DFL loss**
  (Distribution Focal Loss, a more fine-grained measure of exactly how
  precisely each box edge is placed). All three should trend downward
  over training if it's going well; each is tracked separately for the
  training data and the held-out validation data, so a training loss
  that keeps dropping while validation loss stalls or rises is a sign of
  overfitting.
- <a id="term-laplacian-variance"></a>**Laplacian variance (`lap_var`)** — a standard no-reference blur/detail
  metric: apply a Laplacian (edge-detecting) filter to the image, then
  take the variance of the result. Sharp, detailed images have lots of
  strong edges and a high variance; blurry, foggy, or low-texture images
  have few strong edges and a low variance. Used here as a general
  "unreliable image" signal, since fog, lens condensation, and smooth
  low-texture water all suppress it similarly.
- <a id="term-froc"></a>**FROC (free-response receiver operating characteristic)** — recall
  (share of real objects found) plotted against false marks per image as the
  confidence threshold moves. Developed in radiology for lesion detection,
  where findings can appear anywhere on an image and there are no countable
  true negatives; used here for the same reason. See "Across every
  confidence threshold" above.
- <a id="term-mae"></a>**MAE (Mean Absolute Error)** — the average, across every prediction, of
  how far off (in either direction) that prediction was from the real
  value, ignoring sign — e.g. an MAE of 6 surfers means predictions are,
  on average, 6 surfers away from the actual count. Used to judge the
  surfer-count forecast model, not the detector (which uses mAP below).
- <a id="term-map"></a>**mAP (mean Average Precision)** — a standard single-number summary of
  object-detection quality, combining precision and recall across
  different confidence thresholds. Shows up in this model's training
  logs (`results.csv`) but is a *box-level* metric — not directly
  comparable to the *count-level* MAE/bias numbers used elsewhere in this
  project's review work (see `PROJECT_HISTORY.md` for why).
- <a id="term-nms"></a>**NMS (Non-Maximum Suppression)** — the de-duplication step: among a
  group of overlapping boxes (above the IoU threshold), keep only the one
  with the highest confidence and discard the rest. Because IoU compares
  overlap against the *combined* area of both boxes, it does not catch a
  small box nested inside a large one; this pipeline adds a separate
  containment check for that case (see step 6).
- <a id="term-prediction-interval"></a>**Prediction interval** (vs. **confidence interval**) — both describe a
  range of uncertainty, but around different things. A confidence
  interval is about an *average* (e.g. "the average count at 2pm is
  probably between X and Y"); a prediction interval is about *one
  individual future observation* (e.g. "today's count at 2pm will
  probably be between X and Y"). The surfer-count forecast's shaded band
  in the README is a prediction interval — the right concept for a
  one-day forecast, not a long-run average. Built here from **quantile
  regression**: instead of one GBT model predicting the average count, a
  separate GBT is fit for each percentile (e.g. the 10th and 90th), and
  the gap between them forms the interval.
- <a id="term-precision"></a>**Precision** — of everything the model *flagged* as a surfer, what
  fraction actually were surfers. Low precision = lots of false positives.
- <a id="term-recall"></a>**Recall** — of everything that *actually was* a surfer, what fraction
  did the model find. Low recall = lots of missed surfers (undercounting).
- <a id="term-roi"></a>**ROI (Region of Interest)** — the fixed rectangular crop of the raw
  camera frame that the pipeline actually processes (the ocean strip,
  excluding sky/shore clutter above and below it).
- <a id="term-tiling"></a>**Tile / tiling** — splitting one image into smaller overlapping pieces
  before running the model, then merging the results back together. Used
  here because the ROI's wide, short shape and the surfers' small size
  within it make full-frame detection less reliable than tiled detection.
- <a id="term-yolo"></a>**YOLO (You Only Look Once)** — the object-detection model family used
  in this project. [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/)
- <a id="term-yolo-format"></a>**YOLO format (annotation)** — a plain-text annotation format (one
  `.txt` file per image, one line per box: class + normalized center
  x/y/width/height) — simpler than COCO's JSON, and what
  `code/detect_surfers.py`'s ground-truth comparisons were built from.

## Main resources

- [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/) — the
  library and model family used for detection.
- [CVAT](https://www.cvat.ai/) — the annotation tool used to build the
  training/test datasets.
