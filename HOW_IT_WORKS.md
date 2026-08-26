# How the Object Detection Model Works

A plain-language walkthrough of how a raw camera frame becomes a surfer
count, plus a glossary for the terminology used throughout this repo and
in `PROJECT_HISTORY.md`.

## The pipeline, end to end

1. **Capture**: `code/get_clips.py` downloads short video clips from the
   Surfline camera.
2. **Frame + crop**: `code/get_cropped_frame.py` grabs one frame per clip
   and crops it down to a fixed region of interest (ROI) — a wide, short
   strip of the ocean, `1280×180` pixels.
3. **Image-quality gate**: before running the (expensive) model at all,
   each crop is checked for whether it's even usable — see "Image-quality
   gate" below. Frames that fail skip detection entirely; `surfer_count`
   is left blank for them in `data/predictions.csv` rather than guessed.
4. **Tiling**: that strip is an unusual shape for an object-detection model
   (very wide, very short), and surfers are small relative to the whole
   frame. Rather than feed the model the whole strip at once, it's split
   into **4 overlapping horizontal tiles** (376×180px each, 20% overlap),
   matching how the training data was tiled. Smaller, more "normal-shaped"
   tiles make small, distant surfers easier for the model to find.
5. **Detection**: each tile is run through the trained **YOLOv8** model
   separately, producing a list of candidate boxes, each with a
   **confidence score**.
6. **Merge + de-duplicate**: because tiles overlap, a surfer near a tile
   boundary can get detected twice (once in each tile). All the boxes from
   all 4 tiles are translated back into the full-frame's coordinate system,
   then **Non-Maximum Suppression (NMS)** removes the duplicates.
7. **False-positive filtering**: a couple of known static objects in the
   scene (a tree bough on the right edge, a flag/wind sock at the bottom)
   reliably trigger false detections. A small coordinate-based filter drops
   boxes that land in those specific zones — see `code/detect_surfers.py`
   and `PROJECT_HISTORY.md` for exactly how those zones were derived.
8. **Count**: whatever boxes survive all of the above become that frame's
   surfer count, appended to `data/predictions.csv` along with the average
   confidence across the kept boxes.

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
  risks more false positives). This project found lowering it barely
  helps in genuine fog (the model isn't generating candidate boxes there
  at all) but does help meaningfully on otherwise-clear images (see
  `PROJECT_HISTORY.md`).
- **`IOU_NMS` (0.45)**: when two boxes overlap by more than this fraction
  (their **IoU** — see glossary), NMS treats them as duplicates of the
  same object and keeps only the higher-confidence one.

## Predictors: weather, tide, swell, wind, energy, consistency

Separate from detection, `code/get_surf_predictors.py` pulls conditions
data for Jack's from Surfline's public forecast API
(`services.surfline.com/kbyg/spots/forecasts/*`) and writes it to
`data/surfline_predictors.csv`, matched to `predictions.csv` rows by
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

## Known limitations (short version)

Fog, lens condensation, and low light all degrade the model's ability to
generate any candidate boxes at all — not just lower their confidence.
The image-quality gate above (implemented 2026-08-25) catches the clearest
cases of this; the boundary between "moderately foggy but still countable"
and "too foggy to trust" is inherently continuous, not a hard line, so
some frames will still be misclassified either direction. Full
investigation, metrics, and known error rates are in `PROJECT_HISTORY.md`.

---

## Glossary

- **Bounding box** — a rectangle (x1, y1, x2, y2 coordinates) drawn around
  a detected object.
- **Confidence score** — the model's own estimate (0-1) of how sure it is
  that a given box actually contains the object class it's predicting.
- **CVAT** — Computer Vision Annotation Tool, the software used to
  hand-label surfers in the training images. [cvat.ai](https://www.cvat.ai/)
- **COCO format** — a common JSON-based annotation format for object
  detection datasets (named after the "Common Objects in Context" dataset
  that popularized it).
- **Epoch** — one full pass through the entire training dataset during
  training. This model trained for 60 epochs.
- **Ground truth** — the "correct answer" — in this project, the
  hand-drawn CVAT boxes (for the training/test data) or a human's manual
  count (for the later review batches), used to measure how well the
  model is actually doing.
- **Image-quality gate** — the pre-detection check (see above) that skips
  running the model on frames too dark or too low-detail to reliably
  count, based on brightness and `lap_var`.
- **IoU (Intersection over Union)** — a measure of how much two boxes
  overlap: the area they share divided by the total area they cover
  together. 1.0 = identical boxes, 0.0 = no overlap at all. Used both to
  decide if two predicted boxes are "the same" detection (NMS) and to
  decide if a predicted box "matches" a ground-truth box when scoring
  accuracy.
- **Laplacian variance (`lap_var`)** — a standard no-reference blur/detail
  metric: apply a Laplacian (edge-detecting) filter to the image, then
  take the variance of the result. Sharp, detailed images have lots of
  strong edges and a high variance; blurry, foggy, or low-texture images
  have few strong edges and a low variance. Used here as a general
  "unreliable image" signal, since fog, lens condensation, and smooth
  low-texture water all suppress it similarly.
- **mAP (mean Average Precision)** — a standard single-number summary of
  object-detection quality, combining precision and recall across
  different confidence thresholds. Shows up in this model's training
  logs (`results.csv`) but is a *box-level* metric — not directly
  comparable to the *count-level* MAE/bias numbers used elsewhere in this
  project's review work (see `PROJECT_HISTORY.md` for why).
- **NMS (Non-Maximum Suppression)** — the de-duplication step: among a
  group of overlapping boxes (above the IoU threshold), keep only the one
  with the highest confidence and discard the rest.
- **Precision** — of everything the model *flagged* as a surfer, what
  fraction actually were surfers. Low precision = lots of false positives.
- **Recall** — of everything that *actually was* a surfer, what fraction
  did the model find. Low recall = lots of missed surfers (undercounting).
- **ROI (Region of Interest)** — the fixed rectangular crop of the raw
  camera frame that the pipeline actually processes (the ocean strip,
  excluding sky/shore clutter above and below it).
- **Tile / tiling** — splitting one image into smaller overlapping pieces
  before running the model, then merging the results back together. Used
  here because the ROI's wide, short shape and the surfers' small size
  within it make full-frame detection less reliable than tiled detection.
- **YOLO (You Only Look Once)** — the object-detection model family used
  in this project. [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/)
- **YOLO format (annotation)** — a plain-text annotation format (one
  `.txt` file per image, one line per box: class + normalized center
  x/y/width/height) — simpler than COCO's JSON, and what
  `code/detect_surfers.py`'s ground-truth comparisons were built from.

## Main resources

- [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/) — the
  library and model family used for detection.
- [CVAT](https://www.cvat.ai/) — the annotation tool used to build the
  training/test datasets.
