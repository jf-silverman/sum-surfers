# Wave Set and Foam Detection: Research and Plan

2026-10-29. Not started; parked until Joel is ready.

## Summary

Simple OpenCV tools look sufficient for both detectors, but each needs specific
tricks that a first naive test showed.

- **Swell-set detector:** flags when a surfable swell line crosses the lineup.
  Joel's cue: a dark wave face toward shore, a possible breaking foam line on
  top, lighter flat water behind, repeating as bands.
- **Post-wave foam detector:** flags a wide swath of bright, mottled foam,
  mostly near shore, more likely just after a set.
- **Why it matters:** surfers ride out of frame behind the cliff and disappear
  behind wide waves during sets, so counts taken mid-set run low. A set flag
  would let us weight or pick frames, and pairs with the standing-rider idea
  (standing count as a set-in-progress flag).

## What the project already has

From `docs/model_and_feature_ideas.md` (2026-10-01, "Ideas for detecting a wave
passing through the lineup"):

- **A validated foam measure.** Foam pixels = brightness (HSV V) above the
  frame's own median by 30+, low saturation. 9,805 px on a breaking set against
  33 px four minutes later. The threshold must be relative to the frame median,
  or it drifts with the light.
- **Only lineup foam mattered.** Splitting the 180 px strip into thirds, lineup
  foam vs count r = −0.536 (p = 0.040, n = 15, detrended); inshore r = −0.413
  (p = 0.13); top r = −0.240 (p = 0.39).
- **The gap:** a clean unbroken swell line hides sitting surfers but makes no
  foam, so foam alone misses it. Untested swell ideas logged there: a row-wise
  brightness profile, horizontal edges, and using the frames over time.

## What the research says

- **Foam / whitecaps.** The standard method is Callaghan & White's Automated
  Whitecap Extraction (AWE): a brightness threshold chosen per image from the
  intensity distribution, so it follows the light. Later work uses local
  thresholds (Otsu on 64×64 patches) because uneven light breaks a global one,
  and U-Net models. Sun glint is the known failure.
- **Breaking waves on coastal cameras.** The Argus / USGS CoastCam tradition
  works from pixel brightness over time (timestacks). Brightness peaks at the
  foam front just ahead of the crest; the slope of the traces gives wave speed.
  Ten-minute time exposures average breaking into a white band over sandbars.
- **Crest detection.** Published approaches: Radon-transform line detection on
  timestacks, crest tracking with a crest-collapse function, and a CNN that
  finds break points and crests at a New Zealand surf break. No published
  Gabor-filter surf-cam work found; oriented filters are a standard texture
  tool, so that would be our own combination.

## Quick test on four frames (2026-10-29)

`wave_foam_probe_2026-10-29.png` in this folder: four frames from the riding
clips (wave face breaking, lull, set breaking with foam, lull after a set), with
a row brightness profile, a horizontal-edge ratio, and a naive foam mask.

- **Foam works but needs cleanup.** It found the breaking wave, but also the
  horizon and sun glitter on the far water; on 10/26 15:29 the glitter covered
  more pixels than the real foam. Real foam is so mottled it broke into small
  pieces rather than one wide swath.
- **A full-width brightness profile does not work.** The horizon dominates it,
  and a wave line spanning part of the width averages away. The banding is
  visible to the eye but needs a local, oriented measure with static lines
  removed.

## Proposed foam detector

1. Use only the production strip (y 420–600), split into lineup and inshore
   bands. Drops the sky and the far-water glitter.
2. Foam pixels: V above the frame median by 30+ and low saturation
   (`cv2.cvtColor` to HSV). Optionally a local threshold per patch (Otsu) for
   uneven light.
3. Join mottled foam with a wide horizontal closing (`cv2.morphologyEx`,
   `cv2.MORPH_CLOSE`, kernel about 25×5), then `cv2.connectedComponentsWithStats`
   and keep swaths wider than about 15% of the frame width (Joel's "wide swath").
4. Glitter check: glitter stays sparse specks after closing, foam fills its
   area; use each swath's fill ratio (pixels / bounding box) to separate them.
5. Outputs per frame: foam share per band, widest swath width, a yes/no flag.

## Proposed swell-line detector

1. **Remove what does not move.** Horizon, power lines, fence and shoreline are
   static. Subtract a median background built from the clip's own frames, or
   from the 10 frames of a between-hours set (`np.median`, or
   `cv2.createBackgroundSubtractorMOG2`). This uses time, as the coastal-camera
   literature does, and is the key step.
2. **Joel's banding as a signed vertical gradient.** Reading down the image: light
   flat water, then the dark face, then lighter water. A signed `cv2.Sobel` in y
   after `cv2.GaussianBlur` marks the light-to-dark crest edge. A very wide
   horizontal smoothing (`cv2.blur`, about 151×1) keeps long coherent lines and
   averages out short chop.
3. **Orientation check:** `cv2.getGaborKernel` tuned to horizontal at the swell's
   spacing, or a structure tensor from the Sobel pair; keep only near-horizontal
   structure.
4. **Line extraction:** threshold, then keep components spanning 25%+ of the
   width, or `cv2.HoughLinesP` limited to near-horizontal angles. Output: a wave
   line is crossing the lineup, its row and its length.
5. **Sets vs single waves:** one frame says whether a wave is present; a set is
   several in a row, which the 10-frame between-hours sets (6 s apart) show
   directly.

Other OpenCV tools likely useful: `cv2.createCLAHE` (local contrast, already
proven for fog), `cv2.reduce` (row/column profiles), `cv2.calcOpticalFlowFarneback`
(coherent shoreward motion separates a wave from chop).

## How we would know it works

1. Joel marks about 100 frames: swell line on the lineup yes/no, wide foam
   yes/no. The between-hours sets and the riding clips give natural
   before/during/after sequences.
2. Measure each detector's agreement with Joel's marks (precision, recall).
3. The real test: do counts in frames flagged "set in progress" run lower than
   the next lull frames, and by how much?

## Sources

- [Automated Processing of Sea Surface Images for the Determination of Whitecap Coverage (Callaghan & White)](https://research.universityofgalway.ie/en/publications/automated-processing-of-sea-surface-images-for-the-determination-/)
- [Callaghan et al. 2008, GRL](https://data-ww3.ifremer.fr/BIB/Callaghan_etal_GRL2008.pdf)
- [Whitecap extraction under uneven illumination (Remote Sensing 2022)](https://doi.org/10.3390/rs14225691)
- [NOAA whitecap method review](https://repository.library.noaa.gov/view/noaa/60518/noaa_60518_DS1.pdf)
- [NOAA U-Net whitecap extraction](https://repository.library.noaa.gov/view/noaa/71822/noaa_71822_DS1.pdf)
- [Adaptive whitecap threshold (EGU 2020)](https://presentations.copernicus.org/EGU2020/EGU2020-15621_presentation.pdf)
- [Argus image types and conventions](https://www.marinespecies.org/introduced/wiki/Argus_image_types_and_conventions)
- [Argus timestacks and breaking-wave counts (Univ. Bologna)](https://cris.unibo.it/handle/11585/47569)
- [USGS CoastCam timestack data](https://cmgds.marine.usgs.gov/catalog/spcmsc/madbeach_c1_timestacks_metadata.html)
- [Breaking wave height from timex images (Remote Sensing)](https://baes.uc.pt/bitstream/10316/106658/1/Breaking-wave-height-estimation-from-timex-images-Two-methods-for-coastal-video-monitoring-systemsRemote-Sensing.pdf)
- [Radon-transform crest extraction from timestacks (Remote Sensing 2021)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-03372/article_deploy/remotesensing-13-03372.pdf)
- [Surf crest tracking algorithm in wave image](https://www.scientific.net/AMM.482.341)
- [CNN detection of surf break points and crests (ICCE)](https://icce-ojs-tamu.tdl.org/icce/article/download/12664/11937/38907)
- [Gabor wavelength/orientation extraction (USPTO patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10859507)
