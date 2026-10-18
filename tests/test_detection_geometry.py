"""Geometry tests for the detection post-processing.

These cover the code that has actually produced silent bugs in this project:
the 4-tile split, the cross-tile NMS merge, and the nested-box suppression that
was added after the retrained detector started counting one surfer twice. None
of them loads a model or reads a real frame -- they are about arithmetic, which
is the part that fails quietly. A wrong box merge does not raise; it just
returns a number that is slightly off, and nothing downstream can tell.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import detect_surfers as ds  # noqa: E402


# --------------------------------------------------------------------------
# Tiling geometry
# --------------------------------------------------------------------------
def tile_windows(width):
    """The x-ranges tile_image_paths would cut, without touching the disk."""
    out = []
    for i in range(ds.NUM_TILES):
        x0 = i * ds.STEP_X
        x1 = min(x0 + ds.TILE_W, width)
        x0 = max(x1 - ds.TILE_W, 0)
        out.append((x0, x1))
    return out


# 4 tiles of 376 px at a step of 301 reach x=1279, so the rightmost column of a
# 1280-wide ROI is in no tile. Writing this test is what surfaced that. It is
# left as-is rather than fixed: one column cannot hold a surfer, and changing
# the tiling would alter the exact input the production model was trained on,
# which is not worth it for a pixel. The bound below is what makes the test
# useful anyway -- it still fails loudly if someone changes NUM_TILES, TILE_W or
# the overlap and opens a gap that could actually hide a detection.
MAX_UNCOVERED_PX = 2


def test_tiles_cover_the_frame_width_to_within_a_pixel():
    """Almost every column of a 1280-wide ROI must fall inside some tile.

    A real gap here is the worst kind of bug available in this file: surfers in
    the uncovered strip are never shown to the model at all, so they are not
    missed detections -- they are simply absent, and the count is quietly low.
    """
    width = 1280
    covered = np.zeros(width, dtype=bool)
    for x0, x1 in tile_windows(width):
        covered[x0:x1] = True
    uncovered = int((~covered).sum())
    assert uncovered <= MAX_UNCOVERED_PX, (
        f"{uncovered} column(s) of the ROI are in no tile, above the known "
        f"{MAX_UNCOVERED_PX}px right-edge remainder")


def test_tiles_overlap_by_the_configured_amount():
    """Adjacent tiles must actually overlap, or a surfer on a seam is cut in two."""
    windows = tile_windows(1280)
    for (a0, a1), (b0, b1) in zip(windows, windows[1:]):
        overlap = a1 - b0
        assert overlap > 0, "adjacent tiles do not overlap"
        # The last tile is right-aligned when the frame is short, so its overlap
        # can exceed the nominal step; the others should match OVERLAP_PX.
        assert overlap >= ds.OVERLAP_PX


def test_last_tile_is_right_aligned_not_truncated():
    """A short frame must not yield a narrow final tile.

    YOLO would letterbox a narrow tile to a different scale than its siblings,
    so the same surfer would be a different size depending on which tile it
    landed in.
    """
    for width in (1280, 1200, 1100):
        x0, x1 = tile_windows(width)[-1]
        assert x1 - x0 == ds.TILE_W, f"last tile is {x1 - x0}px wide at width {width}"
        assert x1 <= width


# --------------------------------------------------------------------------
# Cross-tile NMS
# --------------------------------------------------------------------------
def test_nms_merges_the_same_surfer_seen_in_two_tiles():
    """The case the overlap exists to create: one surfer, two nearly identical
    boxes from neighbouring tiles. Exactly one should survive, the higher-scoring
    one."""
    boxes = [[300, 80, 320, 100, 0.90], [301, 81, 321, 101, 0.60]]
    kept = ds.nms_across_tiles(boxes, iou_thresh=0.45)
    assert len(kept) == 1
    assert kept[0][4] == pytest.approx(0.90)


def test_nms_keeps_two_genuinely_separate_surfers():
    boxes = [[100, 80, 120, 100, 0.9], [400, 80, 420, 100, 0.9]]
    assert len(ds.nms_across_tiles(boxes, iou_thresh=0.45)) == 2


def test_nms_on_empty_input():
    assert ds.nms_across_tiles([]) == []


# --------------------------------------------------------------------------
# Nested-box suppression (the 2026-09-22 clear-day overcount)
# --------------------------------------------------------------------------
def test_small_box_inside_a_large_one_is_dropped():
    """The bug this function was written for.

    IoU between these two is only ~0.06, so NMS keeps both and the surfer is
    counted twice. Measuring the overlap against the SMALLER area gives 1.0.
    """
    big = [100, 50, 200, 150, 0.80]
    small = [140, 90, 160, 110, 0.60]
    iou_num = (160 - 140) * (110 - 90)
    iou_den = (200 - 100) * (150 - 50) + (160 - 140) * (110 - 90) - iou_num
    assert iou_num / iou_den < 0.45, "test premise: plain NMS would keep both"
    kept = ds.suppress_contained([big, small], thresh=0.7)
    assert len(kept) == 1
    assert kept[0] == big


def test_suppression_keeps_the_higher_confidence_box_not_the_larger():
    """Greedy by confidence, not by size: if the small box scores higher it wins."""
    big = [100, 50, 200, 150, 0.40]
    small = [140, 90, 160, 110, 0.95]
    kept = ds.suppress_contained([big, small], thresh=0.7)
    assert len(kept) == 1
    assert kept[0] == small


def test_side_by_side_surfers_are_not_merged():
    """Touching but distinct boxes must both survive, or a crowded lineup
    collapses into a handful of boxes."""
    a = [100, 50, 140, 90, 0.9]
    b = [138, 50, 178, 90, 0.9]
    assert len(ds.suppress_contained([a, b], thresh=0.7)) == 2


def test_passing_thresh_none_uses_the_default_rather_than_disabling():
    """`thresh=None` means "use CONTAINMENT_THRESH", not "turn this off".

    Worth pinning down: the docstring's "None disables it" refers to setting the
    module constant to None, and reading it the other way -- as this test
    originally did -- gets the behaviour backwards.
    """
    boxes = [[100, 50, 200, 150, 0.8], [140, 90, 160, 110, 0.6]]
    assert len(ds.suppress_contained(boxes, thresh=None)) == 1


def test_suppression_is_disabled_by_a_none_module_constant(monkeypatch):
    monkeypatch.setattr(ds, "CONTAINMENT_THRESH", None)
    boxes = [[100, 50, 200, 150, 0.8], [140, 90, 160, 110, 0.6]]
    assert len(ds.suppress_contained(boxes)) == 2


def test_suppression_is_a_noop_on_fewer_than_two_boxes():
    assert ds.suppress_contained([]) == []
    one = [[1, 2, 3, 4, 0.5]]
    assert ds.suppress_contained(one) == one


# --------------------------------------------------------------------------
# False-positive zones
# --------------------------------------------------------------------------
def test_box_centred_in_the_tree_zone_is_dropped():
    zx1, zy1, zx2, zy2 = ds.TREE_MASK
    cx, cy = (zx1 + zx2) / 2, (zy1 + zy2) / 2
    box = [cx - 5, cy - 5, cx + 5, cy + 5, 0.9]
    assert ds.filter_false_positive_zones([box]) == []


def test_flag_zone_drops_only_low_confidence_boxes():
    """The wind sock is a persistent false positive, but a real surfer can drift
    through the same patch -- so the zone is confidence-gated, not absolute."""
    zx1, zy1, zx2, zy2 = ds.FLAG_MASK
    cx, cy = (zx1 + zx2) / 2, (zy1 + zy2) / 2
    low = [cx - 5, cy - 5, cx + 5, cy + 5, ds.CONF_THRESH_FLAG_MASK - 0.05]
    high = [cx - 5, cy - 5, cx + 5, cy + 5, ds.CONF_THRESH_FLAG_MASK + 0.05]
    assert ds.filter_false_positive_zones([low]) == []
    assert ds.filter_false_positive_zones([high]) == [high]


def test_bluff_zone_needs_the_zone_the_edge_and_the_confidence_together():
    """A person on the cliff path enters the ROI only as the top of a head, so
    the box is always clipped by the bottom edge. All three conditions are
    required -- a surfer in the water at the same x sits clear of the edge, and
    a confident detection there is trusted."""
    zx1, _zy1, zx2, _zy2 = ds.BLUFF_MASK
    cx = (zx1 + zx2) / 2
    conf = ds.CONF_THRESH_BLUFF_MASK - 0.05
    clipped = [cx - 6, 169.0, cx + 6, 180.0, conf]
    assert ds.filter_false_positive_zones([clipped]) == []

    clear_of_edge = [cx - 6, 158.0, cx + 6, ds.BLUFF_EDGE_Y - 1, conf]
    assert ds.filter_false_positive_zones([clear_of_edge]) == [clear_of_edge]

    confident = [cx - 6, 169.0, cx + 6, 180.0, ds.CONF_THRESH_BLUFF_MASK + 0.05]
    assert ds.filter_false_positive_zones([confident]) == [confident]

    left_of_zone = [zx1 - 40, 169.0, zx1 - 28, 180.0, conf]
    assert ds.filter_false_positive_zones([left_of_zone]) == [left_of_zone]


def test_bluff_zone_drops_every_box_the_hand_review_judged():
    """The 24 boxes the rule fires on in a 300-frame sample were all reviewed
    by hand (19 shore heads, 1 duplicate, 4 real surfers). The filter must drop
    exactly that set -- it is the evidence the rule was adopted on."""
    csv = Path(__file__).resolve().parents[1] / "data" / "reviews" / \
        "bluff_head_review" / "bluff_head_review.csv"
    if not csv.exists():
        pytest.skip("review set not present")
    rows = list(csv.open())[1:]
    boxes = []
    for line in rows:
        f = line.rstrip("\n").split(",")
        boxes.append([float(f[-5]), float(f[-4]), float(f[-3]), float(f[-2]), float(f[4])])
    assert len(boxes) == 24
    assert ds.filter_false_positive_zones(boxes) == []


def test_box_outside_every_zone_survives():
    box = [5, 5, 15, 15, 0.9]
    assert ds.filter_false_positive_zones([box]) == [box]


# --- crowd-dependence (added 2026-10-18, both rules default OFF) -------------

def test_crowd_dependence_is_off_by_default():
    """The constants ship inert, so a default run is the measured production one."""
    assert ds.CROWD_THRESH_SCHEDULE is None
    assert ds.CONTAINMENT_DENSITY_SLOPE == 0.0
    assert ds.crowd_threshold(0) == ds.CONF_THRESH
    assert ds.crowd_threshold(500) == ds.CONF_THRESH


def test_crowd_threshold_reads_the_last_matching_step():
    sched = [(0, 0.195), (10, 0.17), (20, 0.15), (30, 0.12)]
    assert ds.crowd_threshold(0, sched) == 0.195
    assert ds.crowd_threshold(9, sched) == 0.195
    assert ds.crowd_threshold(10, sched) == 0.17
    assert ds.crowd_threshold(29, sched) == 0.15
    assert ds.crowd_threshold(1000, sched) == 0.12


def test_local_density_counts_neighbours_not_self():
    # three boxes: two within 40px of each other, one far away
    boxes = [[100, 100, 110, 110, 0.9],
             [115, 100, 125, 110, 0.8],
             [600, 100, 610, 110, 0.7]]
    assert ds.local_density(boxes, radius=40) == [1, 1, 0]
    assert ds.local_density(boxes, radius=5) == [0, 0, 0]


def test_density_slope_leaves_isolated_boxes_at_the_flat_threshold():
    """The point of the slope: it cannot alter a box that has no neighbour.

    Containment suppression was adopted for isolated clear-day false positives,
    so a rule that loosens it in crowds must leave the isolated case alone.
    """
    # a small box fully inside a big one, far from anything else
    boxes = [[100, 100, 140, 140, 0.9], [110, 110, 125, 125, 0.5]]
    flat = ds.suppress_contained(boxes, 0.7, slope=0.0)
    sloped = ds.suppress_contained(boxes, 0.7, slope=0.10)
    assert len(flat) == 1          # contained box dropped
    assert sloped == flat          # and the slope changes nothing here


def test_density_slope_spares_a_duplicate_only_when_it_is_in_a_crowd():
    """With neighbours present the same overlap stops counting as a duplicate."""
    pair = [[100, 100, 140, 140, 0.9], [108, 108, 136, 136, 0.5]]
    # overlap/smaller is above 0.7, so flat suppression drops the small box
    assert len(ds.suppress_contained(pair, 0.7, slope=0.0)) == 1
    # surround them with neighbours: the threshold rises and both survive
    crowd = pair + [[c, 100, c + 10, 110, 0.4] for c in (95, 125, 150)]
    out = ds.suppress_contained(crowd, 0.7, slope=0.10)
    kept = [b for b in out if b in pair]
    assert len(kept) == 2
