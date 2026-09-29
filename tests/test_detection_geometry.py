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


def test_box_outside_every_zone_survives():
    box = [5, 5, 15, 15, 0.9]
    assert ds.filter_false_positive_zones([box]) == [box]
