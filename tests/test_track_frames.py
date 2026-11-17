"""track_frames.present_count (2026-11-16): surfers present at one moment,
including those the detector missed in that frame but caught around it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import track_frames as tf  # noqa: E402


def box(x, y, h=10, w=10):
    return [x, y, x + w, y + h, 0.6]


def test_a_surfer_missed_at_t_but_seen_either_side_is_present():
    frames = [(1.5, [box(100, 50), box(300, 60)]), (2.5, [box(100, 50)]), (3.5, [box(100, 51), box(301, 60)])]
    assert tf.present_count(frames, 2.5) == 2


def test_a_surfer_seen_only_before_t_is_not_present():
    """Left the frame (or paddled off) before the moment: not counted."""
    frames = [(1.5, [box(100, 50), box(300, 60)]), (2.5, [box(100, 50)]), (3.5, [box(100, 51)])]
    assert tf.present_count(frames, 2.5) == 1


def test_sitting_surfers_that_drift_keep_one_track():
    frames = [(t, [box(100 + 2 * t, 50)]) for t in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert tf.present_count(frames, 2.5) == 1
    assert len(tf.track(frames)) == 1


def test_a_fast_standing_rider_is_one_surfer_not_several():
    """2026-11-15: a rider moving ~30 px/s broke into three 'surfers'."""
    frames = [(t, [box(500 + 30 * t, 40, h=22)]) for t in (0.5, 1.0, 1.5, 2.5, 3.5, 4.0, 4.5)]
    assert len(tf.track(frames)) == 1


def test_two_surfers_side_by_side_stay_two():
    frames = [(t, [box(200, 50), box(225, 50)]) for t in (0.5, 1.5, 2.5)]
    assert tf.present_count(frames, 2.5) == 2


def test_the_detector_writes_the_tracked_column():
    import detect_surfers as ds
    assert ds.CSV_HEADER[-1] == "frame_count_tracked"
