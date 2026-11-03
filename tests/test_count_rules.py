"""Counting rules: the hourly median/max (detect_surfers.hybrid_count) and the
evening email's three-look hour count (email_daily_report.hour_counts)."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import detect_surfers as ds  # noqa: E402
import email_daily_report as edr  # noqa: E402


def t(hh, mm):
    return datetime(2026, 10, 31, hh, mm)


# --- hybrid_count -------------------------------------------------------------

def test_hybrid_median_below_the_cut():
    assert ds.hybrid_count([1, 2, 3, 9, 0]) == 2


def test_hybrid_max_from_the_cut_up():
    assert ds.hybrid_count([5, 6, 7, 12, 6]) == 12


def test_hybrid_median_exactly_at_the_cut_takes_the_max():
    assert ds.HYBRID_CUT == 6
    assert ds.hybrid_count([6, 6, 6, 3, 8]) == 8


def test_hybrid_rounds_an_even_length_median():
    assert ds.hybrid_count([1, 2, 3, 4]) == 2   # median 2.5, round-half-even


# --- hour_counts --------------------------------------------------------------

def test_busy_hour_takes_the_max_look():
    actuals = [(t(7, 56), 15.0)]
    subs = [(t(8, 14), 17.0), (t(8, 32), 26.0)]
    (_w, count, looks), = edr.hour_counts(actuals, subs)
    assert looks == [15.0, 17.0, 26.0] and count == 26.0


def test_quiet_hour_takes_the_median_of_every_frame():
    actuals = [(t(12, 17), 3.0)]
    subs = [(t(12, 35), 2.0), (t(12, 53), 4.0)]
    frames = {t(12, 17): [3, 3, 2, 3, 4, 3, 3],
              t(12, 35): [2] * 10,
              t(12, 53): [1, 2, 2, 2, 4, 4, 2, 2, 2, 2]}
    (_w, count, _l), = edr.hour_counts(actuals, subs, frames)
    assert count == 2.0            # median of 27 frames, not the median of the looks (3)


def test_quiet_hour_without_frames_falls_back_to_the_looks():
    actuals = [(t(12, 17), 3.0)]
    subs = [(t(12, 35), 2.0), (t(12, 53), 4.0)]
    (_w, count, _l), = edr.hour_counts(actuals, subs, {})
    assert count == 3.0


def test_one_busy_look_of_three_is_not_a_majority():
    actuals = [(t(9, 44), 2.0)]
    subs = [(t(10, 2), 9.0), (t(10, 20), 1.0)]
    (_w, count, _l), = edr.hour_counts(actuals, subs, {})
    assert count == 2.0


def test_two_looks_split_one_busy_one_quiet_is_not_a_majority():
    actuals = [(t(16, 56), 7.0)]
    subs = [(t(17, 14), 2.0)]       # the +36 min set is missing
    (_w, count, looks), = edr.hour_counts(actuals, subs, {})
    assert looks == [7.0, 2.0] and count == pytest.approx(4.5)


def test_hour_without_between_hours_sets_passes_through():
    (_w, count, looks), = edr.hour_counts([(t(7, 2), 11.0)], [], {})
    assert count == 11.0 and looks == [11.0]


def test_sets_are_assigned_to_the_hour_before_them():
    actuals = [(t(7, 2), 1.0), (t(7, 56), 20.0)]
    subs = [(t(7, 20), 1.0), (t(7, 38), 1.0), (t(8, 14), 22.0), (t(8, 32), 25.0)]
    out = edr.hour_counts(actuals, subs, {})
    assert [len(l) for _w, _c, l in out] == [3, 3]
    assert out[1][1] == 25.0


# --- rejected_frames (the email's list of gate rejections) ---------------------

def test_rejected_frames_lists_hourly_and_between_hours(tmp_path):
    import pandas as pd
    from datetime import date
    preds = tmp_path / "p.csv"
    pd.DataFrame([
        {"date": "2026-11-03", "time_local": "09:08", "filename": "crop2026-11-03_09-08-00.jpg",
         "quality_ok": False, "quality_reason": "dark_or_night"},
        {"date": "2026-11-03", "time_local": "10:02", "filename": "crop2026-11-03_10-02-00.jpg",
         "quality_ok": True, "quality_reason": "ok"},
        {"date": "2026-11-02", "time_local": "06:53", "filename": "crop2026-11-02_06-53-00.jpg",
         "quality_ok": False, "quality_reason": "dark_or_night"},
    ]).to_csv(preds, index=False)
    sub = tmp_path / "s.csv"
    pd.DataFrame([{"date": "2026-11-03", "time_local": "09:26", "quality_ok": False,
                   "quality_reason": "foggy_or_blurred"}]).to_csv(sub, index=False)
    rows = edr.rejected_frames(date(2026, 11, 3), preds, sub)
    assert [r[1] for r in rows] == ["crop2026-11-03_09-08-00.jpg",
                                    "sub2026-11-03_09-26_f05.jpg (between-hours set)"]
    text = edr.rejected_frames_text(rows)
    assert "too dark" in text and "fog or blur" in text
    assert edr.rejected_frames_text([]) == ""
