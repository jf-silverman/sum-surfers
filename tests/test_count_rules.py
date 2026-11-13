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


# --- tide_by_hour (the email chart's tide line) --------------------------------

def test_tide_line_spans_the_forecast_hours(tmp_path, monkeypatch):
    """2026-11-08: drawn from the training table, the line covered only counted
    hours (7-16) while the forecast ran 6-17."""
    import pandas as pd
    from datetime import date
    rec = pd.DataFrame({"hour_local": [f"{h:02d}:00" for h in range(6, 18)],
                        "predicted": 0.0, "tide_ft": [4.0 + 0.1 * h for h in range(6, 18)]})
    rec.to_csv(tmp_path / "forecast_2026-11-08.csv", index=False)
    monkeypatch.setattr(edr.pdp, "FORECASTS_DIR", tmp_path)
    tide = edr.tide_by_hour(date(2026, 11, 8))
    assert [w.hour for w, _ in tide] == list(range(6, 18))


# --- hour_frames (which frame the email shows for an hour) ----------------------

def test_frames_follow_the_table_count(tmp_path, monkeypatch):
    """2026-11-11: the table's 3 pm hour counted 82 from its 3:44 set, but the
    frame picker used the hourly clips alone and called 4:02 pm (68) the
    busiest hour and the biggest miss."""
    import pandas as pd
    from datetime import date
    hourly = {"13:20": 2, "15:08": 48, "16:02": 68}
    pd.DataFrame([{"date": "2026-11-11", "time_local": t, "quality_ok": True, "surfer_count": c,
                   "filename": f"crop2026-11-11_{t.replace(':', '-')}-00.jpg",
                   **{k: c for k in edr.HOURLY_FRAME_COLS}} for t, c in hourly.items()]
                 ).to_csv(tmp_path / "p.csv", index=False)
    sets = {"13:38": 3, "13:56": 13, "15:26": 57, "15:44": 82, "16:20": 64, "16:38": 58}
    rows = []
    for t, c in sets.items():
        frames = [c - 5] * 10
        frames[7] = c
        rows.append({"date": "2026-11-11", "time_local": t, "quality_ok": True, "set_count": c,
                     **{f"count_f{i:02d}": v for i, v in enumerate(frames)}})
    pd.DataFrame(rows).to_csv(tmp_path / "s.csv", index=False)
    monkeypatch.setattr(edr, "PREDICTIONS_CSV", tmp_path / "p.csv")
    monkeypatch.setattr(edr, "SUBHOURLY_CSV", tmp_path / "s.csv")
    hours = {h["hour"].strftime("%H:%M"): h for h in edr.hour_frames(date(2026, 11, 11))}
    assert hours["15:08"]["count"] == 82
    assert hours["15:08"]["path"].name == "sub2026-11-11_15-44_f07.jpg"
    assert hours["16:02"]["path"].name == "crop2026-11-11_16-02-00.jpg"      # its own clip is the max
    assert hours["13:20"]["path"].name == "crop2026-11-11_13-20-00.jpg"      # a median hour keeps its clip
    assert max(hours.values(), key=lambda h: h["count"])["hour"].strftime("%H:%M") == "15:08"


def test_extreme_frames_draws_the_picked_frames(tmp_path, monkeypatch):
    """Smoke test of the drawing step: a variable clash there (the hour entry
    and the image height both named h) crashed the email on 2026-11-13."""
    import numpy as np
    import pandas as pd
    from datetime import date
    crops = tmp_path / "crops"; crops.mkdir()
    rows = []
    for t, c in {"07:02": 0, "13:20": 3, "16:02": 40}.items():
        name = f"crop2026-11-11_{t.replace(':', '-')}-00.jpg"
        import cv2
        cv2.imwrite(str(crops / name), np.zeros((180, 1280, 3), np.uint8))
        rows.append({"date": "2026-11-11", "time_local": t, "quality_ok": True, "surfer_count": c,
                     "filename": name, **{k: c for k in edr.HOURLY_FRAME_COLS}})
    pd.DataFrame(rows).to_csv(tmp_path / "p.csv", index=False)
    monkeypatch.setattr(edr, "PREDICTIONS_CSV", tmp_path / "p.csv")
    monkeypatch.setattr(edr, "SUBHOURLY_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(edr, "OUT_DIR", tmp_path)
    monkeypatch.setattr(edr.pdp.ds, "CROPS_DIR", crops)
    monkeypatch.setattr(edr.pdp, "render_detection_frame",
                        lambda path, model: (np.zeros((180, 1280, 3), np.uint8), 7))
    out = edr.extreme_frames(date(2026, 11, 11), model=object())
    assert [label for label, *_ in out] == ["Busiest hour", "Empty hour", "Quietest hour above zero"]
    assert all(p.exists() for _l, p, _v, _s in out)
