"""Forecast rain and Surfline discussion collectors (2026-11-18)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import fetch_rain_forecast as rf  # noqa: E402
import fetch_surfline_discussion as fsd  # noqa: E402


def test_nws_six_hour_amount_is_spread_over_its_hours_in_local_time():
    props = {"quantitativePrecipitation": {"values": [
                 {"validTime": "2026-11-18T18:00:00+00:00/PT6H", "value": 12.0}]},
             "probabilityOfPrecipitation": {"values": [
                 {"validTime": "2026-11-18T18:00:00+00:00/PT1H", "value": 80}]}}
    h = rf.nws_hourly(props)
    assert sorted(h) == [f"2026-11-18T{x:02d}:00" for x in range(10, 16)]   # 18 UTC = 10 PST
    assert all(v["precip_mm"] == 2.0 for v in h.values())
    assert h["2026-11-18T10:00"]["pop_pct"] == 80 and "pop_pct" not in h["2026-11-18T11:00"]


def test_nws_interval_with_days():
    start, hours = rf._interval("2026-11-18T00:00:00+00:00/P1DT6H")
    assert hours == 30


def test_rain_merge_replaces_same_night_and_keeps_others(tmp_path):
    p = tmp_path / "r.csv"
    row = dict(source="nws", fetched_date="2026-11-18", valid_local="2026-11-19T10:00", precip_mm=1, pop_pct=50)
    rf.merge([row, {**row, "fetched_date": "2026-11-17"}], p)
    assert rf.merge([{**row, "precip_mm": 3}], p) == 2
    assert "3" in p.read_text() and ",1," in p.read_text()


def test_discussion_html_becomes_plain_text():
    assert fsd.plain("<strong>Dawn:</strong> Strong S wind.<br>\n<br>\nJumbled &amp; lumpy. ") == \
        "Dawn: Strong S wind. Jumbled & lumpy."


def test_discussion_keeps_each_nights_snapshot(tmp_path):
    p = tmp_path / "d.csv"
    c = [{"forecastDay": "2026-11-19", "human": True, "dayToWatch": False,
          "forecaster": {"name": "X"}, "headline": "Solid", "observation": "ok"}]
    fsd.merge(fsd.rows_from(c, "2026-11-17"), p)
    assert fsd.merge(fsd.rows_from(c, "2026-11-18"), p) == 2
    assert fsd.merge(fsd.rows_from(c, "2026-11-18"), p) == 2
