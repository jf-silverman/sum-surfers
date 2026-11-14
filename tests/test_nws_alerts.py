"""NWS alert flags (fetch_nws_alerts.py, 2026-11-14)."""

import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import fetch_nws_alerts as nws  # noqa: E402


def alerts_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=nws.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in nws.FIELDS})
    return path


def test_flags_a_warning_overlapping_daylight(tmp_path):
    p = alerts_csv(tmp_path / "a.csv", [
        {"alert_id": "1", "event": "High Surf Warning", "onset": "2026-11-13T18:00:00-08:00",
         "ends": "2026-11-15T04:00:00-08:00"}])
    assert [e for e, *_ in nws.flagged_alerts(date(2026, 11, 14), p)] == ["High Surf Warning"]
    assert nws.flagged_alerts(date(2026, 11, 15), p) == []          # ends at 4 am, before daylight
    text = nws.flag_text(date(2026, 11, 14), p)
    assert text.startswith("NWS High Surf Warning (until Sun 4 AM)")


def test_beach_hazards_and_small_craft_are_not_flagged(tmp_path):
    """Beach Hazards Statements go out on big-swell days that draw surfers."""
    p = alerts_csv(tmp_path / "a.csv", [
        {"alert_id": "1", "event": "Beach Hazard Statement", "onset": "2026-11-07T05:00:00-08:00",
         "ends": "2026-11-09T22:00:00-08:00"},
        {"alert_id": "2", "event": "Small Craft Advisory", "onset": "2026-11-08T05:00:00-08:00",
         "ends": "2026-11-08T22:00:00-08:00"}])
    assert nws.flagged_alerts(date(2026, 11, 8), p) == []
    assert nws.flag_text(date(2026, 11, 8), p) == ""


def test_utc_archive_times_and_duplicates(tmp_path):
    """The IEM archive gives UTC 'Z' times; the live feed gives offsets. The
    same warning from both sources is shown once."""
    p = alerts_csv(tmp_path / "a.csv", [
        {"alert_id": "iem:x", "event": "High Surf Warning", "onset": "2026-11-14T02:00:00Z",
         "ends": "2026-11-15T12:00:00Z", "source": "iem"},
        {"alert_id": "nws:y", "event": "High Surf Warning", "onset": "2026-11-13T18:00:00-08:00",
         "ends": "2026-11-15T04:00:00-08:00", "source": "nws"}])
    assert len(nws.flagged_alerts(date(2026, 11, 14), p)) == 1


def test_no_file_means_no_flag(tmp_path):
    assert nws.flag_text(date(2026, 11, 14), tmp_path / "absent.csv") == ""
