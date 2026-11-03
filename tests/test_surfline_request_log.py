"""The per-request Surfline log (surfline_request_log.py) and the evening
email's Cloudflare note built from it."""

import csv
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import email_daily_report as edr  # noqa: E402
import surfline_request_log as srl  # noqa: E402


def response(status, url, mitigated="", ray="8d1f"):
    return SimpleNamespace(status_code=status, url=url, elapsed=timedelta(milliseconds=120),
                           headers={"cf-mitigated": mitigated, "cf-ray": ray})


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_query_strings_are_never_written(tmp_path):
    """The clip API carries the account token in its query."""
    log = tmp_path / "log.csv"
    url = "https://cameras-service.surfline.com/cameras/abc/clip?accesstoken=SECRET123"
    srl.record("clip_json", url, response(200, url), path=log)
    text = log.read_text()
    assert "SECRET123" not in text and "accesstoken" not in text
    assert rows(log)[0]["endpoint"] == "cameras-service.surfline.com/cameras/abc/clip"


def test_a_block_is_recorded_with_cloudflare_headers(tmp_path):
    log = tmp_path / "log.csv"
    url = "https://services.surfline.com/kbyg/spots/forecasts/tides"
    srl.record("forecast", url, response(403, url, mitigated="challenge", ray="9a9a"), attempt=2, path=log)
    r = rows(log)[0]
    assert (r["status"], r["cf_mitigated"], r["cf_ray"], r["attempt"], r["elapsed_ms"]) == \
        ("403", "challenge", "9a9a", "2", "120")


def test_no_response_is_recorded_as_error(tmp_path):
    log = tmp_path / "log.csv"
    srl.record("forecast", "https://services.surfline.com/kbyg/x", None, path=log)
    assert rows(log)[0]["status"] == "error"


def test_logging_never_raises(tmp_path):
    srl.record("forecast", "https://x", object(), path=tmp_path / "missing_dir" / "a" / "log.csv")
    srl.record("forecast", "https://x", None, path="/dev/null/not_a_dir/log.csv")


def test_blocks_summary_and_email_note(tmp_path):
    log = tmp_path / "log.csv"
    url = "https://services.surfline.com/kbyg/spots/forecasts/"
    for kind, status in [("forecast", 200), ("forecast", 403), ("forecast", 403), ("clip_json", 200)]:
        srl.record(kind, url + "wind", response(status, url + "wind"), path=log)
    b = srl.blocks_on(date.today(), log)
    assert b["refused"] == 2 and b["requests"] == 4 and b["by_kind"] == {"forecast": 2}
    note = edr.cloudflare_note(date.today(), log)
    assert note.startswith("Cloudflare blocked 2 of 4 Surfline requests tonight (2 forecast)")


def test_quiet_night_has_no_note(tmp_path):
    log = tmp_path / "log.csv"
    url = "https://services.surfline.com/kbyg/spots/forecasts/wind"
    srl.record("forecast", url, response(200, url), path=log)
    assert srl.blocks_on(date.today(), log) is None
    assert edr.cloudflare_note(date.today(), log) == ""
    assert edr.cloudflare_note(date.today(), tmp_path / "absent.csv") == ""
