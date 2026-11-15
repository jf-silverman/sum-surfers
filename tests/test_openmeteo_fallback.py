"""Open-Meteo forecast fallback (2026-11-14): its forecast server was
unreachable for three nightly runs (11-12 to 11-14) while the archive server
answered, and each forecast went out missing four weather inputs."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import get_surf_predictors as sp  # noqa: E402

HOURLY = {"time": ["2026-11-15T06:00"], "temperature_2m": [12.0], "relative_humidity_2m": [80],
          "cloud_cover": [50], "weather_code": [3], "surface_pressure": [1015.0]}


class Ok:
    def raise_for_status(self): pass
    def json(self): return {"hourly": HOURLY}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "OPENMETEO_CACHE_PATH", tmp_path / "om.json")
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    return tmp_path / "om.json"


def test_success_saves_the_forecast(cache, monkeypatch):
    monkeypatch.setattr(sp.requests, "get", lambda *a, **k: Ok())
    assert sp.fetch_openmeteo_forecast(8) == HOURLY
    assert sp.load_openmeteo_forecast()[0] == HOURLY


def test_outage_falls_back_to_the_saved_forecast(cache, monkeypatch):
    sp.save_openmeteo_forecast(HOURLY)
    calls = []
    def down(*a, **k):
        calls.append(1); raise requests.exceptions.ReadTimeout("timed out")
    monkeypatch.setattr(sp.requests, "get", down)
    assert sp.fetch_openmeteo_forecast(8) == HOURLY
    assert len(calls) == 1 + len(sp.OPENMETEO_RETRY_WAITS_SEC)


def test_outage_with_nothing_saved_still_raises(cache, monkeypatch):
    monkeypatch.setattr(sp.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ReadTimeout("x")))
    with pytest.raises(requests.exceptions.ReadTimeout):
        sp.fetch_openmeteo_forecast(8)


def test_a_saved_forecast_too_old_is_not_used(cache):
    sp.save_openmeteo_forecast(HOURLY)
    assert sp.load_openmeteo_forecast(max_age_h=-1) is None
