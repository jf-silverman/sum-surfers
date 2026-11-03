"""Forecast publishing guards and records.

- check_forecast_sources: the 2026-10-30 run published a forecast with 6 of 8
  Surfline sources blocked; it must now stop before anything is saved.
- forecast_row: every raw input is kept (2026-11-02, after directions and the
  raw weather string turned out unrecoverable from the scaled values).
- build_forecast_log: still reads old records without the raw_* columns next
  to new ones that have them.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import build_forecast_log as bfl  # noqa: E402
import plot_daily_prediction as pdp  # noqa: E402


# --- check_forecast_sources ---------------------------------------------------

@pytest.mark.parametrize("missing", [[], ["energy"], ["energy", "consistency"]])
def test_small_non_tide_gaps_are_allowed(missing):
    pdp.check_forecast_sources(missing)


@pytest.mark.parametrize("missing", [
    ["tides"],
    ["energy", "consistency", "swells"],
    ["weather", "rating", "tides", "surf", "swells", "wind"],   # 2026-10-30
])
def test_tide_or_three_plus_missing_holds_the_forecast(missing):
    with pytest.raises(SystemExit) as e:
        pdp.check_forecast_sources(missing)
    assert e.value.code == pdp.EXIT_SOURCES_MISSING == 4


def test_partial_failures_are_never_cached(tmp_path, monkeypatch):
    """A cache with any empty endpoint must be refused, not replayed."""
    import json, os, time
    import get_surf_predictors as sp
    path = tmp_path / "cache.json"
    monkeypatch.setattr(sp, "FORECAST_CACHE_PATH", path)
    full = {p: [{"timestamp": 1}] for p in sp.ENDPOINT_PATHS}
    partial = {**full, "tides": []}
    for responses, expect in ((full, True), (partial, False)):
        path.write_text(json.dumps({"fetched_at": "x", "days": 8, "spot_id": sp.SPOT_ID,
                                    "responses": responses}))
        os.utime(path, (time.time(), time.time()))
        assert (sp.load_cached_responses(8) is not None) is expect


# --- forecast_row -------------------------------------------------------------

def hour_record(raw_inputs):
    return {"hour": datetime(2026, 11, 3, 10), "point": 9.0, "point_raw": 10.0,
            "recal_a": -3.4, "recal_b": 1.16,
            "quantiles": {q: 5.0 + i for i, q in enumerate(pdp.FAN_LEVELS)},
            "weather_simple": "CLEAR", "tide_ft": 2.25, "in_training_range": True,
            "crowd_level": 3, "crowd_level_value": 10.0, "crowd_level_low": 2,
            "crowd_level_high": 4, "missing_predictors": [], "raw_inputs": raw_inputs}


def test_every_raw_input_is_written():
    raw = {"wind_direction_deg": 247.1, "primary_swell_direction_deg": 296.8,
           "weather_condition": "SHOWERS", "tide_ft": 2.25, "tide_correction_ft": 1.38081,
           "rating_value": 3, "rating_key": "FAIR", "surf_max_ft": None,
           "nested": {"skip": "me"}}
    row = pdp.forecast_row(date(2026, 11, 3), hour_record(raw))
    assert row["raw_wind_direction_deg"] == 247.1
    assert row["raw_primary_swell_direction_deg"] == 296.8
    assert row["raw_weather_condition"] == "SHOWERS"
    assert row["raw_tide_correction_ft"] == 1.3808          # rounded to 4 places
    assert row["raw_surf_max_ft"] is None                    # kept, so a gap is visible
    assert "raw_nested" not in row                           # non-scalars are left out


def test_shadow_columns_default_to_the_published_forecast():
    row = pdp.forecast_row(date(2026, 11, 3), hour_record({}))
    assert row["shadow_tide_corr_predicted"] == row["predicted"]
    assert row["shadow_tide_corr_predicted_raw"] == row["predicted_raw"]
    assert row["tide_correction_ft"] == 0.0


def test_shadow_columns_carry_the_corrected_forecast():
    r = {**hour_record({}), "tide_correction_ft": 1.381, "shadow_point": 4.2,
         "shadow_point_raw": 5.0}
    row = pdp.forecast_row(date(2026, 11, 3), r)
    assert (row["shadow_tide_corr_predicted"], row["shadow_tide_corr_predicted_raw"],
            row["tide_correction_ft"]) == (4.2, 5.0, 1.381)


def test_record_without_raw_inputs_still_writes():
    row = pdp.forecast_row(date(2026, 11, 3), hour_record(None))
    assert not any(k.startswith("raw_") for k in row)


# --- build_forecast_log --------------------------------------------------------

def test_log_reads_old_and_new_records_together(tmp_path, monkeypatch):
    monkeypatch.setattr(bfl, "FORECASTS_DIR", tmp_path)
    monkeypatch.setattr(bfl, "OUT_CSV", tmp_path / "forecast_log.csv")
    monkeypatch.setattr(bfl, "PREDICTIONS_CSV", tmp_path / "predictions.csv")
    base = {"hour_local": "10:00", "predicted": 9.0, "lower_q10": 2.0, "upper_q90": 15.0,
            "forecast_made_at": "2026-11-01T20:30:00"}
    pd.DataFrame([{**base, "date": "2026-11-02"}]).to_csv(tmp_path / "forecast_2026-11-02.csv", index=False)
    pd.DataFrame([{**base, "date": "2026-11-03", "forecast_made_at": "2026-11-02T20:30:00",
                   "raw_wind_direction_deg": 247.1}]).to_csv(tmp_path / "forecast_2026-11-03.csv", index=False)
    pd.DataFrame([{"date": "2026-11-02", "time_local": "10:05", "quality_ok": True,
                   "surfer_count": 12}]).to_csv(tmp_path / "predictions.csv", index=False)
    bfl.build()
    log = pd.read_csv(tmp_path / "forecast_log.csv")
    assert sorted(log["date"]) == ["2026-11-02", "2026-11-03"]
    scored = log[log["date"] == "2026-11-02"].iloc[0]
    assert scored["actual"] == 12 and bool(scored["inside_band"])


def test_shadow_recalibration_is_not_clamped_to_the_published_band():
    """2026-11-02: clamping into the uncorrected band moved a corrected 0 up to
    the band's lower edge. The shadow call passes (0, inf) as its bounds."""
    recal = {"applied": True, "a": -3.38, "b": 1.159}
    assert pdp.apply_point_recalibration(0.5, 0.0, float("inf"), recal) == 0.0
    assert pdp.apply_point_recalibration(0.5, 4.66, 22.5, recal) == 4.66
    src = Path(pdp.__file__).read_text()
    assert 'apply_point_recalibration(shadow_raw, 0.0, float("inf"), recal)' in src
