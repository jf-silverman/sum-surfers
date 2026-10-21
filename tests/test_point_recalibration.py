"""The nightly point recalibration (plot_daily_prediction, adopted 2026-10-21).

Pure functions over a small CSV, so these run without the data or any model.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import plot_daily_prediction as pdp  # noqa: E402


def write_log(path, raw, actual, n_dates=10, raw_col="predicted", published=None):
    """A forecast_log.csv with len(raw) scored rows spread over n_dates dates."""
    n = len(raw)
    df = pd.DataFrame({
        "date": [f"2026-10-{1 + i % n_dates:02d}" for i in range(n)],
        raw_col: raw,
        "actual": actual,
    })
    if published is not None:
        df["predicted"] = published
    df.to_csv(path, index=False)
    return path


def affine_data(a, b, n=600, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 40, n)
    return raw, a + b * raw + rng.normal(0, 0.5, n)


# --- fitting ---------------------------------------------------------------

def test_recovers_a_known_affine_map(tmp_path):
    raw, actual = affine_data(-3.8, 1.15)
    rc = pdp.fit_point_recalibration(write_log(tmp_path / "log.csv", raw, actual))
    assert rc["applied"]
    assert rc["a"] == pytest.approx(-3.8, abs=0.2)
    assert rc["b"] == pytest.approx(1.15, abs=0.02)
    assert rc["n_rows"] == 600 and rc["n_dates"] == 10


def test_fits_on_the_raw_median_not_the_published_one(tmp_path):
    """The feedback-loop guard, and the reason predicted_raw exists.

    Once live, `predicted` is the CORRECTED median. If the fit read it, the
    next night would fit a correction to the correction and converge on the
    identity whatever the model's real error was. Here the published column is
    already perfectly calibrated (identity would fit it) while the raw one is
    off by a known map; the fit must find the map.
    """
    raw, actual = affine_data(-3.8, 1.15)
    rc = pdp.fit_point_recalibration(write_log(
        tmp_path / "log.csv", raw, actual, raw_col="predicted_raw", published=actual))
    assert rc["b"] == pytest.approx(1.15, abs=0.02)


def test_rows_from_before_predicted_raw_existed_fall_back_to_predicted(tmp_path):
    raw, actual = affine_data(-3.8, 1.15)
    df = pd.DataFrame({"date": [f"2026-10-{1 + i % 10:02d}" for i in range(600)],
                       "predicted": raw, "actual": actual,
                       "predicted_raw": [np.nan] * 300 + list(raw[300:])})
    df.to_csv(tmp_path / "log.csv", index=False)
    rc = pdp.fit_point_recalibration(tmp_path / "log.csv")
    assert rc["n_rows"] == 600 and rc["b"] == pytest.approx(1.15, abs=0.02)


def test_unscored_rows_are_ignored(tmp_path):
    raw, actual = affine_data(-3.8, 1.15, n=700)
    actual = actual.copy()
    actual[600:] = np.nan
    rc = pdp.fit_point_recalibration(write_log(tmp_path / "log.csv", raw, actual))
    assert rc["n_rows"] == 600


@pytest.mark.parametrize("n,n_dates,why", [
    (pdp.RECAL_MIN_ROWS - 1, 10, "scored rows"),
    (600, pdp.RECAL_MIN_DATES - 1, "dates"),
])
def test_identity_when_the_record_is_too_thin(tmp_path, n, n_dates, why):
    raw, actual = affine_data(-3.8, 1.15, n=n)
    rc = pdp.fit_point_recalibration(
        write_log(tmp_path / "log.csv", raw, actual, n_dates=n_dates))
    assert not rc["applied"] and (rc["a"], rc["b"]) == (0.0, 1.0)
    assert why in rc["reason"]


def test_identity_when_the_slope_is_implausible(tmp_path):
    raw, actual = affine_data(0.0, 2.5)
    rc = pdp.fit_point_recalibration(write_log(tmp_path / "log.csv", raw, actual))
    assert not rc["applied"] and "outside" in rc["reason"]


def test_a_missing_log_does_not_stop_the_chart(tmp_path):
    rc = pdp.fit_point_recalibration(tmp_path / "nope.csv")
    assert not rc["applied"] and (rc["a"], rc["b"]) == (0.0, 1.0)


def test_a_constant_prediction_column_does_not_raise(tmp_path):
    rc = pdp.fit_point_recalibration(
        write_log(tmp_path / "log.csv", [5.0] * 600, list(range(600))))
    assert not rc["applied"]


# --- applying --------------------------------------------------------------

ON = dict(a=-3.78, b=1.148, applied=True)


def test_applies_the_map_inside_the_band():
    assert pdp.apply_point_recalibration(20.0, 5.0, 40.0, ON) == pytest.approx(-3.78 + 1.148 * 20)


@pytest.mark.parametrize("raw,lo,hi,expected", [
    (2.0, 1.5, 9.0, 1.5),      # corrected would be -1.48: held at the lower edge
    (40.0, 10.0, 41.0, 41.0),  # corrected would be 42.14: held at the upper edge
    (2.0, 0.0, 9.0, 0.0),      # never negative, even with a zero lower edge
])
def test_clamped_to_the_band_and_never_negative(raw, lo, hi, expected):
    assert pdp.apply_point_recalibration(raw, lo, hi, ON) == pytest.approx(expected)


def test_passthrough_when_not_applied():
    off = dict(a=-3.78, b=1.148, applied=False)
    assert pdp.apply_point_recalibration(12.3, 0.0, 40.0, off) == 12.3


# --- the record --------------------------------------------------------------

def test_forecast_row_records_both_medians_and_the_map():
    r = dict(hour=datetime(2026, 10, 22, 7), point=4.1, point_raw=6.9,
             recal_a=-3.78, recal_b=1.148,
             quantiles={pdp.FAN_LEVELS[0]: 0.0, 0.50: 4.1, pdp.FAN_LEVELS[-1]: 12.0},
             weather_simple="CLEAR", tide_ft=3.1, in_training_range=True,
             crowd_level=2, crowd_level_value=5.0, crowd_level_low=1,
             crowd_level_high=3, missing_predictors=[])
    row = pdp.forecast_row(date(2026, 10, 22), r)
    assert row["predicted"] == 4.1 and row["predicted_raw"] == 6.9
    assert (row["recal_a"], row["recal_b"]) == (-3.78, 1.148)
