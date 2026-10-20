"""Tests for the nightly forecast-band check.

The condition this guards went unnoticed for eight days (2026-10-12 to
2026-10-19): the q0.10 model collapsed to a constant zero, so every band the
email printed had a lower edge that could not be missed from below, which
inflates coverage instead of describing uncertainty. Nothing raised, and the
only visible trace was a NOTE line in a log nobody reads nightly. These tests
cover the two things the check has to get right -- firing when it should, and
staying silent when it should -- plus the derived band label, which is the
other half of the fix (a typed "80%" is what let the label drift from the
levels actually fitted).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import plot_daily_prediction as pdp  # noqa: E402
import email_daily_report as edr  # noqa: E402


FORECAST_COLS = ["date", "hour_local", "predicted", "lower_q10", "upper_q90",
                 "band_lower_q", "band_upper_q"]


def write_forecast(dirpath, name, lowers, lower_q=0.15):
    rows = [{"date": "2026-11-01", "hour_local": f"{7 + i:02d}:00",
             "predicted": lo + 5.0, "lower_q10": lo, "upper_q90": lo + 20.0,
             "band_lower_q": lower_q, "band_upper_q": 0.90}
            for i, lo in enumerate(lowers)]
    pd.DataFrame(rows, columns=FORECAST_COLS).to_csv(dirpath / name, index=False)


def write_features(path, n, n_zero):
    """A training table with a known share of empty hours."""
    counts = [0] * n_zero + [12] * (n - n_zero)
    pd.DataFrame({"date": ["2026-11-01"] * n,
                  "surfer_count": counts}).to_csv(path, index=False)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the check at a scratch forecasts dir and training table."""
    fdir = tmp_path / "forecasts"
    fdir.mkdir()
    feats = tmp_path / "training_features.csv"
    monkeypatch.setattr(pdp, "FORECASTS_DIR", fdir)
    monkeypatch.setattr(edr, "FEATURES_CSV", feats)
    write_features(feats, 100, 5)          # 5% empty: well under the trigger
    return fdir, feats


# --------------------------------------------------------------------------
# The band label is derived, never typed
# --------------------------------------------------------------------------
def test_band_label_matches_the_fitted_levels():
    lo, hi = pdp.FAN_LEVELS[0], pdp.FAN_LEVELS[-1]
    assert pdp.BAND_LABEL == f"{round((hi - lo) * 100)}%"


def test_the_median_is_still_the_middle_level():
    # The point estimate is read as quantiles[0.50]; losing it would make the
    # forecast silently the wrong statistic.
    assert 0.50 in pdp.FAN_LEVELS
    assert pdp.FAN_LEVELS == sorted(pdp.FAN_LEVELS)


# --------------------------------------------------------------------------
# Signal 1: a lower edge sitting on the floor
# --------------------------------------------------------------------------
def test_quiet_when_the_band_is_two_sided(isolated):
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [0.9, 1.7, 3.2, 11.8])
    assert edr.band_health_note() == ""


def test_fires_when_every_lower_edge_is_pinned(isolated):
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [0.0, 0.0, 0.0, 0.0])
    note = edr.band_health_note()
    assert "pinned at zero on 4 of 4 hours" in note


def test_fires_on_a_partial_pin(isolated):
    # The 2026-09-30 episode pinned some hours and not others; a check that
    # only noticed an all-zero day would have missed it.
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [0.0, 2.5, 0.0, 9.0])
    assert "pinned at zero on 2 of 4 hours" in edr.band_health_note()


def test_reads_the_newest_forecast_not_the_oldest(isolated):
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-10-01.csv", [0.0, 0.0])
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.5, 4.0])
    assert edr.band_health_note() == ""


def test_the_log_file_is_not_mistaken_for_a_forecast(isolated):
    # forecast_log.csv lives in the same directory and has the same columns.
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.5, 4.0])
    write_forecast(fdir, "forecast_log.csv", [0.0, 0.0])
    assert edr.band_health_note() == ""


# --------------------------------------------------------------------------
# Signal 2: the empty-hour share that drives it
# --------------------------------------------------------------------------
def test_quiet_below_the_empty_share_trigger(isolated):
    fdir, feats = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.0, 2.0])
    write_features(feats, 100, int(100 * (edr.EMPTY_SHARE_TRIGGER - 0.05)))
    assert edr.band_health_note() == ""


def test_fires_above_the_empty_share_trigger(isolated):
    fdir, feats = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.0, 2.0])
    write_features(feats, 100, 35)
    note = edr.band_health_note()
    assert "35%" in note and "hurdle" in note


def test_the_trigger_boundary_is_inclusive(isolated):
    fdir, feats = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.0, 2.0])
    write_features(feats, 100, int(100 * edr.EMPTY_SHARE_TRIGGER))
    assert edr.band_health_note() != ""


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------
def test_missing_files_do_not_raise(isolated):
    fdir, feats = isolated
    feats.unlink()
    assert edr.band_health_note() == ""


def test_an_unreadable_forecast_is_reported_not_raised(isolated):
    fdir, _ = isolated
    (fdir / "forecast_2026-11-01.csv").write_text("not,a,forecast\n1,2,3\n")
    assert "Could not read" in edr.band_health_note()
