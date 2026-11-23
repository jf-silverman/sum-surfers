"""band_calibration (2026-11-23): forecast band edges calibrated nightly from
the scored record, busy hours only, never narrowing, never fitting on its own output."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import band_calibration as bc  # noqa: E402


def log(n_days=30, lower_gap=4.0, below_share=0.30, **extra):
    """Busy hours forecast 20 with a raw band of 16-30; `below_share` of them come in at 10."""
    rows = []
    rng = np.random.default_rng(0)
    for d in pd.date_range("2026-10-24", periods=n_days):
        for h in range(6):
            actual = 10.0 if rng.random() < below_share else 21.0
            rows.append(dict(date=d.date().isoformat(), forecast_kind="day_ahead", predicted=20.0,
                             lower_q10=20.0 - lower_gap, upper_q90=30.0, actual=actual,
                             band_lower_q=0.15, band_upper_q=0.90, **extra))
    return pd.DataFrame(rows)


TODAY = date(2026, 11, 23)


def test_a_lower_edge_missed_too_often_is_stretched_to_target():
    cal = bc.fit(log(), TODAY, 0.15, 0.90)
    assert cal["applied"] and cal["below_raw"] > 0.2
    lo, hi = bc.apply(20.0, 16.0, 30.0, cal)
    assert lo <= 10.0 and hi == 30.0          # the 10s now fall inside; upper untouched (no misses above)


def test_never_narrows():
    cal = bc.fit(log(below_share=0.0), TODAY, 0.15, 0.90)
    assert cal["k_lower"] == 1.0 and cal["k_upper"] == 1.0


def test_quiet_hours_are_left_alone():
    cal = dict(k_lower=2.0, k_upper=2.0)
    assert bc.apply(3.0, 1.0, 8.0, cal) == (1.0, 8.0)
    assert bc.apply(20.0, 16.0, 30.0, cal) == (12.0, 40.0)


def test_too_little_record_means_no_change():
    cal = bc.fit(log(n_days=5), TODAY, 0.15, 0.90)
    assert not cal["applied"] and cal["k_lower"] == 1.0


def test_other_band_eras_are_ignored():
    cal = bc.fit(log().assign(band_lower_q=0.10), TODAY, 0.15, 0.90)
    assert cal["n_hours"] == 0


def test_fit_reads_raw_edges_not_its_own_output():
    """Published edges already stretched to 0; the raw ones (16) still show the miss."""
    published = log().assign(lower_q10=0.0, lower_raw=16.0, upper_raw=30.0)
    cal = bc.fit(published, TODAY, 0.15, 0.90)
    assert cal["below_raw"] > 0.2 and cal["k_lower"] > 1.0


def test_week_widening_is_undone_for_raw_edges():
    df = pd.DataFrame(dict(predicted=[20.0], lower_q10=[12.0], upper_q90=[40.0], band_factor=[2.0]))
    lo, hi = bc.raw_edges(df)
    assert lo.iloc[0] == 16.0 and hi.iloc[0] == 30.0
