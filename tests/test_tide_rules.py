"""Tide residual rules: daily means, the prior-day carry-forward, the 0.75 ft
Kelvin-wave gate (backfill_tide_residual), and the week chart's low-tide share
with and without the correction (plot_daily_prediction.low_tide_fraction)."""

import csv
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import backfill_tide_residual as btr  # noqa: E402
import plot_daily_prediction as pdp  # noqa: E402


def residual_csv(path, days):
    """days: {date string: (n hours, residual_ft, residual_dev_ft)}."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=btr.CSV_HEADER)
        w.writeheader()
        for d, (n, raw, dev) in days.items():
            for h in range(n):
                w.writerow({"date": d, "hour_local": f"{h:02d}:00", "residual_ft": raw,
                            "residual_dev_ft": dev, "quality": "p"})
    return path


# --- daily means and the prior-day rule ---------------------------------------

def test_days_with_too_few_hours_are_left_out(tmp_path):
    p = residual_csv(tmp_path / "r.csv", {"2026-10-29": (24, 1.3, 0.3),
                                          "2026-10-30": (btr.MIN_DAY_HOURS - 1, 1.5, 0.4)})
    means = btr.daily_dev_means(p)
    assert set(means) == {date(2026, 10, 29)}
    assert btr.daily_dev_means(p, col="residual_ft")[date(2026, 10, 29)] == pytest.approx(1.3)


def test_missing_file_gives_no_means(tmp_path):
    assert btr.daily_dev_means(tmp_path / "absent.csv") == {}


def test_prior_mean_is_the_latest_day_strictly_before():
    means = {date(2026, 10, 29): 0.29, date(2026, 10, 30): 0.39}
    assert btr.prior_dev_mean(date(2026, 10, 31), means) == pytest.approx(0.39)
    assert btr.prior_dev_mean(date(2026, 10, 30), means) == pytest.approx(0.29)


def test_prior_mean_carries_forward_across_a_week():
    means = {date(2026, 10, 30): 0.39}
    for d in range(1, 8):  # days 2-7 of the outlook reuse the last observed day
        assert btr.prior_dev_mean(date(2026, 11, d), means) == pytest.approx(0.39)


def test_no_earlier_day_gives_blank():
    assert btr.prior_dev_mean(date(2026, 10, 1), {date(2026, 10, 5): 0.2}) == ""


# --- the 0.75 ft gate ---------------------------------------------------------

@pytest.mark.parametrize("prior, expected", [
    (0.74, 0.0), (0.75, 0.0), (0.76, 0.76), (1.44, 1.44), (-0.2, 0.0)])
def test_gate_is_strictly_above_the_threshold(prior, expected):
    assert btr.TIDE_CORRECTION_GATE_FT == 0.75
    raw = {date(2026, 10, 31): prior}
    assert btr.tide_correction(date(2026, 11, 1), raw) == pytest.approx(expected)


def test_gate_is_off_with_no_residual_history():
    assert btr.tide_correction(date(2026, 11, 1), {}) == 0.0


# --- week chart low-tide share ------------------------------------------------

def records(tides, correction=0.0):
    return [{"hour": datetime(2026, 11, 3, h), "tide_ft": v, "tide_correction_ft": correction}
            for h, v in tides.items()]


def test_low_tide_share_interpolates_within_the_hour():
    frac = pdp.low_tide_fraction(records({10: 2.0, 11: 4.0}))
    assert frac[10] == pytest.approx(0.5)     # crosses 3 ft halfway through
    assert frac[11] == 0.0                    # last hour: all or nothing


def test_correction_only_counts_when_asked():
    recs = records({10: 2.0, 11: 2.5}, correction=1.0)
    assert pdp.low_tide_fraction(recs)[10] == 1.0
    assert pdp.low_tide_fraction(recs, with_correction=True)[10] == pytest.approx(0.0)


def test_correction_can_leave_an_hour_low():
    recs = records({14: 1.0, 15: 1.2}, correction=1.44)
    assert pdp.low_tide_fraction(recs, with_correction=True)[14] == 1.0
