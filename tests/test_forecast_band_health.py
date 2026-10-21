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


def test_one_source_of_truth_for_the_band():
    """Every band-fitting script reads the pair, rather than repeating it.

    Repeating it is what broke predict_surf_count.py when the lower edge moved:
    it fitted a literal q0.10 with allow_degenerate=False and raised outright,
    while demo_predictions.py silently printed a lower bound of zero. The
    released bundle was re-exported onto the same pair on 2026-10-20, so the
    card's figures and the chart now describe one band.
    """
    import fit_surfer_count_model as fit

    assert pdp.FAN_LEVELS[0] == fit.BAND_LOWER_Q
    assert pdp.FAN_LEVELS[-1] == fit.BAND_UPPER_Q
    assert fit.BAND_LABEL == pdp.BAND_LABEL

    src = Path(__file__).resolve().parent.parent / "code"
    for name in ("predict_surf_count.py", "demo_predictions.py",
                 "export_model_release.py"):
        body = (src / name).read_text()
        assert "BAND_LOWER_Q" in body, f"{name} should read the shared pair"
        assert "0.10, 0.90" not in body, f"{name} still hardcodes a band pair"


def test_the_shipped_release_matches_the_current_band():
    """The published bundle must describe the band the chart draws.

    These drifted apart for one commit on 2026-10-20 -- the chart moved to
    q0.15 while the bundle still shipped q0.10 -- which made every interval
    figure in MODEL_CARD.md describe a band nobody could see.
    """
    import json

    import fit_surfer_count_model as fit

    meta = Path(__file__).resolve().parent.parent / "data" / "model_release" / "release_metadata.json"
    if not meta.exists():
        pytest.skip("no release bundle in this checkout")
    q = json.loads(meta.read_text())["quantiles"]
    assert q["lower"] == fit.BAND_LOWER_Q
    assert q["upper"] == fit.BAND_UPPER_Q


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
    assert "does not vary at all" in note and "every hour at 0.00" in note


def test_fires_when_the_edge_is_flat_but_not_zero(isolated):
    """A collapse need not settle on zero.

    fit_quantile_model_robust reports the constant it actually reached, and the
    conditional q0.05 on the positives collapses to ~1.00 rather than 0. An
    edge flat at 1.00 is no more informative than one flat at 0, so the test is
    variation, not the value.
    """
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [1.0, 1.0, 1.0, 1.0])
    assert "does not vary at all" in edr.band_health_note()


def test_quiet_on_a_partial_pin_because_a_hurdle_zero_is_informative(isolated):
    """Deliberately the opposite of what this asserted before 2026-10-20.

    The old rule fired on any zero, which caught the 2026-09-30 episode. Once
    the edges came from a hurdle fit that rule became a nightly false alarm:
    a hurdle returns zero exactly when P(no surfers) exceeds the band's lower
    quantile, which on the 2026-10-21 outlook is 35 of 90 hours -- dawn and
    dusk in late October, correctly. A varying edge is doing its job even with
    zeros in it; only a near-total pin is a symptom.
    """
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [0.0, 2.5, 0.0, 9.0])
    assert edr.band_health_note() == ""


def test_fires_on_a_near_total_pin(isolated):
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [0.0, 0.0, 0.0, 0.0, 6.5])
    note = edr.band_health_note()
    assert "sits at zero on 4 of 5 hours" in note


def test_the_pinned_share_trigger_is_inclusive(isolated):
    fdir, _ = isolated
    n_zero = int(round(10 * edr.LOWER_EDGE_PINNED_SHARE_TRIGGER))
    edges = [0.0] * n_zero + [float(i + 1) for i in range(10 - n_zero)]
    write_forecast(fdir, "forecast_2026-11-01.csv", edges)
    assert f"sits at zero on {n_zero} of 10 hours" in edr.band_health_note()


def test_a_single_hour_forecast_is_not_called_a_collapse(isolated):
    """One hour trivially "does not vary"; that must not read as a constant."""
    fdir, _ = isolated
    write_forecast(fdir, "forecast_2026-11-01.csv", [7.4])
    assert edr.band_health_note() == ""


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


# --- the hurdle evaluation reads the band from the same place ---------------

def test_hurdle_eval_scores_the_band_production_draws():
    """eval_hurdle_model must score the edges production actually uses.

    It was written when the band was q0.10/q0.90 and hardcoded those, so after
    the move to q0.15 it would have compared a band nobody draws.
    """
    import importlib
    import fit_surfer_count_model as fit
    hurdle = importlib.import_module("eval_hurdle_model")

    assert fit.BAND_LOWER_Q in hurdle.QUANTILES, (
        f"q={fit.BAND_LOWER_Q} is the production lower edge but is not in "
        f"eval_hurdle_model.QUANTILES {hurdle.QUANTILES}")
    assert fit.BAND_UPPER_Q in hurdle.QUANTILES
    pairs = [(lo, hi) for lo, hi, _label in hurdle.INTERVALS]
    assert (fit.BAND_LOWER_Q, fit.BAND_UPPER_Q) in pairs, (
        "the production band pair is not among the scored intervals")
    assert hurdle.QUANTILES == tuple(sorted(hurdle.QUANTILES))


def test_hurdle_eval_uses_a_forward_day_split():
    """Not train_test_split: a random row split leaks the day's own level (B21)."""
    src = (Path(__file__).resolve().parents[1]
           / "code" / "eval_hurdle_model.py").read_text()
    assert "train_test_split" not in src, (
        "eval_hurdle_model is back on a random row split, which leaks (B21)")
    assert 'scheme="forward"' in src
