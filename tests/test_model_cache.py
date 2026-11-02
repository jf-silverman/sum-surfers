"""The forecast's model cache must not outlive a change to its training data.

Regression for the 2026-11-02 defect: the cache was reused whenever the row
count and feature columns were unchanged, so when the Kelvin-wave correction
rewrote October's tide_ft in place, the 11-01 run served a fit that had never
seen the correction with forecast rows that had it applied.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import plot_daily_prediction as pdp  # noqa: E402


def data(n=40):
    X = pd.DataFrame({"tide_ft": [1.0 + 0.1 * i for i in range(n)],
                      "rating_value": [i % 4 + 1 for i in range(n)]})
    y = pd.Series([float(i % 9) for i in range(n)])
    return X, y


def cache_for(X, y, fitted_days_ago=0):
    return {"cache_version": pdp.MODEL_CACHE_VERSION,
            "feature_names": list(X.columns), "n_rows": len(X),
            "data_hash": pdp.training_fingerprint(X, y, len(X)),
            "fitted_at": (datetime.now() - timedelta(days=fitted_days_ago)).isoformat()}


def usable(cache, X, y):
    return pdp.cached_models_are_usable(cache, len(X), X.columns, X, y)


def test_unchanged_data_reuses_the_cache():
    X, y = data()
    ok, _why = usable(cache_for(X, y), X, y)
    assert ok


def test_a_value_changed_in_place_forces_a_refit():
    X, y = data()
    cache = cache_for(X, y)
    X.loc[5, "tide_ft"] += 1.38          # same rows, same columns, new value
    ok, why = usable(cache, X, y)
    assert not ok and "training data changed" in why


def test_a_changed_target_forces_a_refit():
    X, y = data()
    cache = cache_for(X, y)
    y.iloc[3] += 2                      # e.g. a recount migration
    ok, _why = usable(cache, X, y)
    assert not ok


def test_appended_rows_alone_keep_the_cache():
    """New nights add rows; that is what REFIT_AFTER_NEW_ROWS is for, not a change."""
    X, y = data()
    cache = cache_for(X, y)
    X2 = pd.concat([X, X.iloc[:5]], ignore_index=True)
    y2 = pd.concat([y, y.iloc[:5]], ignore_index=True)
    ok, why = usable(cache, X2, y2)
    assert ok, why


def test_old_cache_without_a_fingerprint_is_refused():
    X, y = data()
    cache = cache_for(X, y)
    del cache["data_hash"]
    ok, _why = usable(cache, X, y)
    assert not ok


def test_column_and_age_checks_still_apply():
    X, y = data()
    assert not usable({**cache_for(X, y), "feature_names": ["tide_ft"]}, X, y)[0]
    assert not usable(cache_for(X, y, fitted_days_ago=pdp.REFIT_AFTER_DAYS), X, y)[0]
