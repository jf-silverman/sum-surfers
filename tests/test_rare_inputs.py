"""plot_daily_prediction.rare_inputs (2026-11-19): conditions too rare in the
training data for the quantile models (min_samples_leaf 20) to learn from."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import plot_daily_prediction as pdp  # noqa: E402


def train():
    # 100 hours: wind 0-9 mph mostly, 8 hours at 20+; rating mostly 2-3, 16 at 0
    wind = [5.0] * 92 + [21.0] * 8
    rating = [2] * 50 + [3] * 34 + [0] * 16
    return pd.DataFrame({"wind_speed_mph": wind, "rating_value": rating})


def test_11_18_morning_is_flagged():
    """Regression: 25 mph and VERY POOR were inside the range seen, so the old
    hour-only flag said nothing while the 90% edge reached 28."""
    r = pdp.rare_inputs({"wind_speed_mph": 25.3, "rating_value": 0}, train())
    assert r == ["wind 25 mph (0 h)", "rating VERY POOR (16 h)"]


def test_common_conditions_are_not_flagged():
    assert pdp.rare_inputs({"wind_speed_mph": 5.0, "rating_value": 2}, train()) == []


def test_a_rare_low_value_on_a_high_tail_check_is_ignored():
    t = pd.DataFrame({"wind_speed_mph": [5.0] * 95 + [0.0] * 5})
    assert pdp.rare_inputs({"wind_speed_mph": 0.0}, t) == []


def test_missing_or_blank_inputs_are_skipped():
    assert pdp.rare_inputs({"wind_speed_mph": None, "rating_value": ""}, train()) == []


def test_forecast_row_records_them():
    from datetime import datetime
    r = dict(hour=datetime(2026, 11, 18, 10), point=6.4, point_raw=8.5, quantiles={q: 0 for q in pdp.FAN_LEVELS},
             weather_simple="CLEAR", tide_ft=2.8, in_training_range=True, crowd_level=1, crowd_level_value=1.0,
             crowd_level_low=1, crowd_level_high=4, missing_predictors=[],
             rare_inputs=["wind 25 mph (8 h)", "rating VERY POOR (16 h)"])
    row = pdp.forecast_row(datetime(2026, 11, 18).date(), r)
    assert row["n_rare_inputs"] == 2 and row["rare_inputs"] == "wind 25 mph (8 h);rating VERY POOR (16 h)"
