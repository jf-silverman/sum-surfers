"""Training and forecasting must hand the model the same inputs for the same hour.

`fit_surfer_count_model.load_and_prepare` builds the training matrix from
`training_features.csv`; `predict_surf_count.build_feature_row` builds the
forecast row from live predictors. Any transform applied in one and not the
other (the 2026-11-01 Kelvin-wave tide correction is the latest example) makes
the forecast read a different input from the one the model learned on, with no
error. This builds a tiny training table, runs both paths on the same hours and
compares them column by column.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import fit_surfer_count_model as fit  # noqa: E402
import predict_surf_count as psc  # noqa: E402

# One raw Surfline weather string per simplified category, so the dummy
# columns are exercised in both paths.
RAW_WEATHER = {"CLEAR": "MOSTLY_CLEAR", "CLOUDY_OVERCAST": "OVERCAST",
               "FOG": "MIST", "RAIN": "LIGHT_RAIN"}


def training_table(path):
    rows = []
    for i, (wx, raw) in enumerate(RAW_WEATHER.items()):
        d = datetime(2026, 10, 3 + i, 8 + 2 * i)  # Sat, Sun, Mon, Tue
        rows.append({
            "filename": f"crop{i}.jpg", "date": d.strftime("%Y-%m-%d"),
            "time_local": d.strftime("%H:%M"), "surfer_count": 5 + i,
            "hour_local": d.hour, "month": d.month, "is_weekend": d.weekday() >= 5,
            "weather_simple": wx, "is_night": False,
            "temperature_f": 60 + i, "pressure_mb": 1015 - i, "rating_value": 1 + i % 4,
            "tide_ft": 1.5 + i, "surf_min_ft": 2, "surf_max_ft": 3 + i,
            "primary_swell_height_ft": 3.5, "primary_swell_period_s": 12 + i,
            "wind_speed_mph": 4 + i, "wind_gust_mph": 8 + i,
            "energy_offshore_kj": 600 + 50 * i, "energy_nearshore_kj": 200 + 10 * i,
            "consistency_wave_count": 20 + i,
            "real_temperature_f": 58 + i, "real_humidity_pct": 80 - i,
            "real_cloud_cover_pct": 10 * i, "real_pressure_mb": 1014 + i,
            "good_tide_hours": 4, "good_tide_frac": 0.4, "good_tide_hours_left": 2,
            "tide_resid_dev_prior_ft": 0.1 * i,
            # On for two of the four days, as the 0.75 ft gate would be.
            "tide_correction_ft": [0.0, 1.38, 0.0, 0.9][i],
            "wind_direction_deg": 90 * i + 10, "primary_swell_direction_deg": 280 + 5 * i,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def live_predictors(row, numeric_cols):
    """The predictor dict the forecast path receives for the same hour."""
    p = {c: row[c] for c in numeric_cols}
    p["tide_correction_ft"] = row["tide_correction_ft"]
    p["wind_direction_deg"] = row["wind_direction_deg"]
    p["primary_swell_direction_deg"] = row["primary_swell_direction_deg"]
    p["weather_condition"] = RAW_WEATHER[row["weather_simple"]]
    return p


@pytest.mark.parametrize("correction", [False, True])   # published, shadow
def test_forecast_row_matches_training_row(tmp_path, monkeypatch, correction):
    rows = training_table(tmp_path / "training_features.csv")
    monkeypatch.setattr(fit, "FEATURES_CSV", tmp_path / "training_features.csv")
    X, _y, df, numeric_cols = fit.load_and_prepare(tide_correction=correction)
    assert len(X) == len(rows)

    # Unit scaling, so the comparison is of the inputs, not the standardization.
    mean = pd.Series(0.0, index=numeric_cols)
    std = pd.Series(1.0, index=numeric_cols)
    categories = set(df["weather_simple"])
    for i, row in enumerate(rows):
        when = datetime.strptime(f"{row['date']} {row['time_local']}", "%Y-%m-%d %H:%M")
        served = psc.build_feature_row(when, live_predictors(row, numeric_cols), numeric_cols,
                                       categories, mean, std, X.columns, tide_correction=correction)
        trained = X.iloc[i]
        for col in X.columns:
            assert served.iloc[0][col] == pytest.approx(float(trained[col])), (
                f"row {i}: {col} is {served.iloc[0][col]} when served, {trained[col]} in training")


def test_tide_correction_only_when_asked(tmp_path, monkeypatch):
    """Published (default): harmonic tide. Shadow: tide plus the correction.
    The transform behind the 2026-11-02 defect, in both paths."""
    rows = training_table(tmp_path / "training_features.csv")
    monkeypatch.setattr(fit, "FEATURES_CSV", tmp_path / "training_features.csv")
    X, _y, _df, _nc = fit.load_and_prepare()
    assert X["tide_ft"].iloc[1] == pytest.approx(rows[1]["tide_ft"])
    Xc, _y, _df, _nc = fit.load_and_prepare(tide_correction=True)
    assert Xc["tide_ft"].iloc[1] == pytest.approx(rows[1]["tide_ft"] + 1.38)
    assert Xc["tide_ft"].iloc[0] == pytest.approx(rows[0]["tide_ft"])
