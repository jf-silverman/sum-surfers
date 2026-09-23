"""
forecast_features.py
--------------------
Candidate feature builders for the surfer-count forecast, grouped so that
`eval_forecast_features.py` can switch each group on and off and measure what
it is worth. Nothing here is wired into the live pipeline; this module only
builds columns.

Three groups, all computable the evening before the day being forecast, which
is when `plot_daily_prediction.py` actually makes its call:

  "lunar"  Tide as a cyclical time series rather than a single height. The
           model currently sees `tide_ft` at the hour plus the derived
           `good_tide_*` day features, which say nothing about whether the
           tide is rising or falling, where the hour sits between the day's
           low and high, or whether this is a spring or a neap week.
  "lag"    Previous-day and multi-day-trailing counts, plus a staleness
           column so the model can learn to distrust a lag that sits on the
           far side of a gap.
  "dow"    Day of week as something richer than the current binary
           `is_weekend`, one-hot and cyclical, plus an hour interaction.

Causality — two different arguments, both deliberate:

  The "lag" group reads counts from strictly earlier days only. A row dated D
  never sees any count from D or later.

  The "lunar" group reads the *whole* of day D's tide curve, including hours
  after the row's own hour, and that is legitimate: tide is astronomy, not
  weather. Surfline's `forecasts/tides` endpoint is forward-looking (see
  `get_surf_predictors.py`), so the full curve for tomorrow is in hand
  tonight. Deriving "hours until today's low" from it leaks nothing, because
  the low's timing was knowable before the day began. The same argument does
  NOT extend to counts, which is why the two groups are built separately.

Lunar phase comes from `astral.moon.phase` (already a dependency; no network
call, no new package). It returns 0-27.99 on a 28-unit cycle, 0 at new moon
and 14 at full. Validated against this project's own data: the spring/neap
term below correlates +0.741 with the observed daily tide range across the
117 days that have 8 or more tide readings, which is the sign and rough
magnitude it should have if the astronomy is right.
"""

import datetime

import astral.moon
import numpy as np
import pandas as pd

# astral.moon.phase()'s cycle length in its own units. Checked against the
# alternative of rescaling to the true synodic month (29.530588853 days): 28.0
# scored +0.741 on the tide-range validation above and 29.53 scored +0.728, so
# the native scale is used as-is.
ASTRAL_PHASE_CYCLE = 28.0

# Surfable-tide window used by build_training_features.py's good_tide_* columns.
# Repeated here only to derive "hours until the tide next enters this band";
# this module does not import from that script.
GOOD_TIDE_LO_FT = 0.0
GOOD_TIDE_HI_FT = 3.0


def _moon_phase_frac(date_str):
    """Fraction of the synodic cycle, 0.0 at new moon, 0.5 at full."""
    d = datetime.date.fromisoformat(str(date_str))
    return astral.moon.phase(d) / ASTRAL_PHASE_CYCLE


def build_lunar_tide_features(df):
    """Cyclical/structural tide and lunar columns.

    Returns a DataFrame indexed like `df`.
    """
    out = pd.DataFrame(index=df.index)
    tide = pd.to_numeric(df["tide_ft"], errors="coerce")
    hour = pd.to_numeric(df["hour_local"], errors="coerce")

    # ---- lunar cycle -----------------------------------------------------
    frac = df["date"].map(_moon_phase_frac)
    out["moon_phase_frac"] = frac
    # Synodic month (~29.5 d). Distinguishes new from full, which matters for
    # night-time surfability and for which of the day's two lows is the bigger.
    out["moon_sin"] = np.sin(2 * np.pi * frac)
    out["moon_cos"] = np.cos(2 * np.pi * frac)
    # Spring/neap (~14.8 d): tidal range peaks at BOTH new and full moon, so
    # this is the double-frequency term. +1 at spring, -1 at neap.
    out["spring_neap"] = np.cos(4 * np.pi * frac)
    out["spring_neap_sin"] = np.sin(4 * np.pi * frac)

    # ---- semidiurnal position -------------------------------------------
    # M2, the principal lunar semidiurnal constituent, period 12.4206012 h.
    # Phase is taken from an arbitrary fixed epoch; the model only needs a
    # consistent coordinate, not an absolute one.
    day_num = pd.to_datetime(df["date"]).map(datetime.date.toordinal)
    t_hours = day_num * 24.0 + hour
    for name, period in (("m2", 12.4206012), ("lunar_day", 24.8412024)):
        out[f"{name}_sin"] = np.sin(2 * np.pi * t_hours / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * t_hours / period)

    # ---- shape of today's own tide curve ---------------------------------
    # All of this is forward-knowable astronomy (see module docstring).
    work = pd.DataFrame({"date": df["date"], "hour": hour, "tide": tide})

    # Rate of change: central difference in ft/hour over the day's curve.
    # Surfers care about direction as much as height -- a 2 ft rising tide and
    # a 2 ft falling tide are different sessions, and `tide_ft` cannot tell
    # them apart.
    rate = work.groupby("date", group_keys=False).apply(
        lambda g: g.sort_values("hour")["tide"].diff().reindex(g.index)
    )
    hr_gap = work.groupby("date", group_keys=False).apply(
        lambda g: g.sort_values("hour")["hour"].diff().reindex(g.index)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out["tide_rate_ft_per_hr"] = rate / hr_gap.replace(0, np.nan)
    out["tide_is_rising"] = (out["tide_rate_ft_per_hr"] > 0).astype(float)
    out.loc[out["tide_rate_ft_per_hr"].isna(), "tide_is_rising"] = np.nan

    # Position within the day's range, and the range itself (the direct
    # observable that spring/neap drives).
    day_min = work.groupby("date")["tide"].transform("min")
    day_max = work.groupby("date")["tide"].transform("max")
    out["tide_day_range_ft"] = day_max - day_min
    span = (day_max - day_min).replace(0, np.nan)
    out["tide_pct_of_day_range"] = (tide - day_min) / span

    # Hours from the day's low / to the day's high. Signed, so the model can
    # tell "two hours after the low" from "two hours before it".
    def _extremum_hours(g):
        gs = g.sort_values("hour")
        lo_h = gs.loc[gs["tide"].idxmin(), "hour"]
        hi_h = gs.loc[gs["tide"].idxmax(), "hour"]
        return pd.DataFrame(
            {"h_from_low": g["hour"] - lo_h, "h_from_high": g["hour"] - hi_h},
            index=g.index,
        )

    ext = work.groupby("date", group_keys=False).apply(_extremum_hours)
    out["hours_from_day_low"] = ext["h_from_low"]
    out["hours_from_day_high"] = ext["h_from_high"]
    out["abs_hours_from_day_low"] = ext["h_from_low"].abs()

    # Hours until the tide next falls inside the surfable band, within today.
    def _hours_to_good(g):
        gs = g.sort_values("hour")
        good = gs[(gs["tide"] >= GOOD_TIDE_LO_FT) & (gs["tide"] <= GOOD_TIDE_HI_FT)]
        if good.empty:
            return pd.Series(np.nan, index=g.index)
        res = {}
        for idx, h in zip(gs.index, gs["hour"]):
            later = good.loc[good["hour"] >= h, "hour"]
            res[idx] = (later.iloc[0] - h) if len(later) else np.nan
        return pd.Series(res).reindex(g.index)

    out["hours_to_good_tide"] = work.groupby("date", group_keys=False).apply(_hours_to_good)

    return out


def build_lag_features(df):
    """Previous-day and trailing-window count features.

    Strictly causal: a row dated D reads counts from days < D only. No
    same-day and no previous-hour terms -- the previous-hour mechanism is
    deliberately out of scope for this round.
    """
    out = pd.DataFrame(index=df.index)
    day = pd.to_datetime(df["date"])
    counts = pd.to_numeric(df["surfer_count"], errors="coerce")
    hour = pd.to_numeric(df["hour_local"], errors="coerce")

    work = pd.DataFrame({"day": day, "hour": hour, "count": counts})
    day_stats = work.groupby("day")["count"].agg(["mean", "max", "size"])
    by_day_hour = work.groupby(["day", "hour"])["count"].mean()

    observed_days = np.array(sorted(day_stats.index))

    def _prior_window(d, k):
        """Mean over the k most recent OBSERVED days strictly before d.

        Deliberately counts observed days, not calendar days: after a 95-day
        gap a 7-calendar-day window would be empty, where a 7-observed-day
        window still carries the last week the camera actually ran. Paired
        with days_since_last_obs so the model can discount a stale one.
        """
        prior = observed_days[observed_days < np.datetime64(d)]
        if len(prior) == 0:
            return np.nan, np.nan
        window = prior[-k:]
        return float(day_stats.loc[window, "mean"].mean()), float(
            (np.datetime64(d) - prior[-1]) / np.timedelta64(1, "D")
        )

    cache = {}
    rows = []
    for d, h in zip(day, hour):
        if d not in cache:
            r3, gap = _prior_window(d, 3)
            r7, _ = _prior_window(d, 7)
            d1, d2 = d - pd.Timedelta(days=1), d - pd.Timedelta(days=2)
            cache[d] = dict(
                lag1d_daymean=day_stats["mean"].get(d1, np.nan),
                lag2d_daymean=day_stats["mean"].get(d2, np.nan),
                lag1d_daymax=day_stats["max"].get(d1, np.nan),
                roll3d_daymean=r3,
                roll7d_daymean=r7,
                days_since_last_obs=gap,
            )
        rec = dict(cache[d])
        rec["lag1d_samehour"] = by_day_hour.get((d - pd.Timedelta(days=1), h), np.nan)
        rec["lag2d_samehour"] = by_day_hour.get((d - pd.Timedelta(days=2), h), np.nan)
        rows.append(rec)

    lagged = pd.DataFrame(rows, index=df.index)
    # Trend: is the recent stretch busier than the week around it? NaN-safe by
    # construction (both terms NaN together at the start of the record).
    lagged["roll3d_minus_roll7d"] = lagged["roll3d_daymean"] - lagged["roll7d_daymean"]
    return pd.concat([out, lagged], axis=1)


def build_dow_features(df):
    """Day of week as one-hot plus cyclical, and an hour interaction.

    The live model has only the binary `is_weekend`. This asks whether
    Saturday differs from Sunday, and whether Friday leans weekend-ward.
    """
    out = pd.DataFrame(index=df.index)
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df["day_of_week"].astype(str)
    idx = dow.map({name: i for i, name in enumerate(order)})
    out["dow_idx"] = idx
    out["dow_sin"] = np.sin(2 * np.pi * idx / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx / 7)
    for name in order:
        out[f"dow_{name}"] = (dow == name).astype(float)
    # Interaction: the weekend crowd may not just be bigger but differently
    # shaped across the day (later start, broader midday).
    hour = pd.to_numeric(df["hour_local"], errors="coerce")
    weekend = pd.to_numeric(df["is_weekend"], errors="coerce").astype(float)
    out["weekend_x_hour_sin"] = weekend * np.sin(2 * np.pi * hour / 24)
    out["weekend_x_hour_cos"] = weekend * np.cos(2 * np.pi * hour / 24)
    out["dow_x_hour"] = idx * hour
    return out


BUILDERS = {
    "lunar": build_lunar_tide_features,
    "lag": build_lag_features,
    "dow": build_dow_features,
}

# The 12 columns that carried any permutation importance at all on the forward
# split, out of the 39 the three builders produce. Everything dropped scored
# <= +0.02, and seven columns scored NEGATIVE (moon_phase_frac, moon_sin,
# moon_cos, spring_neap, roll7d_daymean, lag1d_daymean, lag1d_daymax), meaning
# the model did better with them shuffled. The raw one-hot day-of-week columns
# and days_since_last_obs all scored exactly 0.0 -- the trees never split on
# them.
#
# Keeping this list is not an endorsement: the lean set does not beat baseline
# either (see eval_forecast_features.py's stability check). It is here so the
# result is reproducible and so nobody re-derives it from scratch.
LEAN_FEATURES = [
    # previous-day counts -- lag1d_samehour was the single strongest new column
    "lag1d_samehour",
    "lag2d_samehour",
    "lag2d_daymean",
    "roll3d_daymean",
    "roll3d_minus_roll7d",
    # weekend shape across the day; the plain one-hot dow columns were inert
    "weekend_x_hour_sin",
    "weekend_x_hour_cos",
    # tide structure; the plain lunar-phase terms were inert or harmful
    "hours_to_good_tide",
    "tide_rate_ft_per_hr",
    "spring_neap_sin",
    "m2_sin",
    "m2_cos",
]


def build_feature_groups(df, groups=None):
    """{group name: DataFrame of that group's columns}, indexed like `df`."""
    names = list(BUILDERS) if groups is None else list(groups)
    return {name: BUILDERS[name](df) for name in names}
