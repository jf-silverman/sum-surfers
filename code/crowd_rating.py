"""
crowd_rating.py
----------------
The 1-5 crowd rating: band definitions, the statistic the rating is read
off, and the helpers both the daily chart and the 7-day chart use.

**Where the bands came from.** They are the quintiles of the hourly surfer
counts the forecast model is actually fit on (`data/training_features.csv`
after `load_and_prepare()`'s dropna) — measured, not chosen. Re-measure any
time with `python code/eval_crowd_rating.py`, which prints the live quintile
edges next to the ones hardcoded here so drift is visible rather than
silent. Hardcoded rather than computed per run because a rating whose
meaning moves every night is not a rating: "level 4" has to mean the same
crowd in December that it meant in September.

**Which statistic.** Not the median point estimate. The forecast regresses
to the mean (docs/known_bugs.md L02) — it over-predicts quiet hours and
under-predicts crowded ones — so a rating read off the median cannot reach
the top band as often as reality does, and the busiest hours, which are the
ones a surfer most wants warned about, are exactly the ones it would
under-report. `code/eval_crowd_rating.py` measures that on a held-out split
and compares upper-tail readings of the same three quantile models the
chart already fits. RATING_QUANTILE below records the winner; the script's
output is the evidence for it.
"""

# Quintile edges of the measured count distribution (20/40/60/80th
# percentile) as of 2026-09-23: 1,594 hourly rows, 2025-10-11 to 2026-09-22,
# mean 15.27 surfers, 14.4% of hours empty. Edges: 2 / 8 / 16 / 27.
# `max` is inclusive; the top band has none.
CROWD_LEVELS = [
    {"level": 1, "min": 0,  "max": 2,    "name": "Empty",  "blurb": "near-empty water"},
    {"level": 2, "min": 3,  "max": 8,    "name": "Quiet",  "blurb": "a handful out"},
    {"level": 3, "min": 9,  "max": 16,   "name": "Steady", "blurb": "a typical session"},
    {"level": 4, "min": 17, "max": 27,   "name": "Busy",   "blurb": "busy lineup"},
    {"level": 5, "min": 28, "max": None, "name": "Packed", "blurb": "top fifth of all recorded hours"},
]

# The quantile of the predictive distribution the rating is read off.
#
# Measured on 319 held-out hours (code/eval_crowd_rating.py, 2026-09-23;
# 21.3% of those hours are genuinely level 5):
#
#   source   exact  within-1  mean level err  L5 recall  share rated L5
#   q0.50    50.5%     92.2%          +0.09       25.0%            7.8%
#   q0.55    52.0%     92.2%          +0.25       47.1%           13.5%
#   q0.60    50.8%     91.5%          +0.32       54.4%           16.9%
#   q0.70    42.3%     86.2%          +0.60       76.5%           29.5%
#   q0.90    31.7%     75.5%          +0.97       91.2%           47.3%
#
# The median's regression to the mean (docs/known_bugs.md L02) shows up here
# as a frequency error, not a bias in the level number: its mean level error
# is a near-perfect +0.09, yet it calls only 7.8% of hours level 5 where
# reality has 19.9%, and catches just 25% of the genuinely packed ones. A
# rating that misses three quarters of the crowds is the wrong rating, since
# the packed hours are the ones worth being warned about.
#
# 0.60 is the point where that is fixed at almost no cost to agreement: the
# same exact-match rate as the median (50.8% vs 50.5%), within-1 essentially
# unchanged (91.5% vs 92.2%), level-5 recall more than doubled (54.4%), and
# the share of hours rated 5 (16.9%) close to reality's 19.9% instead of a
# third of it. Past 0.65 agreement falls away fast for recall that is bought
# by calling half the week packed.
#
# Interpolated from the 0.10/0.50/0.90 models the chart already fits —
# deliberately NOT a fourth fitted model, because plot_daily_prediction.py
# refits on every run and another fit costs real time on every chart.
RATING_QUANTILE = 0.60


def crowd_level(count):
    """1-5 crowd level for a surfer count (float or int). Never returns None."""
    for band in CROWD_LEVELS:
        if band["max"] is None or count <= band["max"] + 0.5:
            return band["level"]
    return CROWD_LEVELS[-1]["level"]


def level_info(level):
    """The CROWD_LEVELS entry for a level."""
    return CROWD_LEVELS[level - 1]


def level_label(level):
    """e.g. "4 - Busy"."""
    return f"{level} - {level_info(level)['name']}"


def band_text(level, units=True):
    """e.g. "17-27 surfers" — what the level means in surfers."""
    band = level_info(level)
    suffix = " surfers" if units else ""
    if band["max"] is None:
        return f"{band['min']}+{suffix}"
    return f"{band['min']}-{band['max']}{suffix}"


def quantile_at(p, quantiles):
    """Read quantile `p` off a predictive distribution known only at a few
    levels, by linear interpolation of the quantile function between the two
    bracketing fitted levels (and clamping outside their range).

    `quantiles` is {level: value}, e.g. the chart's {0.10: ..., 0.50: ...,
    0.90: ...}. Interpolating is what lets the rating use an upper-tail
    reading without fitting another model — see RATING_QUANTILE.
    """
    levels = sorted(quantiles)
    if p <= levels[0]:
        return float(quantiles[levels[0]])
    if p >= levels[-1]:
        return float(quantiles[levels[-1]])
    for lo, hi in zip(levels, levels[1:]):
        if lo <= p <= hi:
            span = hi - lo
            if span == 0:
                return float(quantiles[lo])
            frac = (p - lo) / span
            return float(quantiles[lo] + frac * (quantiles[hi] - quantiles[lo]))
    return float(quantiles[levels[-1]])


def rating_from_quantiles(quantiles, p=RATING_QUANTILE):
    """(level, value_the_level_was_read_off) for one forecast hour."""
    value = quantile_at(p, quantiles)
    return crowd_level(value), value


def quintile_edges(counts):
    """The measured 20/40/60/80th percentiles of a count series, as the
    integers the bands would use. Lets eval_crowd_rating.py show whether
    CROWD_LEVELS still matches the data it was derived from."""
    return [int(round(float(counts.quantile(q)))) for q in (0.2, 0.4, 0.6, 0.8)]
