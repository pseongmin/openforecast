"""Forecasters for ForecastBench *dataset* questions.

A dataset question asks whether a public series will clear a bar by a resolution
date. Two estimators are combined:

``empirical``  the distribution of the series' own h-step changes, read off the
               cached history strictly before the forecast due date;
``base_rate``  the pooled outcome rate for (source, horizon, threshold) from
               already-resolved rounds, which is the only estimator available
               for sources whose series we cannot fetch.

The empirical estimate is shrunk toward the base rate by its own sample size, so
a series with three usable observations cannot shout down a 300-question prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from core.fb_questions import Question

CLAMP_LOW, CLAMP_HIGH = 0.02, 0.98
# Sample size at which the empirical estimate carries half the weight. Set from
# the walk-forward sweep in forecastbench/backtest.py, not by taste.
SHRINK_N = 40.0


@dataclass(frozen=True)
class Estimate:
    probability: float
    method: str
    n: int


def clamp(p: float) -> float:
    return max(CLAMP_LOW, min(CLAMP_HIGH, p))


def shrink(empirical: Estimate | None, prior: float, shrink_n: float = SHRINK_N) -> Estimate:
    if empirical is None or empirical.n == 0:
        return Estimate(clamp(prior), "prior", 0)
    weight = empirical.n / (empirical.n + shrink_n)
    blended = weight * empirical.probability + (1.0 - weight) * prior
    return Estimate(clamp(blended), f"{empirical.method}+prior", empirical.n)


def horizon_change_probability(
    history: pd.Series,
    asof: pd.Timestamp,
    horizon_days: int,
    threshold_ratio: float,
    allow_equal: bool,
    lookback_years: int = 12,
) -> "Estimate | None":
    """P(value at asof+h clears ``threshold_ratio`` x value at asof).

    Uses overlapping h-calendar-day changes: the resolution dates are calendar
    dates, so a business-day step would systematically mis-set the horizon.
    """
    hist = history[history.index <= asof]
    if len(hist) < 30:
        return None
    hist = hist[hist.index >= asof - pd.Timedelta(days=365 * lookback_years)]
    if len(hist) < 30:
        return None
    values = hist.to_numpy(dtype=float)
    dates = hist.index.to_numpy()
    later_idx = np.searchsorted(dates, dates + np.timedelta64(horizon_days, "D"))
    valid = later_idx < len(values)
    if valid.sum() < 10:
        return None
    base = values[valid]
    future = values[later_idx[valid]]
    ok = (base != 0) & np.isfinite(base) & np.isfinite(future)
    if ok.sum() < 10:
        return None
    ratio = future[ok] / base[ok]
    hits = ratio >= threshold_ratio if allow_equal else ratio > threshold_ratio
    return Estimate(float(hits.mean()), "empirical", int(ok.sum()))


def seasonal_change_probability(
    history: pd.Series,
    asof: pd.Timestamp,
    horizon_days: int,
    threshold_ratio: float,
    allow_equal: bool,
    window_days: int = 12,
) -> "Estimate | None":
    """Same as above but restricted to the same time of year.

    Daily temperature forces this: an unconditional 7-day change says nothing,
    while "is mid-September warmer than early September" is close to a calendar
    fact.
    """
    hist = history[history.index <= asof]
    if len(hist) < 200:
        return None
    target_doy = asof.dayofyear
    doy = hist.index.dayofyear.to_numpy()
    delta = np.minimum(np.abs(doy - target_doy), 365 - np.abs(doy - target_doy))
    idx = np.flatnonzero(delta <= window_days)
    if len(idx) < 20:
        return None
    values = hist.to_numpy(dtype=float)
    dates = hist.index.to_numpy()
    later = np.searchsorted(dates, dates[idx] + np.timedelta64(horizon_days, "D"))
    valid = later < len(values)
    idx, later = idx[valid], later[valid]
    if len(idx) < 15:
        return None
    base, future = values[idx], values[later]
    ok = (base != 0) & np.isfinite(base) & np.isfinite(future)
    if ok.sum() < 15:
        return None
    ratio = future[ok] / base[ok]
    hits = ratio >= threshold_ratio if allow_equal else ratio > threshold_ratio
    return Estimate(float(hits.mean()), "seasonal", int(ok.sum()))


def poisson_count_probability(
    freeze_value: float | None, threshold_ratio: float
) -> "Estimate | None":
    """ACLED-style counts: 'more than k times the reference rate over 30 days'.

    The reference value is one plus the 30-day average over the prior 360 days,
    so a Poisson count with that mean is the honest null. No ACLED history is
    used: the dataset API is not public, and inventing one would be worse than
    the prior.
    """
    if freeze_value is None or freeze_value <= 0:
        return None
    lam = float(freeze_value)
    k = math.floor(threshold_ratio * lam)
    cdf, term = 0.0, math.exp(-lam)
    for i in range(0, k + 1):
        if i > 0:
            term *= lam / i
        cdf += term
    return Estimate(float(max(0.0, 1.0 - cdf)), "poisson", 30)


def forecast(
    question: Question,
    asof: date,
    resolution_date: date,
    history: "pd.Series | None",
    prior: float,
    shrink_n: float = SHRINK_N,
) -> Estimate:
    """Pick the estimator the question's source and shape allow, then shrink."""
    horizon = (resolution_date - asof).days
    ratio = question.relative_threshold
    equal = question.allows_equal
    stamp = pd.Timestamp(asof)

    empirical: Estimate | None = None
    if history is not None and len(history) > 0:
        if question.source == "dbnomics":
            empirical = seasonal_change_probability(history, stamp, horizon, ratio, equal)
        if empirical is None:
            empirical = horizon_change_probability(history, stamp, horizon, ratio, equal)
    elif question.source == "acled":
        empirical = poisson_count_probability(question.freeze_value, ratio)

    return shrink(empirical, prior, shrink_n)
