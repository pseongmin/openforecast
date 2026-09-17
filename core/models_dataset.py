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
# Poisson estimates carry n=30, so shrink_n=20 gives them weight 30/(30+20)=0.6.
ACLED_SHRINK_N = 20.0


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
    window_days: int = 15,
    publication_lag_days: int = 3,
) -> "Estimate | None":
    """Seasonal *and* anomaly-conditioned estimate for daily weather-like series.

    Two facts drive the design (measured on the 2026-08-30 round, where a plain
    seasonal model scored worse than 0.5): the 7-day change in temperature is
    predicted far better by the current anomaly against the day-of-year normal
    (anomalies mean-revert) than by the calendar alone, and the value on the
    forecast due date is not yet published when the forecast is made, so the
    anomaly is read off the last observation available under the publication lag.
    """
    cutoff = asof - pd.Timedelta(days=publication_lag_days)
    hist = history[history.index <= cutoff]
    if len(hist) < 365 * 3:
        return None
    values = hist.to_numpy(dtype=float)
    dates = hist.index
    doy = dates.dayofyear.to_numpy()
    # Day-of-year climatology, smoothed over +-window_days.
    normal = np.full(367, np.nan)
    for d in range(1, 367):
        delta = np.minimum(np.abs(doy - d), 366 - np.abs(doy - d))
        sel = delta <= window_days
        if sel.sum() >= 20:
            normal[d] = np.nanmean(values[sel])
    if np.isnan(normal[asof.dayofyear]):
        return None
    current_anom = values[-1] - normal[doy[-1]]

    # Historical analogues: same season, similar anomaly, and a later value h days on.
    target_doy = asof.dayofyear
    delta = np.minimum(np.abs(doy - target_doy), 366 - np.abs(doy - target_doy))
    idx = np.flatnonzero(delta <= window_days)
    anoms = values[idx] - normal[doy[idx]]
    spread = np.nanstd(values - normal[doy]) or 1.0
    close = np.abs(anoms - current_anom) <= 0.6 * spread
    idx = idx[close]
    if len(idx) < 20:
        return None
    later = np.searchsorted(dates.to_numpy(), dates.to_numpy()[idx] + np.timedelta64(horizon_days + publication_lag_days, "D"))
    base_later = np.searchsorted(dates.to_numpy(), dates.to_numpy()[idx] + np.timedelta64(publication_lag_days, "D"))
    valid = (later < len(values)) & (base_later < len(values))
    idx, later, base_later = idx[valid], later[valid], base_later[valid]
    if len(idx) < 20:
        return None
    base, future = values[base_later], values[later]
    ok = (base != 0) & np.isfinite(base) & np.isfinite(future)
    if ok.sum() < 20:
        return None
    ratio = future[ok] / base[ok]
    hits = ratio >= threshold_ratio if allow_equal else ratio > threshold_ratio
    return Estimate(float(hits.mean()), "seasonal-anomaly", int(ok.sum()))


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


PRIOR_ONLY_SHORT_HORIZON = {
    # Equity 7/30-day outcomes move together across the whole question set, so a
    # per-ticker drift estimate is mostly noise there (measured 2026-09-17:
    # empirical 0.25-0.28 Brier vs pooled prior 0.249).
    "yfinance": 30,
    # Daily temperature 7/30-day questions compare two values that are BOTH
    # unobserved at forecast time (publication lag ~3 days), so they are weather
    # forecasts, not statistics. Until the ensemble-forecast estimator lands
    # (core/models_weather.py), the pooled prior is the honest answer.
    "dbnomics": 30,
}


def use_prior_only(source: str, horizon_days: int) -> bool:
    limit = PRIOR_ONLY_SHORT_HORIZON.get(source)
    return limit is not None and horizon_days <= limit


def forecast(
    question: Question,
    asof: date,
    resolution_date: date,
    history: "pd.Series | None",
    prior: float,
    shrink_n: float = SHRINK_N,
    use_weather_forecast: bool = False,
    wiki_priors=None,
) -> Estimate:
    """Pick the estimator the question's source and shape allow, then shrink.

    ``use_weather_forecast`` switches the live ensemble estimator on for the
    7-day temperature questions; it must stay off in backtests, where the
    ensemble archive is not available and the flat 0.5 is the honest number.
    """
    horizon = (resolution_date - asof).days
    ratio = question.relative_threshold
    equal = question.allows_equal
    stamp = pd.Timestamp(asof)

    if question.source == "wikipedia":
        from core import models_wikipedia

        return models_wikipedia.forecast(question, asof, resolution_date, prior, wiki_priors)

    if question.source == "fred":
        from core import models_fred

        return models_fred.forecast(question, asof, resolution_date, history, prior, shrink_n)

    empirical: Estimate | None = None
    if question.source == "dbnomics" and horizon <= 16 and use_weather_forecast:
        from core import models_weather

        ens = models_weather.forecastbench_probability(question.qid, asof, resolution_date)
        if ens is not None:
            return Estimate(clamp(ens.probability), "ensemble", ens.n_members)
    if use_prior_only(question.source, horizon):
        # Within one round these questions resolve together (one market week,
        # one weather week), so a pooled prior from earlier rounds is a bet on
        # that week, not knowledge: 0.375 vs 0.79 realised on 2026-07-05.
        return Estimate(0.5, "flat", 0)
    if history is not None and len(history) > 0:
        if question.source == "dbnomics":
            empirical = seasonal_change_probability(history, stamp, horizon, ratio, equal)
        if empirical is None:
            empirical = horizon_change_probability(history, stamp, horizon, ratio, equal)
    elif question.source == "acled":
        empirical = poisson_count_probability(question.freeze_value, ratio)
        # The Poisson tail discriminates within a round but is miscalibrated at the
        # centre, so it is blended with the walk-forward rate. Weight 0.6 measured
        # on rounds 2026-07-05..08-30 (0.0721 vs 0.0726 at the default 0.43).
        return shrink(empirical, prior, ACLED_SHRINK_N)

    return shrink(empirical, prior, shrink_n)
