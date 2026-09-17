"""Forecaster for the ForecastBench ``fred`` questions.

The question asks whether a FRED series is higher on the resolution date than on
the forecast due date. Two regimes, measured over the six resolved rounds
2026-06-21..2026-08-30 (n=527):

* **Administered rates that do not move between policy meetings** — the fed funds
  target band, IORB, EFFR, the prime rate, the ECB deposit rate and friends. Over
  the last 40 observations the value changes on fewer than 10 % of days, and the
  question resolved NO 29 times out of 29. "Increase" needs an actual move, and a
  policy rate makes none inside a 7-to-30 day window unless a meeting falls in it.
* **Everything else** — yields, spreads, FX, balance-sheet lines. The empirical
  distribution of the series' own h-day changes is the estimator, shrunk toward
  the walk-forward pooled rate for that horizon.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from core.fb_questions import Question
from core.models_dataset import Estimate, clamp, horizon_change_probability, shrink

FLAT_MOVE_RATE = 0.10      # share of recent observations that change at all
FLAT_LOOKBACK = 40         # observations used to judge it
FLAT_PROBABILITY = 0.03    # a policy rate can move: a meeting may fall in the window


def move_rate(history: pd.Series, lookback: int = FLAT_LOOKBACK) -> float | None:
    """Share of the last ``lookback`` observations that differ from the previous one."""
    tail = history.tail(lookback)
    if len(tail) < 20:
        return None
    return float((tail.diff().abs() > 1e-12).mean())


def forecast(
    question: Question,
    asof: date,
    resolution_date: date,
    history: pd.Series | None,
    prior: float,
    shrink_n: float = 40.0,
) -> Estimate:
    if history is None or len(history) == 0:
        return Estimate(clamp(prior), "prior", 0)
    stamp = pd.Timestamp(asof)
    observable = history[history.index <= stamp]
    if len(observable) < 20:
        return Estimate(clamp(prior), "prior", 0)

    rate = move_rate(observable)
    if rate is not None and rate < FLAT_MOVE_RATE:
        return Estimate(FLAT_PROBABILITY, "structural:administered-rate", len(observable))

    horizon = (resolution_date - asof).days
    empirical = horizon_change_probability(
        observable, stamp, horizon, question.relative_threshold, question.allows_equal
    )
    return shrink(empirical, prior, shrink_n)
