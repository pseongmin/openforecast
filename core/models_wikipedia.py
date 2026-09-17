"""Forecaster for the ForecastBench ``wikipedia`` questions.

These questions are drawn from a handful of templates whose answer is mostly
decided by the calendar, not by judgement. Measured over the six resolved rounds
2026-06-21..2026-08-30 (n=541):

| template                                   |   n | P(yes) |
|--------------------------------------------|-----|--------|
| "will a vaccine have been developed ..."    | 274 | 0.000  |
| "... Elo rating at least X% higher ..."     | 132 | 0.000  |
| "... FIDE ranking as high or higher ..."    | 121 | 0.833  |
| "... still hold the world record ..."       |  14 | 1.000  |

The ranking template splits further on a mechanism: the FIDE rating list is
published monthly, and the Wikipedia page only changes when a new list lands.
With no list release inside the forecast window the value cannot move, so
"as high or higher" is a certain yes — measured 54/54. With one release inside
the window it becomes a real forecast (0.46 at 7 days, 0.76 at 30 days), and the
rate is taken from earlier rounds only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from core.fb_questions import Question
from core.models_dataset import Estimate, clamp

VACCINE = re.compile(r"vaccine have been developed", re.I)
WORLD_RECORD = re.compile(r"still hold the world record", re.I)
PCT_HIGHER = re.compile(r"at least [\d.]+% higher", re.I)
AS_HIGH_OR_HIGHER = re.compile(r"as high or higher", re.I)

# Structural answers: the event cannot happen (or cannot fail) inside a 7-to-30
# day window. Kept away from 0/1 because a benchmark question is never certain.
NEAR_ZERO = 0.02
NEAR_ONE = 0.97


def classify(question: Question) -> str:
    text = question.text
    if VACCINE.search(text):
        return "vaccine"
    if WORLD_RECORD.search(text):
        return "world_record"
    if PCT_HIGHER.search(text):
        return "pct_higher"
    if AS_HIGH_OR_HIGHER.search(text):
        return "as_high_or_higher"
    return "other"


def monthly_releases_in(start: date, end: date) -> int:
    """Number of month boundaries in (start, end] — FIDE publishes on the 1st."""
    count = 0
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if start < cursor <= end:
            count += 1
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    return count


class WikipediaPriors:
    """Walk-forward outcome rates keyed by template, horizon and release count."""

    def __init__(self) -> None:
        self.hits: dict[tuple, list[float]] = {}

    def add(self, question: Question, asof: date, resolution_date: date, outcome: float) -> None:
        for key in self._keys(question, asof, resolution_date):
            self.hits.setdefault(key, []).append(outcome)

    def _keys(self, question: Question, asof: date, resolution_date: date) -> list[tuple]:
        """Lookup order. A window containing a list release is a different question
        from one that cannot move, so a with-release question must never inherit a
        no-release rate: it falls back across horizons within its own release class
        (measured 2026-08-30, where that inheritance cost 0.12 Brier on the round)."""
        kind = classify(question)
        horizon = (resolution_date - asof).days
        has_release = monthly_releases_in(asof, resolution_date) >= 1
        keys = [(kind, horizon, has_release), (kind, has_release)]
        if kind not in ("as_high_or_higher", "pct_higher"):
            keys.append((kind,))
        return keys

    def rate(self, question: Question, asof: date, resolution_date: date, min_n: int = 12) -> tuple[float, int] | None:
        for key in self._keys(question, asof, resolution_date):
            values = self.hits.get(key)
            if values and len(values) >= min_n:
                return sum(values) / len(values), len(values)
        return None


def forecast(
    question: Question,
    asof: date,
    resolution_date: date,
    prior: float,
    priors: WikipediaPriors | None = None,
) -> Estimate:
    kind = classify(question)
    if kind == "vaccine":
        return Estimate(NEAR_ZERO, "structural:vaccine", 0)
    if kind == "world_record":
        return Estimate(NEAR_ONE, "structural:world_record", 0)

    releases = monthly_releases_in(asof, resolution_date)
    if releases == 0:
        # The source page cannot change inside the window, so the value is the
        # same one the question quotes: ties count as yes, a strict increase does not.
        if kind == "as_high_or_higher":
            return Estimate(NEAR_ONE, "structural:no-release", 0)
        if kind == "pct_higher":
            return Estimate(NEAR_ZERO, "structural:no-release", 0)

    if priors is not None:
        hit = priors.rate(question, asof, resolution_date)
        if hit is not None:
            rate, n = hit
            return Estimate(clamp(rate), f"rate:{kind}", n)
    if kind == "as_high_or_higher":
        # One release inside the window: the ranking can move, and it usually does
        # not fall. Never inherit the certain-yes rate of a no-release window.
        return Estimate(0.72, "default:release", 0)
    if kind == "pct_higher":
        return Estimate(0.08, "default:release", 0)
    return Estimate(clamp(prior), "prior", 0)
