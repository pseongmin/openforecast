"""Forecasters for ForecastBench *market* questions.

The question set freezes each market's price ten days before the forecast due
date. A live quote on the due date is strictly more informed, so the forecaster
prefers it when the venue's public API is reachable and falls back to the frozen
value otherwise. Only venues with a public, unauthenticated read endpoint are
queried; nothing is cached across rounds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from core.fb_questions import Question

TIMEOUT = 12
UA = {"User-Agent": "openforecast/0.1 (public benchmark research)"}


@dataclass(frozen=True)
class MarketEstimate:
    probability: float
    method: str


def _clamp(p: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, p))


def manifold_probability(market_id: str) -> float | None:
    try:
        r = requests.get(f"https://api.manifold.markets/v0/market/{market_id}", headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        blob = r.json()
    except (requests.RequestException, ValueError):
        return None
    if blob.get("outcomeType") not in (None, "BINARY"):
        return None
    p = blob.get("probability")
    return float(p) if isinstance(p, (int, float)) else None


def kalshi_probability(ticker: str) -> float | None:
    try:
        r = requests.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}", headers=UA, timeout=TIMEOUT
        )
        if r.status_code != 200:
            return None
        m = r.json().get("market", {})
    except (requests.RequestException, ValueError):
        return None
    # Prefer the mid of the yes book; fall back to last traded price. Prices are in cents.
    bid, ask, last = m.get("yes_bid"), m.get("yes_ask"), m.get("last_price")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid > 0:
        return (bid + ask) / 200.0
    if isinstance(last, (int, float)) and last > 0:
        return last / 100.0
    return None


LIVE_FETCHERS = {
    "manifold": manifold_probability,
    "kalshi": kalshi_probability,
}


def forecast(question: Question, use_live: bool = True) -> MarketEstimate:
    """Live price when the venue is reachable, else the frozen price, else 0.5."""
    if use_live:
        fetcher = LIVE_FETCHERS.get(question.source)
        if fetcher is not None:
            live = fetcher(question.qid)
            if live is not None:
                return MarketEstimate(_clamp(live), f"live:{question.source}")
    if question.freeze_value is not None:
        return MarketEstimate(_clamp(question.freeze_value), "freeze")
    return MarketEstimate(0.5, "uninformed")
