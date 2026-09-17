"""Ensemble-forecast estimator for the daily-temperature questions.

ForecastBench's ``dbnomics`` questions compare a station's daily mean
temperature on the resolution date with the value on the forecast due date.
At forecast time neither value is observed (publication lag ~3 days), so the
question is a weather forecast. Open-Meteo's public ensemble API returns ~50
members of the ECMWF IFS ensemble for up to 15 days; the probability is the
share of members in which the later day is warmer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
TIMEOUT = 30


@dataclass(frozen=True)
class EnsembleResult:
    probability: float
    n_members: int
    day_base: date
    day_target: date


def fetch_daily_mean_members(lat: float, lon: float, forecast_days: int = 16, model: str = "ecmwf_ifs025") -> dict[date, list[float]]:
    """Map each forecast day to the list of member daily-mean temperatures."""
    r = requests.get(
        ENSEMBLE_URL,
        params={
            "latitude": lat, "longitude": lon, "daily": "temperature_2m_mean",
            "models": model, "forecast_days": forecast_days, "timezone": "UTC",
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    daily = r.json()["daily"]
    days = [date.fromisoformat(t) for t in daily["time"]]
    member_keys = [k for k in daily if k.startswith("temperature_2m_mean")]
    out: dict[date, list[float]] = {}
    for i, d in enumerate(days):
        vals = [daily[k][i] for k in member_keys]
        out[d] = [float(v) for v in vals if v is not None]
    return out


def probability_warmer(
    members: dict[date, list[float]], day_base: date, day_target: date, threshold_ratio: float = 1.0
) -> EnsembleResult | None:
    base, target = members.get(day_base), members.get(day_target)
    if not base or not target:
        return None
    n = min(len(base), len(target))
    if n < 10:
        return None
    hits = 0
    for b, t in zip(base[:n], target[:n]):
        # Temperatures can be negative; the question is a plain "higher than"
        # comparison, so the ratio form only matters for the 1.0 threshold used here.
        if t > b * threshold_ratio if b > 0 else t > b:
            hits += 1
    # Laplace-smoothed so one ensemble cannot produce a 0 or 1.
    return EnsembleResult((hits + 1) / (n + 2), n, day_base, day_target)


# --- Station lookup and the ForecastBench adapter -----------------------------

import json
import math
from functools import lru_cache
from pathlib import Path

STATIONS_FILE = Path(__file__).resolve().parents[1] / "cache" / "meteofrance_stations.json"
# Fitted once on 2026-06..09 Open-Meteo previous-run errors (see forecastbench/backtest
# notes): sd of the day-7 minus day-1 forecast-difference error, in degrees C.
DIFF_SIGMA_C = 2.6


@lru_cache(maxsize=1)
def stations() -> dict[str, dict]:
    if not STATIONS_FILE.exists():
        return {}
    return json.loads(STATIONS_FILE.read_text())


def station_id(series_id: str) -> str | None:
    """``meteofrance_TEMPERATURE_celsius.07535.D`` -> ``07535``."""
    parts = series_id.split(".")
    return parts[1] if len(parts) >= 2 and parts[0].startswith("meteofrance") else None


@lru_cache(maxsize=256)
def cached_members(lat: float, lon: float) -> dict[date, list[float]]:
    return fetch_daily_mean_members(lat, lon)


def gaussian_probability(t_target: float, t_base: float, sigma: float = DIFF_SIGMA_C) -> float:
    z = (t_target - t_base) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def forecastbench_probability(series_id: str, day_base: date, day_target: date) -> EnsembleResult | None:
    """P(daily mean on day_target > daily mean on day_base) for a dbnomics station.

    Ensemble members first; if the target is beyond the ensemble range the
    caller falls back to the climatological estimator.
    """
    sid = station_id(series_id)
    meta = stations().get(sid or "")
    if meta is None:
        return None
    try:
        members = cached_members(meta["lat"], meta["lon"])
    except (requests.RequestException, KeyError, ValueError):
        return None
    return probability_warmer(members, day_base, day_target)
