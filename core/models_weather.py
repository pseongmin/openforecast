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

import numpy as np
import pandas as pd
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
    members: dict[date, list[float]], day_base: date, day_target: date, sigma: float | None = None
) -> "EnsembleResult | None":
    """P(daily mean on day_target > daily mean on day_base).

    The ensemble *mean* difference is mapped through a Gaussian whose width was
    measured on 2026-06..09 forecast errors (``DIFF_SIGMA_C``), because the raw
    member share is over-confident: the day-0/1 members are under-dispersed
    (sd 0.5 C vs a 1.1 C realised error) and the sweep in the backtest notes
    showed pooled Brier still falling at sigma 4.0.
    """
    base, target = members.get(day_base), members.get(day_target)
    if not base or not target:
        return None
    n = min(len(base), len(target))
    if n < 10:
        return None
    diff = sum(target[:n]) / n - sum(base[:n]) / n
    p = gaussian_probability(diff, 0.0, sigma if sigma is not None else DIFF_SIGMA_C)
    return EnsembleResult(min(max(p, 0.03), 0.97), n, day_base, day_target)


# --- Station lookup and the ForecastBench adapter -----------------------------

import json
import math
from functools import lru_cache
from pathlib import Path

STATIONS_FILE = Path(__file__).resolve().parents[1] / "cache" / "meteofrance_stations.json"
# Chosen on the 2026-06-21..08-30 rounds (n=286 seven-day questions): pooled Brier
# 0.200 at 2.0 C, 0.193 at 2.6 C, 0.190 at 3.0 C, 0.186 at 4.0 C; the realised error
# sd of the day-7 minus day-1 difference is 2.4 C, so 4.0 C is deliberately wider.
DIFF_SIGMA_C = 4.0


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


# --- Climatology for horizons beyond the ensemble range -----------------------

def _day_of_year_normal(history: pd.Series, day: int, window_days: int = 15) -> float | None:
    """Mean value within +-window_days of that day of year, over the whole history."""
    doy = history.index.dayofyear.to_numpy()
    distance = np.minimum(np.abs(doy - day), 366 - np.abs(doy - day))
    selected = history.to_numpy(dtype=float)[distance <= window_days]
    if selected.size < 20:
        return None
    return float(np.nanmean(selected))


# REJECTED as a shipped estimator (2026-09-17). Fitting (reversion, sigma) on the
# rounds before each scored round and scoring the next one gives pooled Brier
# 0.2864 against 0.2500 for a flat 0.5 — worse, and wildly unstable round to round
# (0.4696 then 0.1108). The in-sample fit that suggested it scored 0.2020, which is
# what an overfit looks like. Kept for reference and used by the experiment only.
def climatological_probability(
    history: pd.Series,
    asof: date,
    resolution_date: date,
    reversion: float,
    sigma: float,
    publication_lag_days: int = 3,
) -> "EnsembleResult | None":
    """P(value on resolution_date > value on asof) from season plus mean reversion.

    Beyond the ensemble's 15-day range the only usable signals are (a) how the
    season itself moves between the two dates and (b) that today's anomaly decays.
    Both constants are supplied by the caller so they can be fitted walk-forward.
    """
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=publication_lag_days)
    observable = history[history.index <= cutoff]
    if len(observable) < 365 * 3:
        return None
    normal_asof = _day_of_year_normal(observable, pd.Timestamp(asof).dayofyear)
    normal_target = _day_of_year_normal(observable, pd.Timestamp(resolution_date).dayofyear)
    if normal_asof is None or normal_target is None:
        return None
    anomaly = float(observable.to_numpy(dtype=float)[-1]) - normal_asof
    expected_change = (normal_target - normal_asof) - reversion * anomaly
    p = gaussian_probability(expected_change, 0.0, sigma)
    return EnsembleResult(min(max(p, 0.05), 0.95), len(observable), asof, resolution_date)
