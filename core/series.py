"""Loaders for the public time series behind ForecastBench dataset questions.

Every loader returns a ``pandas.Series`` indexed by ``datetime64[ns]`` and sorted
ascending. Data comes from the local cache written by ``tools/cache_series.py``;
nothing here reaches the network, so a backtest is reproducible from the cache.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "cache"

# ForecastBench ids map onto one cache directory per source.
CACHE_DIRS = {
    "yfinance": CACHE / "yfinance",
    "fred": CACHE / "fred",
    "dbnomics": CACHE / "dbnomics",
}


@lru_cache(maxsize=4096)
def load(source: str, series_id: str) -> pd.Series | None:
    """Return the cached history for ``series_id``, or None when it is absent."""
    directory = CACHE_DIRS.get(source)
    if directory is None:
        return None
    path = directory / f"{series_id.replace('/', '__')}.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    if frame.empty or frame.shape[1] < 2:
        return None
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, value_col])
    if frame.empty:
        return None
    out = frame.set_index(date_col)[value_col].sort_index()
    return out[~out.index.duplicated(keep="last")]


def as_of(series: pd.Series, date: pd.Timestamp) -> pd.Series:
    """Truncate to rows observable on ``date`` — the backtest's only cutoff rule."""
    return series[series.index <= date]


def coverage() -> dict[str, int]:
    return {
        source: len(list(directory.glob("*.csv"))) if directory.exists() else 0
        for source, directory in CACHE_DIRS.items()
    }
