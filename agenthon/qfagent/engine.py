"""Forecast engine for Agenthon 2026 Track 2.

The metric is S = 0.5 CRPS + 0.3 variogram + 0.2 tail penalty (lower is better),
so the object being built is a *joint predictive distribution*, delivered as
Monte-Carlo draws. Three properties follow directly from that metric and drive
the design:

1. Draws must be jointly sampled across assets and horizons — an independently
   sampled matrix is penalised by the variogram term even when every margin is
   right.
2. The spread has to be honest. The tail term is pinball loss at 1/5/95/99 %,
   which punishes thin tails harder than wide ones.
3. The centre should sit at the random walk unless something argues otherwise;
   a drift that is not earned is a bias.

The engine is a stationary block bootstrap over the panel's own history, which
preserves autocorrelation and cross-asset dependence without assuming normality,
followed by an optional scenario mixture that the text layer can populate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """One branch of a mixture: a weight, a drift shift and a spread multiplier.

    ``drift`` is expressed in units of the horizon's own standard deviation, so a
    text signal cannot accidentally move a quiet series as far as a violent one.
    """

    name: str
    weight: float
    drift_in_sd: float = 0.0
    spread_multiple: float = 1.0
    established: bool = False


@dataclass
class ForecastRequest:
    asset_ids: list[str]
    horizons: list[int]
    target_type: str  # "level" or "log_return"
    n_draws: int = 1000
    scenarios: list[Scenario] = field(default_factory=list)
    seed: int = 20260917


def _returns(panel: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """Per-asset step series the bootstrap resamples.

    For ``level`` panels the step is the first difference; for ``log_return``
    panels the rows already are daily simple returns, so the step is log(1+r).
    """
    if target_type == "log_return":
        return np.log1p(panel.clip(lower=-0.9999))
    return panel.diff()


def stationary_block_bootstrap(
    steps: pd.DataFrame, horizon: int, n_draws: int, rng: np.random.Generator,
    mean_block: int = 10,
) -> np.ndarray:
    """Return an (n_draws, n_assets) array of cumulative h-step changes.

    Blocks are sampled jointly across columns — the whole row block moves
    together — which is what preserves cross-asset dependence.
    """
    values = steps.to_numpy(dtype=float)
    values = values[~np.isnan(values).any(axis=1)]
    n_obs, n_assets = values.shape
    if n_obs < 30:
        raise ValueError(f"need >=30 clean observations, have {n_obs}")

    out = np.empty((n_draws, n_assets), dtype=float)
    p_restart = 1.0 / mean_block
    for d in range(n_draws):
        total = np.zeros(n_assets)
        idx = rng.integers(n_obs)
        for _ in range(horizon):
            total += values[idx]
            if rng.random() < p_restart:
                idx = rng.integers(n_obs)
            else:
                idx = (idx + 1) % n_obs
        out[d] = total
    return out


def apply_scenarios(
    changes: np.ndarray, scenarios: list[Scenario], rng: np.random.Generator
) -> np.ndarray:
    """Turn one cloud into a weighted mixture of shifted/scaled clouds."""
    if not scenarios:
        return changes
    weights = np.array([s.weight for s in scenarios], dtype=float)
    if weights.sum() <= 0:
        return changes
    weights = weights / weights.sum()
    assignment = rng.choice(len(scenarios), size=len(changes), p=weights)
    sd = changes.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    centre = changes.mean(axis=0)
    out = changes.copy()
    for i, scenario in enumerate(scenarios):
        mask = assignment == i
        if not mask.any():
            continue
        shifted = (out[mask] - centre) * scenario.spread_multiple + centre
        out[mask] = shifted + scenario.drift_in_sd * sd
    return out


def forecast_draws(
    panel: pd.DataFrame, request: ForecastRequest
) -> pd.DataFrame:
    """Build the submission frame: one row per (draw, asset, horizon).

    ``panel`` is wide: index = date, columns = asset ids, values = the series.
    """
    rng = np.random.default_rng(request.seed)
    assets = [a for a in request.asset_ids if a in panel.columns]
    if not assets:
        raise ValueError(f"none of {request.asset_ids} present in panel columns")
    steps = _returns(panel[assets], request.target_type)
    anchor = panel[assets].ffill().iloc[-1].to_numpy(dtype=float)

    rows = []
    for horizon in request.horizons:
        changes = stationary_block_bootstrap(steps, horizon, request.n_draws, rng)
        changes = apply_scenarios(changes, request.scenarios, rng)
        if request.target_type == "log_return":
            values = changes  # target is the cumulative log return itself
        else:
            values = anchor + changes
        for j, asset in enumerate(assets):
            rows.append(
                pd.DataFrame(
                    {
                        "draw": np.arange(request.n_draws, dtype="int32"),
                        "asset": asset,
                        "horizon": np.int32(horizon),
                        "value": values[:, j].astype("float64"),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    return frame.astype({"draw": "int32", "horizon": "int32", "value": "float64"})
