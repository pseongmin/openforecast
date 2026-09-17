"""Walk-forward check for the 30-day temperature questions.

The climatological estimator has two constants (anomaly reversion, spread). They
are re-fitted on the rounds BEFORE each scored round, so the reported number is
not the in-sample fit that first suggested the model.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import date
from pathlib import Path

from core import models_weather, series
from core.fb_questions import QuestionSet

WORKSPACE = Path(__file__).resolve().parents[3]
HISTORY = Path(os.environ.get("FB_HISTORY", WORKSPACE / "data" / "forecastbench_history"))
ROUNDS = ["2026-06-21", "2026-07-05", "2026-07-19", "2026-08-02", "2026-08-16", "2026-08-30"]
REVERSIONS = [0.0, 0.3, 0.5, 0.7, 1.0]
SIGMAS = [2.0, 3.0, 4.0, 5.0, 6.0]
HORIZON = 30


def rows_for(round_name: str) -> list[tuple]:
    qs = QuestionSet.load(HISTORY / f"q_{round_name}.json")
    resolutions = json.load(open(HISTORY / f"r_{round_name}.json"))["resolutions"]
    by_key = {(q.qid, q.source): q for q in qs.questions}
    out = []
    for r in resolutions:
        q = by_key.get((str(r["id"]), r["source"]))
        if q is None or q.source != "dbnomics" or r.get("resolved_to") is None:
            continue
        rd = date.fromisoformat(r["resolution_date"])
        if (rd - qs.forecast_due_date).days != HORIZON:
            continue
        history = series.load("dbnomics", q.qid)
        if history is None:
            continue
        out.append((q, qs.forecast_due_date, rd, float(r["resolved_to"]), history))
    return out


def score(rows, reversion: float, sigma: float) -> float | None:
    scored = []
    for _, asof, rd, y, history in rows:
        est = models_weather.climatological_probability(history, asof, rd, reversion, sigma)
        if est is None:
            continue
        scored.append((est.probability, y))
    if not scored:
        return None
    return statistics.mean((p - y) ** 2 for p, y in scored)


def main() -> int:
    cached = {name: rows_for(name) for name in ROUNDS}
    pooled_flat, pooled_model = [], []
    for i, name in enumerate(ROUNDS):
        if i < 2:
            continue
        train = [row for earlier in ROUNDS[:i] for row in cached[earlier]]
        best = min(
            ((score(train, rev, sig) or 9, rev, sig) for rev in REVERSIONS for sig in SIGMAS),
            key=lambda t: t[0],
        )
        _, rev, sig = best
        rows = cached[name]
        model = score(rows, rev, sig)
        n = sum(1 for _, a, rd, _, h in rows if models_weather.climatological_probability(h, a, rd, rev, sig))
        shown = f"{model:.4f}" if model is not None else "   n/a"
        print(f"{name}  n={n:3d}  flat=0.2500  climatology={shown}  (fitted on earlier rounds: reversion={rev}, sigma={sig})")
        if model is None:
            continue
        for _, asof, rd, y, history in rows:
            est = models_weather.climatological_probability(history, asof, rd, rev, sig)
            if est is None:
                continue
            pooled_model.append((est.probability, y))
            pooled_flat.append((0.5, y))
    f = statistics.mean((p - y) ** 2 for p, y in pooled_flat)
    m = statistics.mean((p - y) ** 2 for p, y in pooled_model)
    print(f"POOLED n={len(pooled_model)}  flat={f:.4f}  climatology={m:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
