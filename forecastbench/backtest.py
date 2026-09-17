"""Walk-forward evaluation of the ForecastBench forecasters on resolved rounds.

For each round R the priors come from rounds strictly before R, and the series
histories are truncated at R's forecast due date. That is the only way a number
from this script can be compared with the public leaderboard.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from datetime import date
from pathlib import Path

import pandas as pd

from core import models_dataset, models_market, series
from core.fb_questions import MARKET_SOURCES, Question, QuestionSet


def load_resolutions(path: Path) -> list[dict]:
    return json.loads(path.read_text())["resolutions"]


def brier_index(mean_brier: float) -> float:
    """Leaderboard-style scale: always-0.5 scores 50, perfect scores 100."""
    return 100.0 * (1.0 - 2.0 * mean_brier)


class PriorTable:
    """Pooled outcome rate by (source, horizon-days, threshold) from earlier rounds."""

    def __init__(self) -> None:
        self.hits: dict[tuple, list[float]] = collections.defaultdict(list)

    def add(self, question: Question, horizon: int, outcome: float) -> None:
        key = (question.source, horizon, round(question.relative_threshold, 3))
        self.hits[key].append(outcome)
        self.hits[(question.source, horizon)].append(outcome)
        self.hits[(question.source,)].append(outcome)

    def prior(self, question: Question, horizon: int, min_n: int = 20) -> float:
        for key in (
            (question.source, horizon, round(question.relative_threshold, 3)),
            (question.source, horizon),
            (question.source,),
        ):
            v = self.hits.get(key)
            if v and len(v) >= min_n:
                return statistics.mean(v)
        return 0.5


def evaluate_round(
    qs: QuestionSet, resolutions: list[dict], priors: PriorTable, shrink_n: float, use_live: bool
) -> dict:
    by_key = {(q.qid, q.source): q for q in qs.questions}
    rows = []
    for r in resolutions:
        q = by_key.get((str(r["id"]), r["source"]))
        if q is None or r.get("resolved_to") is None:
            continue
        y = float(r["resolved_to"])
        if q.is_market:
            est = models_market.forecast(q, use_live=use_live)
            rows.append(("market", q.source, None, est.probability, y, est.method))
        else:
            rd = date.fromisoformat(r["resolution_date"])
            horizon = (rd - qs.forecast_due_date).days
            hist = series.load(q.source, q.qid)
            prior = priors.prior(q, horizon)
            est = models_dataset.forecast(q, qs.forecast_due_date, rd, hist, prior, shrink_n)
            rows.append(("dataset", q.source, horizon, est.probability, y, est.method))
    return summarise(rows)


def summarise(rows: list[tuple]) -> dict:
    out: dict = {}
    for kind in ("market", "dataset"):
        sub = [r for r in rows if r[0] == kind]
        if not sub:
            continue
        mb = statistics.mean((p - y) ** 2 for _, _, _, p, y, _ in sub)
        out[kind] = {"n": len(sub), "brier": round(mb, 4), "index": round(brier_index(mb), 1)}
        per_src: dict = {}
        for src in sorted({r[1] for r in sub}):
            s2 = [r for r in sub if r[1] == src]
            mb2 = statistics.mean((p - y) ** 2 for _, _, _, p, y, _ in s2)
            methods = collections.Counter(r[5].split("+")[0] for r in s2)
            per_src[src] = {"n": len(s2), "brier": round(mb2, 4), "methods": dict(methods)}
        out[kind]["by_source"] = per_src
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with q_<date>.json and r_<date>.json")
    ap.add_argument("--shrink-n", type=float, default=models_dataset.SHRINK_N)
    ap.add_argument("--live", action="store_true", help="query live market prices (not for backtests)")
    args = ap.parse_args()

    rounds = sorted(p.name[2:12] for p in args.data.glob("q_*.json") if (args.data / f"r_{p.name[2:12]}.json").exists())
    priors = PriorTable()
    print(f"rounds: {rounds}  cache: {series.coverage()}")
    for i, d in enumerate(rounds):
        qs = QuestionSet.load(args.data / f"q_{d}.json")
        res = load_resolutions(args.data / f"r_{d}.json")
        if i >= 1:
            summary = evaluate_round(qs, res, priors, args.shrink_n, args.live)
            ds, mk = summary.get("dataset", {}), summary.get("market", {})
            print(
                f"{d}  dataset n={ds.get('n', 0):4d} brier={ds.get('brier', float('nan')):.4f} idx={ds.get('index', float('nan')):5.1f}"
                f"  | market n={mk.get('n', 0):4d} brier={mk.get('brier', float('nan')):.4f} idx={mk.get('index', float('nan')):5.1f}"
            )
            for src, v in ds.get("by_source", {}).items():
                print(f"          {src:10s} n={v['n']:3d} brier={v['brier']:.4f} methods={v['methods']}")
        by_key = {(q.qid, q.source): q for q in qs.questions}
        for r in res:
            q = by_key.get((str(r["id"]), r["source"]))
            if q is None or q.is_market or r.get("resolved_to") is None:
                continue
            horizon = (date.fromisoformat(r["resolution_date"]) - qs.forecast_due_date).days
            priors.add(q, horizon, float(r["resolved_to"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
