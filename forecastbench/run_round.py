"""Produce a ForecastBench forecast set for one question set.

    python forecastbench/run_round.py --question-set 2026-09-27-llm.json \
        --organization "Anonymous N" --model "openforecast v0.1" --out out/

Every question receives a forecast (market: one; dataset: one per resolution
date), so the 95 % coverage rule cannot be missed by construction. Priors come
from the resolved rounds under ``--history``; series histories come from the
local cache, truncated at the forecast due date.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from core import models_dataset, models_market, series
from core.fb_questions import QuestionSet
from forecastbench.backtest import PriorTable, load_resolutions


def build_priors(history_dir: Path, before: date) -> PriorTable:
    priors = PriorTable()
    for qpath in sorted(history_dir.glob("q_*.json")):
        stamp = qpath.name[2:12]
        rpath = history_dir / f"r_{stamp}.json"
        if not rpath.exists() or date.fromisoformat(stamp) >= before:
            continue
        qs = QuestionSet.load(qpath)
        by_key = {(q.qid, q.source): q for q in qs.questions}
        for r in load_resolutions(rpath):
            q = by_key.get((str(r["id"]), r["source"]))
            if q is None or q.is_market or r.get("resolved_to") is None:
                continue
            horizon = (date.fromisoformat(r["resolution_date"]) - qs.forecast_due_date).days
            priors.add(q, horizon, float(r["resolved_to"]))
    return priors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-set", type=Path, required=True)
    ap.add_argument("--history", type=Path, required=True, help="dir with past q_/r_ files")
    ap.add_argument("--organization", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-organization", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-live", action="store_true", help="skip live market quotes")
    args = ap.parse_args()

    qs = QuestionSet.load(args.question_set)
    priors = build_priors(args.history, qs.forecast_due_date)
    forecasts: list[dict] = []
    methods: dict[str, int] = {}

    for q in qs.questions:
        if q.is_market:
            est = models_market.forecast(q, use_live=not args.no_live)
            forecasts.append(
                {"id": q.qid, "source": q.source, "forecast": round(est.probability, 4),
                 "resolution_date": None, "reasoning": est.method}
            )
            methods[est.method] = methods.get(est.method, 0) + 1
            continue
        hist = series.load(q.source, q.qid)
        for rd in q.resolution_dates:
            horizon = (rd - qs.forecast_due_date).days
            prior = priors.prior(q, horizon)
            est = models_dataset.forecast(q, qs.forecast_due_date, rd, hist, prior, use_weather_forecast=not args.no_live)
            forecasts.append(
                {"id": q.qid, "source": q.source, "forecast": round(est.probability, 4),
                 "resolution_date": rd.isoformat(), "reasoning": f"{est.method} n={est.n}"}
            )
            methods[est.method] = methods.get(est.method, 0) + 1

    payload = {
        "organization": args.organization,
        "model": args.model,
        "model_organization": args.model_organization or args.organization,
        "question_set": qs.name,
        "forecasts": forecasts,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{qs.forecast_due_date.isoformat()}.{args.organization.replace(' ', '_')}.1.json"
    (args.out / name).write_text(json.dumps(payload, indent=1))
    n_market = sum(1 for f in forecasts if f["resolution_date"] is None)
    print(f"wrote {args.out / name}: {len(forecasts)} forecasts ({n_market} market, {len(forecasts) - n_market} dataset rows)")
    print("methods:", dict(sorted(methods.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
