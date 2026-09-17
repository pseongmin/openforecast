"""Walk-forward comparison for the wikipedia questions: current model vs template model."""
from __future__ import annotations

import glob
import json
import os
import statistics
from datetime import date
from pathlib import Path

from core import models_dataset, models_wikipedia
from core.fb_questions import QuestionSet
from forecastbench.backtest import PriorTable

WORKSPACE = Path(__file__).resolve().parents[3]  # repo/forecastbench/experiments/<file>
HISTORY = Path(os.environ.get("FB_HISTORY", WORKSPACE / "data" / "forecastbench_history"))
PUBLIC = Path(os.environ.get("FB_PUBLIC", WORKSPACE / "data" / "fb_public" / "forecastbench-processed-forecast-sets"))
ROUNDS = ["2026-06-21", "2026-07-05", "2026-07-19", "2026-08-02", "2026-08-16", "2026-08-30"]


def brier(rows: list[tuple[float, float]]) -> float:
    return statistics.mean((p - y) ** 2 for p, y in rows) if rows else float("nan")


def competitor_scores(round_name: str) -> list[tuple[float, str]]:
    out = []
    for path in glob.glob(str(PUBLIC / round_name / "*external*.json")):
        rows = json.load(open(path))["forecasts"]
        wiki = [(float(r["forecast"]), float(r["resolved_to"])) for r in rows
                if r["source"] == "wikipedia" and r.get("resolved") and r.get("resolved_to") is not None
                and not r.get("imputed")]
        if len(wiki) >= 40:
            out.append((brier(wiki), os.path.basename(path)))
    return sorted(out)[:3]


def main() -> int:
    priors, wiki_priors = PriorTable(), models_wikipedia.WikipediaPriors()
    pooled_before: list[tuple[float, float]] = []
    pooled_after: list[tuple[float, float]] = []
    for i, name in enumerate(ROUNDS):
        qs = QuestionSet.load(HISTORY / f"q_{name}.json")
        resolutions = json.load(open(HISTORY / f"r_{name}.json"))["resolutions"]
        by_key = {(q.qid, q.source): q for q in qs.questions}
        if i >= 1:
            before, after = [], []
            for r in resolutions:
                q = by_key.get((str(r["id"]), r["source"]))
                if q is None or q.source != "wikipedia" or r.get("resolved_to") is None:
                    continue
                rd = date.fromisoformat(r["resolution_date"])
                horizon = (rd - qs.forecast_due_date).days
                prior = priors.prior(q, horizon)
                y = float(r["resolved_to"])
                before.append((models_dataset.forecast(q, qs.forecast_due_date, rd, None, prior).probability, y))
                after.append((models_wikipedia.forecast(q, qs.forecast_due_date, rd, prior, wiki_priors).probability, y))
            pooled_before += before
            pooled_after += after
            top = competitor_scores(name)
            top_text = "  ".join(f"{b:.4f}" for b, _ in top)
            print(f"{name}  n={len(before):3d}  current={brier(before):.4f}  template={brier(after):.4f}  top3=[{top_text}]")
        for r in resolutions:
            q = by_key.get((str(r["id"]), r["source"]))
            if q is None or q.is_market or r.get("resolved_to") is None:
                continue
            rd = date.fromisoformat(r["resolution_date"])
            priors.add(q, (rd - qs.forecast_due_date).days, float(r["resolved_to"]))
            if q.source == "wikipedia":
                wiki_priors.add(q, qs.forecast_due_date, rd, float(r["resolved_to"]))
    print(f"POOLED n={len(pooled_before)}  current={brier(pooled_before):.4f}  template={brier(pooled_after):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
