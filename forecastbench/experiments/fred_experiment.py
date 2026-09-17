"""Walk-forward comparison for the fred questions: current model vs regime-split model."""
from __future__ import annotations

import glob
import json
import os
import statistics
from datetime import date
from pathlib import Path

from core import models_dataset, models_fred, series
from core.fb_questions import QuestionSet
from forecastbench.backtest import PriorTable

WORKSPACE = Path(__file__).resolve().parents[3]  # repo/forecastbench/experiments/<file>
HISTORY = Path(os.environ.get("FB_HISTORY", WORKSPACE / "data" / "forecastbench_history"))
PUBLIC = Path(os.environ.get("FB_PUBLIC", WORKSPACE / "data" / "fb_public" / "forecastbench-processed-forecast-sets"))
ROUNDS = ["2026-06-21", "2026-07-05", "2026-07-19", "2026-08-02", "2026-08-16", "2026-08-30"]


def brier(rows):
    return statistics.mean((p - y) ** 2 for p, y in rows) if rows else float("nan")


def competitor_scores(round_name: str, source: str, min_rows: int = 40):
    out = []
    for path in glob.glob(str(PUBLIC / round_name / "*external*.json")):
        rows = json.load(open(path))["forecasts"]
        sub = [(float(r["forecast"]), float(r["resolved_to"])) for r in rows
               if r["source"] == source and r.get("resolved") and r.get("resolved_to") is not None
               and not r.get("imputed")]
        if len(sub) >= min_rows:
            out.append((brier(sub), os.path.basename(path)))
    return sorted(out)[:3]


def main() -> int:
    priors = PriorTable()
    pooled_before, pooled_after = [], []
    for i, name in enumerate(ROUNDS):
        qs = QuestionSet.load(HISTORY / f"q_{name}.json")
        resolutions = json.load(open(HISTORY / f"r_{name}.json"))["resolutions"]
        by_key = {(q.qid, q.source): q for q in qs.questions}
        if i >= 1:
            before, after = [], []
            for r in resolutions:
                q = by_key.get((str(r["id"]), r["source"]))
                if q is None or q.source != "fred" or r.get("resolved_to") is None:
                    continue
                rd = date.fromisoformat(r["resolution_date"])
                prior = priors.prior(q, (rd - qs.forecast_due_date).days)
                hist = series.load("fred", q.qid)
                y = float(r["resolved_to"])
                before.append((models_dataset.forecast(q, qs.forecast_due_date, rd, hist, prior).probability, y))
                after.append((models_fred.forecast(q, qs.forecast_due_date, rd, hist, prior).probability, y))
            pooled_before += before
            pooled_after += after
            top = "  ".join(f"{b:.4f}" for b, _ in competitor_scores(name, "fred"))
            print(f"{name}  n={len(before):3d}  current={brier(before):.4f}  regime={brier(after):.4f}  top3=[{top}]")
        for r in resolutions:
            q = by_key.get((str(r["id"]), r["source"]))
            if q is None or q.is_market or r.get("resolved_to") is None:
                continue
            priors.add(q, (date.fromisoformat(r["resolution_date"]) - qs.forecast_due_date).days, float(r["resolved_to"]))
    print(f"POOLED n={len(pooled_before)}  current={brier(pooled_before):.4f}  regime={brier(pooled_after):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
