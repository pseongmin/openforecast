"""Parse a ForecastBench question set into the shapes the forecasters need."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MARKET_SOURCES = ("kalshi", "manifold", "metaculus", "polymarket")
DATASET_SOURCES = ("acled", "dbnomics", "fred", "wikipedia", "yfinance")

# ForecastBench phrases each dataset question from a small set of templates. The
# comparison the question asks for decides which model applies, so it is parsed
# once here rather than re-guessed inside every forecaster.
PCT_HIGHER = re.compile(r"at least ([\d.]+)% higher", re.I)
N_TIMES = re.compile(r"more than ([a-z\-]+|[\d.]+) times as many", re.I)
WORD_NUMBERS = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "twenty": 20.0, "fifty": 50.0, "one-hundred": 100.0, "a hundred": 100.0,
}


@dataclass(frozen=True)
class Question:
    qid: str
    source: str
    text: str
    url: str
    freeze_value: float | None
    resolution_dates: tuple[date, ...]
    raw: dict

    @property
    def is_market(self) -> bool:
        return self.source in MARKET_SOURCES

    @property
    def relative_threshold(self) -> float:
        """Multiplicative bar the future value must clear (1.0 = simply higher)."""
        m = PCT_HIGHER.search(self.text)
        if m:
            return 1.0 + float(m.group(1)) / 100.0
        m = N_TIMES.search(self.text)
        if m:
            token = m.group(1).strip().lower()
            return WORD_NUMBERS.get(token, _safe_float(token, 1.0))
        return 1.0

    @property
    def allows_equal(self) -> bool:
        """'as high or higher' resolves yes on a tie; 'higher than' does not."""
        return "as high or higher" in self.text.lower() or "or higher" in self.text.lower()


def _safe_float(token: str, default: float) -> float:
    try:
        return float(token)
    except ValueError:
        return default


@dataclass(frozen=True)
class QuestionSet:
    forecast_due_date: date
    name: str
    questions: tuple[Question, ...]

    @classmethod
    def load(cls, path: str | Path) -> "QuestionSet":
        blob = json.loads(Path(path).read_text())
        due = date.fromisoformat(blob["forecast_due_date"])
        out: list[Question] = []
        for row in blob["questions"]:
            if isinstance(row["id"], list):
                continue  # combination question: excluded from the benchmark since 2025-10
            dates = row.get("resolution_dates")
            parsed = (
                tuple(date.fromisoformat(d) for d in dates)
                if isinstance(dates, list)
                else ()
            )
            out.append(
                Question(
                    qid=str(row["id"]),
                    source=row["source"],
                    text=row["question"],
                    url=row.get("url", ""),
                    freeze_value=_as_float(row.get("freeze_datetime_value")),
                    resolution_dates=parsed,
                    raw=row,
                )
            )
        return cls(forecast_due_date=due, name=blob["question_set"], questions=tuple(out))

    def market(self) -> list[Question]:
        return [q for q in self.questions if q.is_market]

    def dataset(self) -> list[Question]:
        return [q for q in self.questions if not q.is_market]


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
