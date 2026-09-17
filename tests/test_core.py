import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import models_dataset
from core.fb_questions import Question, QuestionSet
from tools.leak_guard import scan


def _q(text: str, source: str = "yfinance", fv=1.0) -> Question:
    return Question(qid="X", source=source, text=text, url="", freeze_value=fv, resolution_dates=(), raw={})


def test_relative_threshold_parses_percent_and_multiples():
    assert _q("Will X have an Elo rating that's at least 1% higher than today?").relative_threshold == pytest.approx(1.01)
    assert _q("Will there be more than ten times as many 'Protests'").relative_threshold == 10.0
    assert _q("Will X's close be higher than today's?").relative_threshold == 1.0


def test_allows_equal_only_on_or_higher_wording():
    assert _q("ranking as high or higher than").allows_equal is True
    assert _q("close price higher than").allows_equal is False


def test_question_set_drops_combination_questions(tmp_path: Path):
    blob = {
        "forecast_due_date": "2026-09-27",
        "question_set": "2026-09-27-llm.json",
        "questions": [
            {"id": "AAPL", "source": "yfinance", "question": "q", "resolution_dates": ["2026-10-04"]},
            {"id": ["a", "b"], "source": "yfinance", "question": "combo", "resolution_dates": "N/A"},
            {"id": "123", "source": "metaculus", "question": "m", "resolution_dates": "N/A"},
        ],
    }
    p = tmp_path / "q.json"
    p.write_text(json.dumps(blob))
    qs = QuestionSet.load(p)
    assert [q.qid for q in qs.questions] == ["AAPL", "123"]
    assert qs.dataset()[0].resolution_dates == (date(2026, 10, 4),)
    assert qs.market()[0].is_market


def test_poisson_probability_is_monotone_in_threshold():
    p1 = models_dataset.poisson_count_probability(5.0, 1.0).probability
    p10 = models_dataset.poisson_count_probability(5.0, 10.0).probability
    assert 0 <= p10 < p1 < 1
    assert models_dataset.poisson_count_probability(0.0, 1.0) is None


def test_shrink_moves_toward_prior_with_small_n():
    emp = models_dataset.Estimate(0.9, "empirical", 4)
    small = models_dataset.shrink(emp, prior=0.1, shrink_n=40).probability
    big = models_dataset.shrink(models_dataset.Estimate(0.9, "empirical", 4000), 0.1, 40).probability
    assert 0.1 < small < 0.3
    assert big > 0.85


def test_horizon_change_uses_calendar_days_and_cutoff():
    idx = pd.date_range("2015-01-01", "2026-09-01", freq="D")
    rng = np.random.default_rng(1)
    s = pd.Series(np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(idx)))), index=idx)
    est = models_dataset.horizon_change_probability(s, pd.Timestamp("2026-09-01"), 7, 1.0, False)
    assert est is not None and 0.4 < est.probability < 0.7
    assert models_dataset.horizon_change_probability(s, pd.Timestamp("2014-01-01"), 7, 1.0, False) is None


def test_leak_guard_catches_planted_marker(tmp_path: Path):
    bad = tmp_path / "bad.md"
    # Build the markers at runtime so this file never carries them literally.
    bad.write_text("see " + "/home/" + "someone/secret and host " + "ksad" + "lq001")
    assert len(scan([str(bad)])) == 2
