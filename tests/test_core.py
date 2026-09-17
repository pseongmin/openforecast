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


def test_weather_probability_is_tempered_and_bounded():
    from datetime import date as _date

    from core.models_weather import probability_warmer

    members = {_date(2026, 9, 27): [18.0] * 50, _date(2026, 10, 4): [22.0] * 50}
    r = probability_warmer(members, _date(2026, 9, 27), _date(2026, 10, 4))
    assert r is not None and 0.75 < r.probability < 0.9  # +4 C at sigma 4 -> ~0.84
    assert probability_warmer(members, _date(2026, 9, 27), _date(2026, 10, 20)) is None


def test_wikipedia_templates_and_release_rule():
    from datetime import date as _date

    from core import models_wikipedia as W

    vaccine = _q("According to Wikipedia, will a vaccine have been developed for X by {resolution_date}?", "wikipedia")
    rank = _q("According to Wikipedia, will A B have a FIDE ranking on {resolution_date} as high or higher than their ranking on {forecast_due_date}?", "wikipedia")
    elo = _q("According to Wikipedia, will A B have an Elo rating on {resolution_date} that's at least 1% higher than on {forecast_due_date}?", "wikipedia")
    assert W.classify(vaccine) == "vaccine"
    assert W.classify(rank) == "as_high_or_higher"
    assert W.classify(elo) == "pct_higher"
    # A window with no month boundary cannot see a new FIDE list.
    assert W.monthly_releases_in(_date(2026, 6, 21), _date(2026, 6, 28)) == 0
    assert W.monthly_releases_in(_date(2026, 8, 30), _date(2026, 9, 6)) == 1
    assert W.forecast(rank, _date(2026, 6, 21), _date(2026, 6, 28), 0.5).probability > 0.9
    assert W.forecast(elo, _date(2026, 6, 21), _date(2026, 6, 28), 0.5).probability < 0.1
    assert W.forecast(vaccine, _date(2026, 8, 30), _date(2026, 9, 6), 0.5).probability < 0.05


def test_wikipedia_release_window_never_inherits_the_certain_rate():
    from datetime import date as _date

    from core import models_wikipedia as W

    rank = _q("will A B have a FIDE ranking on {resolution_date} as high or higher than their ranking on {forecast_due_date}?", "wikipedia")
    priors = W.WikipediaPriors()
    for _ in range(40):  # a history of no-release windows, all yes
        priors.add(rank, _date(2026, 6, 21), _date(2026, 6, 28), 1.0)
    p = W.forecast(rank, _date(2026, 8, 30), _date(2026, 9, 6), 0.5, priors).probability
    assert p < 0.9, p


def test_acled_blend_weight_is_the_measured_one():
    from core import models_dataset as M

    est = M.Estimate(0.46, "poisson", 30)
    blended = M.shrink(est, prior=0.15, shrink_n=M.ACLED_SHRINK_N).probability
    assert abs(blended - (0.6 * 0.46 + 0.4 * 0.15)) < 1e-9


def test_fred_flat_series_is_structural_no():
    from datetime import date as _date

    import numpy as np
    import pandas as pd

    from core import models_fred

    idx = pd.date_range("2026-01-01", "2026-09-01", freq="D")
    flat = pd.Series(np.full(len(idx), 4.25), index=idx)
    q = _q("Will the value have increased by {resolution_date}?", "fred")
    est = models_fred.forecast(q, _date(2026, 9, 1), _date(2026, 9, 8), flat, prior=0.5)
    assert est.probability < 0.05 and "administered" in est.method
    moving = pd.Series(np.linspace(4.0, 5.0, len(idx)), index=idx)
    est2 = models_fred.forecast(q, _date(2026, 9, 1), _date(2026, 9, 8), moving, prior=0.5)
    assert est2.probability > 0.5
