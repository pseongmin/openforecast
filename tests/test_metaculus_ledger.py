"""Ledger math for the zero-cost Metaculus profile (needs forecasting-tools)."""
import importlib

import pytest

pytest.importorskip("forecasting_tools")


def test_free_ledger_budget_math(tmp_path, monkeypatch):

    monkeypatch.setenv("OPENFORECAST_LEDGER", str(tmp_path / "ledger.json"))
    monkeypatch.setenv("METACULUS_TOKEN", "x")
    bot = importlib.import_module("metaculus.bot")
    importlib.reload(bot)
    today, used = bot._ledger_today()
    assert used == 0
    bot._ledger_add(9)
    _, used = bot._ledger_today()
    assert used == 9
    assert (bot.FREE_DAILY_REQUESTS - used) // bot.FREE_REQUESTS_PER_QUESTION == (45 - 9) // 3
