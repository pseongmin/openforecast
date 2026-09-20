"""Prove the submission path works without spending a model call.

Posts a deliberate 0.5 forecast plus an explanatory comment on one binary question
in the bot-testing-area tournament, then reads the forecast back from the API. The
testing area exists for exactly this and feeds no leaderboard.

    python metaculus/plumbing_check.py            # dry run: fetch and report only
    python metaculus/plumbing_check.py --publish  # actually post, then verify
"""
from __future__ import annotations

import argparse
import os

import dotenv
import requests
from forecasting_tools import BinaryQuestion, MetaculusApi

dotenv.load_dotenv()
API = "https://www.metaculus.com/api"
TOURNAMENT = "bot-testing-area"
NOTE = (
    "Plumbing check from the openforecast bot: this is a deliberate 0.5 with no "
    "research behind it, posted to confirm the token, the question fetch and the "
    "submission path work end to end. https://github.com/pseongmin/openforecast"
)


def _headers() -> dict:
    token = os.environ["METACULUS_TOKEN"]
    return {"Authorization": f"Token {token}", "User-Agent": "openforecast/0.1"}


def read_back(question_id: int) -> dict | None:
    """Read the bot's own latest forecast on a question straight from the API."""
    r = requests.get(f"{API}/posts/{question_id}/", headers=_headers(), timeout=30)
    if r.status_code != 200:
        return None
    question = (r.json() or {}).get("question") or {}
    forecasts = (question.get("my_forecasts") or {}).get("latest")
    return forecasts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    questions = MetaculusApi.get_all_open_questions_from_tournament(TOURNAMENT)
    binaries = [q for q in questions if isinstance(q, BinaryQuestion)]
    print(f"open questions in {TOURNAMENT}: {len(questions)} ({len(binaries)} binary)")
    if not binaries:
        print("no binary question open in the testing area; nothing to check")
        return 1
    target = binaries[0]
    print(f"target: {target.page_url}\n  {target.question_text[:90]}")
    if not args.publish:
        print("dry run: nothing posted (pass --publish to post)")
        return 0

    MetaculusApi.post_binary_question_prediction(target.id_of_question, 0.5)
    MetaculusApi.post_question_comment(target.id_of_post, NOTE)
    latest = read_back(target.id_of_post)
    print(f"read back: {latest}")
    if latest:
        print("PASS: the forecast is stored on Metaculus")
        return 0
    print("posted, but the read-back returned nothing — check the bot profile by hand")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
