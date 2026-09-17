"""openforecast bot for the Metaculus FutureEval Fall 2026 tournament.

Design notes
------------
The Spring 2026 bot-maker survey found two things that correlated with score:
using a frontier model, and spending the effort on *research* rather than on
prompt gymnastics. This bot therefore keeps the forecasting prompt short and
spends its budget on (a) several independent research passes and (b) an
explicit reference-class / status-quo decomposition before the number.

Everything here is public: it reads Metaculus questions, calls a model through
OpenRouter (or any provider key present in the environment), and posts a
forecast plus its reasoning comment.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

import dotenv
from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# The forecasting instruction that every question type inherits. Kept in one
# place so a change cannot land on binary questions only.
BASE_RULES = """
Work as a superforecaster would:
1. State the status quo outcome — what happens if nothing changes. Most questions
   resolve to the status quo, because the world changes slowly.
2. Name a reference class and its base rate, with the count behind it.
3. List the evidence that moves you off that base rate, and by how much. Weak
   evidence should move you a little; only decisive evidence moves you a lot.
4. State the time left and what would have to happen in it for the other outcome.
5. Give your final answer in the exact format requested.
Do not assume the question has already resolved. Today's date is given below.
"""


class OpenForecastBot(ForecastBot):
    """Research-heavy bot: several research passes, several forecasts, median."""

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            prompt = clean_indents(
                f"""
                You are a research assistant for a superforecaster. Today is
                {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.

                Gather what a forecaster needs for this question and nothing else:
                - the current state of play, with dates and numbers;
                - the historical base rate of this kind of event, with the sample;
                - the most recent developments, newest first, each dated;
                - the strongest case for YES and the strongest case for NO.

                Do not give a probability. Do not speculate past the evidence.

                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Background: {question.background_info}
                Fine print: {question.fine_print}
                """
            )
            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                research = await SmartSearcher(
                    model=researcher.removeprefix("smart-searcher/"),
                    temperature=0,
                    num_searches_to_run=3,
                    num_sites_per_search=10,
                ).invoke(prompt)
            else:
                research = ""
            logger.info("research for %s: %d chars", question.page_url, len(research))
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            {BASE_RULES}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Background: {question.background_info}

            Research:
            {research}

            Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.

            End with exactly: "Probability: ZZ%" (an integer 1-99).
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsed: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        # Clamp: a 0 or 1 on a live question is never justified and is scored brutally.
        probability = max(0.01, min(0.99, parsed.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=probability, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            {BASE_RULES}

            Question: {question.question_text}
            Options, in order: {question.options}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Background: {question.background_info}

            Research:
            {research}

            Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
            Leave real probability on every option — unexpected outcomes happen.

            End with one line per option, in the given order:
            Option_A: XX%
            Option_B: XX%
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsed: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=(
                "The option names must match this list exactly, in this order: "
                f"{question.options}"
            ),
        )
        return ReasonedPrediction(prediction_value=parsed, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        bounds = self._bound_messages(question)
        prompt = clean_indents(
            f"""
            {BASE_RULES}

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'infer from the question'}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Background: {question.background_info}
            {bounds}

            Research:
            {research}

            Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
            Set wide 10/90 bounds: the world is more surprising than it looks.

            End with exactly these six lines, strictly increasing:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        percentiles = await structure_output(
            reasoning, list[type(None)] if False else list, model=self.get_llm("parser", "llm")
        ) if False else None
        distribution = NumericDistribution.from_question(
            await structure_output(
                reasoning,
                list[__import__("forecasting_tools").Percentile],
                model=self.get_llm("parser", "llm"),
            ),
            question,
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)

    def _bound_messages(self, question: NumericQuestion) -> str:
        lower = (
            f"The outcome cannot be lower than {question.lower_bound}."
            if not question.open_lower_bound
            else f"Values below {question.lower_bound} are possible."
        )
        upper = (
            f"The outcome cannot be higher than {question.upper_bound}."
            if not question.open_upper_bound
            else f"Values above {question.upper_bound} are possible."
        )
        return f"{lower}\n{upper}"


def build_bot(publish: bool) -> OpenForecastBot:
    """Model choice is environment-driven so the same code runs on any credits."""
    model_name = os.getenv("OPENFORECAST_MODEL", "openrouter/anthropic/claude-sonnet-4.5")
    parser_name = os.getenv("OPENFORECAST_PARSER", "openrouter/openai/gpt-4o-mini")
    researcher = os.getenv("OPENFORECAST_RESEARCHER", "asknews/news-summaries")
    return OpenForecastBot(
        research_reports_per_question=int(os.getenv("OPENFORECAST_RESEARCH_PASSES", "2")),
        predictions_per_research_report=int(os.getenv("OPENFORECAST_PREDICTIONS", "3")),
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model=model_name, temperature=0.3, timeout=90, allowed_tries=2),
            "parser": parser_name,
            "researcher": researcher,
            "summarizer": parser_name,
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the openforecast Metaculus bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "minibench", "test", "cup"],
        default="tournament",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not publish to Metaculus")
    args = parser.parse_args()

    bot = build_bot(publish=not args.dry_run)
    client = MetaculusClient()
    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "test": ["bot-testing-area"],
        "cup": [client.CURRENT_METACULUS_CUP_ID],
    }[args.mode]
    if args.mode in ("test", "cup"):
        bot.skip_previously_forecasted_questions = False

    reports = []
    for target in targets:
        reports += asyncio.run(bot.forecast_on_tournament(target, return_exceptions=True))
    bot.log_report_summary(reports)


if __name__ == "__main__":
    main()
