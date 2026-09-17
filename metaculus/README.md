# Metaculus FutureEval bot

`bot.py` forecasts binary, multiple-choice and numeric questions on the current FutureEval
seasonal tournament, MiniBench, the bot-testing-area, or the Metaculus Cup.

```
pip install forecasting-tools python-dotenv
python metaculus/bot.py --mode test --dry-run     # no publishing
python metaculus/bot.py --mode tournament          # publishes
```

Environment: `METACULUS_TOKEN`, `OPENROUTER_API_KEY` (or another provider key that
`forecasting-tools` understands), optional `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET`,
optional `OPENFORECAST_MODEL`, `OPENFORECAST_PARSER`, `OPENFORECAST_RESEARCHER`,
`OPENFORECAST_RESEARCH_PASSES`, `OPENFORECAST_PREDICTIONS`.

The bot never assumes a question has resolved, clamps binary answers to [1 %, 99 %], and
posts its reasoning as the forecast comment so the run is auditable on the question page.
