# ForecastBench pipeline

Round procedure (question set drops 00:00 UTC on the due date; upload by 23:59:59 UTC):

```
forecastbench/round.sh 2026-09-27 "Anonymous N" "openforecast v0.1"
# -> out/2026-09-27.Anonymous_N.1.json  (upload to the GCS folder ForecastBench assigns)
```

What the forecasters do (walk-forward numbers from `backtest.py` on the 2026-06-21..08-30 rounds):

| Source | Estimator | Evidence (Brier; 0.5 scores 0.25) |
|---|---|---|
| kalshi, manifold | live public quote on the due date, else the frozen price | 0.034 to 0.059 |
| metaculus, polymarket | frozen price (venue APIs not reachable from the runner) | same |
| acled | Poisson tail on the question's own reference rate, shrunk to the pooled rate | 0.05 to 0.10 |
| wikipedia | pooled rate by horizon and threshold from resolved rounds | 0.12 to 0.19 |
| fred | empirical h-day change distribution of the series, shrunk to the pooled rate | 0.22 to 0.27 |
| yfinance, h <= 30 d | flat 0.5 (all tickers resolve with the same market week) | pooled prior averaged 0.266, worse than 0.25 |
| yfinance, h > 30 d | empirical h-day change distribution (drift matters) | not resolved yet |
| dbnomics temperature, h <= 16 d | Open-Meteo ECMWF ensemble mean difference through a Gaussian of width 4 C | 0.186 on 286 questions using archived deterministic runs |
| dbnomics temperature, longer | same-season, anomaly-matched analogues | not resolved yet |

Dataset-question index (100 x (1 - 2 x Brier)) per round with the live weather estimator off:
60.7, 60.4, 61.8, 61.2, 62.0. The weather estimator adds about 2 points on the 7-day board.
