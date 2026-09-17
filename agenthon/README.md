# Agenthon 2026 — Track 2 forecasting agent

`qfagent/` implements the `forecast` verb the harness runs:

```
forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet
```

Engine: stationary block bootstrap over the panel's own history, sampled jointly across assets
(cross-asset dependence survives), plus a conservative scenario mixture driven by the dated text
corpus (direction only when lopsided, tails widened on shock vocabulary, never narrowed). Targets
absent from the panel (transfer cards) are proxied by the panel-average step series and flagged in
the rationale.

Local check (offline, the official scorer's gates g0-g3):

```
docker build --platform linux/amd64 -t openforecast-t2:dev agenthon
docker run --rm --network=none -v <unit>:/input:ro -v <out>:/output openforecast-t2:dev \
  forecast --panels /input/ --text /input/text/ --asof <data_cutoff> --out /output/forecast.parquet
python scoring/scoring.py score --card <unit>/card.toml --forecast <out>/forecast.parquet   # in the track repo, Python 3.13
```

Measured 2026-09-17: 103/103 public practice units admissible.

Submission (after the image is pushed to a public registry):

1. `docker push ghcr.io/<user>/openforecast-t2:v0.1` and copy the digest into `submission.template.json`
   (`image.digest`), save as `submission.json`.
2. `qfbench2 submission pack --team-number <N> --team-key-file <mode-600 file> --descriptor submission.json --out submission.zip`
   (the toolkit derives `team_id`, seals `descriptor_digest`, and writes `team-claim.json`).
3. Upload `submission.zip` on the track's CodaBench page (Development: 5 per day, 20 total).

Category is `api` with `models: []` — the forecaster is deterministic and calls no model.
