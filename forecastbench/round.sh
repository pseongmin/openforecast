#!/bin/bash
# One ForecastBench round, end to end:
#   1. refresh the resolution sets of past rounds (priors),
#   2. download the question set for DUE (default: today, UTC),
#   3. write the forecast set to OUT.
# Usage: forecastbench/round.sh [DUE=YYYY-MM-DD] [ORG="Anonymous N"] [MODEL="openforecast v0.1"]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/../.venv/bin/python}"
HIST="${HIST:-$REPO/../data/forecastbench_history}"
OUT="${OUT:-$REPO/out}"
DUE="${1:-$(date -u +%F)}"
ORG="${2:-openforecast-dryrun}"
MODEL="${3:-openforecast v0.1}"
RAW="https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
mkdir -p "$HIST" "$OUT"

for q in "$HIST"/q_*.json; do
  d=$(basename "$q" .json); d=${d#q_}
  curl -sfL "$RAW/resolution_sets/${d}_resolution_set.json" -o "$HIST/r_${d}.json.tmp" && mv "$HIST/r_${d}.json.tmp" "$HIST/r_${d}.json"
done

QS="$HIST/q_${DUE}.json"
for attempt in $(seq 1 30); do
  curl -sfL "$RAW/question_sets/${DUE}-llm.json" -o "$QS.tmp" && mv "$QS.tmp" "$QS" && break
  echo "question set ${DUE} not published yet (attempt $attempt); waiting 60 s"; sleep 60
done
[ -f "$QS" ] || { echo "no question set for ${DUE}"; exit 2; }

cd "$REPO" && PYTHONPATH="$REPO" "$PY" forecastbench/run_round.py --question-set "$QS" --history "$HIST" \
  --organization "$ORG" --model "$MODEL" --out "$OUT"
