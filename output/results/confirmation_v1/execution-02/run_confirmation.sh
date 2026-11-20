#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$(realpath -- "$0")")/../../../.."
base="$PWD/output/results/confirmation_v1"
execution="$base/execution-02"
run="$base/run-01"
analysis="$base/analysis-01"
trap 'result=$?; printf "%s %s\n" "$(date -Is)" "$result" > "$execution/exit.txt"; exit "$result"' EXIT
python3 -B "$execution/resume.py" > "$execution/queue.log" 2>&1
python3 -B -m src.py.experiments.formal_v1 confirmation analyze --run "$run" --output "$analysis" > "$execution/analysis.log" 2>&1
python3 -B "$execution/verify_confirmation.py" "$analysis" > "$execution/verification.json.tmp"
mv -- "$execution/verification.json.tmp" "$execution/verification.json"
