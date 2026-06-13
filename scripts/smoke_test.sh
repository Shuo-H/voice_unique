#!/usr/bin/env bash
# Quick end-to-end smoke test on a SMALL subset (2 M + 2 F speakers per DR, TRAIN only ~= 32 spk / 320 utts).
# Verifies extraction -> analyze -> classify -> sentinel pipeline runs without crashing and emits sane outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA='CONDA_NO_PLUGINS=true C:/ProgramData/anaconda3/Scripts/conda.exe run -n voice_unique python'
run() { echo ">>> $*"; eval $CONDA "$@"; }

export TIMIT_SMOKE_N=2
export TIMIT_OUTDIR=features_smoke
export TIMIT_RESULTS=results_smoke
export TIMIT_NPERM=50      # reduced permutation null for speed during smoke
export TIMIT_NBOOT=200     # reduced bootstrap for speed during smoke
export TIMIT_NFOLDS=5

echo "=== SMOKE: env check ==="
run scripts/check_env.py

echo "=== SMOKE: extraction (subset) ==="
run scripts/extract_features.py

echo "=== SMOKE: analyze ==="
run scripts/analyze.py

echo "=== SMOKE: classify ==="
run scripts/classify.py

echo "=== SMOKE: outputs ==="
ls -la "$TIMIT_OUTDIR" "$TIMIT_RESULTS"
echo "--- coverage head ---"; head -20 "$TIMIT_RESULTS/coverage.csv"
echo "--- analyze_summary ---"; cat "$TIMIT_RESULTS/analyze_summary.json"
echo "=== SMOKE COMPLETE ==="
