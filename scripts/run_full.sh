#!/usr/bin/env bash
# Full end-to-end run on the complete TIMIT corpus (6,300 utts).
# Sentinel-driven: extract -> analyze (s1-5,7) -> classify (s6) -> assemble report.
# Resumable: re-running skips finished shards and skips extraction if sentinel present.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA='CONDA_NO_PLUGINS=true C:/ProgramData/anaconda3/Scripts/conda.exe run -n voice_unique python'
run() { echo ">>> $*"; eval $CONDA "$@"; }

export TIMIT_OUTDIR=features
export TIMIT_RESULTS=results
export TIMIT_REPORT=report_TIMIT_v2.md
export TIMIT_NPERM=200
export TIMIT_NBOOT=1000
export TIMIT_NFOLDS=5

echo "=== FULL: env check ==="
run scripts/check_env.py

echo "=== FULL: extraction (sentinel-gated, resumable) ==="
run scripts/extract_features.py

if [ ! -f features/_EXTRACTION_DONE ]; then
  echo "ERROR: extraction sentinel missing; aborting downstream." >&2
  exit 1
fi

echo "=== FULL: self-verify row count / decode failures ==="
run scripts/verify.py

echo "=== FULL: analyze (s1-5,7) ==="
run scripts/analyze.py

echo "=== FULL: classify (s6) ==="
run scripts/classify.py

echo "=== FULL: assemble report ==="
run scripts/assemble_report.py

echo "=== FULL: library versions + wall clock -> run.log ==="
run scripts/check_env.py
echo "=== FULL RUN COMPLETE ==="
