#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODE="${1:-smoke}"
export PYTHONUTF8=1

usage() {
  cat <<'EOF'
Usage: bash quick_test.sh [smoke|cached]

Modes:
  smoke   Fast checks only: compile, import, cheap CLIs, regenerate reports from existing artifacts.
  cached  smoke + rerun analysis steps only when the needed cached parquet files already exist.

Examples:
  bash quick_test.sh
  PYTHON=C:/Users/shuoo/.conda/envs/voice_unique/python.exe bash quick_test.sh
  PYTHON=C:/Users/shuoo/.conda/envs/voice_unique/python.exe bash quick_test.sh cached
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_quiet() {
  log "$*"
  "$@" >/dev/null
}

have_file() {
  [[ -f "$ROOT/$1" ]]
}

case "$MODE" in
  smoke|cached) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

log "repo root: $ROOT"
"$PYTHON" -c "import sys; print('[env] python =', sys.executable)"

run_quiet "$PYTHON" -c "import numpy, pandas, scipy, sklearn, pyarrow, librosa, soundfile, matplotlib; print('deps ok')"

log "compile all Python files"
"$PYTHON" -m compileall -q \
  "$ROOT/collision_experiment" \
  "$ROOT/mi_experiment" \
  "$ROOT/jointbits_experiment"

log "import all repo modules"
ROOT="$ROOT" "$PYTHON" - <<'PY'
import importlib
import os
import sys

root = os.environ["ROOT"]
for rel in ("collision_experiment", "mi_experiment", "jointbits_experiment"):
    sys.path.insert(0, os.path.join(root, rel))

modules = [
    "collision", "features", "extract_stage", "analyze", "run_experiment",
    "mi_core", "mi_features", "mi_extract", "mi_analyze", "mi_report", "run_mi",
    "jb_core", "jb_extract_timit", "jb_matched_robust", "jb_report", "jb_run", "run_jointbits",
]

for name in modules:
    importlib.import_module(name)
    print(f"[import] ok {name}")
PY

run_quiet "$PYTHON" "$ROOT/collision_experiment/collision.py"
run_quiet "$PYTHON" "$ROOT/collision_experiment/run_experiment.py" --help
run_quiet "$PYTHON" "$ROOT/mi_experiment/run_mi.py" --help

if have_file "mi_experiment/artifacts/analysis_summary.json"; then
  run_quiet "$PYTHON" "$ROOT/mi_experiment/mi_report.py"
else
  log "skip mi_report.py (missing mi_experiment/artifacts/analysis_summary.json)"
fi

if have_file "jointbits_experiment/results.json"; then
  run_quiet "$PYTHON" "$ROOT/jointbits_experiment/jb_report.py"
else
  log "skip jb_report.py (missing jointbits_experiment/results.json)"
fi

if [[ "$MODE" == "cached" ]]; then
  if have_file "collision_experiment/features.parquet"; then
    run_quiet "$PYTHON" "$ROOT/collision_experiment/analyze.py"
  else
    log "skip collision analyze (missing collision_experiment/features.parquet)"
  fi

  if have_file "mi_experiment/features.parquet"; then
    run_quiet "$PYTHON" "$ROOT/mi_experiment/mi_analyze.py"
  else
    log "skip mi_analyze.py (missing mi_experiment/features.parquet)"
  fi

  if have_file "mi_experiment/features.parquet" && have_file "jointbits_experiment/timit_features.parquet"; then
    run_quiet "$PYTHON" "$ROOT/jointbits_experiment/jb_run.py"
  else
    log "skip jb_run.py (missing cached parquet inputs)"
  fi
fi

log "quick test finished: $MODE"