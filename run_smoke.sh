#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SMOKE TEST for the TIMIT 40-feature battery.
# Runs the SAME pipeline scripts (check_env / extract_all / analyze / classify)
# UNCHANGED, but against a small balanced speaker subset, inside an isolated
# ./smoke/ working dir. It never writes into ./results, ./features, or the
# read-only corpus. Goal: confirm every stage runs end-to-end in ~2-3 min.
#
# Usage:
#   bash run_smoke.sh              # 8 male + 8 female speakers (160 utts)
#   SMOKE_N_PER_SEX=4 bash run_smoke.sh   # smaller/faster (8 speakers, 80 utts)
#
# NOTE: numbers produced here are from a TINY subset and are NOT the study
# results -- this only verifies the code is runnable. The real run uses the
# full corpus via build_manifest.py + extract_all.py + analyze.py + classify.py.
# ---------------------------------------------------------------------------
set -euo pipefail

CONDA="C:/ProgramData/anaconda3/Scripts/conda.exe"
ENVN="voice_unique"
RUN() { CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENVN" python "$@"; }

BATTERY="$(cd "$(dirname "$0")" && pwd)"
SMOKE="$BATTERY/smoke"
export SMOKE_N_PER_SEX="${SMOKE_N_PER_SEX:-8}"

echo "==================================================================="
echo " TIMIT battery SMOKE TEST   (N_PER_SEX=$SMOKE_N_PER_SEX, isolated in ./smoke)"
echo "==================================================================="

# --- stage 0: clean isolated workspace + copy the real scripts in unchanged ---
echo; echo ">>> [0] preparing isolated ./smoke workspace"
rm -rf "$SMOKE"
mkdir -p "$SMOKE/results" "$SMOKE/features"
cp "$BATTERY"/feat_lib.py "$BATTERY"/check_env.py "$BATTERY"/extract_all.py \
   "$BATTERY"/analyze.py "$BATTERY"/classify.py "$BATTERY"/smoke_subset.py "$SMOKE"/
cd "$SMOKE"

# --- stage 1: environment + package check (must print RESULT: ALL_PRESENT) ---
echo; echo ">>> [1] env / package check"
RUN check_env.py

# --- stage 2: build subset manifest + speaker_meta ---
echo; echo ">>> [2] build subset manifest (walks corpus, read-only)"
RUN smoke_subset.py

# --- stage 3: feature extraction (sharded -> features_per_utt.parquet) ---
echo; echo ">>> [3] feature extraction"
RUN extract_all.py
test -f features/_EXTRACTION_DONE && echo "    sentinel OK: $(cat features/_EXTRACTION_DONE)"
echo "    extract_time:"; cat results/extract_time.txt | sed 's/^/      /'

# --- stage 4: analysis sections 1-5,7 ---
echo; echo ">>> [4] analyze.py (sections 1-5,7)"
RUN analyze.py

# --- stage 5: classifier section 6 ---
echo; echo ">>> [5] classify.py (section 6)"
RUN classify.py 2>/dev/null   # hide sklearn FutureWarnings; remove 2>/dev/null to see them

# --- stage 6: show the machine-readable artifacts that were produced ---
echo; echo ">>> [6] artifacts written under ./smoke/results :"
ls -1 results/*.csv results/*.json results/*.txt 2>/dev/null | sed 's/^/      /'
echo
echo "==================================================================="
echo " SMOKE TEST PASSED -- every stage ran to completion."
echo " (Report assembly is done by hand from results/ in the full run;"
echo "  there is no report script to exercise here.)"
echo "==================================================================="
