#!/usr/bin/env bash
# quick_test.sh -- end-to-end smoke test of the cv17_v2 battery on a tiny subset.
#
# Downloads ONE Common Voice shard, carves out a small speaker subset, then runs
# the full pipeline (extract -> step3 -> step4 -> step5 -> step6 -> report) in an
# ISOLATED temp dir so the committed result files are never touched.  This proves
# every step runs and that `features.parquet` is regenerable from the script.
#
# The numbers it produces are NOT meaningful (subset is far too small) -- it is a
# plumbing/regression check only.
#
# Usage:
#   bash quick_test.sh
#   PYTHON=/c/Users/you/.conda/envs/voice_unique/python.exe bash quick_test.sh
#   KEEP=1 bash quick_test.sh        # keep the temp run dir for inspection
#
# Env knobs (all optional):
#   PYTHON         python interpreter with the DSP deps (default: python)
#   N_SPEAKERS     subset speaker count, all with >=10 clips (default: 110)
#   CAP            clips per speaker in the subset            (default: 15)
#   KEEP=1         do not delete the temp run dir
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
N_SPEAKERS="${N_SPEAKERS:-110}"
CAP="${CAP:-15}"
SHARD="en/validated-00000-of-00138.parquet"
RAW_CACHE="$HERE/../cv_cache"          # where the real 493 MB shard is cached
export PYTHONUTF8=1

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

# Isolated workspace: copy every .py here so all relative reads/writes + the
# `import features/common/mi_core/jb_core/collision` stay self-contained.
RUN="$(mktemp -d)"
SUBCACHE="$RUN/cache/en"
cleanup() { [[ "${KEEP:-0}" == "1" ]] && { log "kept temp dir: $RUN"; return; }; rm -rf "$RUN"; }
trap cleanup EXIT

log "python      : $("$PYTHON" -c 'import sys; print(sys.executable)')"
log "temp run dir: $RUN"
"$PYTHON" -c "import numpy,pandas,scipy,sklearn,pyarrow,librosa,soundfile" || die "missing deps (use the voice_unique env)"

# 0) make sure the raw shard is present (downloads ~493 MB once)
if [[ ! -f "$RAW_CACHE/$SHARD" ]]; then
  log "downloading $SHARD (~493 MB, one time) ..."
  "$PYTHON" - "$RAW_CACHE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("fixie-ai/common_voice_17_0", "en/validated-00000-of-00138.parquet",
                repo_type="dataset", local_dir=sys.argv[1])
PY
else
  log "raw shard already cached: $RAW_CACHE/$SHARD"
fi

# 1) carve a subset: N_SPEAKERS speakers with >=10 clips, capped at CAP clips each
log "building subset cache: $N_SPEAKERS speakers x <=$CAP clips"
mkdir -p "$SUBCACHE"
"$PYTHON" - "$RAW_CACHE/$SHARD" "$SUBCACHE/$( basename "$SHARD" )" "$N_SPEAKERS" "$CAP" <<'PY'
import sys, collections, pyarrow as pa, pyarrow.parquet as pq
src, dst, n_spk, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t = pq.read_table(src)
cid = t.column("client_id").to_pylist()
counts = collections.Counter(cid)
keep = [k for k, v in counts.items() if v >= 10][:n_spk]
keepset = set(keep)
per = collections.Counter()
rows = []
for i, c in enumerate(cid):
    if c in keepset and per[c] < cap:
        rows.append(i); per[c] += 1
pq.write_table(t.take(rows), dst)
print(f"[subset] {len(keep)} speakers, {len(rows)} clips -> {dst}")
PY

# 2) copy the pipeline into the isolated dir and run every step there
cp "$HERE"/*.py "$RUN"/
cd "$RUN"
export CV_CACHE="$SUBCACHE"

declare -a STEPS=(
  "extract_v2.py|features.parquet|extract"
  "step3_fratios.py|fratios.csv|step3 F-ratios"
  "step4_mi.py|usable_bits.csv|step4 MI"
  "step5_pr.py|pr_effective_dim.csv|step5 PR"
  "step6_classifier.py|classifiers.csv|step6 classifier"
  "report_v2.py|report.md|step7 report"
)
for s in "${STEPS[@]}"; do
  IFS='|' read -r script out label <<< "$s"
  log "running $label ($script)"
  "$PYTHON" "$script" >"$RUN/$script.log" 2>&1 \
    || { tail -20 "$RUN/$script.log" >&2; die "$label crashed -- see $RUN/$script.log"; }
  [[ -s "$RUN/$out" ]] || die "$label produced no $out"
  printf '   [ok] %-16s -> %s\n' "$label" "$out"
done

log "ALL STEPS PASSED  (subset: $(wc -c <"$RUN/features.parquet") bytes features.parquet)"
echo "  report head:"; sed -n '1,3p' "$RUN/report.md" | sed 's/^/    /'
