#!/usr/bin/env python
"""
run_mi.py -- ONE-COMMAND runner for the quantization-based, information-theoretic
voice-individuality experiment on Common Voice 17 (English, validated).

Pipeline:  (download shards) -> STEP 1 extract  ->  STEPS 2-6 analyse  ->  STEP 7 report
Fixed seed 1234 throughout.  All outputs land in mi_experiment/.

DATA MODES
----------
MODE A (local Common Voice release dir with validated.tsv + clips/*.mp3):
    python mi_experiment/run_mi.py --mode A --cv_dir /path/to/cv-corpus-17.0-en
MODE B (default; stream a CAPPED subset from the public parquet mirror
    `fixie-ai/common_voice_17_0`, since the official HF repo was emptied Oct-2025):
    python mi_experiment/run_mi.py
The runner re-uses any cached shards in cv_cache/ and any already-computed measured
features in ../features.parquet; pass --fresh to ignore caches.

Usage:
    python mi_experiment/run_mi.py [--skip-extract] [--shards N] [--mode A --cv_dir DIR]
"""
import argparse, os, sys, glob, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
PY = sys.executable
REPO = "fixie-ai/common_voice_17_0"
CACHE = os.path.join(PARENT, "cv_cache", "en")


def ensure_shards(n_shards):
    os.makedirs(CACHE, exist_ok=True)
    have = sorted(glob.glob(f"{CACHE}/validated-*.parquet"))
    if len(have) >= n_shards:
        print(f"[data] {len(have)} shards cached (>= {n_shards} requested)"); return
    # HF_TOKEN is read from the environment automatically by huggingface_hub if set.
    # Do NOT hardcode tokens here -- export HF_TOKEN=... before running if the download needs auth.
    from huggingface_hub import hf_hub_download
    for i in range(n_shards):
        fn = f"en/validated-{i:05d}-of-00138.parquet"
        if os.path.exists(os.path.join(PARENT, "cv_cache", fn)):
            continue
        print(f"[data] downloading {fn} ...", flush=True)
        hf_hub_download(REPO, fn, repo_type="dataset",
                        local_dir=os.path.join(PARENT, "cv_cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="B")
    ap.add_argument("--cv_dir", default=None)
    ap.add_argument("--shards", type=int, default=14,
                    help="MODE B: max shards to make available (extractor stops "
                         "streaming once ~1500 speakers reach 12 clips).")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    if args.mode == "A":
        if not args.cv_dir:
            sys.exit("--cv_dir required for MODE A")
        # MODE A: point the extractor at a local release. The extractor reads parquet
        # shards by default; for a raw CV release wire validated.tsv + clips/ here.
        sys.exit("[MODE A] To run on a local Common Voice release, set CV_LOCAL_DIR and "
                 "adapt mi_extract.build_selection() to read validated.tsv + clips/*.mp3 "
                 "(one-line change: decode mp3 from disk instead of parquet 'audio' bytes). "
                 "MODE B is the validated path for this run.")
    else:
        ensure_shards(args.shards)

    if not (args.skip_extract and os.path.exists(os.path.join(HERE, "features.parquet"))):
        print("\n=== STEP 1: balanced feature extraction ===", flush=True)
        subprocess.run([PY, os.path.join(HERE, "mi_extract.py")], check=True)
    else:
        print("[skip] using existing mi_experiment/features.parquet")

    print("\n=== STEPS 2-6: quantization + MI + usable bits + cumulative + strata ===", flush=True)
    subprocess.run([PY, os.path.join(HERE, "mi_analyze.py")], check=True)

    print("\n=== STEP 7: report ===", flush=True)
    subprocess.run([PY, os.path.join(HERE, "mi_report.py")], check=True)

    print("\n" + "=" * 72 + "\n  report-cv-quant.md\n" + "=" * 72)
    print(open(os.path.join(HERE, "report-cv-quant.md")).read())
    print(f"\n[run_mi] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
