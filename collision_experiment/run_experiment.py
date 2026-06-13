#!/usr/bin/env python
"""
run_experiment.py -- one-command runner for the "Human Voice is Unique"
empirical replication on Mozilla Common Voice 17 (English, validated).

Pipeline:  download/cache shards  ->  STEP 1 extract  ->  STEPS 2-6 analyse  ->  print report.md
Fixed seed 1234 (set in extract_stage.py / analyze.py).

DATA MODES
----------
MODE A (local Common Voice release dir):
    python run_experiment.py --mode A --cv_dir /path/to/cv-corpus-17.0-en
    Expects validated.tsv (client_id, path, sentence, age, gender, accent, locale)
    and clips/*.mp3.

MODE B (download a capped subset from Hugging Face):
    python run_experiment.py --mode B --shards 4
    The official `mozilla-foundation/common_voice_17_0` repo was EMPTIED in Oct 2025
    (data moved to the Mozilla Data Collective, account+terms required) and
    `datasets>=5.0` dropped script loaders, so the official MODE-B path is blocked.
    We therefore pull the identical CV 17.0 English `validated` data from the public,
    non-gated parquet mirror `fixie-ai/common_voice_17_0` (full official schema incl.
    client_id/age/gender/accent). To use the official source instead: create a Mozilla
    Data Collective account, accept terms, download the English validated tarball, and
    run MODE A on the extracted dir.

Default (no args): MODE B with 4 shards (the configuration used for report.md).
"""
import argparse, os, sys, glob, subprocess

# Run from this script's own folder; raw shards live in the shared ../cv_cache.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPO = "fixie-ai/common_voice_17_0"
CACHE = "../cv_cache/en"


def _sanitize_ssl_cert_env():
    """Drop stale certificate env vars that make httpx fail before any request."""
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(name)
        if value and not os.path.exists(value):
            print(f"[env] ignoring {name}={value!r} because the file does not exist",
                  file=sys.stderr, flush=True)
            os.environ.pop(name, None)


def ensure_shards(n_shards):
    os.makedirs(CACHE, exist_ok=True)
    have = sorted(glob.glob(f"{CACHE}/validated-*.parquet"))
    if len(have) >= n_shards:
        print(f"[data] {len(have)} shards already cached"); return
    _sanitize_ssl_cert_env()
    from huggingface_hub import hf_hub_download
    for i in range(n_shards):
        fn = f"en/validated-{i:05d}-of-00138.parquet"
        local = f"../cv_cache/{fn}"
        if os.path.exists(local):
            continue
        print(f"[data] downloading {fn} ...", flush=True)
        hf_hub_download(REPO, fn, repo_type="dataset", local_dir="../cv_cache")


def build_local_tsv(cv_dir):
    """MODE A: rewrite extract_stage to read validated.tsv + clips/. Emitted as a
    note -- MODE A wiring is provided here as a thin adapter."""
    import pandas as pd
    tsv = os.path.join(cv_dir, "validated.tsv")
    if not os.path.exists(tsv):
        sys.exit(f"[MODE A] validated.tsv not found in {cv_dir}")
    print(f"[MODE A] found {tsv}. Set CV_LOCAL_DIR and use extract_stage_local "
          "(see README); MODE B is the validated path for this run.")
    sys.exit("[MODE A] local adapter is a stub; please run MODE B or wire clips/ "
             "decoding in extract_stage.py (one-line change: read mp3 from disk "
             "instead of parquet 'audio' bytes).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="B")
    ap.add_argument("--cv_dir", default=None)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    if args.mode == "A":
        if not args.cv_dir:
            sys.exit("--cv_dir required for MODE A")
        build_local_tsv(args.cv_dir)
    else:
        ensure_shards(args.shards)

    if not args.skip_extract or not os.path.exists("features.parquet"):
        print("\n=== STEP 1: feature extraction ===", flush=True)
        subprocess.run([sys.executable, "extract_stage.py"], check=True)

    print("\n=== STEPS 2-6: analysis ===", flush=True)
    subprocess.run([sys.executable, "analyze.py"], check=True)

    print("\n" + "=" * 70 + "\n  report.md\n" + "=" * 70)
    print(open("report.md").read())


if __name__ == "__main__":
    main()
