#!/usr/bin/env python3
"""
Voice-uniqueness experiment -- single-entry orchestrator.

Runs STEP 1 (feature extraction) then STEPS 2-6 (analysis + report).

Usage:
    # point at the TIMIT dir that directly contains TRAIN/ and TEST/
    export VU_TIMIT_ROOT="/path/to/timit_LDC93S1/timit/TIMIT"
    export VU_OUT="/path/to/output_dir"           # optional, defaults to ./out
    python run_experiment.py                       # full run
    python run_experiment.py --analyze-only        # reuse existing features.parquet

Dependencies:
    pip install numpy scipy pandas pyarrow scikit-learn librosa soundfile \
                praat-parselmouth sphfile matplotlib

Reproducibility: fixed seed 1234 throughout (see vu_extract.py / vu_analyze.py).
"""
import os, sys

def main():
    analyze_only = "--analyze-only" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.environ.get("VU_OUT")
    if not out:
        parent = os.path.abspath(os.path.join(here, ".."))
        out = parent if os.path.exists(os.path.join(parent, "features.parquet")) else os.getcwd()
        os.environ["VU_OUT"] = out          # propagate to vu_extract / vu_analyze
    feats = os.path.join(out, "features.parquet")

    if not analyze_only:
        import vu_extract
        vu_extract.main()
    else:
        if not os.path.exists(feats):
            sys.exit(f"[run] --analyze-only but {feats} not found; run extraction first.")

    import vu_analyze
    vu_analyze.main()

    rep = os.path.join(out, "report.md")
    try:                                   # make Windows consoles (cp1252) UTF-8 safe
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("\n" + "=" * 80 + "\nREPORT (" + rep + ")\n" + "=" * 80)
    with open(rep, encoding="utf-8") as fh:
        print(fh.read())

if __name__ == "__main__":
    main()
