"""
run_jointbits.py -- one-command driver for the JOINT usable-speaker-information
lower-bound experiment (CV vs TIMIT).

  python jointbits_experiment/run_jointbits.py [--extract-timit]

Steps:
  0. (optional) extract the 28 measured features on TIMIT  -> timit_features.parquet
  1. run the full analysis (Steps 1-7)                     -> CSVs, figs/, results.json
  2. assemble the report                                   -> report-jointbits-cv.md

Reuses the Common Voice features.parquet from mi_experiment/ (no CV re-extraction).
Seed 1234 everywhere.
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sh(script):
    print(f"\n=== running {script} ===", flush=True)
    subprocess.run([PY, os.path.join(HERE, script)], check=True)


def main():
    if "--extract-timit" in sys.argv or not os.path.exists(
            os.path.join(HERE, "timit_features.parquet")):
        sh("jb_extract_timit.py")
    sh("jb_run.py")
    sh("jb_report.py")
    print("\nAll done -> jointbits_experiment/report-jointbits-cv.md")


if __name__ == "__main__":
    main()
