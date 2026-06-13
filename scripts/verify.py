"""Self-verification before report: row count, decode failures, all-NaN columns, measured count.
STOP (exit 1) on a hard failure."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_40

OUTDIR = os.environ.get("TIMIT_OUTDIR", "features")
EXPECT_ROWS = int(os.environ.get("TIMIT_EXPECT_ROWS", "6300"))


def main():
    df = pd.read_parquet(os.path.join(OUTDIR, "features_per_utt.parquet"))
    problems = []
    n = len(df)
    print(f"rows = {n} (expected {EXPECT_ROWS})")
    if n != EXPECT_ROWS:
        problems.append(f"row count {n} != {EXPECT_ROWS}")
    dec_fail = int((~df["decode_ok"]).sum())
    print(f"decode failures = {dec_fail}")
    if dec_fail != 0:
        problems.append(f"{dec_fail} decode failures")
    dec = df[df["decode_ok"]]
    allnan = [f for f in FEATURES_40 if dec[f].notna().sum() == 0]
    measured = [f for f in FEATURES_40 if dec[f].notna().sum() > 0]
    print(f"measured = {len(measured)}/40; all-NaN columns = {allnan}")
    print(f"speakers = {dec['speaker'].nunique()}")
    # duplicates
    dup = df["utt_id"].duplicated().sum()
    if dup:
        problems.append(f"{dup} duplicate utt_ids")
    if problems:
        print("VERIFY FAILED: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
