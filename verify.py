import pandas as pd, numpy as np
from feat_lib import FEATURES_40, NOT_ATTEMPTED
df = pd.read_parquet("features/features_per_utt.parquet")
print("rows", len(df), "unique_utt", df["utt_id"].nunique(),
      "speakers", df["speaker"].nunique())
print("decode_fail_sum", int(df["_decode_fail"].sum()),
      "sr_mismatch_sum", int(df["_sr_mismatch"].sum()))
allnan = [f for f in FEATURES_40 if not np.isfinite(df[f].to_numpy(float)).any()]
print("all_nan_cols", sorted(allnan))
print("all_nan==NOT_ATTEMPTED", set(allnan) == set(NOT_ATTEMPTED))
unexpected = set(allnan) - set(NOT_ATTEMPTED)
print("unexpected_all_nan", sorted(unexpected))
