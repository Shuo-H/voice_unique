"""Download the pre-computed 40-feature cache from Hugging Face so you can SKIP
the ~80-minute extraction and run analyze.py / classify.py directly.

Places `features_per_utt.parquet` into ./features/ and writes the
_EXTRACTION_DONE sentinel (so extract_all.py will no-op if it is ever run).

Repo is overridable:  HF_FEATURES_REPO=<owner>/<dataset>  python fetch_features.py
"""
import os
from huggingface_hub import hf_hub_download

REPO_ID = os.environ.get("HF_FEATURES_REPO", "shuohann/timit-40feature-battery")
FILENAME = "features_per_utt.parquet"

os.makedirs("features", exist_ok=True)
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                       repo_type="dataset", local_dir="features")
# extract_all.py expects the parquet at exactly features/features_per_utt.parquet
print(f"downloaded {REPO_ID}:{FILENAME} -> {path}")

with open(os.path.join("features", "_EXTRACTION_DONE"), "w") as fh:
    fh.write(f"fetched from huggingface dataset {REPO_ID}\n")
print("wrote features/_EXTRACTION_DONE -> extraction can be skipped")
print("next: run analyze.py then classify.py")
