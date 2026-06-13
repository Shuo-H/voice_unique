"""Sharded, resumable feature extraction over TIMIT.
Env vars:
  TIMIT_SMOKE_N   if set (int), smoke mode: that many M and that many F speakers per DR, TRAIN only.
  TIMIT_OUTDIR    output base dir (default ./features). Smoke runs use ./features_smoke.
Writes shards atomically, logs progress to run.log, merges to features_per_utt.parquet,
then writes sentinel _EXTRACTION_DONE.
"""
import os, sys, time, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import enumerate_utterances, FEATURES_40, parse_spkrinfo
from featurelib import extract_utt

BATCH = 200


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with open("run.log", "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    print(line, end="")


def atomic_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def done_utt_ids(shard_dir):
    done = set()
    for sh in glob.glob(os.path.join(shard_dir, "shard_*.parquet")):
        try:
            ids = pd.read_parquet(sh, columns=["utt_id"])["utt_id"].tolist()
            done.update(ids)
        except Exception:
            pass
    return done


def main():
    smoke_n = os.environ.get("TIMIT_SMOKE_N")
    smoke_n = int(smoke_n) if smoke_n else None
    outdir = os.environ.get("TIMIT_OUTDIR", "features_smoke" if smoke_n else "features")
    shard_dir = os.path.join(outdir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    meta = parse_spkrinfo()
    utts = enumerate_utterances(smoke_per_sex_per_dr=smoke_n)
    total = len(utts)
    log(f"=== extract start: mode={'SMOKE n='+str(smoke_n) if smoke_n else 'FULL'}, total={total} utts, outdir={outdir} ===")

    sentinel = os.path.join(outdir, "_EXTRACTION_DONE")
    merged = os.path.join(outdir, "features_per_utt.parquet")
    if os.path.exists(sentinel) and os.path.exists(merged):
        log("sentinel present, nothing to do.")
        return

    done = done_utt_ids(shard_dir)
    log(f"resume: {len(done)} utts already done in shards")
    todo = [u for u in utts if u["utt_id"] not in done]

    t0 = time.time()
    n_done = len(done)
    decode_fail = 0
    shard_idx = len(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
    last_log = time.time()

    buf = []
    for u in todo:
        rec = {"utt_id": u["utt_id"], "split": u["split"], "dr": u["dr"],
               "speaker": u["speaker"], "sex": u["sex"]}
        m = meta.get(u["speaker"], {})
        rec["height_cm"] = m.get("height_cm")
        rec["age_years"] = m.get("age_years")
        rec["decode_ok"] = True
        try:
            feats, dur = extract_utt(u["wav"], u["phn"])
            rec.update(feats)
            rec["duration_s"] = dur
        except Exception as e:
            decode_fail += 1
            rec["decode_ok"] = False
            rec["duration_s"] = np.nan
            for k in FEATURES_40:
                rec[k] = np.nan
            log(f"DECODE/EXTRACT FAIL {u['utt_id']}: {e}")
        buf.append(rec)
        n_done += 1

        if len(buf) >= BATCH:
            shard_idx += 1
            atomic_parquet(pd.DataFrame(buf), os.path.join(shard_dir, f"shard_{shard_idx:03d}.parquet"))
            buf = []
        if time.time() - last_log > 60 or n_done == total:
            el = time.time() - t0
            rate = (n_done - len(done)) / el if el > 0 else 0
            log(f"{n_done}/{total} utts, {el:.0f}s, {rate:.1f} utt/s, stage=extract, decode_fail={decode_fail}")
            last_log = time.time()

    if buf:
        shard_idx += 1
        atomic_parquet(pd.DataFrame(buf), os.path.join(shard_dir, f"shard_{shard_idx:03d}.parquet"))

    # merge
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))]
    full = pd.concat(parts, ignore_index=True).drop_duplicates("utt_id").reset_index(drop=True)
    atomic_parquet(full, merged)
    with open(sentinel, "w") as f:
        f.write(f"rows={len(full)} decode_fail_this_run={decode_fail} time={time.time()-t0:.0f}s\n")
    log(f"=== extract DONE: merged {len(full)} rows -> {merged}; decode_fail_this_run={decode_fail} ===")


if __name__ == "__main__":
    main()
