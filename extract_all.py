"""Sharded, resumable, progress-logging feature extraction.
- Each batch writes its own shard atomically to features/shards/shard_<NN>.parquet
  (write .tmp then os.replace).
- On (re)start, utterances already present in completed shards are skipped.
- Progress logged to run.log every >=60s or >=250 new utterances.
- At the end, shards are merged -> features/features_per_utt.parquet (atomic).
Single fixed RNG seed 1234. Corpus is READ-ONLY (only WAV/PHN are read)."""
import os, csv, time, sys, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
import feat_lib as fl

SEED = 1234
OUTDIR = "features"
SHARDDIR = os.path.join(OUTDIR, "shards")
CACHE = os.path.join(OUTDIR, "features_per_utt.parquet")
NPROC = 16
BATCH = 200
TOTAL = 6300


def log(msg):
    line = f"[extract {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("run.log", "a") as fh:
        fh.write(line + "\n")
        fh.flush()


def work(r):
    try:
        out, diag = fl.extract_one(r["wav"], r["phn"])
    except Exception as e:
        out = fl._nan_dict()
        diag = {"decode_fail": 1, "err": str(e)[:120], "sr": 0,
                "n_samples": 0, "sr_mismatch": 0}
    rec = {"utt_id": r["utt_id"], "speaker": r["speaker"], "sex": r["sex"],
           "split": r["split"], "dialect": r["dialect"], "utt": r["utt"]}
    rec.update(out)
    rec["_decode_fail"] = diag.get("decode_fail", 0)
    rec["_sr"] = diag.get("sr", 0)
    rec["_sr_mismatch"] = diag.get("sr_mismatch", 0)
    rec["_n_samples"] = diag.get("n_samples", 0)
    return rec


def load_done_ids():
    done = set()
    for s in glob.glob(os.path.join(SHARDDIR, "shard_*.parquet")):
        try:
            done |= set(pd.read_parquet(s, columns=["utt_id"])["utt_id"].tolist())
        except Exception:
            pass
    return done


def next_shard_index():
    idx = 0
    for s in glob.glob(os.path.join(SHARDDIR, "shard_*.parquet")):
        try:
            n = int(os.path.basename(s).split("_")[1].split(".")[0])
            idx = max(idx, n + 1)
        except Exception:
            pass
    return idx


def write_shard(buf, idx):
    df = pd.DataFrame(buf)
    tmp = os.path.join(SHARDDIR, f"shard_{idx:03d}.parquet.tmp")
    fin = os.path.join(SHARDDIR, f"shard_{idx:03d}.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fin)


def merge_and_validate(t0):
    shards = sorted(glob.glob(os.path.join(SHARDDIR, "shard_*.parquet")))
    parts = [pd.read_parquet(s) for s in shards]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("utt_id").sort_values("utt_id").reset_index(drop=True)
    tmp = CACHE + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE)
    el = time.time() - t0
    nfail = int(df["_decode_fail"].sum())
    nmis = int(df["_sr_mismatch"].sum())
    with open("results/extract_time.txt", "w") as fh:
        fh.write(f"extract_wallclock_sec={el:.1f}\n")
        fh.write(f"n_utts={len(df)}\n")
        fh.write(f"decode_failures={nfail}\n")
        fh.write(f"sr_mismatch={nmis}\n")
        fh.write(f"nproc={NPROC}\n")
    log(f"MERGED {len(df)} rows -> {CACHE} decode_fail={nfail} "
        f"sr_mismatch={nmis} elapsed={el:.0f}s")
    sent = os.path.join(OUTDIR, "_EXTRACTION_DONE")
    tmp_s = sent + ".tmp"
    with open(tmp_s, "w") as fh:
        fh.write(f"rows={len(df)} decode_fail={nfail} sr_mismatch={nmis}\n")
    os.replace(tmp_s, sent)
    return df


def main():
    np.random.seed(SEED)
    os.makedirs(SHARDDIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if os.path.exists(CACHE):
        log(f"CACHE_EXISTS {CACHE} -- skipping extraction")
        return
    rows = []
    with open("results/manifest.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: r["utt_id"])

    done = load_done_ids()
    todo = [r for r in rows if r["utt_id"] not in done]
    log(f"resume: {len(done)} already done, {len(todo)} to do "
        f"(total {len(rows)}), {NPROC} procs")
    t0 = time.time()
    if not todo:
        merge_and_validate(t0)
        return

    shard_idx = next_shard_index()
    buf = []
    n_done = len(done)
    last_log = time.time()
    last_log_count = 0
    since_shard = 0
    with Pool(NPROC) as pool:
        for rec in pool.imap_unordered(work, todo, chunksize=4):
            buf.append(rec)
            n_done += 1
            since_shard += 1
            if since_shard >= BATCH:
                write_shard(buf, shard_idx)
                log(f"shard_{shard_idx:03d} written; {n_done}/{len(rows)} "
                    f"utts, {time.time()-t0:.0f}s, stage=extract")
                shard_idx += 1
                buf = []
                since_shard = 0
            elif (time.time() - last_log >= 60) or (n_done - last_log_count >= 250):
                el = time.time() - t0
                rate = (n_done - len(done)) / el if el > 0 else 0
                log(f"{n_done}/{len(rows)} utts, {el:.0f}s, "
                    f"{rate:.2f} utt/s, stage=extract")
                last_log = time.time()
                last_log_count = n_done
    if buf:
        write_shard(buf, shard_idx)
        log(f"final shard_{shard_idx:03d} written; {n_done}/{len(rows)} utts")
    merge_and_validate(t0)
    log("EXTRACT_COMPLETE")


if __name__ == "__main__":
    main()
