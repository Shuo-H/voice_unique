"""
extract_stage.py -- STEP 1 of the experiment.

Pools the locally-cached Common Voice 17 'en/validated' parquet shards, applies
the speaker filter (>=5 clips, cap 30/speaker via a seeded random sample),
extracts the 41 canonical features per utterance in parallel, and writes:
  - features.parquet   (long format: speaker_id, sex, accent, age, utt_id, feature, value)
  - coverage.csv       (per-feature fraction of utterances successfully computed)
  - artifacts/speaker_manifest.csv  (speaker -> n_clips kept, sex, accent, age)

Seed 1234 throughout.
"""
import os, io, glob, json, time, collections, random
import numpy as np, pandas as pd
import soundfile as sf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
import features as F

# Run from this script's own folder; raw shards live in the shared ../cv_cache.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 1234
MIN_CLIPS = 5
CAP = 30
N_WORKERS = 12
CACHE = "../cv_cache/en"


def modal(lst):
    lst = [x for x in lst if x not in (None, "")]
    return collections.Counter(lst).most_common(1)[0][0] if lst else None


def build_selection():
    """Metadata pass: pick speakers with >=5 clips, cap 30 (seeded)."""
    shards = sorted(glob.glob(f"{CACHE}/validated-*.parquet"))
    print(f"[meta] pooling {len(shards)} shards", flush=True)
    rows_by_spk = collections.defaultdict(list)   # cid -> [(shard_i, local_row)]
    age_by_spk = collections.defaultdict(list)
    gen_by_spk = collections.defaultdict(list)
    acc_by_spk = collections.defaultdict(list)
    paths = {}                                    # (shard_i, row) -> mp3 path
    total = 0
    for si, sh in enumerate(shards):
        t = pq.read_table(sh, columns=["client_id", "age", "gender", "accent", "path"])
        cid = t.column("client_id").to_pylist()
        age = t.column("age").to_pylist()
        gen = t.column("gender").to_pylist()
        acc = t.column("accent").to_pylist()
        pth = t.column("path").to_pylist()
        for r, c in enumerate(cid):
            rows_by_spk[c].append((si, r))
            if age[r]: age_by_spk[c].append(age[r])
            if gen[r]: gen_by_spk[c].append(gen[r])
            if acc[r]: acc_by_spk[c].append(acc[r])
            paths[(si, r)] = pth[r]
        total += len(cid)
    print(f"[meta] {total} clips, {len(rows_by_spk)} distinct speakers", flush=True)

    rng = random.Random(SEED)
    kept = {}            # cid -> list of (shard_i, row) selected (<=CAP)
    meta = {}            # cid -> dict(sex, accent, age, n_total)
    for c, rl in rows_by_spk.items():
        if len(rl) < MIN_CLIPS:
            continue
        sel = rl if len(rl) <= CAP else rng.sample(rl, CAP)
        kept[c] = sel
        meta[c] = dict(sex=modal(gen_by_spk[c]), accent=modal(acc_by_spk[c]),
                       age=modal(age_by_spk[c]), n_total=len(rl), n_kept=len(sel))
    nclips = sum(len(v) for v in kept.values())
    print(f"[meta] kept {len(kept)} speakers (>= {MIN_CLIPS} clips), "
          f"{nclips} clips after cap {CAP}", flush=True)
    return shards, kept, meta, paths


def _worker(args):
    utt_id, audio_bytes = args
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        feats = F.extract_features(np.asarray(y), sr)
    except Exception:
        feats = {k: float("nan") for k in F.FEATURES_41 + F.AUX}
    return utt_id, feats


def main():
    t0 = time.time()
    shards, kept, meta, paths = build_selection()

    # invert: per shard, which rows to pull
    rows_per_shard = collections.defaultdict(dict)   # shard_i -> {row: (cid, utt_id)}
    for c, sel in kept.items():
        for (si, r) in sel:
            rows_per_shard[si][r] = (c, paths[(si, r)])

    long_records = []          # speaker_id, sex, accent, age, utt_id, feature, value
    n_done = 0
    n_total = sum(len(v) for v in rows_per_shard.values())

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for si, sh in enumerate(shards):
            want = rows_per_shard.get(si, {})
            if not want:
                continue
            tbl = pq.read_table(sh, columns=["audio"])
            audio = tbl.column("audio")
            tasks = []
            for r, (cid, utt) in want.items():
                b = audio[r]["bytes"].as_py()
                tasks.append((utt, b, cid))
            # submit
            futs = {ex.submit(_worker, (utt, b)): cid for (utt, b, cid) in tasks}
            for fut in as_completed(futs):
                cid = futs[fut]
                utt, feats = fut.result()
                m = meta[cid]
                for k in F.FEATURES_41 + F.AUX:
                    long_records.append((cid, m["sex"], m["accent"], m["age"],
                                         utt, k, feats[k]))
                n_done += 1
                if n_done % 1000 == 0:
                    el = time.time() - t0
                    print(f"[extract] {n_done}/{n_total} clips "
                          f"({el:.0f}s, {n_done/el:.1f} clip/s)", flush=True)
            del tbl, audio

    df = pd.DataFrame(long_records, columns=["speaker_id", "sex", "accent",
                                             "age", "utt_id", "feature", "value"])
    df.to_parquet("features.parquet", index=False)
    print(f"[save] features.parquet: {len(df)} rows, "
          f"{df.utt_id.nunique()} utts, {df.speaker_id.nunique()} speakers", flush=True)

    # coverage over the 41 (+HNR aux): fraction of utts with non-NaN value
    cov_rows = []
    n_utts = df.utt_id.nunique()
    for feat in F.FEATURES_41 + F.AUX:
        sub = df[df.feature == feat]
        frac = float(sub["value"].notna().mean()) if len(sub) else 0.0
        cov_rows.append(dict(feature=feat,
                             group=("aux_HNR" if feat in F.AUX else "canonical41"),
                             coverage=round(frac, 4),
                             status=("NOT MEASURED" if frac == 0 else "measured")))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv("coverage.csv", index=False)
    print("[save] coverage.csv", flush=True)

    # speaker manifest
    man = pd.DataFrame([dict(speaker_id=c, **meta[c]) for c in kept])
    os.makedirs("artifacts", exist_ok=True)
    man.to_csv("artifacts/speaker_manifest.csv", index=False)
    # dataset summary
    summ = dict(seed=SEED, n_shards=len(shards), min_clips=MIN_CLIPS, cap=CAP,
                n_speakers=len(kept), n_clips=int(n_total),
                clips_per_speaker_kept=man["n_kept"].describe().to_dict(),
                sex_counts=man["sex"].value_counts(dropna=False).to_dict(),
                accent_counts=man["accent"].value_counts(dropna=False).head(10).to_dict(),
                age_counts=man["age"].value_counts(dropna=False).to_dict(),
                elapsed_s=round(time.time() - t0, 1))
    json.dump(summ, open("artifacts/dataset_summary.json", "w"), indent=2, default=str)
    print(f"[done] {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
