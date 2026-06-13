"""
extract_v2.py -- STEP 0/1 of the v2 battery.

Pools ALL locally-cached Common Voice 17 'en/validated' parquet shards, applies
the speaker filter (>=5 clips, cap 30/speaker via a seeded random sample),
extracts the 40 canonical features (VTLE REMOVED) per utterance in parallel,
and writes:
  - features.parquet   (long: speaker_id, sex, accent, age, utt_id, feature, value)
  - coverage.csv       (per-feature fraction of utterances successfully computed)
  - artifacts/speaker_manifest.csv  (speaker -> n_clips kept, sex, accent, age)
  - artifacts/dataset_summary.json  (provenance + scale + metadata distributions)

Seed 1234 throughout.  VTLE is excluded entirely; VOT is computed-as-NaN
(NOT MEASURED, no phone alignments).  Features are NEVER imputed.
"""
import os, io, glob, json, time, collections, random
import numpy as np, pandas as pd
import soundfile as sf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
import features as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 1234
MIN_CLIPS = 5
CAP = 30
N_WORKERS = 16
CACHE = "../cv_cache/en"
OUT_FEATS = F.FEATURES_40 + F.AUX        # 40 canonical + HNR aux (VTLE excluded)


def modal(lst):
    lst = [x for x in lst if x not in (None, "")]
    return collections.Counter(lst).most_common(1)[0][0] if lst else None


def build_selection():
    """Metadata pass over ALL cached shards: pick speakers with >=5 clips,
    cap 30 (seeded). Returns shards, kept, meta, paths, scan_stats."""
    shards = sorted(glob.glob(f"{CACHE}/validated-*.parquet"))
    print(f"[meta] pooling {len(shards)} shards", flush=True)
    rows_by_spk = collections.defaultdict(list)
    age_by_spk = collections.defaultdict(list)
    gen_by_spk = collections.defaultdict(list)
    acc_by_spk = collections.defaultdict(list)
    paths = {}
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
        print(f"[meta]  scanned shard {si} ({len(cid)} clips)", flush=True)
    n_distinct = len(rows_by_spk)
    print(f"[meta] {total} clips, {n_distinct} distinct speakers", flush=True)

    rng = random.Random(SEED)
    kept, meta = {}, {}
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
    scan_stats = dict(n_shards=len(shards), n_distinct_scanned=n_distinct,
                      n_clips_scanned=total)
    return shards, kept, meta, paths, scan_stats


def _worker(args):
    utt_id, audio_bytes = args
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        feats = F.extract_features(np.asarray(y), sr)
    except Exception:
        feats = {k: float("nan") for k in F.FEATURES_41 + F.AUX}
    return utt_id, {k: feats[k] for k in OUT_FEATS}


def main():
    t0 = time.time()
    shards, kept, meta, paths, scan_stats = build_selection()

    rows_per_shard = collections.defaultdict(dict)
    for c, sel in kept.items():
        for (si, r) in sel:
            rows_per_shard[si][r] = (c, paths[(si, r)])

    long_records = []
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
            futs = {ex.submit(_worker, (utt, b)): cid for (utt, b, cid) in tasks}
            for fut in as_completed(futs):
                cid = futs[fut]
                utt, feats = fut.result()
                m = meta[cid]
                for k in OUT_FEATS:
                    long_records.append((cid, m["sex"], m["accent"], m["age"],
                                         utt, k, feats[k]))
                n_done += 1
                if n_done % 2000 == 0:
                    el = time.time() - t0
                    print(f"[extract] {n_done}/{n_total} clips "
                          f"({el:.0f}s, {n_done/el:.1f} clip/s)", flush=True)
            del tbl, audio

    df = pd.DataFrame(long_records, columns=["speaker_id", "sex", "accent",
                                             "age", "utt_id", "feature", "value"])
    df.to_parquet("features.parquet", index=False)
    print(f"[save] features.parquet: {len(df)} rows, "
          f"{df.utt_id.nunique()} utts, {df.speaker_id.nunique()} speakers", flush=True)

    # coverage over the 40 (+HNR aux)
    cov_rows = []
    for feat in OUT_FEATS:
        sub = df[df.feature == feat]
        frac = float(sub["value"].notna().mean()) if len(sub) else 0.0
        cov_rows.append(dict(feature=feat, display=F.disp(feat),
                             group=("aux_HNR" if feat in F.AUX else F.V2_GROUP.get(feat)),
                             coverage=round(frac, 4),
                             status=("NOT MEASURED" if frac == 0 else "measured")))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv("coverage.csv", index=False)
    print("[save] coverage.csv", flush=True)

    man = pd.DataFrame([dict(speaker_id=c, **meta[c]) for c in kept])
    os.makedirs("artifacts", exist_ok=True)
    man.to_csv("artifacts/speaker_manifest.csv", index=False)
    summ = dict(seed=SEED, min_clips=MIN_CLIPS, cap=CAP, **scan_stats,
                n_speakers=len(kept), n_clips=int(n_total),
                clips_per_speaker_kept=man["n_kept"].describe().to_dict(),
                sex_counts=man["sex"].value_counts(dropna=False).to_dict(),
                accent_counts=man["accent"].value_counts(dropna=False).head(12).to_dict(),
                age_counts=man["age"].value_counts(dropna=False).to_dict(),
                n_speakers_ge10=int((man["n_kept"] >= 10).sum()),
                elapsed_s=round(time.time() - t0, 1))
    json.dump(summ, open("artifacts/dataset_summary.json", "w"), indent=2, default=str)
    n_meas = int((cov[cov.group != "aux_HNR"].coverage >= 0.80).sum())
    print(f"[done] {time.time()-t0:.0f}s total; measured-of-40 (cov>=0.80) = {n_meas}",
          flush=True)


if __name__ == "__main__":
    main()
