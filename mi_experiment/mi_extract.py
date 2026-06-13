"""
mi_extract.py -- STEP 1: balanced per-utterance feature extraction.

Design (critical for unbiased MI):
  * Group clips by client_id (== speaker, an assumption stated in the report).
  * CLIPS_PER_SPEAKER = 12. Keep only speakers with >= 12 validated, decodable
    clips; from each retained speaker RANDOMLY SAMPLE EXACTLY 12 clips (seed 1234)
    -> a BALANCED design (every speaker contributes the same count).
  * Target ~1500 retained speakers; include parquet shards in order until the
    count of >=12-clip speakers reaches the target (or shards run out).

Outputs (all under mi_experiment/):
  features.parquet            long: speaker_id, sex, accent, age, utt_id, feature, value
  coverage.csv                per feature: coverage fraction, measured vs NOT MEASURED
  artifacts/speaker_manifest.csv
  artifacts/dataset_summary.json
  artifacts/selection.csv     speaker_id, utt_id (the 12 chosen clips/speaker)

Reuses already-computed measured features from the prior run's ../features.parquet
(keyed by mp3 basename) to avoid recomputation; any clip not in that cache is
decoded from the parquet MP3 bytes via soundfile (libsndfile, no ffmpeg) and
extracted fresh.  Features are never imputed.
"""
import os, io, sys, glob, json, time, collections, hashlib
import numpy as np, pandas as pd
import soundfile as sf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mi_features as MF

SEED = 1234
CLIPS_PER_SPEAKER = 12
TARGET_SPEAKERS = 1500
N_WORKERS = max(2, (os.cpu_count() or 4) - 2)
CACHE_DIR = os.path.join(PARENT, "cv_cache", "en")
OUT_DIR = HERE
PRIOR_PARQUET = os.path.join(PARENT, "features.parquet")  # reuse source


def modal(lst):
    lst = [x for x in lst if x not in (None, "")]
    return collections.Counter(lst).most_common(1)[0][0] if lst else None


def _spk_rng(cid):
    h = int(hashlib.sha256(cid.encode()).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng([SEED, h])


def build_selection():
    """Metadata pass: include shards in order until >=12-clip speakers reaches
    TARGET; then sample exactly 12 clips/speaker (per-speaker deterministic rng)."""
    shards = sorted(glob.glob(f"{CACHE_DIR}/validated-*.parquet"))
    if not shards:
        sys.exit(f"[fatal] no shards in {CACHE_DIR}")
    rows_by_spk = collections.defaultdict(list)     # cid -> [(shard_i, row, basename)]
    age_by, gen_by, acc_by = (collections.defaultdict(list) for _ in range(3))
    used_shards = 0
    for si, sh in enumerate(shards):
        t = pq.read_table(sh, columns=["client_id", "age", "gender", "accent", "path"])
        cid = t.column("client_id").to_pylist()
        age = t.column("age").to_pylist()
        gen = t.column("gender").to_pylist()
        acc = t.column("accent").to_pylist()
        pth = t.column("path").to_pylist()
        for r in range(len(cid)):
            c = cid[r]
            rows_by_spk[c].append((si, r, os.path.basename(pth[r])))
            if age[r]: age_by[c].append(age[r])
            if gen[r]: gen_by[c].append(gen[r])
            if acc[r]: acc_by[c].append(acc[r])
        used_shards = si + 1
        n_qual = sum(1 for v in rows_by_spk.values() if len(v) >= CLIPS_PER_SPEAKER)
        print(f"[meta] shard {si:02d}: {len(rows_by_spk)} speakers, "
              f"{n_qual} with >={CLIPS_PER_SPEAKER} clips", flush=True)
        if n_qual >= TARGET_SPEAKERS:
            break

    qual = {c: v for c, v in rows_by_spk.items() if len(v) >= CLIPS_PER_SPEAKER}
    print(f"[meta] using {used_shards} shards; {len(qual)} qualifying speakers "
          f"(>={CLIPS_PER_SPEAKER} clips)", flush=True)

    selection = {}   # cid -> list[(shard_i, row, basename)] length 12
    meta = {}
    for c in sorted(qual):                       # sorted -> stable
        clips = sorted(qual[c], key=lambda x: x[2])  # sort by basename
        idx = _spk_rng(c).choice(len(clips), size=CLIPS_PER_SPEAKER, replace=False)
        selection[c] = [clips[i] for i in sorted(idx.tolist())]
        meta[c] = dict(sex=modal(gen_by[c]), accent=modal(acc_by[c]),
                       age=modal(age_by[c]), n_total=len(qual[c]),
                       n_kept=CLIPS_PER_SPEAKER)
    return shards, used_shards, selection, meta


def load_prior_cache():
    """basename -> {feature: value} for the 28 measured features, from the prior run."""
    if not os.path.exists(PRIOR_PARQUET):
        print("[cache] no prior features.parquet; extracting all clips fresh", flush=True)
        return {}
    df = pd.read_parquet(PRIOR_PARQUET, columns=["utt_id", "feature", "value"])
    df = df[df["feature"].isin(MF.MEASURED)]
    df["base"] = df["utt_id"].map(os.path.basename)
    cache = {}
    for base, sub in df.groupby("base"):
        cache[base] = dict(zip(sub["feature"], sub["value"]))
    print(f"[cache] loaded {len(cache)} clips from prior features.parquet", flush=True)
    return cache


def _worker(args):
    base, audio_bytes = args
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        feats = MF.extract_measured(np.asarray(y), sr)
    except Exception:
        feats = {k: float("nan") for k in MF.MEASURED}
    return base, feats


def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT_DIR, "artifacts"), exist_ok=True)
    shards, used_shards, selection, meta = build_selection()
    cache = load_prior_cache()

    # which (shard,row,base) need fresh extraction (not in cache)
    need_by_shard = collections.defaultdict(list)   # si -> [(row, base)]
    base2cid = {}
    n_reuse = n_need = 0
    for c, clips in selection.items():
        for (si, r, base) in clips:
            base2cid[base] = c
            if base in cache and all(k in cache[base] for k in MF.MEASURED):
                n_reuse += 1
            else:
                need_by_shard[si].append((r, base))
                n_need += 1
    print(f"[extract] reuse {n_reuse} cached clips, extract {n_need} fresh "
          f"({N_WORKERS} workers)", flush=True)

    feats_by_base = {}
    # seed reused values
    for c, clips in selection.items():
        for (si, r, base) in clips:
            if base in cache and all(k in cache[base] for k in MF.MEASURED):
                feats_by_base[base] = {k: cache[base][k] for k in MF.MEASURED}

    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for si in sorted(need_by_shard):
            want = need_by_shard[si]
            tbl = pq.read_table(shards[si], columns=["audio"])
            audio = tbl.column("audio")
            futs = {}
            for (r, base) in want:
                b = audio[r]["bytes"].as_py()
                futs[ex.submit(_worker, (base, b))] = base
            for fut in as_completed(futs):
                base, feats = fut.result()
                feats_by_base[base] = feats
                n_done += 1
                if n_done % 1000 == 0:
                    el = time.time() - t0
                    print(f"[extract] {n_done}/{n_need} fresh "
                          f"({el:.0f}s, {n_done/max(el,1e-9):.1f} clip/s)", flush=True)
            del tbl, audio

    # ---- assemble long-format dataframe ----
    rec = []
    sel_rows = []
    for c in sorted(selection):
        m = meta[c]
        for (si, r, base) in selection[c]:
            sel_rows.append((c, base))
            fv = feats_by_base.get(base, {k: float("nan") for k in MF.MEASURED})
            for k in MF.MEASURED:
                rec.append((c, m["sex"], m["accent"], m["age"], base, k, fv.get(k, float("nan"))))
    df = pd.DataFrame(rec, columns=["speaker_id", "sex", "accent", "age",
                                    "utt_id", "feature", "value"])
    df.to_parquet(os.path.join(OUT_DIR, "features.parquet"), index=False)
    S = df["speaker_id"].nunique(); N = df["utt_id"].nunique()
    print(f"[save] features.parquet: {len(df)} rows, S={S} speakers, N={N} utts", flush=True)

    # ---- coverage: 28 measured + 14 NOT MEASURED ----
    cov_rows = []
    for feat in MF.MEASURED:
        sub = df[df.feature == feat]
        frac = float(sub["value"].notna().mean()) if len(sub) else 0.0
        cov_rows.append(dict(feature=feat, group="measured", coverage=round(frac, 4),
                             status="measured" if frac > 0 else "NOT MEASURED"))
    for feat in MF.NOT_MEASURED:
        cov_rows.append(dict(feature=feat, group="glottal/inverse-filtering",
                             coverage=0.0, status="NOT MEASURED"))
    pd.DataFrame(cov_rows).to_csv(os.path.join(OUT_DIR, "coverage.csv"), index=False)
    print("[save] coverage.csv", flush=True)

    # ---- speaker manifest + selection + dataset summary ----
    man = pd.DataFrame([dict(speaker_id=c, **meta[c]) for c in sorted(selection)])
    man.to_csv(os.path.join(OUT_DIR, "artifacts", "speaker_manifest.csv"), index=False)
    pd.DataFrame(sel_rows, columns=["speaker_id", "utt_id"]).to_csv(
        os.path.join(OUT_DIR, "artifacts", "selection.csv"), index=False)
    summ = dict(seed=SEED, clips_per_speaker=CLIPS_PER_SPEAKER,
                target_speakers=TARGET_SPEAKERS, shards_used=used_shards,
                S_speakers=int(S), N_utterances=int(N),
                H_speaker_ceiling_bits=float(np.log2(S)),
                n_reused=n_reuse, n_extracted=n_need,
                sex_counts=man["sex"].value_counts(dropna=False).to_dict(),
                accent_counts=man["accent"].value_counts(dropna=False).head(12).to_dict(),
                age_counts=man["age"].value_counts(dropna=False).to_dict(),
                measured=MF.MEASURED, not_measured=MF.NOT_MEASURED,
                elapsed_s=round(time.time() - t0, 1))
    json.dump(summ, open(os.path.join(OUT_DIR, "artifacts", "dataset_summary.json"), "w"),
              indent=2, default=str)
    print(f"[done] S={S}, N={N}, log2(S)={np.log2(S):.3f} bits, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
