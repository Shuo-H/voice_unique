"""
jb_extract_timit.py -- extract the SAME 28 measured acoustic features used in the
Common Voice MI run on the TIMIT corpus, so the cross-corpus joint-bits contrast
(Step 6) is apples-to-apples (identical extractor, identical 10 utts/speaker).

Reuses mi_experiment/mi_features.extract_measured (which itself reuses the
validated Praat/DSP routines in collision_experiment/features.py).  TIMIT wavs
are already 16 kHz mono PCM, so no SPHERE conversion is needed.

Output: timit_features.parquet in the SAME long format as the CV features.parquet
    columns: speaker_id, sex, accent, age, utt_id, feature, value
where for TIMIT:
    speaker_id = TIMIT speaker code (e.g. FJWB0), globally unique
    sex        = 'male' / 'female'  (first char M/F of speaker code)
    accent     = dialect region 'DR1'..'DR8'      (TIMIT's analogue of accent)
    age        = NaN  (not used)
    utt_id     = f'{speaker_id}_{wav_basename}'    (globally unique)
    value      = feature value (np.nan on extraction failure; never imputed)

Run:  python jointbits_experiment/jb_extract_timit.py
Seed 1234 is irrelevant to extraction (deterministic) but recorded for provenance.
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import soundfile as sf
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# make the shared extractor + its feature routines importable
sys.path.insert(0, os.path.join(ROOT, "collision_experiment"))  # features.py
sys.path.insert(0, os.path.join(ROOT, "mi_experiment"))         # mi_features.py
import mi_features as mf  # noqa: E402

TIMIT_ROOT = "/Users/ziyue/Projects/CoLMbo/TIMIT_wav"
OUT = os.path.join(HERE, "timit_features.parquet")
SEED = 1234
N_WORKERS = 14


def enumerate_wavs():
    """Return list of (speaker_id, sex, accent, utt_id, path) for all TIMIT
    TRAIN+TEST wavs (10 per speaker, 630 speakers)."""
    rows = []
    for split in ("TRAIN", "TEST"):
        pat = os.path.join(TIMIT_ROOT, split, "DR*", "*", "*.wav")
        for path in glob.glob(pat):
            # .../<DR>/<SPK>/<UTT>.wav
            parts = path.split(os.sep)
            utt = os.path.splitext(parts[-1])[0]
            spk = parts[-2]
            dr = parts[-3]
            sex = "male" if spk[0].upper() == "M" else "female"
            rows.append((spk, sex, dr, f"{spk}_{utt}", path))
    return rows


def _worker(rec):
    spk, sex, dr, utt_id, path = rec
    try:
        y, srate = sf.read(path)
        feats = mf.extract_measured(y, srate)
    except Exception:
        feats = {k: float("nan") for k in mf.MEASURED}
    out = []
    for k in mf.MEASURED:
        out.append((spk, sex, dr, np.nan, utt_id, k, float(feats[k])))
    return out


def main():
    recs = enumerate_wavs()
    spks = sorted({r[0] for r in recs})
    print(f"[timit] {len(recs)} wavs, {len(spks)} speakers "
          f"(unique speaker codes: {len(spks)})")
    # sanity: exactly 10 utts/speaker expected
    from collections import Counter
    c = Counter(r[0] for r in recs)
    dist = Counter(c.values())
    print(f"[timit] utts/speaker distribution: {dict(dist)}")

    t0 = time.time()
    all_rows = []
    with Pool(N_WORKERS) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_worker, recs, chunksize=8)):
            all_rows.extend(chunk)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"[timit] {i+1}/{len(recs)} utts  ({el:.0f}s, "
                      f"{(i+1)/el:.1f} utt/s)")
    df = pd.DataFrame(all_rows, columns=[
        "speaker_id", "sex", "accent", "age", "utt_id", "feature", "value"])
    # enforce dtypes consistent with CV parquet
    for c_ in ("speaker_id", "sex", "accent", "utt_id", "feature"):
        df[c_] = df[c_].astype(str)
    df["value"] = df["value"].astype("float64")
    df.to_parquet(OUT, index=False)
    el = time.time() - t0
    print(f"[timit] wrote {OUT}  shape={df.shape}  in {el:.0f}s")
    # quick coverage glance
    cov = df.assign(ok=df["value"].notna()).groupby("feature")["ok"].mean()
    print("[timit] per-feature coverage min/median/max: "
          f"{cov.min():.4f}/{cov.median():.4f}/{cov.max():.4f}")
    print("[timit] lowest-coverage features:")
    print(cov.sort_values().head(6).to_string())


if __name__ == "__main__":
    main()
