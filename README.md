# TIMIT — 40-feature voice-distinctiveness battery

Measures a fixed set of **40 canonical voice features** per utterance on the
[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) corpus (6,300 utterances, 630
speakers) and quantifies **how distinctive speakers are** through five lenses:
per-feature resolution (F-ratios), usable bit depth (mutual information),
effective dimensionality (participation ratio), a held-out speaker-ID classifier
(Fano / cross-entropy bit bounds), and a collision-probability cross-check.

The headline deliverable is **[`report_TIMIT_v2.md`](report_TIMIT_v2.md)**. Every
number in it is read back from machine-readable tables in **`results/`** (a strict
"source-of-truth firewall" — nothing in the report is estimated or hand-typed).

---

## 1. What you need

| Requirement | Detail |
|---|---|
| **TIMIT corpus** | LDC-licensed, **not** included. Default location: `../data/TIMIT` (i.e. `C:\Users\shuoo\Desktop\voice_unique\data\TIMIT`). Layout: `TRAIN/`+`TEST/` → `DR1..DR8/` → `{M,F}xxx0/` speaker dirs → 10 utterances each as `.WAV`/`.PHN`/`.TXT`/`.WRD` + `DOC/SPKRINFO.TXT`. |
| **Python** | 3.10 (validated on 3.10.20). |
| **Packages** | see [`requirements.txt`](requirements.txt) — `sphfile`, `praat-parselmouth`, `librosa`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `pyarrow`. |
| **Corpus is READ-ONLY** | the pipeline only ever reads `.WAV` + `.PHN` (+ `SPKRINFO.TXT`); it never writes into the corpus tree. |

### Set up the environment
```bash
conda create -n voice_unique python=3.10
conda activate voice_unique
pip install -r requirements.txt
```

> **Corpus path is hard-coded.** The scripts that walk the corpus
> (`build_manifest.py`, `build_spkrinfo.py`, `smoke_subset.py`) define
> `CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"` at the top. If your
> corpus lives elsewhere, edit that one constant in those three files.

---

## 2. Quick start — is it runnable? (≈3 min)

Before committing to the full ~80-minute extraction, run the **smoke test**. It
runs the *exact same pipeline scripts, unchanged*, against a small balanced
speaker subset inside a disposable `./smoke/` workspace — it never touches the
real `results/`, `features/`, or the corpus.

```bash
bash run_smoke.sh                     # 8 male + 8 female speakers (160 utts)
SMOKE_N_PER_SEX=4 bash run_smoke.sh   # smaller / faster (8 speakers, 80 utts)
```
A pass means every stage (env check → manifest → extract → analyze → classify)
ran to completion. ⚠️ The **numbers it prints are tiny-subset artifacts, not the
study results** (e.g. it may retain 29/30 features because VOT dips under the 90%
coverage gate on a small sample). The smoke test only certifies runnability.
Delete the scratch dir anytime: `rm -rf smoke/`.

---

## 3. Full run — step by step

Every command goes **through the conda env, applied per command** (don't rely on
`conda activate` persisting across separate shells). On the validated Windows
machine the exact, battle-tested invocation is:

```bash
CONDA_NO_PLUGINS=true "C:/ProgramData/anaconda3/Scripts/conda.exe" run -n voice_unique python <script>.py
```
Why this exact form: `conda` is **not** on PATH (use the full path);
`CONDA_NO_PLUGINS=true` stops an interactive plugin prompt from hanging a
non-interactive shell; and **never** pass multi-line code via `python -c` on
Windows (the shell mangles it) — always run a `.py` file. On Linux/macOS with
`conda` on PATH you can simplify to `conda run -n voice_unique python <script>.py`.

Run the stages in order:

| # | Command (`… python <script>`) | Reads | Writes |
|---|---|---|---|
| 0 | `check_env.py` | — | prints package versions; exits non-zero if any are missing |
| 1 | `build_manifest.py` | corpus | `results/manifest.csv` (6,300 rows: ids, sex, abs WAV/PHN paths) |
| 2 | `build_spkrinfo.py` | `DOC/SPKRINFO.TXT` | `results/speaker_meta.csv` (sex, dialect, height_cm, age_yr) |
| 3 | `extract_all.py` | manifest + audio | `features/shards/*.parquet` → merged `features/features_per_utt.parquet` + `_EXTRACTION_DONE` sentinel; `results/extract_time.txt` |
| 4 | `analyze.py` | features + speaker_meta | `results/{coverage,binning,f_ratio,usable_bits}.csv`, `results/{effective_dim,collision,analyze_summary}.json` (report §1–5,7) |
| 5 | `classify.py` | features | `results/classifier.json` (report §6) |
| 6 | *(manual)* assemble report | `results/*` | `report_TIMIT_v2.md` |

So a full run is:
```bash
C="CONDA_NO_PLUGINS=true \"C:/ProgramData/anaconda3/Scripts/conda.exe\" run -n voice_unique python"
eval "$C check_env.py"
eval "$C build_manifest.py"
eval "$C build_spkrinfo.py"
eval "$C extract_all.py"      # the long one: decodes 6,300 files, ~80 min
eval "$C analyze.py"
eval "$C classify.py"
```

### About the long extraction (stage 3)
- **Sharded & resumable.** Utterances are processed in batches; each batch is
  written atomically to `features/shards/shard_<NN>.parquet`. On restart the
  script scans existing shards, skips finished `utt_id`s, and resumes — a
  crash/disconnect loses at most one in-flight batch. When all 6,300 are done it
  merges to `features/features_per_utt.parquet` and drops the
  `features/_EXTRACTION_DONE` sentinel.
- **Progress, not babysitting.** It appends `<n>/6300 utts … stage=extract`
  lines to `run.log` every ~60 s / 250 utts. Tail that instead of blocking.
- **Reuse.** If `features/features_per_utt.parquet` already exists, `extract_all.py`
  exits immediately — so re-running stages 4–5 after a stats tweak never
  re-decodes audio.

### Verifying a run
```bash
# row count, decode failures, and that the only all-NaN columns are the
# 10 deliberately not-attempted features:
CONDA_NO_PLUGINS=true "C:/ProgramData/anaconda3/Scripts/conda.exe" run -n voice_unique python verify.py
```
Expected: `rows 6300 … decode_fail_sum 0 … unexpected_all_nan []`.

---

## 4. The 40 features (30 measured, 10 not-attempted)

Measured per utterance, then aggregated to per-speaker mean + within-speaker
variance. **VTLE is deliberately excluded** and is not a feature.

- **Glottal source (12):** F0, jitter, shimmer, GCT\*, CQ\*, MFDR\*, SQ\*, NAQ\*, SHR, IHI\*, VFP\*, semitone_SD_F0
- **Vocal-tract filter (11):** F1–F5, B1–B5, Nasality\*
- **Spectral envelope (10):** spectral_skewness, spectral_kurtosis, spectral_entropy, spectral_rolloff, spectral_flux, alpha_ratio, LHR, SPI, GNE, SSPF\*
- **Articulatory/prosodic (7):** CPP, dCPP, RMS, AMD, speech_rate, VOT, BGD\*

\* = **NOT-MEASURED** (10 total): the 7 EGG/inverse-filter glottal features
(GCT, CQ, MFDR, SQ, NAQ, IHI, VFP), `Nasality` (needs a nasometer), and the two
ambiguously-defined `SSPF`/`BGD`. These are emitted as NaN with 0 coverage and
**excluded everywhere — never imputed**. So a clean run reports **30/40 measured**.
The not-attempted set is declared in `feat_lib.NOT_ATTEMPTED`.

---

## 5. Repo map

```
PROMPT_TIMIT_experiments_claudecode.md   the full experiment specification
feat_lib.py            core DSP: extract_one(wav, phn) -> the 40 features + diagnostics
build_manifest.py      walk corpus            -> results/manifest.csv
build_spkrinfo.py      parse SPKRINFO.TXT     -> results/speaker_meta.csv
check_env.py           package/version guard
extract_all.py         sharded resumable extraction -> features/features_per_utt.parquet
analyze.py             report sections 1-5,7  -> results/*.csv, results/*.json
classify.py            report section 6       -> results/classifier.json
verify.py              post-run sanity checks (rows / decode fails / NaN columns)
test_extract.py        debug: dump features for the first 6 utterances
finalize_log.py        append library versions + wall-clock to run.log
smoke_subset.py        build a small balanced subset manifest (used by smoke test)
run_smoke.sh           end-to-end smoke test on a subset (isolated in ./smoke)
report_TIMIT_v2.md     the final report (numbers sourced only from results/)
results/               machine-readable output tables (the source of truth)
features/              cached parquet feature matrix + shards   (git-ignored)
run.log                run/provenance log                        (git-ignored)
```

---

## 6. Conventions & reproducibility

- **Single fixed RNG seed `1234`** everywhere (extraction order, permutation
  nulls, bootstraps, CV folds, sklearn `random_state`).
- **No fabrication.** If the corpus path is missing/empty or a required package
  is unavailable, the pipeline **stops and reports** rather than estimating. Every
  reported number traces to a specific file+field under `results/`.
- **Atomic writes.** Every artifact is written `*.tmp` then `os.replace`d, so a
  kill mid-write never corrupts a file and any fresh run can resume purely from
  what's on disk.
- **Single-session caveat.** TIMIT is single-session, so within-speaker variance
  omits day-to-day/health/channel/affective variation — all F-ratios are
  **optimistic upper bounds** and q_max values are optimistic. The report states
  this explicitly.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` | Call conda by full path: `"C:/ProgramData/anaconda3/Scripts/conda.exe"`. |
| Shell hangs on a conda call | Prefix every call with `CONDA_NO_PLUGINS=true`. |
| `FATAL: missing split dir …` | Wrong corpus path — fix `CORPUS` in `build_manifest.py` / `build_spkrinfo.py` / `smoke_subset.py`. |
| `MISSING <pkg>` from `check_env.py` | Package not installed in the env — `pip install -r requirements.txt` (do **not** install into base Python). |
| `extract_all.py` returns instantly | `features/features_per_utt.parquet` already exists; delete it (or `features/`) to force a fresh extraction. |
| sklearn `FutureWarning: multi_class …` | Harmless (slated for removal in sklearn 1.8); `classify.py` still produces correct results. |
| `n_splits=5 … least populated class` (smoke only) | Subset too small — raise `SMOKE_N_PER_SEX`. |
