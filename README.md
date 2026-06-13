# TIMIT — 40-feature voice-distinctiveness battery (v3)

Measures a fixed set of **40 canonical voice features** per utterance on the
[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) corpus (6,300 utterances, 630
speakers) and quantifies **how distinctive speakers are** through five lenses:
per-feature resolution (F-ratios), usable bit depth (mutual information),
effective dimensionality (participation ratio), a held-out speaker-ID classifier
(Fano / cross-entropy bit bounds), and a collision-probability cross-check.

The headline deliverable is **[`report_TIMIT_v2.md`](report_TIMIT_v2.md)**. Every
number in it is read back from machine-readable tables in **`results/`** (a strict
"source-of-truth firewall" — nothing in the report is estimated or hand-typed).

> **What's new in v3:** all **40/40** features are genuinely computed (v2 left 10
> as not-attempted). The voice-source family (GCT, CQ, MFDR, SQ, NAQ) is derived
> from approximate IAIF glottal inverse filtering, and IHI, VFP, Nasality, SSPF,
> BGD use documented operational proxies (see [`scripts/featurelib.py`](scripts/featurelib.py)).
> Every value is real DSP on the audio/PHN — nothing is imputed.

---

## 1. Headline results (latest full run)

From [`report_TIMIT_v2.md`](report_TIMIT_v2.md) — 6,300 utterances, 630 speakers,
**0 decode failures**, extraction ~13 min. All numbers traced to `results/`.

| metric | value |
|---|---|
| **Measured features** | **40 / 40** |
| **F0 F-ratio** | pooled **9.564** → within-sex **2.39** (M 1.366 / F 3.413); q_max(pooled) = 3 |
| **PR (pooled)** | **9.242**  CI [8.757, 9.542] |
| **PR (within-sex)** | mean **11.572** (M 11.275 / F 11.869) |
| **PR (parent-residual, sex+age+height)** | **12.670**  CI [11.860, 12.989] |
| **Classifier top-1** | logreg 0.6888 · MLP 0.6618 · **LDA 0.7364** |
| **Bit lower bounds** | **Fano 6.017 bits** · cross-entropy 7.177 bits  (H(speaker) = 9.299) |
| **Total summed per-feature usable bits** | 22.21 (optimistic; correlated) |

**Two signatures worth quoting.** (1) F0's pooled F-ratio (9.56) collapses to
~2.4 within sex — sex carries most of its between-speaker variance; effective
dimensionality *rises* pooled → within-sex → parent-residual (9.24 → 11.57 →
12.67), i.e. removing the shared parents (sex, age, height) exposes *more*
independent identity dimensions, not fewer. (2) The MLP under-performs the linear
models — the **capacity-inversion / data-starvation signature** expected at
~10 utts/speaker. All F-ratios/q_max are optimistic upper bounds (single-session
TIMIT) and all bit bounds are floors.

---

## 2. What you need

| Requirement | Detail |
|---|---|
| **TIMIT corpus** | LDC-licensed, **not** included. Default location: `C:\Users\shuoo\Desktop\voice_unique\data\TIMIT`. Layout: `TRAIN/`+`TEST/` → `DR1..DR8/` → `{M,F}xxx0/` speaker dirs → 10 utterances each as `.WAV`/`.PHN`/`.TXT`/`.WRD` + `DOC/SPKRINFO.TXT`. |
| **Python** | 3.10 (validated on 3.10.20). |
| **Packages** | see [`requirements.txt`](requirements.txt) — `sphfile`, `praat-parselmouth`, `librosa`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `pyarrow`. |
| **Corpus is READ-ONLY** | the pipeline only ever reads `.WAV` + `.PHN` (+ `SPKRINFO.TXT`); it never writes into the corpus tree. |

### Set up the environment
```bash
conda create -n voice_unique python=3.10
conda activate voice_unique
pip install -r requirements.txt
```

> **Corpus path is hard-coded** in [`scripts/common.py`](scripts/common.py) as
> `CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"`. If your corpus
> lives elsewhere, edit that one constant.

---

## 3. Quick start — is it runnable? (≈3 min)

Before committing to the full ~13-minute extraction, run the **smoke test**. It
runs the *exact same pipeline scripts, unchanged*, against a small balanced
speaker subset, isolated in `features_smoke/` + `results_smoke/` — it never
touches the real `results/`, `features/`, or the corpus.

```bash
bash scripts/smoke_test.sh                 # 2 M + 2 F per dialect region, TRAIN only (~32 spk / 320 utts)
TIMIT_SMOKE_N=1 bash scripts/smoke_test.sh # smaller / faster (16 speakers)
```
A pass means every stage (env check → extract → analyze → classify) ran to
completion. ⚠️ The **numbers it prints are tiny-subset artifacts, not the study
results**. The smoke test only certifies runnability. Delete the scratch dirs
anytime: `rm -rf features_smoke results_smoke`.

---

## 4. Full run — step by step

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

The whole pipeline is wrapped in one script:
```bash
bash scripts/run_full.sh        # check_env -> extract -> verify -> analyze -> classify -> report
```

Or run the stages individually:

| # | Command (`… python scripts/<script>`) | Reads | Writes |
|---|---|---|---|
| 0 | `check_env.py` | — | package versions → `run.log`; exits non-zero if any are missing |
| 1 | `extract_features.py` | corpus audio + PHN + SPKRINFO | `features/shards/*.parquet` → merged `features/features_per_utt.parquet` + `_EXTRACTION_DONE` sentinel |
| 2 | `verify.py` | features | row count / decode-failure / all-NaN sanity (stops on failure) |
| 3 | `analyze.py` | features | `results/{coverage,f_ratio,usable_bits}.csv`, `results/{binning,effective_dim,collision,analyze_summary}.json` (report §1–5,7) |
| 4 | `classify.py` | features | `results/classifier.json` (report §6) |
| 5 | `assemble_report.py` | `results/*` | `report_TIMIT_v2.md` |

Tunable env vars (defaults in parentheses): `TIMIT_OUTDIR` (`features`),
`TIMIT_RESULTS` (`results`), `TIMIT_REPORT` (`report_TIMIT_v2.md`),
`TIMIT_NPERM` (`200`, MI permutation null), `TIMIT_NBOOT` (`1000`, PR bootstrap),
`TIMIT_NFOLDS` (`5`).

### About the long extraction (stage 1)
- **Sharded & resumable.** Utterances are processed in batches; each batch is
  written atomically to `features/shards/shard_<NN>.parquet`. On restart the
  script scans existing shards, skips finished `utt_id`s, and resumes — a
  crash/disconnect loses at most one in-flight batch. When all 6,300 are done it
  merges to `features/features_per_utt.parquet` and drops the
  `features/_EXTRACTION_DONE` sentinel.
- **Progress, not babysitting.** It appends `<n>/6300 utts … stage=extract`
  lines to `run.log` every ~60 s / 250 utts. Tail that instead of blocking.
- **Reuse.** If `features/_EXTRACTION_DONE` + the merged parquet exist,
  `extract_features.py` exits immediately — so re-running stages 3–5 after a
  stats tweak never re-decodes audio. The committed
  `features/features_per_utt.parquet` lets you reproduce §1–7 with **no corpus
  and no extraction**.

---

## 5. The 40 features (all 40 measured)

Measured per utterance, then aggregated to per-speaker mean + within-speaker
variance. **VTLE is deliberately excluded** and is not a feature.

- **Glottal source (12):** F0, jitter, shimmer, GCT†, CQ†, MFDR†, SQ†, NAQ†, SHR, IHI‡, VFP‡, semitone_SD_F0
- **Vocal-tract filter (11):** F1–F5, B1–B5, Nasality‡
- **Spectral envelope (10):** spectral_skewness, spectral_kurtosis, spectral_entropy, spectral_rolloff, spectral_flux, alpha_ratio, LHR, SPI, GNE, SSPF‡
- **Articulatory/prosodic (7):** CPP, dCPP, RMS, AMD, speech_rate, VOT, BGD‡

**† approximate IAIF** glottal inverse filtering + GCI/threshold cycle
segmentation. **‡ documented operational proxy** (IHI = inter-harmonic intensity
ratio; VFP = vocal-fry probability; Nasality = low-frequency nasal-band energy
ratio; SSPF = spectral slope/tilt; BGD = boundary-gap duration from PHN). All 40
are genuinely computed from the audio/PHN — **never imputed**; coverage reflects
how often each estimator yields a valid value (lowest: VOT 0.97, BGD 0.98).
Definitions live in the docstrings of [`scripts/featurelib.py`](scripts/featurelib.py).

---

## 6. Repo map

```
PROMPT_TIMIT_experiments_claudecode.md   the full experiment specification
scripts/
  common.py            corpus walk + SPKRINFO parse + the 40 feature names + seed 1234
  featurelib.py        core DSP: extract_utt(wav, phn) -> the 40 features + diagnostics
  check_env.py         package/version guard -> run.log
  extract_features.py  sharded resumable extraction -> features/features_per_utt.parquet (+ sentinel)
  analyze.py           report sections 1-5,7  -> results/*.csv, results/*.json
  classify.py          report section 6       -> results/classifier.json
  verify.py            post-run sanity checks (rows / decode fails / NaN columns)
  assemble_report.py   read results/*         -> report_TIMIT_v2.md
  run_full.sh          full end-to-end pipeline
  smoke_test.sh        end-to-end smoke test on a balanced subset (isolated dirs)
report_TIMIT_v2.md     the final report (numbers sourced only from results/)
results/               machine-readable output tables (the source of truth)
features/features_per_utt.parquet   cached 40-feature matrix (committed; ~2.4 MB)
features/shards/, run.log, *_smoke/                                  (git-ignored)
requirements.txt       pinned, validated package versions
```

---

## 7. Conventions & reproducibility

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

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` | Call conda by full path: `"C:/ProgramData/anaconda3/Scripts/conda.exe"`. |
| Shell hangs on a conda call | Prefix every call with `CONDA_NO_PLUGINS=true`. |
| Extraction far slower than ~8 utt/s | Orphaned `python.exe` workers from repeated `conda run` calls can starve the CPU — kill stray processes. |
| `MISSING <pkg>` from `check_env.py` | Package not installed in the env — `pip install -r requirements.txt` (do **not** install into base Python). |
| `extract_features.py` returns instantly | `features/_EXTRACTION_DONE` + parquet already exist; delete `features/` to force a fresh extraction. |
| sklearn `FutureWarning: multi_class …` | Harmless (slated for removal in sklearn 1.8); `classify.py` still produces correct results. |
| `n_splits=5 … least populated class` (smoke only) | Subset too small — raise `TIMIT_SMOKE_N`. |
