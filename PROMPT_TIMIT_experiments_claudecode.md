# Experiment prompt — TIMIT (40-feature voice-distinctiveness battery)

You are an empirical research assistant. Run the complete analysis below on the **TIMIT** corpus, then write the report exactly as specified in the **Report** section. Do not fabricate, impute, or estimate any value you did not compute — log missing values as missing and exclude them. Use a single fixed RNG seed **1234** for every randomized procedure (extraction order, quantile ties, permutation nulls, bootstraps, CV folds, classifier initialization, sklearn `random_state`).

## Environment & outputs (execution contract — read first)
- **TIMIT corpus root:** `C:\Users\shuoo\Desktop\voice_unique\data\TIMIT` (verified present on this machine). **Exact layout — walk it directly, no discovery needed:**
  - Two splits: `TRAIN/` and `TEST/`.
  - Each split → 8 dialect regions `DR1/` … `DR8/`.
  - Each region → speaker dirs named `{SEX}{ID}` where SEX ∈ {`M`,`F`} and ID is 3 letters + 1 digit, e.g. `FCJF0`, `MABC0`. The leading letter IS the sex label (use it for within-sex analyses). Total 630 speaker dirs.
  - Each speaker dir → 10 utterances, prefixes `SA1`, `SA2`, `SI####`, `SX###`, each present as four files: `.WAV` (NIST SPHERE `NIST_1A`, decode with `sphfile`), `.PHN` (phone alignments, sample indices @16 kHz), `.TXT`, `.WRD`. You only need `.WAV` + `.PHN`. → 6,300 utterances total.
  - Speaker metadata: `DOC/SPKRINFO.TXT` (sex, dialect region, use, recording/birth dates, height) — parse for sex/height/age body-size proxies.
- **Run from a clean working directory**, not inside the corpus. Treat the corpus tree as **read-only**; never write into it.
- **Environment:** run inside the existing conda env **`voice_unique`**. Execute every command through that env, applied per command (do not rely on a `conda activate` persisting across separate bash calls). Do NOT create new envs or pip-install into base/system Python. Assume `sphfile`, `praat-parselmouth`, `librosa`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `pyarrow` are already installed in `voice_unique`.
  - **Working invocation on this machine (use exactly this form):**
    ```bash
    CONDA_NO_PLUGINS=true "C:/ProgramData/anaconda3/Scripts/conda.exe" run -n voice_unique python <script>.py
    ```
  - **Why this exact form (lessons from prior runs — do not relearn them):**
    1. **`conda` is NOT on PATH.** Call the full path `C:/ProgramData/anaconda3/Scripts/conda.exe`; a bare `conda` fails.
    2. **`CONDA_NO_PLUGINS=true` is REQUIRED.** Without it the conda error-reporting plugin can fire an interactive prompt that hangs the non-interactive bash call.
    3. **NEVER use `conda run ... python -c "<multi-line code>"` on Windows** — multi-line `-c` strings get mangled by the shell and fail. Always write the code to a `.py` file and run the file. (The contract requires script files anyway.)
  - Before extraction, run a one-line import + version check inside the env (as a script file) and record it in `run.log`; if any required package is genuinely missing, **STOP and report it** rather than installing — never impute or work around a missing tool.
- **Report output:** write the final Markdown report to `./report_TIMIT_v2.md`.
- **Feature cache:** cache the per-utterance 40-feature matrix to `./features/` (e.g. `features_per_utt.parquet`) and reuse it on reruns, so re-decoding 6,300 audio files is not required when only the downstream statistics change.
- **Intermediate numbers:** save each table as a machine-readable file under `./results/` — e.g. `coverage.csv`, `f_ratio.csv`, `usable_bits.csv`, `effective_dim.json`, `classifier.json` — so every number can be verified and reused later without re-running.
- **Source-of-truth firewall:** every computed/measured number in the report comes ONLY from `./results/`. Never report a number you did not compute.
- **Reproducibility:** keep the single fixed seed **1234** everywhere (specified throughout below). At the end of the run, print installed library versions and total wall-clock time to `run.log`.
- **No-fabrication guard:** if the corpus path is missing or empty, or a required tool is unavailable, **STOP and report the problem** — do not estimate, simulate, or fill in any value you did not compute.

## Execution robustness (long extraction + handoff — read before launching)
Feature extraction decodes 6,300 audio files and takes tens of minutes. Do **NOT** sit in a blocking `while [ ! -f ... ]; do sleep; done` wait loop watching it — that wastes turns and a single crash/disconnect loses all work. Structure the long run so it survives a crash and hands off automatically:

1. **Sharded, resumable extraction.** Process utterances in batches; after each batch write a self-contained shard to `./features/shards/shard_<NN>.parquet` **atomically** (write `shard_<NN>.parquet.tmp`, then `os.replace` to the final name). On (re)start, scan existing shards, collect the `utt_id`s already done, and skip them — so a restart resumes exactly where it left off instead of redoing finished work. When all utterances are done, merge the shards into `./features/features_per_utt.parquet` (atomic tmp→replace), then write a completion sentinel `./features/_EXTRACTION_DONE`.
2. **Log progress, not liveness.** Append a completed-count line to `run.log` every ~60 s or ~250 utts: `<n_done>/<total> utts, <elapsed>s, <rate> utt/s, stage=extract`, flushed to disk. This lets a resuming turn read true progress without watching the process. Detect a genuine stall (no new completed utts for >10 min while CPU is idle) → **STOP and report**, don't keep waiting.
3. **Sentinel-triggered handoff, then return control.** After extraction, auto-run the downstream stages in order: `analyze.py` (§1–5,7) → `classify.py` (§6) → assemble `./report_TIMIT_v2.md`. Drive this off the sentinel: on each resume, check whether `./features/_EXTRACTION_DONE` exists and continue from the correct stage. **Do not babysit** — launch the long job in the background, then return control; re-engage on the next turn and check the sentinel rather than blocking.
4. **Self-verification before declaring done.** Before writing the report, confirm: row count == 6,300; no unexpected all-NaN columns beyond the not-attempted set; decode failures == 0; measured-out-of-40 count present; capacity-inversion check ran. Each headline number must trace to a specific file+field under `./results/`.
5. **Crash/disconnect safety.** Every artifact is written via tmp→`os.replace` so a kill mid-write never corrupts a file. Because shards + sentinel are on disk, any fresh turn can reconstruct state purely from the filesystem — no in-memory state is required to resume.

## 0. Corpus and provenance
- Corpus: **TIMIT**, 6,300 utterances, 630 speakers, 10 utterances/speaker, 16 kHz.
- Audio: NIST SPHERE (`NIST_1A`); decode with `sphfile`. Verify every signal is non-empty and 16 kHz before extraction. Report decode failures (expected: 0).
- Speaker label: TIMIT speaker ID. Sex label: from the TIMIT directory convention (`DR*/{M,F}xxx0`) — record per-speaker sex (M/F) for the within-sex analyses below.
- Record extraction wall-clock.

## 1. Feature set — THE 40 CANONICAL FEATURES (VTLE has been removed)
Measure each of these 40 features per utterance, then form a per-speaker mean (and per-speaker within-utterance variance). **VTLE / vocal-tract-length estimate is NOT a feature in this list — do not include it.** If you compute VTL internally for any reason, exclude it from all feature tables, rankings, dimensionality, and classifier inputs.

**Glottal source (12):** F0, jitter, shimmer, GCT, CQ, MFDR, SQ, NAQ, SHR, IHI, VFP, semitone_SD_F0
**Vocal-tract filter (11):** F1, F2, F3, F4, F5, B1, B2, B3, B4, B5, Nasality
**Spectral envelope (10):** spectral_skewness, spectral_kurtosis, spectral_entropy, spectral_rolloff, spectral_flux, alpha_ratio, LHR, SPI, GNE, SSPF
**Articulatory/prosodic (7):** CPP, dCPP, RMS, AMD, speech_rate, VOT, BGD

Tooling guidance (use what you did before; substitute equivalent validated tools if needed):
- parselmouth/Praat: F0, F1–F5, B1–B5, jitter, shimmer, HNR (note: HNR is not in the 40; you may compute it as a diagnostic but exclude from the feature set).
- librosa + custom DSP: spectral moments/entropy/rolloff/flux, alpha_ratio, LHR, RMS, AMD, CPP, dCPP.
- TIMIT `.PHN` alignments: speech_rate, VOT, BGD.
- Voice-source / inverse-filtering features (GCT, CQ, MFDR, SQ, NAQ, SHR, IHI, VFP, SPI, GNE, SSPF, Nasality): compute only those you can extract **reliably**; for any you cannot, log 0 coverage and EXCLUDE — never impute.

Report, for every one of the 40 features: coverage fraction (fraction of utterances with a successful value) and measured/NOT-MEASURED status. State the count measured out of 40 (prior runs reached ~29/40 on TIMIT after VTLE removal — report your actual number).

## 2. Population distributions and quantile bins
For each MEASURED feature, form the across-speaker distribution from the 630 (or actual N) per-speaker means; compute q-quantile bin edges for q ∈ {2,3,5,10}. Note any feature where quantile edges collapse (degenerate ties reducing realized bin count below q) and report the realized bin count.

## 3. F-ratios and usable resolution — POOLED **and** WITHIN-SEX
For each measured feature compute:
- `within_var` = mean over speakers of the within-speaker variance (across that speaker's utterances);
- `between_var` = variance of the per-speaker means;
- `F_ratio = between_var / within_var`; also the one-way ANOVA F and p.
- `q_max` = largest q ∈ {2,3,5,10} whose mean across-speaker bin-crossing rate stays < 0.20.

Compute this **three times**:
  (a) **Pooled** over all speakers.
  (b) **Within male speakers only.**
  (c) **Within female speakers only.**
For (b) and (c), recompute the across-speaker distribution, bin edges, between/within variance, F_ratio, and q_max using only that sex's speakers. Report a combined within-sex F_ratio per feature as the pooled-within-sex value (average of male and female between/within decompositions, or report both columns).

**Deliverable:** a table sorted by pooled F_ratio with columns: feature, within_var, between_var, F_ratio(pooled), q_max(pooled), F_ratio(male), F_ratio(female), q_max(within-sex). State explicitly whether within-sex F-ratios are systematically higher or lower than pooled, and by how much for the top features (especially F0 and the formants).

**Caveat to state in the report:** TIMIT is single-session, so within-speaker variance omits day-to-day/health/channel/affective variation; all F_ratios are OPTIMISTIC UPPER BOUNDS and q_max values are optimistic.

## 4. Per-feature usable bit depth (mutual information)
For each measured feature, quantize and compute mutual information between cell assignment and speaker identity, **Miller–Madow corrected against a permutation null** (≥200 shuffles, seed 1234): `I_corrected = max(0, I_mm − I_null_mean)`. Report `b*` = argmax over b ∈ {1..8} of I_corrected, the realized `q_eff(b*)`, `I_corrected(b*)` in bits, normalized MI, and permutation p. Sort by usable bits. Report total summed usable bits across features (note this is an optimistic over-count because features are correlated).

## 5. Effective dimensionality — POOLED, WITHIN-SEX, and PARENT-RESIDUAL
Estimate effective dimensionality as the participation ratio `PR = (Σλ_i)^2 / Σλ_i^2` of the eigenvalues of the per-speaker feature covariance (z-scored features). 95% CIs from 1,000 speaker-level bootstrap resamples (seed 1234). Compute PR for:
  (a) **Pooled** over all speakers.
  (b) **Within-sex** (compute PR within male and within female separately, report both and their mean).
  (c) **Parent-residual (KEY NEW ANALYSIS):** regress each feature (per-speaker mean vector) on the available shared-parent variables — at minimum **sex**; include any age or body-size proxy if available in TIMIT metadata — then compute the participation ratio on the **residuals**. This removes the shared-parent contribution and estimates the effective dimensionality that survives after the dominant confounders are removed.

**Deliverable:** report PR(pooled), PR(within-sex, male/female/mean), PR(parent-residual), each with 95% CI. State clearly how much the participation ratio rises from pooled → within-sex → parent-residual. (Hypothesis under test: parent-residual PR is substantially higher than the pooled value, i.e. effective dimensionality survives after the dominant shared-parent confounders are removed.)

## 6. Joint usable speaker bits — held-out classifier lower bound
Keep features with ≥90% coverage; listwise-delete incomplete rows; report rows and speakers retained (prior run: 5,035/6,300 utts, 629 speakers). Chance = 1/S; H(speaker)=log2(S).
- Utterance-disjoint stratified 5-fold CV (each speaker split across folds; no utterance shared train/test). z-score on train-fold statistics only.
- Train three classifiers: (A) regularized logistic regression, (B) higher-capacity MLP, (C) shrinkage LDA. Report top-1 accuracy + 95% CI + per-fold mean±std + log-loss (bits).
- Convert to bit lower bounds via **Fano**: `I_lower = H(speaker) − [H_b(P_e) + P_e·log2(S−1)]`; and **cross-entropy**: `H(speaker) − mean test log-loss (bits)`. Report both, with 95% CIs.
- Note the capacity-inversion check: report whether the MLP underperforms the linear models (expected at ~8 utts/speaker — the data-starvation signature).
- Headline: the larger Fano lower bound; note all bounds are floors (a stronger classifier raises them) and remain below the H(speaker) sample ceiling.

## 7. Collision-metric sanity (optional cross-check)
Using the measured pooled q_max and PR, and separately the parent-residual PR, plug into the collision formulae at n=10^10 and report P(E), P(M), P(B) at the measured operating points. This is a sanity cross-check, not a headline.

## Report
Write a single Markdown report titled **"TIMIT — 40-feature distinctiveness battery (v2)"** with these sections, each containing the tables/numbers above: (0) provenance + seed + decode failures + extraction time; (1) coverage table + measured-out-of-40 count; (2) binning notes; (3) **F-ratio table with pooled AND within-sex columns** + explicit statement of the pooled-vs-within-sex difference; (4) usable-bit-depth table; (5) **effective-dimensionality block reporting PR pooled / within-sex / parent-residual with CIs** and the rise across the three; (6) classifier table + Fano/xent bounds + capacity-inversion note; (7) optional collision cross-check. End with a "Headline numbers" block listing, for direct quotation in a paper: measured-features-out-of-40, F0 F-ratio (pooled and within-sex) and q_max, PR(pooled), PR(within-sex), PR(parent-residual), classifier top-1 accuracies, and the Fano/xent bit bounds. Report every number to the precision you computed it. Flag every NOT-MEASURED feature explicitly.