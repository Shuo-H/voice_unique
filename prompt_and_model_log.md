# PROMPT

# Experiment prompt — Common Voice 17 (40-feature voice-distinctiveness battery)

You are an empirical research assistant. Run the complete analysis below on **Mozilla Common Voice 17 (English)**, then write the report exactly as specified in the **Report** section. Do not fabricate, impute, or estimate any value you did not compute — log missing values as missing and exclude them. Use a single fixed RNG seed **1234** for every randomized procedure.

## 0. Corpus, provenance, and scale
- Target data: Common Voice 17.0, English, `validated` split. The official `mozilla-foundation/common_voice_17_0` repo was emptied (Oct 2025) and moved to the Mozilla Data Collective (gated). Use the public non-gated parquet mirror **`fixie-ai/common_voice_17_0`**, which preserves the official schema (`client_id, path, audio, sentence, up_votes, down_votes, age, gender, accent, locale, segment, variant`). If an official local release is available, you may use it instead — state which you used.
- Decode MP3 via soundfile/libsndfile; resample to 16 kHz mono. **`client_id` is the speaker label.**
- **SCALE UP from the prior run (which used ~1,755 speakers).** Pool **as many `en/validated` parquet shards as needed to reach at least 5,000 qualifying speakers** (more is better; target 5,000–10,000 if compute allows). Report the number of shards scanned, distinct client_ids scanned, and final speaker/clip counts.
- Speaker filter: keep client_ids with **≥ 5 validated clips**; cap at **30 clips/speaker** (seeded random subsample). Report clips/speaker min/median/mean/max.
- Record per-speaker metadata distributions actually present: **gender/sex** (male_masculine / female_feminine / NaN), **age** buckets, top **accents**. These are the shared-parent variables for the conditioning analyses below.
- State the multi-session caveat: CV within-speaker variance is genuinely multi-session/multi-channel (different devices, rooms, days), so F-ratios here are REALISTIC (not optimistic upper bounds like TIMIT), if anything conservative.

## 1. Feature set — THE 40 CANONICAL FEATURES (VTLE removed)
Measure each of these 40 per clip, then per-speaker mean and within-speaker variance. **VTLE is NOT a feature — exclude it entirely.** VOT generally cannot be measured on CV (no phone alignments) — expect to exclude it; log as missing, never impute.

**Glottal source (12):** F0, jitter, shimmer, GCT, CQ, MFDR, SQ, NAQ, SHR, IHI, VFP, semitone_SD_F0
**Vocal-tract filter (11):** F1, F2, F3, F4, F5, B1, B2, B3, B4, B5, Nasality
**Spectral envelope (10):** spectral_skewness, spectral_kurtosis, spectral_entropy, spectral_rolloff, spectral_flux, alpha_ratio, LHR, SPI, GNE, SSPF
**Articulatory/prosodic (7):** CPP, dCPP, RMS, AMD, speech_rate, VOT, BGD

Use parselmouth/Praat + librosa + custom DSP as for TIMIT. For features requiring phone alignments (VOT, BGD, speech_rate) that CV lacks, compute what you can (e.g. an energy-based speech_rate proxy if defensible, else exclude) and clearly mark the rest missing. Report coverage (≥80% = used downstream) and measured-out-of-40 count (prior run: 39/40, excluding VOT — report your actual number).

## 2. Population distributions and quantile bins
For each measured feature, across-speaker distribution from per-speaker means; q-quantile bin edges for q ∈ {2,3,5,10}; report degenerate/collapsed bins.

## 3. F-ratios and usable resolution — POOLED **and** WITHIN-SEX
Exactly as for TIMIT: `within_var` (mean of within-speaker variances), `between_var` (variance of per-speaker means), `F_ratio`, ANOVA F/p, and `q_max` (largest q with mean bin-crossing rate < 0.20). Compute **pooled**, **within male**, and **within female** (using the CV `gender` field; drop NaN-gender speakers for the within-sex computation only). Deliverable: table sorted by pooled F_ratio with columns feature, within_var, between_var, F_ratio(pooled), q_max(pooled), F_ratio(male), F_ratio(female), q_max(within-sex). State whether within-sex F-ratios exceed pooled and by how much for the top features. Expected on CV: most features support only q_max ≤ 2 (multi-session variability); report the actual q≥3 / q=2 failure counts.

## 4. Per-feature usable bit depth (mutual information)
Same as TIMIT: Miller–Madow-corrected MI vs ≥200-shuffle permutation null (seed 1234); `b*`, `q_eff(b*)`, `I_corrected(b*)` bits, NMI, perm p; sorted by usable bits; total summed usable bits (optimistic over-count).

## 5. Effective dimensionality — POOLED, WITHIN-SEX, and PARENT-RESIDUAL
Participation ratio `PR = (Σλ_i)^2 / Σλ_i^2` of the per-speaker feature covariance (z-scored); 95% CIs from 1,000 speaker-level bootstraps (seed 1234). Compute:
  (a) **Pooled** (prior run: PR ≈ 12).
  (b) **Within-sex** (male and female separately + mean).
  (c) **Parent-residual (KEY NEW ANALYSIS):** regress each feature on the available shared parents — **sex, age bucket, and accent** (CV has all three; encode age/accent as categoricals) — then compute PR on the residuals. This is the empirically-grounded estimate of the effective dimensionality surviving after the dominant confounders are removed.

**Deliverable:** PR(pooled), PR(within-sex), PR(parent-residual), each with CI, and the rise across the three. CV is the better corpus for this test because it has explicit age and accent labels in addition to sex.

## 6. Joint usable speaker bits — held-out classifier lower bound
With the larger speaker set, run the same classifier battery as TIMIT: keep ≥90%-coverage features, listwise-delete, report retained clips/speakers. Utterance-disjoint stratified 5-fold CV, z-scored on train folds only. Three classifiers: (A) regularized logistic regression, (B) MLP, (C) shrinkage LDA. Report top-1 accuracy + CI + per-fold + log-loss(bits). Fano and cross-entropy bit lower bounds (formulae as in TIMIT prompt), with CIs. Note: with more speakers and ~8–11 clips/speaker, report whether the capacity inversion (MLP < linear) persists. Headline: larger Fano lower bound; all bounds are floors below H(speaker)=log2(S).

## 7. Collision-metric sanity (optional cross-check)
Plug measured pooled q_max and PR, and separately parent-residual PR, into the collision formulae at n=10^10; report P(E), P(M), P(B). Sanity cross-check only.

## Report
Write a single Markdown report titled **"Common Voice 17 — 40-feature distinctiveness battery (v2)"** with sections mirroring the TIMIT report: (0) provenance + mirror used + **final speaker/clip counts (target ≥5,000 speakers)** + metadata distributions + multi-session caveat; (1) coverage + measured-out-of-40; (2) binning notes; (3) **F-ratio table pooled AND within-sex** + difference statement; (4) usable-bit-depth table; (5) **PR pooled / within-sex / parent-residual with CIs** + the rise across the three (emphasize parent-residual uses sex+age+accent); (6) classifier + Fano/xent bounds + capacity-inversion note; (7) optional collision cross-check. End with a "Headline numbers" block for direct quotation: final speaker count, measured-out-of-40, F0 F-ratio (pooled and within-sex) and q_max, q≥3 failure count, PR(pooled), PR(within-sex), PR(parent-residual), classifier accuracies, Fano/xent bounds. Report every number to computed precision; flag every NOT-MEASURED feature.


# MODEL LOG
I'll start by understanding what's already in this project — the prompt references prior TIMIT and CV runs, so there's likely reusable feature-extraction code I should build on rather than rewrite.

There's substantial prior infrastructure. Let me map it in detail — feature extraction code, the CV cache, and what the prior runs produced.

This is a v2 of the existing `collision_experiment` (the 40-feature battery). Let me read the foundational code I'll be building on.

The 41-feature extractor and CV extraction pipeline already exist. Now let me read the analysis code (F-ratios, PR, MI), the collision formulae, the classifier code, and the prior report to mirror its structure.

Now the MI methodology and classifier (joint-bits) methodology, plus prior run's actual counts.

Let me reconcile the prior collision run's actual feature coverage and check the compute environment before committing to a scale.

Environment confirmed: 18 cores, 48 GB RAM, and the `fable5_test` venv has all packages (parselmouth, librosa, soundfile, sklearn, pyarrow). Prior 4-shard run gave 39/40 measured (only VOT missing). Now let me measure how many qualifying speakers the 14 cached shards yield, to decide whether I need to download more.

14 cached shards already yield **6,601 qualifying speakers** (39,832 scanned, 80,137 clips after cap) — comfortably above the ≥5,000 target. The prompt says "more is better; target 5,000–10,000." Let me test download speed for additional shards to decide whether to push toward ~10,000.

Let me set up task tracking for this multi-step experiment and test the download speed for one additional shard in parallel.

Download is fast (~16s/shard at 31 MB/s). I'll push toward the upper end of the target (~10k speakers) by adding ~10 more shards. Let me set up task tracking, then download shards 15–23 to reach 24 shards total.

Download running in background. Let me build the v2 pipeline now. First, set up the directory and the 40-feature extractor (reusing the proven `features.py`, VTLE excluded).

Now add the 40-feature canonical definitions (VTLE excluded, regrouped per the new prompt) and a display-name map to the extractor.

Download of shards 15–23 completed. Let me read the copied file's header, then add the 40-feature definitions.

Now let me verify the download and rescan metadata with all 24 shards.

**24 shards → 10,676 qualifying speakers** (51,330 scanned, 137,675 clips after cap; 5,105 speakers with ≥10 clips for the classifier). That's at the top of the 5–10k target. Locking in 24 shards. Let me update the task and write the extraction driver.

Before the ~50-min full run, let me smoke-test the extractor on a handful of real clips to confirm the 40-feature output and coverage look right.

Extractor works cleanly: VTLE excluded, VOT all-NaN (NOT MEASURED), SSPF partial coverage, all other 38 finite. Let me launch the full extraction in the background (~50 min), then write the analysis scripts while it runs.

Extraction launched in background. Now, while it runs (~50 min), I'll write the analysis scripts. First let me study the exact MI usable-bit-depth methodology and the jointbits driver so I replicate them faithfully.

I now fully understand the MI usable-bit-depth methodology. Let me read the jointbits driver to replicate Step 6 faithfully, then write all analysis scripts.

The extraction wrapper returned, but the actual `nohup`'d Python should still be running detached. Let me verify it's alive and progressing.

Extraction is humming at 44.7 clips/s (~51 min total). Confirmed: 10,676 speakers, 137,675 clips. While it runs, I'll write and de-risk all analysis scripts against the prior 1,755-speaker parquet (same schema). Let me check the prior classifier run's scale/runtime first, then write `common.py`.

Prior CV classifier (S=1599, 28 feat): logreg 0.5515, MLP 0.5103 (inversion present), Fano 4.88, xent 6.92 bits. Our v2 scales to S≈5,105 with 40 features. Now let me write the analysis modules. Starting with shared loaders.

Now the Step 3 F-ratio module (pooled + within-sex + q_max), plus Step 2 bins:

Let me dry-run Step 3 against the prior 1,755-speaker parquet (same schema) to catch bugs before the real data lands.

Step 3 validated: F0 pooled F=3.3 (q_max=2), male F=1.2 — within-sex lower for F0 as theory predicts; q_max dist {1:22, 2:17}, none ≥3. Now the MI usable-bit-depth module (Step 4):

Dry-run Step 4's MI pipeline on the prior parquet (reduced perms for speed):

Step 4 validated (F0 b*=2, 0.65 bits, p<0.001). Now the effective-dimensionality module — PR pooled, within-sex, and the key parent-residual analysis (Step 5):

Let me dry-run Step 5 (PR pooled/within-sex/parent-residual) on the prior parquet:

Step 5 validated: pooled PR=12.53 (matches prior ≈12), rising to within-sex ~13.0 and parent-residual 13.49 — the expected de-correlation rise; sex explains up to 50% of a feature's variance. Now the classifier battery (Step 6), CV-only:

Now the report assembler with the Step 7 collision cross-check built in:

All analysis scripts written and the first three dry-validated. Let me check extraction progress.

Extraction at 20k/137.7k (~44 min to go). Let me use the time to validate the one untested integration — Step 6's classifier pipeline — against the prior parquet (LDA only, for speed):

Found a real bug that would bite the live run: `jb.load_wide` auto-discovers all features including all-NaN VOT (dropped by the pivot) and aux HNR. I must restrict Step 6 to the canonical 40 present in the wide table. Let me fix it.

Now re-validate Step 6 with the fix (prior parquet has the same all-NaN VOT issue):

Step 6 fixed and validated (LDA acc 0.57, Fano 4.33 bits, fast). All scripts ready. The extraction is nohup-detached (~40 min left), so I'll wait on the output file appearing. Let me load the Monitor tool to block until it's done.

Per the Monitor guidance, for a single "done" signal I'll use a background `until` loop that exits when the extraction process finishes — I'll be notified once.

Everything is staged. Here's where things stand while the extraction finishes:

**Done:**
- Pooled **24 CV17 `en/validated` shards** → **10,676 qualifying speakers** / 137,675 clips (51,330 client_ids scanned). Comfortably in the 5–10k target.
- Built `cv17_v2_experiment/` reusing the proven 40-feature extractor (**VTLE excluded**, VOT NOT MEASURED).
- Wrote and **dry-validated against the prior 1,755-speaker parquet** all five analysis modules:
  - **Step 3** F-ratios (pooled + within-sex + q_max) — confirmed F0 pooled F=3.3 vs male F=1.2, q_max dist {1:22, 2:17}.
  - **Step 4** MI usable bit depth — confirmed F0 b*=2, 0.65 bits, p<0.001.
  - **Step 5** PR pooled/within-sex/**parent-residual** — confirmed pooled 12.5 → residual 13.5 rise.
  - **Step 6** classifier joint-bits — caught and fixed a real bug (all-NaN VOT + aux HNR leaking into the feature set); LDA validated.
  - **Step 7 + report** assembler with collision cross-check.

**Extraction in progress:** ~42 clips/s, 137,675 clips total (~50 min). A background waiter will notify me the moment it finishes, then I'll run Steps 3–5, launch the classifier battery (Step 6, the long pole) in the background, and assemble the report.

I'll continue automatically when extraction completes.

Extraction finished. Let me verify the outputs and final counts.

Extraction complete: **10,676 speakers / 137,675 clips**, **39/40 measured** (only VOT missing; SSPF=0.886 clears threshold). Metadata looks rich (US 2,581 / England 811 / India 450 accents; full age range). Let me run Step 3 (F-ratios pooled + within-sex):

Step 3 done (56s). Real-data results: q_max(pooled) dist {1:17, 2:21, **3:1**} — F0 reaches q_max=3 at this scale; **38/39 fail q≥3**; within-sex F-ratios lower than pooled for 38/39 features. Now Step 4 (MI usable bit depth):

Step 4 done (16s): total optimistic usable bits = 11.71; top F0, RMS, CPP, GCT, AlphaRatio; all 39 permutation-significant. Now Step 5 (PR pooled/within-sex/parent-residual with bootstraps):

Step 5 done (18s): **PR(pooled) 12.95 [12.78,13.07] → within-sex 13.33 → parent-residual 13.57 [13.39,13.71]** — a clean, CI-separated rise (sex+age+accent removal de-correlates the features). Now the long pole, Step 6 classifier battery (S=5,105). Launching in background:

Step 6 running: S=5,095, N=50,950, 38 features (SSPF dropped), ceiling 12.315 bits. The 5,095-class logreg/MLP are the slow part. Let me set up a completion waiter:

While the classifier trains, let me verify the Step 3–5 outputs are sound and ready for the report.

Outputs are clean and self-consistent. Highlights: F0 leads (F=3.47, q_max=3); usable bits all b*=2 with F0 at 0.662 bits; PR rises 12.95→13.33→13.57 with non-overlapping CIs; sex+age+accent explain 33% of F0's between-speaker variance. Let me check the classifier's progress:

Steps 3–5 are complete and verified. Step 6 (the 5,095-class classifier battery) is training — the multinomial logreg over ~5k classes is the slow part. A completion waiter will notify me when it finishes, then I'll assemble the final report.

Interim results so far:
- **F-ratios:** F0 leads (pooled 3.47, q_max=3); 38/39 features fail q≥3; within-sex F-ratios lower than pooled for 38/39 (sex inflates pooled separation).
- **Usable bits:** all features peak at b*=2 (q_eff=4); F0 0.66 bits; total optimistic sum 11.71 bits; all 39 permutation-significant.
- **Effective dimensionality:** PR 12.95 (pooled) → 13.33 (within-sex) → **13.57 (parent-residual, sex+age+accent)**, with non-overlapping CIs.

I'll continue automatically when the classifier finishes.


is it still running?

It just **finished** — all three classifiers are done (completed at 11:42, 41 min total):

| classifier | top-1 acc | Fano ≥ | xent ≥ |
|---|---|---|---|
| logreg | **0.5475** | **5.749 bits** | 8.451 |
| MLP | 0.5343 | 5.583 | 7.982 |
| LDA | 0.5228 | 5.440 | 6.479 |

**Capacity inversion persists** (MLP 0.5343 < logreg 0.5475). Now generating Step 7 + the final report:

Report assembled (238 lines). Let me review the full output for correctness before presenting it.

The report is complete and well-formed. One cosmetic inconsistency: the §3 "for example" table shows `AlphaRatio` (internal name) instead of the display name `alpha_ratio`. Let me fix that and regenerate.

Fixed. Both remaining tasks are done. Let me mark them complete and record a memory of this v2 run.

Done — the complete battery ran end to end on real Common Voice 17 audio. Full report: [cv17_v2_experiment/report.md](cv17_v2_experiment/report.md).

## What ran
Pooled **24 `fixie-ai/common_voice_17_0` en/validated shards** → scanned 51,330 client_ids → **10,676 speakers / 137,675 clips** (≥5 clips, cap 30, seed 1234). Decoded MP3→16 kHz mono, extracted the **40 canonical features (VTLE excluded)**, **39/40 measured** (only VOT missing — no phone alignments; SSPF 0.886 clears threshold). Every number is computed; nothing imputed.

## Headline numbers
| Quantity | Result |
|---|---|
| Speakers / clips | **10,676 / 137,675** (24 shards) |
| Measured of 40 | **39** (VOT not measurable) |
| F0 F-ratio | pooled **3.47** (male 1.45, female 2.18), q_max=3 |
| q≥3 failure count | **38 of 39** (q_max=1: 17, q_max=2: 21) |
| **PR(pooled)** | **12.95** [12.78, 13.07] |
| PR(within-sex mean) | 13.33 |
| **PR(parent-residual, sex+age+accent)** | **13.57** [13.39, 13.71] — +0.62 rise, CI-separated |
| Total per-feature usable bits (optimistic) | **11.71 bits** (F0 top, 0.66 bits) |
| Classifier top-1 (S=5,095) | logreg **0.5475** / MLP 0.5343 / LDA 0.5228 |
| **Joint bit floors** | **Fano ≥ 5.749, xent ≥ 8.451 bits** (ceiling 12.31) |
| Capacity inversion | **persists** (MLP < logreg) |

## Key findings vs the prompt's expectations
- **Within-sex F-ratios are *lower* than pooled for 38/39 features** — pooling across sexes inflates between-speaker variance, so the sex-linked source/filter features look more discriminative than they are within a sex. F0: 3.47 pooled → 1.45 (M) / 2.18 (F).
- **The parent-residual rise is the new result:** removing sex+age+accent *de-correlates* the features and *raises* effective dimensionality (12.95 → 13.57). Even after stripping all three confounders, 39 measured axes carry only ~14 independent dimensions — far below nominal independence. (CV uniquely supports this because it has explicit age + accent labels.)
- **Bigger Fano bound at scale:** 5.75 bits vs the prior run's 4.88, and the capacity inversion survives — the regularized linear model still beats the MLP at ~10 clips/speaker over thousands of classes.
- **Collision cross-check:** plugging measured PR (~13) and q≤2 into the paper's formulae makes population collisions certain (P(B)=1, P(E)=1) — the "astronomically unique" figures are an artifact of the independence + high-q assumptions.

## Artifacts (all in `cv17_v2_experiment/`)
`report.md`, `coverage.csv`, `fratios.csv`, `usable_bits.csv` + `mi_by_feature_bit.csv`, `pr_effective_dim.csv`, `classifiers.csv` + `classifier_results.json`, `collision_crosscheck.csv`, `bins.json`, plus `artifacts/` (dataset summary, parent R², per-step JSON summaries) and the reusable scripts (`extract_v2.py`, `common.py`, `step3–6`, `report_v2.py`).

One note on the only judgment call I made without asking: I balanced the MI step at **5 clips/speaker** (not 12 as in the prior run) specifically so the bit-depth analysis keeps the *full* 10,676-speaker set rather than dropping to the ~3,500 speakers with ≥12 clips — it's documented in §4. Everything else follows the spec's defaults.

