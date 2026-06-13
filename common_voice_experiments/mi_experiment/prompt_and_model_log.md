# PROMPT
You are running a quantization-based, information-theoretic voice-individuality experiment on
Mozilla Common Voice. The goal: measure how many BITS each acoustic feature carries about
speaker identity, as a function of quantization depth, using mutual information between speaker
ID and quantization-bin assignment, with rigorous finite-sample bias correction. Work in
Python. Be rigorous and honest: the headline metric is bias-corrected MI above a permutation
null — never report raw plug-in MI as the result. Never impute or fabricate a feature you
cannot compute; log it missing. Fixed random seed 1234 everywhere (numpy default_rng, all
shuffles, all bootstraps, all subsampling); record it in the report.

=========================
DATA — two modes, auto-select
=========================
MODE A (local): a Common Voice release directory at <PATH IF I GIVE ONE>, containing
validated.tsv (columns include client_id, path, sentence, age, gender, accent/accents, locale)
and a clips/ folder of mp3 files.
MODE B (download): if no path is given, stream a CAPPED subset from Hugging Face `datasets`,
dataset "mozilla-foundation/common_voice_17_0", config "en", split "validated", streaming=True.
Accept terms programmatically if required; if blocked, STOP and print the exact manual step.
Collect clips until you have enough speakers meeting the filtering rule below (target ~1500
retained speakers; stop streaming once reached). Cache decoded audio to ./cv_cache.
Decode mp3 via librosa (ffmpeg backend); resample to 16 kHz mono. Verify each signal is
non-empty and finite before extraction; skip and log unreadable clips.

=========================
ENVIRONMENT
=========================
If any packages are required, please create a virtual environment named fable5_test and install them there.
HF-token: <redacted -- set via HF_TOKEN env var>
=========================
SPEAKER FILTERING & BALANCING (critical for unbiased MI)
=========================
Group clips by client_id (treat client_id as speaker identity — note this assumption in the
report; one client_id is assumed one speaker).
- Set CLIPS_PER_SPEAKER = 12. Keep only speakers with >= 12 validated, successfully-decoded
  clips. From each retained speaker, RANDOMLY SAMPLE EXACTLY 12 clips (seed 1234). This yields
  a BALANCED design: every speaker contributes the same count, so the speaker x bin contingency
  table and the Miller-Madow correction are well-calibrated across speakers.
- Record: final speaker count S, total utterances N = 12*S, and retained-speaker metadata
  (gender, age band, accent group) for stratified analysis.
- If MODE B can't reach ~1500 speakers at 12 clips each, proceed with what you have but report
  S prominently — MI estimates and especially joint bits are sample-limited and S is the ceiling.
H(speaker) = log2(S) is the absolute MI ceiling; reference it throughout.

=========================
STEP 1 — FEATURE EXTRACTION (per utterance)
=========================
Extract as many of these features as you can compute reliably, PER UTTERANCE (not per-speaker
means — within-speaker spread is essential):
  parselmouth/Praat: F0, F1-F5, B1-B5, jitter, shimmer, HNR.
  librosa + DSP: spectral_skewness, spectral_kurtosis, spectral_entropy, spectral_rolloff,
    spectral_flux, alpha_ratio, LHR, RMS, AMD, CPP, dCPP.
  formant-based: VTLE (vocal-tract-length estimate from formants).
  prosodic (no forced alignment available in CV, so derive what is robust): speech_rate proxy
    (syllable-nucleus rate via energy/voicing peaks), semitone_SD_F0.
Glottal-source / inverse-filtering features (GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE,
Nasality, SSPF, VOT, BGD): attempt only if a validated tool is available; otherwise log as
NOT MEASURED with 0 coverage. DO NOT fabricate them.
- For dynamic features (jitter, shimmer, flux, dCPP) keep the per-utterance value.
- Save long-format features.parquet (speaker_id, gender, age, accent, utt_id, feature, value).
- Save coverage.csv: per feature, fraction of utterances successfully computed; mark
  measured vs NOT MEASURED. Drop any feature with <90% coverage from the MI analysis and say so.

=========================
STEP 2 — QUANTIZATION GRID
=========================
For each measured feature and each bit depth b in {1,2,3,4,5,6,7,8} (q = 2^b bins):
bin the per-utterance values with q-quantile (equal-frequency) edges over the pooled
across-utterance distribution, so bins are marginally equiprobable. Merge degenerate/empty
bins for low-entropy features and record the effective bin count q_eff(b) (may be < 2^b); use
q_eff in all entropy ceilings and corrections. Save bins per (feature,b) to bins.json.

=========================
STEP 3 — MUTUAL INFORMATION: RAW + BIAS-CORRECTED + PERMUTATION NULL
=========================
For each (feature, b):
  - Build the S x q_eff contingency table (speaker x bin) over all N utterances.
  - I_raw(speaker; bin) in bits (plug-in).
  - Miller-Madow correction via MM-corrected H(speaker), H(bin), H(speaker,bin):
    I_mm = H_mm(speaker) + H_mm(bin) - H_mm(speaker,bin).
  - Permutation null: shuffle the speaker_id column across all N utterances 200x (seed 1234),
    recompute MI each time; record I_null_mean, I_null_p95, and permutation p = fraction of
    null MI >= I_raw.
  - HEADLINE: I_corrected = max(0, I_mm - I_null_mean)  [bits above chance].
    Also NMI_corrected = I_corrected / log2(S).
  - Save all columns to mi_by_feature_bit.csv:
    feature, b, q_eff, I_raw, I_mm, I_null_mean, I_null_p95, perm_p, I_corrected, NMI_corrected.

=========================
STEP 4 — USABLE BIT DEPTH PER FEATURE
=========================
For each feature: plot I_corrected, I_raw, I_null_mean vs b -> figs/mi_<feature>.png.
b* = argmax_b I_corrected (the depth past which finer bins add noise, not speaker info).
Table usable_bits.csv: feature, b*, q_eff(b*), I_corrected(b*), NMI_corrected(b*), perm_p(b*),
sorted by I_corrected(b*) descending. This replaces variance-based q_max with a distribution-
free, partition-based usable resolution.

=========================
STEP 5 — JOINT / CUMULATIVE BITS (bias-aware, sample-capped)
=========================
Greedy forward selection at fixed b=2 per feature (keeps joint cell count estimable):
  - Start empty. At each step add the feature whose addition maximizes I_corrected(speaker;
    joint-bin) of the selected set, using the SAME Miller-Madow + permutation-null correction
    on the joint contingency table.
  - Stop when the best marginal corrected gain <= its permutation p95 (no real added info).
  - Record selection order and cumulative I_corrected (bits) vs #features -> cumulative_bits.csv
    and figs/cumulative_bits.png.
This cumulative-bits curve is the information-theoretic analogue of effective dimensionality:
total USABLE speaker bits, not a variance count. Saturation point = empirical usable joint
dimensionality (in bits) over the measured features.
CAVEAT TO STATE: cumulative corrected MI cannot exceed log2(S); as joint cells grow the null
rises, so saturation is partly sample-limited. Report where it flattens AND the log2(S) ceiling;
present the saturation as an upper-bounded ESTIMATE, not a measured population value.

=========================
STEP 6 — STRATIFIED & HOMOGENEOUS-COHORT ANALYSIS
=========================
Re-run Steps 3-5 within each sex stratum (and, where a single accent group has >= 300 retained
speakers, within that accent group). Report whether per-feature usable bits and cumulative bits
DROP within a homogeneous cohort vs the pooled population — this empirically tests the framework's
'low effective dimensionality among similar speakers' prediction. Save stratified tables.

=========================
STEP 7 — REPORT
=========================
Write report-cv-quant.md containing:
  - Data summary: S speakers, N utterances, 12 clips/speaker (balanced), feature coverage,
    measured vs NOT MEASURED list.
  - Per-feature usable-bits table (Step 4), highlighting the top features by corrected bits.
  - Cumulative-bits curve and its saturation point, WITH the log2(S) ceiling caveat (Step 5).
  - Stratified / homogeneous-cohort comparison (Step 6).
  - Honest limitations: (1) mp3 compression degrades high-frequency & source-periodicity
    features (jitter, shimmer, CPP, flux, rolloff) — absolute bits are biased LOW for those and
    the absolute scale is not comparable to clean-audio corpora; relative structure is more
    robust. (2) Finite-sample MI bias — state that MM + permutation null are why, and that
    residual bias still inflates absolute bits. (3) client_id = speaker assumption.
    (4) log2(S) ceiling on joint bits. (5) speech_rate is a proxy (no forced alignment).
    (6) self-reported, uneven demographic metadata.
  - File manifest.
Save all CSVs, JSONs, figures, and a single runnable script (extract + analyze + report).
Print report-cv-quant.md at the end.

DO NOT: report raw MI as the discriminability result; impute missing features; let q_eff differ
from 2^b without logging; claim usable bits higher than the permutation null supports; or treat
the cumulative-bits saturation as a population measurement rather than a sample-capped estimate.


# MODEL LOG
I'll start by understanding the current state of the project directory, environment, and the prior context from memory.

This is a continuation of a prior project. There's already substantial work here — but importantly, the **prior** experiment measured effective dimensionality / F-ratios / collisions, while the **current** task is a different analysis: information-theoretic mutual information in bits with Miller-Madow correction and permutation nulls, on a **balanced 12-clips/speaker** design. Let me inspect what's reusable before deciding the approach.

Prior work computed 41 features (the "Voice-is-Unique" replication), including glottal-source features via custom DSP. The **current** task is stricter on two fronts that I need to verify: (1) glottal features must be NOT MEASURED unless from a *validated* tool — no fabrication — and (2) a strictly **balanced 12-clips/speaker** design. Let me inspect how features were computed and the actual clips-per-speaker distribution, since those decide whether I reuse or re-extract.

Now I have the full picture. Let me record the key findings before deciding the approach:

1. **Feature extractor reuse**: `features.py` already computes all features the current task wants. But it also computes 14 glottal/inverse-filtering features (GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE, Nasality, SSPF, BGD) via *self-described "best-effort" DSP approximations* — not a validated inverse-filtering tool. The current spec is explicit: those must be **NOT MEASURED** unless a validated tool exists. That's the honest call.

2. **Balanced-design problem**: the existing `features.parquet` is *unbalanced* (1,755 speakers, avg 10.7 clips, median 8). Only **479 speakers** have ≥12 clips — far below the ~1,500 target. So I need to stream more data.

Let me check the cache and prior streaming machinery before finalizing the plan.

I now fully understand the prior setup:
- **Decoding** uses `soundfile` directly on parquet MP3 bytes — no ffmpeg needed (good, since ffmpeg isn't installed).
- 4 shards are cached → only 479 speakers with ≥12 clips. I need ~13–14 shards to approach the ~1,500-speaker target.
- The extractor already computes all 28 *legitimately measured* features; I'll **exclude the 14 glottal/inverse-filtering features as NOT MEASURED** (no validated tool — exactly what the spec mandates).

Decisions I'm making (all resolvable from the spec, so I won't interrupt you):
1. **Glottal/inverse-filtering features → NOT MEASURED** (no validated tool; the prior project's DSP proxies don't qualify under this spec's honesty rule).
2. **Stream more shards** to reach ~1,500 balanced speakers (the spec's target).
3. **Write outputs to a new `mi_experiment/` subdir** so I don't clobber the prior (differently-designed) experiment's artifacts.

Let me set up tracking, kick off the shard downloads in the background, and verify the environment.

Environment is ready. Let me create the task list and kick off the shard downloads in the background while I write the pipeline.

Downloads are running in the background. Let me mark task 1 in-progress and write the pipeline while shards stream. First, the lean extractor that computes **only the 28 legitimately measured features** (reusing the validated routines from `features.py`).

Let me smoke-test the extractor and check download progress.

Extractor produces sensible values. Downloads are progressing (9 shards). Now let me write the Step 1 extraction stage — balanced selection + parallel extraction with reuse of already-computed features.

Downloads finished. Let me verify shard count, then write the core analysis module (the statistically delicate part).

14 shards cached; basename join key is clean (`common_voice_en_NNNNN.mp3`) and matches the prior parquet. Let me launch the balanced extraction in the background and monitor the speaker-count scan.

14 shards cached; basename join key is clean (`common_voice_en_NNNNN.mp3`) and matches the prior parquet. Let me launch the balanced extraction in the background and monitor the speaker-count scan.

Let me check extraction progress, then write the core analysis module.

S=1599 qualifying speakers (10 shards), N≈19,188, ceiling log2(S)≈10.64 bits. Extraction is running (~12k fresh clips, reusing 6,816). Let me update tasks and write the core analysis module while it finishes.

Now the statistically critical part. Let me write the MI estimator core as a separate, unit-testable module, then the analysis driver.

The estimator behaves correctly: independent → I_corrected=0 (perm_p=0.095, not significant); deterministic → recovers ~2 bits (perm_p=0); noisy-informative → 0.98 bits, significant. But the **degenerate binning** collapsed a 2-value feature to 1 bin when it should be 2. The fix: when a feature has ≤ q distinct values, use one bin per distinct value (equal-frequency is impossible there anyway). This only matters for low-cardinality features; continuous ones still get equal-frequency quantiles.

The estimator behaves correctly: independent → I_corrected=0 (perm_p=0.095, not significant); deterministic → recovers ~2 bits (perm_p=0); noisy-informative → 0.98 bits, significant. But the **degenerate binning** collapsed a 2-value feature to 1 bin when it should be 2. The fix: when a feature has ≤ q distinct values, use one bin per distinct value (equal-frequency is impossible there anyway). This only matters for low-cardinality features; continuous ones still get equal-frequency quantiles.

Extraction finished. Let me verify the quantize fix and inspect the extraction outputs.

Extraction is solid: **S=1599, N=19,188, balanced 12/speaker, all 28 measured features ≥99.99% coverage, 14 glottal features correctly logged NOT MEASURED.** The quantize fix works. Strata available: male (938), female (338), US-English (509) — all ≥300.

Let me mark Step 1 done and write the analysis driver (Steps 2–6).

Before the full (~10 min) run, let me smoke-test the analysis plumbing on a small subset with reduced permutations to catch bugs early.

The error is a multiprocessing-via-stdin artifact (spawned children can't re-import `<stdin>`), not a logic bug — the real file-based run is unaffected. Let me write the smoke test as a proper file.

Smoke test works end-to-end (3s for 80 speakers). The math behaves correctly: F0 is the top discriminator, perm_p=0 for real features, and the joint curve saturates as cells outgrow the sample (exactly the sample-limitation the spec warns about). 

One refinement: the cumulative curve can peak then decline (the joint null rises as cells grow), so I should report the **peak** as the saturation estimate, not the value at the stop step. Let me fix that.

Now let me run the full analysis (Steps 2–6) in the background and monitor progress.

While the analysis runs, let me write the report generator (reads all artifacts → `report-cv-quant.md`) and the single-command orchestrator.

Analysis finished fast. Let me check the results.

The analysis ran cleanly (19s total), but the cumulative curve peaks at only 2 features for the pooled set — I want to verify that's a genuine sample-limitation finding and not a bug. Let me inspect the actual numbers.

The method is working correctly. The per-feature results are clean and interpretable: **F0 peaks at b\*=3 (q=8) with 0.887 bits**, then declines as finer bins add only noise — exactly the "usable bit depth" concept. All 28 features are significant (perm_p=0). The cumulative curve peaking at ~2 features is the genuine sample-limitation (the null `I_null_mean` rises to meet `I_mm` as joint cells explode: 0.20→1.02→2.40→6.70) — precisely what the spec anticipates.

Let me verify the figures/strata outputs, then improve the cumulative figure to show *why* it saturates (overlay I_mm vs null), and regenerate.

All artifacts generated correctly (28 per-feature figures, bins.json with 28 features, strata tables). Let me clean the smoke-test leftover and enrich the cumulative figure to show the saturation mechanism (I_mm converging onto the rising null), then regenerate.

Let me write the single-command orchestrator, clean the smoke leftovers, and re-run the analysis.

Analysis regenerated. Let me update tasks and generate the report.

The report is complete and covers all 7 required sections. Now, per the plan (and since ultracode is on), let me run an **adversarial multi-agent review** of the methodology and claims — the MI bias-correction math, the permutation-null invariance, the greedy/saturation logic, spec compliance, and whether the report overclaims. Each finding gets independently verified before I act on it.

The adversarial review is running in the background (4 reviewers × verifiers). While it works, let me do my own sanity check on a subtle point a reviewer is likely to probe — whether the Step 5 complete-case restriction meaningfully unbalances the design.

Confirmed: the complete-case restriction drops exactly **1 utterance** (one clip where F0 failed) — 1598 speakers keep all 12 clips, 1 keeps 11. The balanced design is essentially perfectly preserved, so that's a non-issue. Let me check on the review workflow's progress.

The 4 reviewers are deep into their audits. Let me wait for the workflow to complete (it'll notify me) and check the journal once more.

The pipeline is built and has produced a complete, honest result; the adversarial review workflow is running its 4 reviewers + verifiers in the background. Here's where things stand while I wait for it to finish:

**Built a fresh information-theoretic pipeline** (`mi_experiment/`, kept separate from your prior d_eff replication so nothing was clobbered):

- **Balanced design**: streamed 10 CV17 shards → **1,599 speakers, exactly 12 clips each, N=19,188** (seed 1234). Ceiling H(speaker)=log₂(S)=**10.64 bits**.
- **28 features MEASURED** (Praat F0/F1–5/B1–5/jitter/shimmer/HNR; DSP spectral + α-ratio/LHR/RMS/AMD/CPP/dCPP; VTLE; SpeechRate, SemitoneSDF0), all ≥99.99% coverage. **14 glottal/inverse-filtering features → NOT MEASURED** (no validated tool — I did *not* reuse the prior project's best-effort DSP proxies, per your honesty rule).
- **Headline (bias-corrected MI, never raw)**: best single feature **F0 = 0.887 bits at b\*=3 (q=8)** = 8.3% of ceiling; all 28 features significant (perm_p=0). Most features peak at **b\*=2 (q=4)** — finer bins add noise, not speaker info.
- **Cumulative joint bits**: sample-limited — peaks at **~1.12 bits / 2 features** then declines as the permutation null rises to meet I_mm (12 clips/speaker is too sparse for high-dim joint cells). Reported as an upper-bounded estimate with the log₂(S) caveat, exactly as the spec requires.
- **Stratified**: cohorts sit below pooled (male 0.86, female 0.85, US-Eng 1.05 vs pooled 1.12), consistent with lower effective dimensionality among similar speakers (confound: lower S/ceiling, noted).

I verified the estimator math myself (independent→I_corrected=0; deterministic→recovers bits; perm-null invariance) and confirmed the complete-case restriction drops just 1 utterance. I'll address any confirmed findings from the adversarial review as soon as it returns, then finalize.

The review returned 6 confirmed findings. Let me read the full output to see all of them.

Excellent — this is exactly why the adversarial pass was worth running. The reviewers (with independent verification) found one **major** honesty issue and several minor/nit robustness gaps. Let me triage:

| # | Severity | Finding | Action |
|---|----------|---------|--------|
| 1 | **Major** | Section 5 says cohort bit-drops "confirm" the homogeneity prediction, but it's confounded by S/ceiling — and ceiling-normalized NMI actually contradicts it (US *exceeds* pooled) | **Add matched-S random-subsample control + report NMI + honest language** |
| 2 | Minor | `I_corrected` subtracts plug-in null from MM estimate (conservative ~20% understatement) | Spec mandates this formula — keep headline, but add MM-consistent diagnostic column + docstring note |
| 3 | Minor | `mi_metrics` crashes if a feature is all-NaN in a cohort (N=0) — latent, never fires now | Guard N==0 / q_eff≤1 |
| 4 | Nit | `quantize` q_eff can overstate occupied bins (point mass on edge) — latent | Densify → faithful q_eff |
| 5 | Nit | Step-5 stop rule is a documented reinterpretation | Add note; results robust |
| 6 | Nit | NMI divides by log2(full S) ignoring NaN-dropped speakers — zero impact now | Use occupied-speaker count |

Finding #1 is the important one. Let me fix the code (the matched-S control directly de-confounds the homogeneity claim). Starting with `mi_core.py` (#2, #3, #4):

Let me verify the core still passes its self-tests, then update the analysis driver.

Self-tests pass. Now the analysis driver: wire the new diagnostic columns, fix the NMI denominator (#6), and — the important one — refactor the greedy into a reusable function so I can add **matched-S random-subsample controls** (#1).

Now refactor the Step 5 greedy into a reusable function (so the matched-S controls can call it) and add the diagnostic NMI column. Let me replace the Step 5 block in `run_cohort`:

Now replace the old inline Step 5 block in `run_cohort` with a call to the new function:

Now add the matched-S random-subsample control to `main()` and surface NMI. Let me read the current `main()` to edit precisely.

Now re-run the full analysis with the controls and diagnostics:

Analysis re-ran with identical pooled/strata numbers (confirming the core changes were behavior-preserving on this data). Now the key new evidence — the matched-S control comparison:

This is a clean, decisive result. The **matched-S control** completely resolves the major finding: comparing ceiling-normalized NMI of each homogeneous cohort against equal-size *random* subsamples:

| Cohort | Homogeneous NMI | Random control NMI (±sd) | Below control? |
|---|---|---|---|
| sex:male | 0.087 | 0.114 ± 0.0006 | **yes, ~24%** |
| sex:female | 0.102 | 0.133 ± 0.0022 | **yes, ~24%** |
| accent:US | 0.116 | 0.126 ± 0.0022 | **yes, ~7%** |

All three homogeneous cohorts fall **below** their size-matched random controls (far outside the tiny control SD) — so the "low effective dimensionality among similar speakers" prediction is now **confirmed in a de-confounded way**, fixing the earlier overclaim (which had wrongly compared against higher-S pooled). The mmnull diagnostic shows the conservative headline understates per-feature bits by ~20-25% (F0 0.887→1.111).

Let me rewrite the report's Section 5 (and add the mmnull + stop-rule notes). Let me read the current report sections to edit.

Now add the mmnull transparency note in Section 3 and the stop-rule note in Section 4. Let me view those parts of the report builder:

Now make the three report edits. First, the mmnull transparency note in Section 3:

Now rewrite Section 5 with the de-confounded matched-S control as the centerpiece:

Let me add the new control-comparison artifact to the manifest, then regenerate the report:

Section 5 is now rigorous and de-confounded. Let me validate the single-command orchestrator end-to-end (`run_mi.py --skip-extract` chains analyze + report), confirming the deliverable runs as one command.

The single-command pipeline runs clean end-to-end and is fully deterministic. Let me do a final focused adversarial check on the *fix* itself (the revised Section 5 + control code), and review the report's limitations/manifest in parallel.

The limitations are solid. The final verification agent is running. Let me confirm the deliverables directory is clean and complete while I wait.

All deliverables are present and cleanly organized (6 scripts, balanced features.parquet, all spec'd CSVs/JSON, bins.json, 33 figures, per-cohort tables, the new control comparison). The verification agent is auditing the fix in the background. Let me record this new experiment in memory while I wait (it's a distinct analysis from the prior d_eff replication).

Memory recorded. Everything is built, run, and reproducible; I'm now waiting on the final verification agent that's adversarially auditing the Section 5 fix (the matched-S control claim and code). I'll address anything it surfaces, then give you the wrap-up. Standing by for its result.

The verification confirms the fix is substantively correct — control code is bug-free, all numbers match, and the de-confounded claim holds 3/3. It caught **one residual wording overclaim**: "many control-SDs below the control mean" overstates strength because the 5 control draws overlap heavily (~59% for male), making the SD artificially tiny. That's a fair, precise catch — the directional NMI gaps carry the claim without the misleading SD-distance framing. Let me fix the wording.

The wording is fixed (no more "control-SDs" significance framing; the heavy-overlap caveat is now explicit). All six review findings plus the residual wording catch are resolved. Let me do a final consistency check and confirm the workspace is clean.

Everything checks out: self-tests pass, stray files are gone, all 7 report sections and key numbers are intact. Let me mark the final task complete.

The experiment is complete, reviewed, and hardened. Here's the wrap-up.

## What I built

A fresh quantization-based, information-theoretic voice-individuality pipeline in **`mi_experiment/`** (kept separate from your prior d_eff replication so nothing was clobbered). One command reproduces everything: `python mi_experiment/run_mi.py`.

**Data (Mode B, balanced):** streamed 10 of `fixie-ai/common_voice_17_0` English-validated shards (MP3 decoded via soundfile, no ffmpeg) → **1,599 speakers × exactly 12 clips = 19,188 utterances**, seed 1234. Ceiling **H(speaker)=log₂(S)=10.64 bits**.

**Honesty on features:** 28 features MEASURED (≥99.99% coverage); the **14 glottal/inverse-filtering features (NAQ, CQ, GCT, SHR, GNE, …, VOT, BGD) logged NOT MEASURED** — no validated inverse-filtering toolkit exists here, and I deliberately did *not* reuse the prior project's best-effort DSP proxies for them.

**Headline results (bias-corrected MI in bits, never raw):**
- Best single feature **F0 = 0.887 bits at b\*=3 (q=8)** = 8.3% of the ceiling; all 28 features significant (perm_p < 0.005). Most features peak at b\*=2 — finer quantization adds noise, not speaker info.
- **Cumulative joint bits are sample-limited**: the curve peaks at ~1.12 bits / 2 features then declines as the permutation null rises to meet I_mm (12 clips/speaker can't densely fill high-dim joint cells). Reported as an upper-bounded estimate with the log₂(S) caveat, exactly as specified.
- **Homogeneous-cohort prediction confirmed (de-confounded)**: every cohort (male/female/US-English) yields fewer ceiling-normalized usable bits than a **size-matched random speaker set** (male 0.087 vs 0.114, female 0.102 vs 0.133, US 0.116 vs 0.126).

## Verification

I ran a 4-dimension adversarial review workflow (estimator math, pipeline, spec compliance, report honesty) with independent per-finding verifiers. It surfaced **one major issue** — Section 5 originally "confirmed" the homogeneity prediction by comparing cohort bits to the *higher-S pooled* set, a sample-size confound (ceiling-normalized NMI actually showed US *exceeding* pooled). I fixed it properly by **adding a matched-S random-subsample control** that isolates homogeneity from S. A second agent verified the fix and caught a residual wording overclaim ("many control-SDs below") — fixed, since the 5 overlapping draws make the SD artificially tiny. Minor/nit findings (N=0 guard, faithful q_eff, MM-consistent-null diagnostic column, occupied-speaker NMI denominator, stop-rule note) are all addressed.

## Deliverables (all in `mi_experiment/`)

[report-cv-quant.md](mi_experiment/report-cv-quant.md) · `features.parquet` · `coverage.csv` · `bins.json` · `mi_by_feature_bit.csv` · `usable_bits.csv` · `cumulative_bits.csv` · per-cohort tables · `artifacts/stratified_{usable_comparison,saturation,control_comparison}.csv` · 33 figures (`figs/mi_<feature>.png`, `cumulative_bits.png`, overlay) · pipeline scripts (`mi_features/mi_extract/mi_core/mi_analyze/mi_report/run_mi.py`).

One honest framing to keep in mind: the **per-feature usable-bits (Step 4) is the robust result**; the **joint/cumulative bits (Step 5) is the most sample-limited part** — the 12-clips/speaker balanced design inherently caps how deep the joint estimate can go, which the report states plainly.