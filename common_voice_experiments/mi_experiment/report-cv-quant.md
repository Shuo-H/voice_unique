# Quantization-Based Information-Theoretic Voice Individuality on Common Voice 17

**Headline metric:** bias-corrected mutual information (bits) between speaker identity and per-utterance feature *quantization bin*, above a permutation null. Raw plug-in MI is reported only as a diagnostic, never as the result.

- **Speakers S = 1599**  |  **Utterances N = 19188** (balanced: exactly 12 clips/speaker)
- **Absolute MI ceiling H(speaker) = log2(S) = 10.643 bits** (uniform over speakers by construction)
- Random seed **1234** everywhere (numpy default_rng; all shuffles, bootstraps, subsampling). Permutation null = 200 shuffles per estimate.
- Data: `fixie-ai/common_voice_17_0` English `validated` (public parquet mirror of CV 17.0), 10 shards streamed; MP3 decoded via soundfile (libsndfile, no ffmpeg), resampled to 16 kHz mono.

## 1. Data summary

Grouping clips by `client_id` (**assumed = one speaker**; stated as a limitation). Kept only speakers with >= 12 validated, decodable clips, then **randomly sampled exactly 12 clips/speaker (seed 1234)** for a balanced design. Reached 1599 speakers (target ~1500).

- Reused 6816 already-extracted clips + extracted 12372 fresh; features never imputed (NaN on failure, counted missing).

**Demographics (self-reported, uneven):**

- Sex (speakers): male_masculine=938, female_feminine=338, NaN=323
- Age (speakers): twenties=543, thirties=302, NaN=286, fourties=152, teens=139, fifties=96, sixties=60, seventies=17, eighties=4
- Accent (top): NaN=549, United States English=509, England English=182, India and South Asia (India, Pakistan, Sri Lanka)=98, Canadian English=81, Australian English=53, Southern African (South Africa, Zimbabwe, Namibia)=23, New Zealand English=16

### Feature coverage (measured vs NOT MEASURED)

**28 features MEASURED** (coverage >= 90% required; all listed features clear it):

| feature | group | coverage | status |
| --- | --- | --- | --- |
| F0 | measured | 1.000 | measured |
| F1 | measured | 1.000 | measured |
| F2 | measured | 1.000 | measured |
| F3 | measured | 1.000 | measured |
| F4 | measured | 1.000 | measured |
| F5 | measured | 1.000 | measured |
| B1 | measured | 1.000 | measured |
| B2 | measured | 1.000 | measured |
| B3 | measured | 1.000 | measured |
| B4 | measured | 1.000 | measured |
| B5 | measured | 1.000 | measured |
| jitter | measured | 1.000 | measured |
| shimmer | measured | 1.000 | measured |
| HNR | measured | 1.000 | measured |
| SpectralSkewness | measured | 1.000 | measured |
| SpectralKurtosis | measured | 1.000 | measured |
| SpectralEntropy | measured | 1.000 | measured |
| SpectralRolloff | measured | 1.000 | measured |
| SpectralFlux | measured | 1.000 | measured |
| AlphaRatio | measured | 1.000 | measured |
| LHR | measured | 1.000 | measured |
| RMS | measured | 1.000 | measured |
| AMD | measured | 1.000 | measured |
| CPP | measured | 1.000 | measured |
| dCPP | measured | 1.000 | measured |
| VTLE | measured | 1.000 | measured |
| SpeechRate | measured | 1.000 | measured |
| SemitoneSDF0 | measured | 1.000 | measured |

**14 features NOT MEASURED (0 coverage) — logged, not fabricated:**

> The glottal-source / inverse-filtering family (GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE, Nasality, SSPF, VOT, BGD) requires a *validated* glottal inverse-filtering toolkit (e.g. COVAREP / Aparat / a validated IAIF implementation), which is not available in this environment. Per the experiment's honesty rule these are reported as NOT MEASURED with 0 coverage rather than approximated with best-effort DSP. VOT additionally requires forced alignment (unavailable for Common Voice).

## 2. Method (per feature, per bit depth b in {1..8}, q = 2^b)

1. **Quantize** per-utterance values with q-quantile (equal-frequency) edges over the pooled distribution, so bins are marginally equiprobable. Degenerate/duplicate edges merged; effective bin count **q_eff(b)** recorded (`bins.json`). Low-cardinality features use one bin per distinct value.
2. **I_raw** = plug-in MI = H(spk)+H(bin)-H(spk,bin) in bits (upward biased).
3. **I_mm** = Miller-Madow: each entropy gets +(K-1)/(2N), K = occupied cells.
4. **Permutation null** (200x, seed 1234): shuffle the speaker column across all N utterances, recompute plug-in MI. Under a label shuffle the speaker- and bin-marginals are invariant, so only H(spk,bin) changes. Gives I_null_mean, I_null_p95, and perm_p = fraction(null MI >= I_raw).
5. **HEADLINE I_corrected = max(0, I_mm - I_null_mean)** [bits above chance]; **NMI_corrected = I_corrected / log2(S)**.

> Note: I_corrected subtracts *both* the Miller-Madow analytic bias term *and* the empirical permutation-null floor. This is deliberately **conservative** (it can double-subtract bias), so absolute I_corrected is a lower-leaning estimate; the permutation p-value certifies significance independently.

## 3. Per-feature usable bit depth (Step 4)

`b* = argmax_b I_corrected` — the depth past which finer bins add noise, not speaker information. Sorted by corrected bits (descending).

| feature | b_star | q_eff | I_corrected | NMI_corrected | perm_p |
| --- | --- | --- | --- | --- | --- |
| F0 | 3 | 8 | 0.887 | 0.083 | 0.000 |
| RMS | 3 | 8 | 0.732 | 0.069 | 0.000 |
| CPP | 2 | 4 | 0.679 | 0.064 | 0.000 |
| dCPP | 2 | 4 | 0.592 | 0.056 | 0.000 |
| HNR | 2 | 4 | 0.572 | 0.054 | 0.000 |
| AlphaRatio | 2 | 4 | 0.563 | 0.053 | 0.000 |
| LHR | 2 | 4 | 0.538 | 0.051 | 0.000 |
| SpectralRolloff | 2 | 4 | 0.512 | 0.048 | 0.000 |
| SpectralEntropy | 2 | 4 | 0.497 | 0.047 | 0.000 |
| SpectralKurtosis | 2 | 4 | 0.491 | 0.046 | 0.000 |
| VTLE | 2 | 4 | 0.479 | 0.045 | 0.000 |
| F4 | 2 | 4 | 0.479 | 0.045 | 0.000 |
| SpectralSkewness | 2 | 4 | 0.469 | 0.044 | 0.000 |
| F5 | 2 | 4 | 0.468 | 0.044 | 0.000 |
| shimmer | 2 | 4 | 0.451 | 0.042 | 0.000 |
| SpectralFlux | 2 | 4 | 0.450 | 0.042 | 0.000 |
| B1 | 2 | 4 | 0.426 | 0.040 | 0.000 |
| F3 | 2 | 4 | 0.407 | 0.038 | 0.000 |
| B4 | 2 | 4 | 0.397 | 0.037 | 0.000 |
| F1 | 2 | 4 | 0.389 | 0.037 | 0.000 |
| B2 | 2 | 4 | 0.364 | 0.034 | 0.000 |
| B5 | 2 | 4 | 0.352 | 0.033 | 0.000 |
| B3 | 2 | 4 | 0.348 | 0.033 | 0.000 |
| F2 | 2 | 4 | 0.343 | 0.032 | 0.000 |
| jitter | 2 | 4 | 0.323 | 0.030 | 0.000 |
| SemitoneSDF0 | 2 | 4 | 0.236 | 0.022 | 0.000 |
| AMD | 2 | 4 | 0.197 | 0.018 | 0.000 |
| SpeechRate | 2 | 4 | 0.131 | 0.012 | 0.000 |

**Top features:** F0 (0.887 bits, b*=3), RMS (0.732 bits, b*=3), CPP (0.679 bits, b*=2), dCPP (0.592 bits, b*=2), HNR (0.572 bits, b*=2).
28/28 measured features carry significant speaker information (perm_p < 0.05; all 200-shuffle nulls fall below I_raw, so perm_p < 1/200 = 0.005). Best single feature = F0 at 0.887 bits = 8.3% of the 10.64-bit ceiling. Per-feature MI-vs-b curves: `figs/mi_<feature>.png`.

> **Conservatism check.** The headline I_corrected subtracts the *plug-in* permutation null from the *Miller-Madow* point estimate, which can double-count bias and lean the absolute bits **low**. A self-consistent variant subtracting a Miller-Madow-corrected null (column `I_corrected_mmnull`) is ~20-25% higher (e.g. F0: 0.887 -> 1.111 bits). Significance (perm_p) is computed self-consistently (plug-in I_raw vs plug-in null) and is unaffected. Reported headline bits are therefore a lower-leaning estimate.

## 4. Joint / cumulative usable bits (Step 5)

Greedy forward selection at fixed **b=2 (q=4 bins/feature)**, each step adding the feature that maximizes the joint-bin I_corrected (same Miller-Madow + permutation-null correction on the joint contingency table). Stop rule: marginal corrected gain <= the permutation noise band (I_null_p95 - I_null_mean).

> **Stop-rule note.** 'Permutation p95' is operationalized as the joint null's noise band (I_null_p95 - I_null_mean) at each step; the literal alternative (a permutation null of the step-to-step *gain*) and the simpler gain<=0 rule both fire at the same step here, and the reported peak is robust to the choice.

| step | feature | cum_I_corrected | marginal_gain | I_null_mean | I_null_p95 | perm_p | q_eff_joint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | F0 | 0.874 | 0.874 | 0.200 | 0.207 | 0.000 | 4 |
| 2 | RMS | 1.121 | 0.247 | 1.016 | 1.026 | 0.000 | 16 |
| 3 | CPP | 0.825 | -0.295 | 2.399 | 2.405 | 0.000 | 64 |
| 4 | dCPP | 0.525 | -0.300 | 3.498 | 3.503 | 0.000 | 224 |
| 5 | HNR | 0.215 | -0.310 | 5.160 | 5.163 | 0.000 | 782 |
| 6 | SpectralEntropy | 0.031 | -0.184 | 6.701 | 6.703 | 0.000 | 2301 |

**Saturation (peak) = 1.121 corrected bits at 2 features**, vs the **log2(S) = 10.643 bit ceiling**. See `figs/cumulative_bits.png`.

> **Ceiling caveat (required):** cumulative corrected MI cannot exceed log2(S). As joint cells grow (q_eff_joint above), the permutation null I_null_mean rises and the corrected curve peaks then declines — so the saturation point is **partly sample-limited**, not a population constant. It is an **upper-bounded ESTIMATE of usable joint speaker bits over the measured features**, not a measured population value. With more speakers (higher ceiling) and more clips/speaker (denser joint cells) the peak would move right and up.

## 5. Stratified / homogeneous-cohort analysis (Step 6)

Steps 3-5 re-run within each sex stratum and each accent group with >= 300 speakers. The framework predicts **lower** effective dimensionality (fewer *usable* bits) among acoustically similar speakers.

**Cumulative-bits saturation by cohort** (NMI = saturation_bits / log2(S), the ceiling-normalized usable joint information):

| cohort | S | logS_ceiling | saturation_features | saturation_bits | saturation_NMI | stop_step |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | 1599 | 10.643 | 2 | 1.121 | 0.105 | 3 |
| sex:male | 938 | 9.873 | 2 | 0.860 | 0.087 | 3 |
| sex:female | 338 | 8.401 | 2 | 0.854 | 0.102 | 3 |
| accent:US | 509 | 8.992 | 2 | 1.046 | 0.116 | 3 |

> **Why absolute bits cannot be compared to pooled directly.** Each cohort has fewer speakers (smaller S) than pooled, hence a lower log2(S) ceiling, smaller N = 12*S, and a relatively higher joint permutation-null floor (the same Section-4 sample-capping confound). A smaller *random* speaker set would show lower absolute saturation bits too — so a raw drop vs pooled does NOT isolate acoustic homogeneity, and on raw absolute bits US (1.05) even approaches pooled (1.12). The valid test is a **size-matched random control**.

**Matched-S control (the de-confounded test):** for each cohort we drew 5 random speaker subsets of the SAME size from the full pool (seed 1234) and ran the identical greedy cumulative analysis. Comparison on ceiling-normalized NMI:

| cohort | S | homogeneous_NMI | ctrl_NMI | homogeneous_below_control |
| --- | --- | --- | --- | --- |
| sex:male | 938 | 0.087 | 0.114 ± 0.001 | True |
| sex:female | 338 | 0.102 | 0.133 ± 0.002 | True |
| accent:US | 509 | 0.116 | 0.126 ± 0.002 | True |

**All 3/3 homogeneous cohorts fall BELOW their size-matched random controls** in ceiling-normalized usable joint bits (sex:male -24%, sex:female -24%, accent:US -8%). Controlling for sample size, acoustically homogeneous cohorts therefore yield **fewer usable speaker bits than a random speaker set of equal size** — a de-confounded confirmation of the low-effective-dimensionality-among-similar-speakers prediction. (Naively comparing cohort absolute bits to the larger-S pooled set would have been confounded and is NOT the basis for this claim.)

> The N=5 control draws overlap heavily (e.g. the male control resamples 938 of 1599 speakers each time, ~59% expected overlap), so the reported control SD understates true sampling variability and is **not** used as a significance figure; the claim rests on the directional NMI gaps, robust across all three cohorts (US -8% is the weakest leg).

**Per-feature usable corrected bits, pooled vs cohorts (top features):**

| feature | Ic_pooled | Ic_sex:male | Ic_sex:female | Ic_accent:US |
| --- | --- | --- | --- | --- |
| F0 | 0.887 | 0.488 | 0.607 | 0.858 |
| RMS | 0.732 | 0.683 | 0.682 | 0.657 |
| CPP | 0.679 | 0.545 | 0.427 | 0.657 |
| dCPP | 0.592 | 0.522 | 0.480 | 0.596 |
| HNR | 0.572 | 0.455 | 0.436 | 0.596 |
| AlphaRatio | 0.563 | 0.590 | 0.486 | 0.562 |
| LHR | 0.538 | 0.545 | 0.524 | 0.547 |
| SpectralRolloff | 0.512 | 0.492 | 0.544 | 0.487 |
| SpectralEntropy | 0.497 | 0.491 | 0.482 | 0.478 |
| SpectralKurtosis | 0.491 | 0.492 | 0.480 | 0.467 |
| VTLE | 0.479 | 0.435 | 0.420 | 0.488 |
| F4 | 0.479 | 0.398 | 0.418 | 0.504 |

Note F0's usable bits drop sharply within sex strata (pooled 0.887 -> male 0.488, female 0.607): once sex is fixed, much of F0's speaker information is gone, consistent with sex explaining a large share of pitch variance. Overlay of all cumulative curves: `figs/cumulative_bits_overlay.png`.

## 6. Honest limitations

1. **MP3 compression** degrades high-frequency and source-periodicity features (jitter, shimmer, CPP/dCPP, spectral flux, rolloff): their absolute bits are biased **low** and the absolute scale is **not comparable** to clean-audio corpora. Relative structure (ordering of features, cohort contrasts) is more robust than absolute bits.
2. **Finite-sample MI bias.** Plug-in MI is upward-biased; Miller-Madow + the permutation null are applied precisely because of this, but **residual bias still inflates absolute bits**, especially for high b and for the joint table.
3. **`client_id` = speaker** is assumed (one account = one speaker); mislabeling would inflate apparent individuality.
4. **log2(S) = 10.64 bit ceiling** caps all (especially joint/cumulative) bits; the cumulative saturation is sample-capped (Section 4 caveat).
5. **SpeechRate is a proxy** (syllable-nucleus rate from the energy/voicing envelope; no forced alignment is available for Common Voice).
6. **Self-reported, uneven demographic metadata** (large NaN fractions for sex/age/accent); strata are imbalanced.
7. **14 glottal/inverse-filtering features are NOT MEASURED** (no validated tool), so the source-related dimension of voice individuality is under-sampled here; absolute joint bits would likely be higher with a validated glottal toolkit.

## 7. File manifest

- `mi_experiment/features.parquet` — long-format per-utterance features (balanced)
- `mi_experiment/coverage.csv` — per-feature coverage, measured vs NOT MEASURED
- `mi_experiment/bins.json` — quantization edges + q_eff per (feature, b)
- `mi_experiment/mi_by_feature_bit.csv` — Step 3: I_raw/I_mm/null/perm_p/I_corrected/NMI per (feature,b)
- `mi_experiment/usable_bits.csv` — Step 4: b*, q_eff, I_corrected, NMI, perm_p (sorted)
- `mi_experiment/cumulative_bits.csv` — Step 5: greedy joint cumulative corrected bits
- `mi_experiment/mi_by_feature_bit_<cohort>.csv / usable_bits_<cohort>.csv / cumulative_bits_<cohort>.csv` — Step 6: per-stratum tables (sex:male, sex:female, accent:US)
- `mi_experiment/artifacts/stratified_usable_comparison.csv` — pooled vs cohort per-feature bits
- `mi_experiment/artifacts/stratified_saturation.csv` — cohort cumulative saturation + ceilings + NMI
- `mi_experiment/artifacts/stratified_control_comparison.csv` — homogeneous cohort vs matched-S random control (NMI)
- `mi_experiment/artifacts/dataset_summary.json / analysis_summary.json` — run metadata
- `mi_experiment/artifacts/speaker_manifest.csv / selection.csv` — retained speakers + chosen 12 clips each
- `mi_experiment/figs/mi_<feature>.png` — per-feature I_corrected/I_raw/I_null vs b
- `mi_experiment/figs/cumulative_bits.png + _overlay.png` — cumulative curve + cohort overlay
- `mi_experiment/{mi_features,mi_extract,mi_core,mi_analyze,mi_report,run_mi}.py` — the runnable pipeline

*Pooled analysis wall-clock: 31.3 s. Reproduce end-to-end: `python mi_experiment/run_mi.py`.*
