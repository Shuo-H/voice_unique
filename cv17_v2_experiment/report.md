# Common Voice 17 — 40-feature distinctiveness battery (v2)

_Fixed seed **1234**. Generated 2026-06-13 12:12. Within-speaker variance is genuinely multi-session/multi-channel; F-ratios are realistic (if anything conservative), not optimistic upper bounds._

## 0. Corpus, provenance, and scale

**Data / mirror.** Common Voice 17.0, English, `validated` split. The official `mozilla-foundation/common_voice_17_0` repo was emptied (Oct 2025) and moved to the gated Mozilla Data Collective, so we used the public non-gated parquet mirror **`fixie-ai/common_voice_17_0`**, which preserves the official schema (`client_id, path, audio, sentence, up_votes, down_votes, age, gender, accent, locale, segment, variant`). MP3 decoded via soundfile/libsndfile, resampled to **16 kHz mono**. **`client_id` is the speaker label.**

**Scale (scaled up from the prior ~1,755-speaker run).** Pooled **24 `en/validated` parquet shards**; scanned **51,330 distinct client_ids** / **312,936 clips**. Speaker filter: kept client_ids with **≥ 5 validated clips**, capped at **30 clips/speaker** (seeded subsample). **Final: 10,676 speakers / 137,675 clips.**

Clips/speaker (kept): min 5, median 9, mean 12.9, max 30. Speakers with ≥10 clips (classifier-eligible): **5105**.

**Per-speaker metadata distributions (modal label per speaker):**

- **Sex/gender:** {'male_masculine': 4801, 'NaN': 4294, 'female_feminine': 1581}
- **Age buckets:** {'NaN': 4144, 'twenties': 2777, 'thirties': 1428, 'teens': 763, 'fourties': 683, 'fifties': 480, 'sixties': 285, 'seventies': 100, 'eighties': 13, 'nineties': 3}
- **Top accents:** {'NaN': 5588, 'United States English': 2581, 'England English': 811, 'India and South Asia (India, Pakistan, Sri Lanka)': 450, 'Canadian English': 376, 'Australian English': 268, 'Southern African (South Africa, Zimbabwe, Namibia)': 99, 'Scottish English': 71, 'New Zealand English': 71, 'Irish English': 65, 'Filipino': 44, 'Hong Kong English': 32}

> **Multi-session caveat.** CV is crowd-sourced: a speaker's clips span different devices, rooms, and days, so within-speaker variance is genuinely multi-session/multi-channel. Unlike TIMIT (single-session read sentences), the F-ratios below are **realistic, not optimistic upper bounds** — if anything conservative.

## 1. Feature coverage — measured out of 40

**VTLE is excluded entirely** (not a feature in v2). **VOT is NOT MEASURED** (no phone alignments on CV). Features are **never imputed**; failures are NaN. Coverage = fraction of utterances with a successfully-computed value.

**Measured (coverage ≥ 80%, used downstream): 39 of 40.**
**Not measured (coverage < 80%): VOT (0.00).**

| # | feature | group | coverage | status |
|---|---|---|---:|---|
| 1 | F0 | glottal_source | 1.000 | measured |
| 2 | jitter | glottal_source | 1.000 | measured |
| 3 | shimmer | glottal_source | 1.000 | measured |
| 4 | GCT | glottal_source | 1.000 | measured |
| 5 | CQ | glottal_source | 1.000 | measured |
| 6 | MFDR | glottal_source | 1.000 | measured |
| 7 | SQ | glottal_source | 1.000 | measured |
| 8 | NAQ | glottal_source | 1.000 | measured |
| 9 | SHR | glottal_source | 1.000 | measured |
| 10 | IHI | glottal_source | 0.998 | measured |
| 11 | VFP | glottal_source | 1.000 | measured |
| 12 | semitone_SD_F0 | glottal_source | 1.000 | measured |
| 13 | F1 | vocal_tract_filter | 1.000 | measured |
| 14 | F2 | vocal_tract_filter | 1.000 | measured |
| 15 | F3 | vocal_tract_filter | 1.000 | measured |
| 16 | F4 | vocal_tract_filter | 1.000 | measured |
| 17 | F5 | vocal_tract_filter | 1.000 | measured |
| 18 | B1 | vocal_tract_filter | 1.000 | measured |
| 19 | B2 | vocal_tract_filter | 1.000 | measured |
| 20 | B3 | vocal_tract_filter | 1.000 | measured |
| 21 | B4 | vocal_tract_filter | 1.000 | measured |
| 22 | B5 | vocal_tract_filter | 1.000 | measured |
| 23 | Nasality | vocal_tract_filter | 1.000 | measured |
| 24 | spectral_skewness | spectral_envelope | 1.000 | measured |
| 25 | spectral_kurtosis | spectral_envelope | 1.000 | measured |
| 26 | spectral_entropy | spectral_envelope | 1.000 | measured |
| 27 | spectral_rolloff | spectral_envelope | 1.000 | measured |
| 28 | spectral_flux | spectral_envelope | 1.000 | measured |
| 29 | alpha_ratio | spectral_envelope | 1.000 | measured |
| 30 | LHR | spectral_envelope | 1.000 | measured |
| 31 | SPI | spectral_envelope | 1.000 | measured |
| 32 | GNE | spectral_envelope | 1.000 | measured |
| 33 | SSPF | spectral_envelope | 0.886 | measured |
| 34 | CPP | articulatory_prosodic | 1.000 | measured |
| 35 | dCPP | articulatory_prosodic | 1.000 | measured |
| 36 | RMS | articulatory_prosodic | 1.000 | measured |
| 37 | AMD | articulatory_prosodic | 1.000 | measured |
| 38 | speech_rate | articulatory_prosodic | 1.000 | measured |
| 39 | VOT | articulatory_prosodic | 0.000 | NOT MEASURED |
| 40 | BGD | articulatory_prosodic | 1.000 | measured |
| – | HNR (aux) | aux_HNR | 1.000 | measured |

## 2. Population distributions & quantile bins

Across-speaker distributions are built from per-speaker means. Equiprobable q-quantile bin edges for q ∈ {2,3,5,10} are in `bins.json`; collapsed (degenerate) bins are logged in `artifacts/bin_degeneracy.csv`. With thousands of distinct per-speaker means, equiprobable edges are non-degenerate for the continuous features at all q; any collapse is noted there.

## 3. F-ratios and usable resolution — POOLED and WITHIN-SEX

`within_var` = mean over speakers of within-speaker variance; `between_var` = variance of per-speaker means; `F_ratio` = between/within; one-way ANOVA across speakers; `q_max` = largest q∈{2,3,5,10} with mean bin-crossing rate < 0.20. Within-sex uses CV `gender` (NaN-gender dropped for that computation only); `q_max(within)` = min over the two sexes. Sorted by pooled F_ratio.

| feature | within_var | between_var | F(pooled) | q_max(pool) | F(male) | F(female) | q_max(within) | ANOVA F | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 490.242 | 1699.954 | 3.47 | 3 | 1.45 | 2.18 | 2 | 44.6 | 0.0e+00 |
| RMS | 0.000 | 0.001 | 3.28 | 2 | 2.66 | 3.30 | 2 | 35.8 | 0.0e+00 |
| CPP | 0.008 | 0.022 | 2.69 | 2 | 2.11 | 1.78 | 2 | 33.7 | 0.0e+00 |
| alpha_ratio | 6.995 | 16.624 | 2.38 | 2 | 2.39 | 1.94 | 2 | 27.9 | 0.0e+00 |
| dCPP | 0.000 | 0.001 | 2.35 | 2 | 1.98 | 1.93 | 2 | 28.9 | 0.0e+00 |
| LHR | 15.953 | 36.764 | 2.30 | 2 | 2.18 | 2.28 | 2 | 27.5 | 0.0e+00 |
| spectral_rolloff | 207423.358 | 462790.186 | 2.23 | 2 | 2.07 | 2.12 | 2 | 26.2 | 0.0e+00 |
| SPI | 9.470 | 20.677 | 2.18 | 2 | 2.14 | 1.86 | 2 | 26.1 | 0.0e+00 |
| GCT | 0.515 | 1.050 | 2.04 | 2 | 1.27 | 1.44 | 2 | 25.2 | 0.0e+00 |
| spectral_flux | 0.001 | 0.001 | 1.97 | 2 | 1.89 | 1.73 | 1 | 22.6 | 0.0e+00 |
| spectral_entropy | 0.001 | 0.002 | 1.96 | 2 | 1.92 | 1.72 | 2 | 23.3 | 0.0e+00 |
| F4 | 15328.004 | 26691.067 | 1.74 | 2 | 1.45 | 1.63 | 1 | 20.9 | 0.0e+00 |
| F5 | 17324.960 | 29809.081 | 1.72 | 2 | 1.72 | 1.24 | 1 | 20.0 | 0.0e+00 |
| B4 | 4783.055 | 8058.858 | 1.68 | 2 | 1.52 | 1.72 | 1 | 19.6 | 0.0e+00 |
| shimmer | 0.000 | 0.001 | 1.59 | 2 | 1.22 | 1.51 | 1 | 19.1 | 0.0e+00 |
| F3 | 14918.562 | 23615.571 | 1.58 | 2 | 1.36 | 1.58 | 1 | 18.1 | 0.0e+00 |
| CQ | 0.004 | 0.006 | 1.52 | 2 | 1.50 | 1.16 | 1 | 17.8 | 0.0e+00 |
| F2 | 12505.324 | 18634.452 | 1.49 | 2 | 1.43 | 1.25 | 1 | 17.1 | 0.0e+00 |
| B1 | 6390.693 | 9349.765 | 1.46 | 2 | 1.39 | 1.23 | 1 | 17.6 | 0.0e+00 |
| B5 | 10340.183 | 14180.428 | 1.37 | 2 | 1.32 | 1.14 | 1 | 15.7 | 0.0e+00 |
| B3 | 5251.247 | 7125.869 | 1.36 | 1 | 1.29 | 1.30 | 1 | 15.8 | 0.0e+00 |
| F1 | 13839.100 | 18272.520 | 1.32 | 1 | 1.31 | 1.06 | 1 | 15.3 | 0.0e+00 |
| B2 | 5822.355 | 7605.793 | 1.31 | 1 | 1.19 | 1.32 | 1 | 15.5 | 0.0e+00 |
| VFP | 0.003 | 0.003 | 1.22 | 1 | 1.15 | 1.12 | 1 | 12.8 | 0.0e+00 |
| Nasality | 22.171 | 24.611 | 1.11 | 1 | 1.01 | 1.16 | 1 | 13.1 | 0.0e+00 |
| jitter | 0.000 | 0.000 | 1.06 | 1 | 0.87 | 0.86 | 1 | 12.9 | 0.0e+00 |
| semitone_SD_F0 | 2.797 | 2.287 | 0.82 | 1 | 0.82 | 0.59 | 1 | 9.6 | 0.0e+00 |
| NAQ | 0.002 | 0.002 | 0.80 | 1 | 0.77 | 0.47 | 1 | 9.6 | 0.0e+00 |
| AMD | 0.056 | 0.043 | 0.75 | 1 | 0.66 | 0.67 | 1 | 8.6 | 0.0e+00 |
| speech_rate | 0.439 | 0.306 | 0.70 | 1 | 0.63 | 0.64 | 1 | 7.8 | 0.0e+00 |
| IHI | 0.000 | 0.000 | 0.68 | 1 | 0.38 | 0.40 | 1 | 8.4 | 0.0e+00 |
| SQ | 4.352 | 2.759 | 0.63 | 1 | 0.57 | 0.59 | 1 | 7.4 | 0.0e+00 |
| MFDR | 0.004 | 0.002 | 0.61 | 1 | 0.58 | 0.57 | 1 | 6.9 | 0.0e+00 |
| SHR | 0.067 | 0.039 | 0.58 | 1 | 0.52 | 0.51 | 1 | 6.9 | 0.0e+00 |
| SSPF | 1196369.010 | 678475.331 | 0.57 | 1 | 0.50 | 0.49 | 1 | 6.0 | 0.0e+00 |
| GNE | 0.000 | 0.000 | 0.52 | 1 | 0.48 | 0.50 | 1 | 5.9 | 0.0e+00 |
| BGD | 1.497 | 0.694 | 0.46 | 1 | 0.43 | 0.41 | 1 | 5.2 | 0.0e+00 |
| spectral_skewness | 2.125 | 0.853 | 0.40 | 2 | 0.43 | 0.37 | 2 | 4.6 | 0.0e+00 |
| spectral_kurtosis | 32016058.690 | 5726691.274 | 0.18 | 2 | 0.19 | 0.16 | 2 | 1.8 | 0.0e+00 |

**Usable resolution.** q_max(pooled) distribution: {'1': 17, '2': 21, '3': 1}. **17 features cannot hold even q=2 (q_max=1); 21 reach q=2; 1 reach q≥3** — i.e. the **q≥3 failure count is 38 of 39**. On realistic multi-session audio the usable per-feature resolution is q ≤ 2; the paper's q=5–10 is not supported.

**Within-sex vs pooled.** Mean within-sex F exceeds pooled F for **1 of 39** features. The exception is the sex-linked source/filter features: pooling across sexes inflates their between-speaker variance, so their pooled F-ratio is the *higher* number and within-sex is lower. For example:

| feature | F(pooled) | F(male) | F(female) |
|---|---:|---:|---:|
| F0 | 3.47 | 1.45 | 2.18 |
| RMS | 3.28 | 2.66 | 3.30 |
| CPP | 2.69 | 2.11 | 1.78 |
| alpha_ratio | 2.38 | 2.39 | 1.94 |
| dCPP | 2.35 | 1.98 | 1.93 |

## 4. Per-feature usable bit depth (mutual information)

Balanced **5 clips/speaker** over all **10,676 speakers** (uniform prior; N=53,380 utts; ceiling log2(S)=13.382 bits). For each feature and bit depth b∈{1..8} (q=2^b equal-frequency bins): Miller–Madow MI, permutation null (200 shuffles, seed 1234), `I_corrected = max(0, I_mm − I_null_mean)`. b* = argmax_b I_corrected. Sorted by usable bits.

| feature | b* | q_eff | I_corrected (bits) | NMI | perm p |
|---|---:|---:|---:|---:|---:|
| F0 | 2 | 4 | 0.6619 | 0.0495 | 0.000 |
| RMS | 2 | 4 | 0.6371 | 0.0476 | 0.000 |
| CPP | 2 | 4 | 0.5002 | 0.0374 | 0.000 |
| GCT | 2 | 4 | 0.4721 | 0.0353 | 0.000 |
| alpha_ratio | 2 | 4 | 0.4418 | 0.0330 | 0.000 |
| LHR | 2 | 4 | 0.4318 | 0.0323 | 0.000 |
| dCPP | 2 | 4 | 0.4316 | 0.0323 | 0.000 |
| SPI | 2 | 4 | 0.4103 | 0.0307 | 0.000 |
| spectral_rolloff | 2 | 4 | 0.3969 | 0.0297 | 0.000 |
| spectral_kurtosis | 2 | 4 | 0.3912 | 0.0292 | 0.000 |
| spectral_skewness | 2 | 4 | 0.3772 | 0.0282 | 0.000 |
| spectral_entropy | 2 | 4 | 0.3727 | 0.0278 | 0.000 |
| F5 | 2 | 4 | 0.3703 | 0.0277 | 0.000 |
| F4 | 2 | 4 | 0.3517 | 0.0263 | 0.000 |
| F3 | 2 | 4 | 0.3503 | 0.0262 | 0.000 |
| spectral_flux | 2 | 4 | 0.3487 | 0.0261 | 0.000 |
| shimmer | 2 | 4 | 0.3328 | 0.0249 | 0.000 |
| CQ | 2 | 4 | 0.3256 | 0.0243 | 0.000 |
| B1 | 2 | 4 | 0.3222 | 0.0241 | 0.000 |
| F1 | 2 | 4 | 0.3074 | 0.0230 | 0.000 |
| B4 | 2 | 4 | 0.2985 | 0.0223 | 0.000 |
| F2 | 2 | 4 | 0.2930 | 0.0219 | 0.000 |
| B5 | 2 | 4 | 0.2699 | 0.0202 | 0.000 |
| B2 | 2 | 4 | 0.2650 | 0.0198 | 0.000 |
| Nasality | 2 | 4 | 0.2570 | 0.0192 | 0.000 |
| B3 | 1 | 2 | 0.2559 | 0.0191 | 0.000 |
| jitter | 1 | 2 | 0.2176 | 0.0163 | 0.000 |
| NAQ | 1 | 2 | 0.1998 | 0.0149 | 0.000 |
| SSPF | 2 | 4 | 0.1976 | 0.0148 | 0.000 |
| SQ | 2 | 4 | 0.1708 | 0.0128 | 0.000 |
| IHI | 1 | 2 | 0.1659 | 0.0124 | 0.000 |
| AMD | 1 | 2 | 0.1457 | 0.0109 | 0.000 |
| VFP | 2 | 2 | 0.1398 | 0.0104 | 0.000 |
| SHR | 1 | 2 | 0.1370 | 0.0102 | 0.000 |
| semitone_SD_F0 | 2 | 4 | 0.1282 | 0.0096 | 0.000 |
| MFDR | 1 | 2 | 0.1063 | 0.0079 | 0.000 |
| speech_rate | 1 | 2 | 0.1057 | 0.0079 | 0.000 |
| BGD | 1 | 2 | 0.0611 | 0.0046 | 0.000 |
| GNE | 1 | 2 | 0.0599 | 0.0045 | 0.000 |

**Total summed usable bits (optimistic over-count, ignores redundancy): 11.708 bits** across 39 features (39 permutation-significant at p<0.05). This sum double-counts correlated information; the joint classifier bound in §6 is the honest figure.

## 5. Effective dimensionality — POOLED, WITHIN-SEX, PARENT-RESIDUAL

Participation ratio PR = (Σλ)²/Σλ² of the z-scored per-speaker correlation matrix; 95% CIs from 1000 speaker-level bootstraps (seed 1234). **Parent-residual** regresses each feature on the shared parents **sex + age bucket + accent** (categorical dummies; missing→'unknown', rare accents→'other'), then takes PR of the residuals — the effective dimensionality surviving after the dominant confounders are removed. CV is the better corpus for this because it has explicit age and accent labels in addition to sex.

| analysis | n speakers | PR | 95% CI |
|---|---:|---:|---|
| pooled | 10,551 | 12.95 | [12.78, 13.07] |
| within_sex_male | 4,744 | 13.18 | [12.87, 13.34] |
| within_sex_female | 1,571 | 13.48 | [12.98, 13.70] |
| within_sex_mean |  | 13.33 | — |
| parent_residual | 10,551 | 13.57 | [13.39, 13.71] |

**The rise.** PR(pooled) = **12.95** → PR(within-sex mean) = **13.33** → PR(parent-residual) = **13.57** (+0.62 over pooled). Removing sex/age/accent *de-correlates* the features and *raises* effective dimensionality: the shared parents are themselves correlation-inducing axes (mean parent R² = 0.054 across features). Even after removing all three confounders the 39 measured axes carry only ~14 independent dimensions — far below nominal independence.

## 6. Joint usable speaker bits — held-out classifier lower bound

Kept ≥90%-coverage features (38; dropped ['SSPF']), listwise-complete, balanced **10 clips/speaker**. **S = 5,095 speakers, N = 50,950 clips**, ceiling H(speaker)=log2(S) = **12.315 bits**. Utterance-disjoint stratified 5-fold CV, z-scored on train folds only. All bounds are FLOORS below the ceiling.

| classifier | top-1 acc | acc 95% CI | per-fold acc | log-loss (bits) | Fano ≥ (bits) | xent ≥ (bits) |
|---|---:|---:|---:|---:|---:|---:|
| A: multinomial logreg (L2) | 0.5475 | [0.5435,0.5521] | 0.5475±0.0046 | 3.864 | 5.749 [5.70,5.81] | 8.451 [8.42,8.48] |
| B: small MLP | 0.5343 | [0.5299,0.5386] | 0.5343±0.0059 | 4.333 | 5.583 [5.53,5.64] | 7.982 [7.92,8.04] |
| C: shrinkage-LDA (Ledoit-Wolf) | 0.5228 | [0.5183,0.5272] | 0.5228±0.0037 | 5.836 | 5.440 [5.38,5.50] | 6.479 [6.40,6.56] |

**Headline.** Strongest bound from **logreg**: Fano ≥ **5.749 bits**, cross-entropy ≥ **8.451 bits** — both floors below the H(speaker)=12.31-bit ceiling. With the larger speaker set the Fano lower bound is substantially larger than the prior run's.

**Capacity inversion:** MLP top-1 (0.5343) < logreg top-1 (0.5475) — inversion **persists** at this scale. The nonlinear MLP underperforms the regularized linear model, consistent with the regularized-linear model being better matched to ~10 clips/speaker and thousands of classes.

## 7. Collision-metric cross-check (optional)

Plugging measured pooled q_max (geo-mean q≈1.494) and PR into the paper's collision formulae at n=1e+10, p=1e-09. m=q^d; P(E)=1−(1−1/m)^(n−1); P(M)=1/m; P(B)=1−∏(1−i/m). Sanity cross-check only.

| config | q | d_used | m=q^d | P(E) | P(M) | P(B) |
|---|---:|---:|---:|---:|---:|---:|
| full_independence_k | 2.00 | 39.00 | 5.50e+11 | 1.80e-02 | 1.82e-12 | 1.00e+00 |
| full_independence_k | 3.00 | 39.00 | 4.05e+18 | 2.47e-09 | 2.47e-19 | 1.00e+00 |
| PR_pooled @ q=2 | 2.00 | 12.95 | 7.93e+03 | 1.00e+00 | 1.26e-04 | 1.00e+00 |
| PR_pooled @ q_geo(qmax) | 1.49 | 12.95 | 1.81e+02 | 1.00e+00 | 5.52e-03 | 1.00e+00 |
| PR_parent_residual @ q=2 | 2.00 | 13.57 | 1.21e+04 | 1.00e+00 | 8.23e-05 | 1.00e+00 |
| PR_parent_residual @ q_geo(qmax) | 1.49 | 13.57 | 2.32e+02 | 1.00e+00 | 4.31e-03 | 1.00e+00 |

With the *measured* low effective dimensionality and q≤2 usable resolution, the collision metrics move from the paper's 'astronomically unique' regime toward near-certain population collisions — the paper's tiny collision probabilities are an artifact of the independence + high-q assumptions, which these measurements do not support. (PR is a *linear* redundancy measure, so this is a collision-pessimistic summary; the sample-scale speakers remain highly separable, per §6.)

## Headline numbers (for direct quotation)

- **Final speaker count:** 10,676 speakers / 137,675 clips (24 shards, 51,330 client_ids scanned).
- **Measured out of 40:** 39/40 (excluded: VOT; VTLE removed by design).
- **F0 F-ratio:** pooled 3.47 (male 1.45, female 2.18); q_max(pooled) = 3.
- **q≥3 failure count:** 38 of 39 measured features cannot support q≥3 (q_max=1: 17, q_max=2: 21).
- **PR(pooled):** 12.95 [12.78, 13.07].
- **PR(within-sex mean):** 13.33.
- **PR(parent-residual, sex+age+accent):** 13.57 (rise of +0.62 over pooled).
- **Total per-feature usable bits (optimistic sum):** 11.708 bits; top feature F0 at 0.662 bits.
- **Classifier top-1 accuracy** (S=5,095): logreg 0.5475, MLP 0.5343, LDA 0.5228 (chance 1.96e-04).
- **Joint bit lower bounds:** Fano ≥ 5.749 bits, cross-entropy ≥ 8.451 bits (ceiling log2(S)=12.31); capacity inversion persists.

_Every number above is computed (seed 1234); NOT-MEASURED features are flagged. VTLE excluded by design; VOT not measurable without phone alignments._
