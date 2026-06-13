# TIMIT — 40-feature distinctiveness battery (v2)

*Generated 2026-06-14 05:27:55 · RNG seed = 1234 (fixed everywhere) · all numbers traced to `./results/`*

## 0. Corpus and provenance

| field | value |
|---|---|
| Corpus | TIMIT (NIST SPHERE NIST_1A, 16 kHz) |
| Utterances (rows) | 6300 |
| Speakers | 630 |
| Decode failures | 0 |
| Extraction sentinel | `rows=6300 decode_fail_this_run=0 time=781s` |
| Seed | 1234 |
| Source-of-truth | every number below comes from `./results/` |

## 1. Feature coverage (measured out of 40)

**Measured: 40/40.** VTLE excluded by design. Coverage = fraction of utterances with a successful value (never imputed).

| feature | coverage | status |
|---|---|---|
| F0 | 1.000 | MEASURED |
| jitter | 1.000 | MEASURED |
| shimmer | 1.000 | MEASURED |
| GCT | 1.000 | MEASURED |
| CQ | 1.000 | MEASURED |
| MFDR | 1.000 | MEASURED |
| SQ | 1.000 | MEASURED |
| NAQ | 1.000 | MEASURED |
| SHR | 1.000 | MEASURED |
| IHI | 1.000 | MEASURED |
| VFP | 1.000 | MEASURED |
| semitone_SD_F0 | 1.000 | MEASURED |
| F1 | 1.000 | MEASURED |
| F2 | 1.000 | MEASURED |
| F3 | 1.000 | MEASURED |
| F4 | 1.000 | MEASURED |
| F5 | 1.000 | MEASURED |
| B1 | 1.000 | MEASURED |
| B2 | 1.000 | MEASURED |
| B3 | 1.000 | MEASURED |
| B4 | 1.000 | MEASURED |
| B5 | 1.000 | MEASURED |
| Nasality | 1.000 | MEASURED |
| spectral_skewness | 1.000 | MEASURED |
| spectral_kurtosis | 1.000 | MEASURED |
| spectral_entropy | 1.000 | MEASURED |
| spectral_rolloff | 1.000 | MEASURED |
| spectral_flux | 1.000 | MEASURED |
| alpha_ratio | 1.000 | MEASURED |
| LHR | 1.000 | MEASURED |
| SPI | 1.000 | MEASURED |
| GNE | 1.000 | MEASURED |
| SSPF | 1.000 | MEASURED |
| CPP | 1.000 | MEASURED |
| dCPP | 1.000 | MEASURED |
| RMS | 1.000 | MEASURED |
| AMD | 1.000 | MEASURED |
| speech_rate | 1.000 | MEASURED |
| VOT | 0.971 | MEASURED |
| BGD | 0.984 | MEASURED |

NOT-MEASURED (0 coverage): none — all 40 measured.

## 2. Population distributions and quantile bins

Quantile bin edges computed for q ∈ {2,3,5,10} from the across-speaker per-speaker-mean distribution. Degenerate (collapsed) bins:

- none — all features realized the requested bin count at every q.

## 3. F-ratios and usable resolution — pooled AND within-sex

Sorted by pooled F-ratio. within_var = mean within-speaker variance; between_var = variance of per-speaker means; F_ratio = between/within. q_max = largest q∈{2,3,5,10} with mean bin-crossing rate < 0.20.

| feature | within_var | between_var | F_ratio(pooled) | q_max(pooled) | F(ANOVA) | p | F_ratio(M) | F_ratio(F) | F_ratio(within-sex) | q_max(M) | q_max(F) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 168 | 1.61e+03 | 9.564 | 3 | 95.64 | 0 | 1.366 | 3.413 | 2.39 | 2 | 2 |
| F4 | 6.9e+03 | 4.24e+04 | 6.141 | 3 | 61.41 | 0 | 2.616 | 2.473 | 2.545 | 2 | 2 |
| F5 | 9.12e+03 | 4.47e+04 | 4.9 | 3 | 48.98 | 0 | 5.232 | 3.359 | 4.296 | 3 | 2 |
| GCT | 9.43e-09 | 3.74e-08 | 3.97 | 2 | 39.7 | 0 | 2.348 | 0.9104 | 1.629 | 2 | — |
| RMS | 5.43e-06 | 1.9e-05 | 3.496 | 2 | 34.96 | 0 | 3.481 | 3.54 | 3.511 | 2 | 2 |
| VFP | 0.00357 | 0.0114 | 3.186 | 2 | 31.86 | 0 | 3.18 | 0.5369 | 1.859 | 2 | — |
| B1 | 2.8e+03 | 7.82e+03 | 2.789 | 2 | 27.89 | 0 | 1.736 | 0.6076 | 1.172 | 2 | — |
| CPP | 0.0383 | 0.101 | 2.644 | 2 | 26.44 | 0 | 1.821 | 1.153 | 1.487 | 2 | — |
| MFDR | 5.88e+03 | 1.46e+04 | 2.491 | 2 | 24.91 | 0 | 2.317 | 3.001 | 2.659 | 2 | 2 |
| SHR | 0.000258 | 0.000612 | 2.37 | 2 | 23.7 | 0 | 1.162 | 1.849 | 1.506 | — | 2 |
| CQ | 0.000238 | 0.00056 | 2.353 | 2 | 23.53 | 0 | 1.829 | 1.215 | 1.522 | 2 | — |
| spectral_flux | 1.29e-05 | 2.84e-05 | 2.211 | 2 | 22.11 | 0 | 1.365 | 1.354 | 1.359 | — | — |
| B4 | 9.03e+03 | 1.81e+04 | 1.999 | 2 | 19.99 | 0 | 1.8 | 1.054 | 1.427 | 2 | — |
| dCPP | 0.00237 | 0.00459 | 1.935 | 2 | 19.35 | 0 | 1.693 | 1.693 | 1.693 | 2 | 2 |
| B5 | 2.03e+04 | 3.32e+04 | 1.636 | 2 | 16.36 | 0 | 2.252 | 0.6593 | 1.456 | 2 | — |
| shimmer | 0.000208 | 0.000321 | 1.542 | 2 | 15.42 | 0 | 0.7635 | 0.6494 | 0.7064 | — | — |
| F3 | 1.14e+04 | 1.74e+04 | 1.525 | — | 15.25 | 0 | 0.8967 | 1.064 | 0.9802 | — | — |
| NAQ | 0.000253 | 0.000352 | 1.391 | 2 | 13.91 | 0 | 1.085 | 1.371 | 1.228 | — | — |
| alpha_ratio | 2.15 | 2.86 | 1.331 | — | 13.31 | 0 | 1.37 | 1.086 | 1.228 | — | — |
| spectral_entropy | 0.001 | 0.00122 | 1.222 | — | 12.22 | 0 | 0.7964 | 0.6444 | 0.7204 | — | — |
| SQ | 3.8 | 4.21 | 1.107 | 2 | 11.07 | 0 | 0.6178 | 0.5886 | 0.6032 | — | — |
| B3 | 7.5e+03 | 7.63e+03 | 1.017 | — | 10.17 | 0 | 1.006 | 1.043 | 1.024 | — | — |
| F1 | 1.77e+03 | 1.71e+03 | 0.9638 | — | 9.638 | 0 | 1.051 | 0.5819 | 0.8162 | — | — |
| jitter | 1.25e-05 | 1.2e-05 | 0.9595 | — | 9.595 | 0 | 0.8582 | 0.5426 | 0.7004 | — | — |
| spectral_kurtosis | 2.62 | 2.4 | 0.9163 | — | 9.163 | 0 | 0.8055 | 0.8819 | 0.8437 | — | — |
| SSPF | 0.666 | 0.61 | 0.9149 | — | 9.149 | 0 | 0.7489 | 0.6638 | 0.7064 | — | — |
| B2 | 5.25e+03 | 4.59e+03 | 0.8755 | — | 8.755 | 0 | 0.6492 | 1.053 | 0.8512 | — | — |
| F2 | 1.29e+04 | 1.05e+04 | 0.8148 | — | 8.148 | 0 | 0.3583 | 0.4988 | 0.4285 | — | — |
| LHR | 7.05 | 5.54 | 0.7861 | — | 7.861 | 0 | 0.8268 | 0.6787 | 0.7527 | — | — |
| SPI | 5.68 | 4.45 | 0.783 | — | 7.83 | 0 | 0.768 | 0.7735 | 0.7707 | — | — |
| Nasality | 3.3 | 2.57 | 0.7773 | — | 7.773 | 0 | 0.5746 | 1.272 | 0.9232 | — | — |
| spectral_skewness | 0.0906 | 0.0695 | 0.7669 | — | 7.669 | 0 | 0.6637 | 0.6687 | 0.6662 | — | — |
| speech_rate | 2.24 | 1.71 | 0.7609 | — | 7.609 | 0 | 0.7635 | 0.7106 | 0.7371 | — | — |
| spectral_rolloff | 1.7e+05 | 9.13e+04 | 0.5381 | — | 5.381 | 2.46e-269 | 0.4165 | 0.3738 | 0.3951 | — | — |
| semitone_SD_F0 | 2.68 | 1.32 | 0.494 | — | 4.94 | 1.92e-238 | 0.4825 | 0.5583 | 0.5204 | — | — |
| GNE | 0.000719 | 0.000348 | 0.4837 | — | 4.837 | 3.65e-231 | 0.3457 | 0.5182 | 0.432 | — | — |
| BGD | 0.000411 | 0.000132 | 0.3219 | — | 3.154 | 8.61e-112 | 0.3061 | 0.3649 | 0.3355 | — | — |
| AMD | 0.0096 | 0.00273 | 0.2845 | — | 2.845 | 5.29e-91 | 0.2723 | 0.3011 | 0.2867 | — | — |
| IHI | 9.2 | 2.46 | 0.2678 | — | 2.678 | 7.47e-80 | 0.1771 | 0.2705 | 0.2238 | — | — |
| VOT | 0.00016 | 3.76e-05 | 0.2342 | — | 2.27 | 1.84e-53 | 0.2286 | 0.2425 | 0.2356 | — | — |

**Pooled vs within-sex:** median within-sex F-ratio is 0.86× the pooled value (within-sex lower overall). 
For **F0**, pooled F-ratio = 9.564 collapses to male=1.366, female=3.413 (within-sex 2.39) — sex carries most of F0's between-speaker variance. 
Formants show the same pooled→within-sex shrinkage (see F1–F4 rows).

**Caveat:** TIMIT is single-session, so within-speaker variance omits day-to-day/health/channel/affective variation. All F-ratios are OPTIMISTIC UPPER BOUNDS and q_max values are optimistic.

## 4. Per-feature usable bit depth (Miller–Madow MI vs permutation null)

`I_corrected = max(0, I_mm − I_null_mean)`, ≥200 shuffles, seed 1234. Sorted by usable bits.

| feature | b* | q_eff | usable bits | norm MI | perm p |
|---|---|---|---|---|---|
| F0 | 3 | 8 | 1.393 | 0.15 | 0.00498 |
| F4 | 3 | 8 | 1.157 | 0.124 | 0.00498 |
| F5 | 3 | 8 | 1.126 | 0.121 | 0.00498 |
| RMS | 3 | 8 | 0.9801 | 0.105 | 0.00498 |
| GCT | 3 | 8 | 0.9529 | 0.102 | 0.00498 |
| CPP | 3 | 8 | 0.7977 | 0.0858 | 0.00498 |
| B1 | 3 | 8 | 0.7965 | 0.0857 | 0.00498 |
| MFDR | 3 | 8 | 0.7709 | 0.0829 | 0.00498 |
| CQ | 3 | 8 | 0.7573 | 0.0814 | 0.00498 |
| SHR | 3 | 8 | 0.7155 | 0.0769 | 0.00498 |
| B4 | 2 | 4 | 0.6934 | 0.0746 | 0.00498 |
| spectral_flux | 3 | 8 | 0.6895 | 0.0741 | 0.00498 |
| dCPP | 3 | 8 | 0.676 | 0.0727 | 0.00498 |
| B5 | 3 | 8 | 0.6552 | 0.0705 | 0.00498 |
| shimmer | 2 | 4 | 0.6223 | 0.0669 | 0.00498 |
| NAQ | 3 | 8 | 0.5838 | 0.0628 | 0.00498 |
| F3 | 3 | 8 | 0.5809 | 0.0625 | 0.00498 |
| alpha_ratio | 3 | 8 | 0.5782 | 0.0622 | 0.00498 |
| VFP | 3 | 5 | 0.5603 | 0.0603 | 0.00498 |
| SQ | 2 | 4 | 0.5488 | 0.059 | 0.00498 |
| spectral_entropy | 2 | 4 | 0.4874 | 0.0524 | 0.00498 |
| B3 | 3 | 8 | 0.4632 | 0.0498 | 0.00498 |
| semitone_SD_F0 | 3 | 8 | 0.455 | 0.0489 | 0.00498 |
| SSPF | 2 | 4 | 0.4226 | 0.0454 | 0.00498 |
| spectral_kurtosis | 3 | 8 | 0.4214 | 0.0453 | 0.00498 |
| F1 | 3 | 8 | 0.3979 | 0.0428 | 0.00498 |
| jitter | 2 | 4 | 0.3932 | 0.0423 | 0.00498 |
| F2 | 2 | 4 | 0.3793 | 0.0408 | 0.00498 |
| LHR | 3 | 8 | 0.3666 | 0.0394 | 0.00498 |
| SPI | 2 | 4 | 0.3658 | 0.0393 | 0.00498 |
| B2 | 2 | 4 | 0.3623 | 0.039 | 0.00498 |
| spectral_skewness | 2 | 4 | 0.3532 | 0.038 | 0.00498 |
| Nasality | 2 | 4 | 0.3342 | 0.0359 | 0.00498 |
| speech_rate | 2 | 4 | 0.3326 | 0.0358 | 0.00498 |
| spectral_rolloff | 2 | 4 | 0.2413 | 0.0259 | 0.00498 |
| GNE | 3 | 8 | 0.228 | 0.0245 | 0.00498 |
| BGD | 2 | 4 | 0.1889 | 0.0203 | 0.00498 |
| IHI | 3 | 8 | 0.1575 | 0.0169 | 0.00498 |
| AMD | 3 | 8 | 0.1408 | 0.0151 | 0.00498 |
| VOT | 3 | 8 | 0.08647 | 0.0093 | 0.00498 |

**Total summed usable bits = 22.21** across features (OPTIMISTIC over-count: features are correlated). H(speaker) = 9.299 bits for S = 630 speakers.

## 5. Effective dimensionality — participation ratio (PR)

PR = (Σλ)²/Σλ² of the z-scored per-speaker covariance; 95% CI from 1000 speaker bootstraps (seed 1234). 40 complete features used.

| analysis | PR | 95% CI | n speakers |
|---|---|---|---|
| Pooled | 9.242 | [8.757, 9.542] | 630 |
| Within-sex male | 11.275 | [10.394, 11.672] | 438 |
| Within-sex female | 11.869 | [10.505, 12.109] | 192 |
| Within-sex mean | 11.572 | — | — |
| Parent-residual (sex+age+height) | 12.670 | [11.860, 12.989] | 629 |

**Rise across analyses:** pooled PR = 9.24 → within-sex mean = 11.57 (+2.33) → parent-residual = 12.67 (+3.43 vs pooled). The parent-residual PR is the empirically-grounded analogue of the manuscript's d_eff lower bound: removing the shared-parent confounders (sex, age, height) raises effective dimensionality.

## 6. Joint usable speaker bits — held-out classifier lower bound

Features with ≥90% coverage; listwise-deleted rows. Retained **6105 utts / 630 speakers**, 40 features. Chance = 0.00159; H(speaker) = 9.299 bits. Utterance-disjoint stratified 5-fold CV, z-scored on train folds only.

| model | top-1 acc | 95% CI | per-fold mean±std | log-loss (bits) | Fano bits | x-ent bits |
|---|---|---|---|---|---|---|
| logreg | 0.6888 | [0.6761, 0.7014] | 0.6888±0.0144 | 2.143 | 5.511 | 7.157 |
| mlp | 0.6618 | [0.6530, 0.6705] | 0.6618±0.0100 | 2.122 | 5.231 | 7.177 |
| lda | 0.7364 | [0.7242, 0.7487] | 0.7364±0.0140 | 2.203 | 6.017 | 7.096 |

**Capacity-inversion check:** MLP UNDER-performs the linear models — the data-starvation signature at ~8–10 utts/speaker is present.

**Headline (Fano):** 6.017 bits; **headline (cross-entropy):** 7.177 bits. Both are FLOORS (a stronger classifier raises them) and remain below the H(speaker) = 9.299-bit sample ceiling.

## 7. Collision-metric sanity cross-check (optional, illustrative)

Illustrative birthday model at n = 1e10 identities, N_cells = q_max^PR with operating q_max = 2. Not a headline.

| operating point | q_max | PR | log10(N_cells) | P(E) exp. pairs | P(M) any collision | P(B) per-pair |
|---|---|---|---|---|---|---|
| pooled_PR | 2 | 9.242 | 2.782 | 8.26e+16 | 1 | 0.00165 |
| parent_residual_PR | 2 | 12.670 | 3.814 | 7.67e+15 | 1 | 0.000153 |

## Headline numbers (for direct quotation)

- **Measured features:** 40/40
- **F0 F-ratio:** pooled = 9.564, within-sex = 2.39 (M 1.366 / F 3.413); q_max(pooled) = 3
- **PR(pooled):** 9.242  CI [8.757, 9.542]
- **PR(within-sex):** mean 11.572 (M 11.275 / F 11.869)
- **PR(parent-residual):** 12.670  CI [11.860, 12.989]  (parents: sex, age, height)
- **Classifier top-1 accuracy:** logreg=0.6888, mlp=0.6618, lda=0.7364
- **Bit lower bounds:** Fano = 6.017 bits, cross-entropy = 7.177 bits (H(speaker) = 9.299)
- **Total summed per-feature usable bits:** 22.21 (optimistic; correlated)

*All F-ratios / q_max are optimistic upper bounds (single-session TIMIT). All bit bounds are floors. NOT-MEASURED features (if any) flagged in §1.*
