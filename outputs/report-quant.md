# Information-Theoretic Speaker Discriminability on TIMIT

*Reproducibility:* single RNG `numpy.default_rng(1234)` drives all 200-shuffle permutation nulls. Per-utterance values (630 speakers x 10 utts), 40 measured features. **Headline metric is bias-corrected** `I_corrected = max(0, I_mm - I_null_mean)` (Miller-Madow + permutation null); raw MI is shown only for context.

> Note on coverage: the prior extraction measured **40 features** (not 30); this analysis uses all 40. Two of the paper's 42 columns (VFI, Nasality) remain NOT MEASURED and are excluded — never imputed.

## Usable bit depth per feature (Step 3)

`b*` = argmax over b in {1..8} of I_corrected; finer binning beyond b* adds only sampling noise. `q_eff` is the realized bin count after merging degenerate quantile edges (logged whenever < 2^b). Sorted by usable bits.

| feature | b* | q_eff(b*) | I_corrected(b*) bits | NMI_corrected | perm_p |
|---|---:|---:|---:|---:|---:|
| F0 | 3 | 8 | 1.444 | 0.1553 | 0.000 |
| CPP | 2 | 4 | 1.081 | 0.1162 | 0.000 |
| F5 | 3 | 8 | 1.079 | 0.1161 | 0.000 |
| F4 | 2 | 4 | 1.041 | 0.1119 | 0.000 |
| dCPP | 2 | 4 | 0.996 | 0.1071 | 0.000 |
| GCT | 2 | 4 | 0.938 | 0.1009 | 0.000 |
| VTLE | 2 | 4 | 0.765 | 0.0822 | 0.000 |
| RMS | 2 | 4 | 0.745 | 0.0801 | 0.000 |
| SHR | 2 | 4 | 0.744 | 0.0800 | 0.000 |
| HNR | 2 | 4 | 0.717 | 0.0771 | 0.000 |
| NAQ | 2 | 4 | 0.690 | 0.0742 | 0.000 |
| F3 | 2 | 4 | 0.633 | 0.0681 | 0.000 |
| MFDR | 2 | 4 | 0.502 | 0.0540 | 0.000 |
| CQ | 2 | 4 | 0.500 | 0.0537 | 0.000 |
| spectral_entropy | 2 | 4 | 0.490 | 0.0526 | 0.000 |
| B4 | 2 | 4 | 0.455 | 0.0489 | 0.000 |
| SSPF | 2 | 4 | 0.434 | 0.0466 | 0.000 |
| shimmer | 2 | 4 | 0.398 | 0.0428 | 0.000 |
| SQ | 2 | 4 | 0.371 | 0.0399 | 0.000 |
| alpha_ratio | 2 | 4 | 0.366 | 0.0394 | 0.000 |
| B3 | 2 | 4 | 0.361 | 0.0388 | 0.000 |
| B5 | 2 | 4 | 0.351 | 0.0378 | 0.000 |
| spectral_rolloff | 2 | 4 | 0.319 | 0.0343 | 0.000 |
| F2 | 2 | 4 | 0.308 | 0.0331 | 0.000 |
| B2 | 2 | 4 | 0.293 | 0.0316 | 0.000 |
| F1 | 2 | 4 | 0.262 | 0.0282 | 0.000 |
| spectral_kurtosis | 2 | 4 | 0.241 | 0.0259 | 0.000 |
| IHI | 2 | 4 | 0.238 | 0.0256 | 0.000 |
| spectral_skewness | 2 | 4 | 0.238 | 0.0256 | 0.000 |
| B1 | 2 | 4 | 0.232 | 0.0249 | 0.000 |
| LHR | 2 | 4 | 0.230 | 0.0247 | 0.000 |
| semitone_SD_F0 | 2 | 4 | 0.227 | 0.0245 | 0.000 |
| jitter | 2 | 4 | 0.178 | 0.0191 | 0.000 |
| SPI | 1 | 2 | 0.166 | 0.0178 | 0.000 |
| speech_rate | 1 | 2 | 0.134 | 0.0144 | 0.000 |
| GNE | 1 | 2 | 0.076 | 0.0082 | 0.000 |
| spectral_flux | 1 | 2 | 0.067 | 0.0072 | 0.000 |
| AMD | 1 | 2 | 0.031 | 0.0033 | 0.000 |
| BGD | 1 | 2 | 0.000 | 0.0000 | 0.000 |
| VOT | 1 | 2 | 0.000 | 0.0000 | 0.000 |

**Top feature:** F0 carries 1.444 corrected bits about speaker identity at b*=3 (15.5% of the log2(630)=9.30-bit ceiling). All per-feature bit counts are small fractions of that ceiling — no single feature comes close to identifying a speaker among 630.

Bin-deficiency (q_eff < 2^b at some depth) occurred for 3 feature(s), logged in `mi_per_feature_full.csv` (column `deficient`). Entropy ceilings use q_eff.

## Variance metric vs information metric (Step 4)

Spearman rank correlation between F_ratio (variance separability) and I_corrected(b*) (partition information): **rho = 0.973** (p = 0.000). They are positively but imperfectly related — the two are not the same quantity.

| feature | F_ratio | variance_q_max | I_corrected(b*) bits | b* | rank_F | rank_bits | Δrank | disagree |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| F0 | 35.28 | 5 | 1.444 | 3 | 1 | 1 | +0 |  |
| CPP | 11.24 | 3 | 1.081 | 2 | 3 | 2 | +1 |  |
| F5 | 10.59 | 3 | 1.079 | 3 | 4 | 3 | +1 |  |
| F4 | 13.00 | 3 | 1.041 | 2 | 2 | 4 | -2 |  |
| dCPP | 8.94 | 3 | 0.996 | 2 | 5 | 5 | +0 |  |
| GCT | 5.89 | 3 | 0.938 | 2 | 7 | 6 | +1 |  |
| VTLE | 6.13 | 2 | 0.765 | 2 | 6 | 7 | -1 |  |
| RMS | 3.67 | 2 | 0.745 | 2 | 10 | 8 | +2 |  |
| SHR | 3.84 | 2 | 0.744 | 2 | 9 | 9 | +0 |  |
| HNR | 3.86 | 2 | 0.717 | 2 | 8 | 10 | -2 |  |
| NAQ | 3.40 | 2 | 0.690 | 2 | 12 | 11 | +1 |  |
| F3 | 3.50 | 2 | 0.633 | 2 | 11 | 12 | -1 |  |
| MFDR | 1.99 | 2 | 0.502 | 2 | 15 | 13 | +2 |  |
| CQ | 2.05 | 2 | 0.500 | 2 | 14 | 14 | +0 |  |
| spectral_entropy | 2.18 | 2 | 0.490 | 2 | 13 | 15 | -2 |  |
| B4 | 1.73 | 2 | 0.455 | 2 | 16 | 16 | +0 |  |
| SSPF | 1.62 | 2 | 0.434 | 2 | 17 | 17 | +0 |  |
| shimmer | 1.53 | 2 | 0.398 | 2 | 18 | 18 | +0 |  |
| SQ | 0.81 | 0 | 0.371 | 2 | 32 | 19 | +13 | **YES** |
| alpha_ratio | 1.44 | 0 | 0.366 | 2 | 20 | 20 | +0 |  |
| B3 | 1.38 | 0 | 0.361 | 2 | 22 | 21 | +1 |  |
| B5 | 1.28 | 0 | 0.351 | 2 | 24 | 22 | +2 |  |
| spectral_rolloff | 1.37 | 0 | 0.319 | 2 | 23 | 23 | +0 |  |
| F2 | 1.40 | 0 | 0.308 | 2 | 21 | 24 | -3 |  |
| B2 | 1.45 | 0 | 0.293 | 2 | 19 | 25 | -6 |  |
| F1 | 1.19 | 0 | 0.262 | 2 | 25 | 26 | -1 |  |
| spectral_kurtosis | 1.05 | 0 | 0.241 | 2 | 29 | 27 | +2 |  |
| IHI | 1.07 | 0 | 0.238 | 2 | 28 | 28 | +0 |  |
| spectral_skewness | 1.09 | 0 | 0.238 | 2 | 27 | 29 | -2 |  |
| B1 | 1.17 | 0 | 0.232 | 2 | 26 | 30 | -4 |  |
| LHR | 0.98 | 0 | 0.230 | 2 | 30 | 31 | -1 |  |
| semitone_SD_F0 | 0.72 | 0 | 0.227 | 2 | 34 | 32 | +2 |  |
| jitter | 0.85 | 0 | 0.178 | 2 | 31 | 33 | -2 |  |
| SPI | 0.79 | 0 | 0.166 | 1 | 33 | 34 | -1 |  |
| speech_rate | 0.72 | 0 | 0.134 | 1 | 35 | 35 | +0 |  |
| GNE | 0.44 | 0 | 0.076 | 1 | 37 | 36 | +1 |  |
| spectral_flux | 0.47 | 0 | 0.067 | 1 | 36 | 37 | -1 |  |
| AMD | 0.32 | 0 | 0.031 | 1 | 38 | 38 | +0 |  |
| BGD | 0.16 | 0 | 0.000 | 1 | 40 | 39 | +1 |  |
| VOT | 0.18 | 0 | 0.000 | 1 | 39 | 39 | +0 |  |

Overall the two metrics are strongly concordant (Spearman rho=0.973), so the headline is *agreement*: features that separate speakers by variance also tend to carry speaker bits. The interesting cases are the rank divergences (|Δrank| >= 10, i.e. ~a quarter of the 40 features), flagged below.

**Disagreements (key result — variance- and partition-separability are not the same quantity):**
- `SQ`: **more informative than its variance suggests** — bits rank #19 vs F_ratio rank #32 (F_ratio=0.81, I_corrected=0.371 bits, b*=2).

Mechanism: F_ratio rewards a large between/within *variance* ratio, which a few outlying speakers or a heavy tail can inflate without cleanly partitioning the population; corrected MI instead rewards a feature that splits speakers into distinguishable equiprobable bins. A feature can score well on one and not the other.

## Joint usable bits — greedy forward selection (Step 5)

Fixed depth b=2 per feature (q_eff<=4), on 6300 utterances listwise-complete across all 40 features (630 speakers). Each step adds the feature maximizing joint I_corrected (same MM + permutation correction on the joint table); stop when the marginal corrected gain falls to or below the permutation noise floor (p95 - mean).

| step | feature added | #joint bins | cumulative I_corrected (bits) | marginal gain | noise floor |
|---:|---|---:|---:|---:|---:|
| 1 | F0 | 4 | 1.305 | 1.305 | 0.012 |
| 2 | F5 | 16 | 1.524 | 0.219 | 0.015 |
| 3 | CPP  (STOP: gain<=floor, not added) | 47 | 1.297 | -0.227 | 0.013 |

**Selection order:** F0 -> F5

**Saturation:** the curve flattens after **2 features** at **1.524 cumulative corrected bits**. This is the information-theoretic analogue of d_eff: total *usable* speaker bits, not a variance-axis count.

> **Sample-ceiling caveat (do not over-read):** cumulative corrected MI can never exceed H(speaker)=log2(630)=9.30 bits, and as joint cells multiply (4^k) with only N=6300 samples the permutation null rises and the MM correction grows, so part of this saturation is *sample-limited*, not purely a property of the voice. The flattening point is a lower bound on where real joint information stops being estimable here, not a hard physiological limit.

## Honest limitations

**Finite-sample MI bias.** Plug-in MI is upward-biased: with 630 speakers x q bins the contingency table is sparse, so raw MI overstates information. That is exactly why we (a) Miller-Madow-correct every entropy and (b) subtract a 200-fold permutation null; we report I_corrected, never raw MI. Residual bias still inflates absolute bits, so treat the numbers as upper-ish estimates and trust the rankings and the permutation p-values more than the third decimal.

**Single-session within-speaker variance.** TIMIT gives one recording session per speaker, so within-speaker spread excludes day-to-day, health, and emotional variability. Usable bits and b* are therefore an **OPTIMISTIC upper bound**: cross-session data would increase within-speaker bin-crossing, lowering both b* and I_corrected. (Conversely, the 10 utterances are different sentences, so some within-speaker spread is phonetic content rather than identity noise.)

**Coverage.** 40/42 candidate features measured; VFI and Nasality not measured and excluded (not imputed). Glottal-flow features are approximate single-pass IAIF estimates carried over from the prior run.

**Sample ceiling.** All joint bits are bounded by log2(630)=9.30 bits; the greedy curve's flattening is partly this ceiling and the rising joint-cell null, not solely the acoustics.


*Artifacts:* mi_per_feature_full.csv, usable_bits.csv, fratio_vs_bits.csv, joint_greedy.csv, figs/mi_*.png, figs/joint_cumulative_bits.png, report-quant.md. Seed=1234.
