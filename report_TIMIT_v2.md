# TIMIT — 40-feature distinctiveness battery (v2)

*Every measured/computed number below is read from machine-readable artifacts under
`./results/` (source-of-truth firewall). Per-utterance feature matrix cached at
`./features/features_per_utt.parquet`. Single fixed RNG seed **1234** for all
randomized procedures. No value in this report was estimated, imputed, or
simulated; not-measured features are flagged explicitly.*

---

## 0. Corpus and provenance

| Item | Value | Source |
|---|---|---|
| Corpus | TIMIT (NIST SPHERE `NIST_1A`, 16 kHz) | `results/manifest.csv` |
| Utterances | **6,300** (unique `utt_id` = 6,300) | `features/features_per_utt.parquet` |
| Speakers | **630** (M = 438, F = 192) | `run.log`, `results/speaker_meta.csv` |
| Utterances/speaker | 10 | manifest |
| Splits / regions | TRAIN+TEST, DR1–DR8 | manifest |
| **Decode failures** | **0** | `results/extract_time.txt` |
| Sample-rate mismatches | 0 | `results/extract_time.txt` |
| Extraction wall-clock | **4,949 s** (≈82.5 min), 16 processes | `results/extract_time.txt` |
| RNG seed | 1234 (everywhere) | — |
| Environment | conda env `voice_unique`, Python 3.10.20, Windows | `run.log` |

Library versions (from `run.log`): sphfile 1.0.0, praat-parselmouth 0.4.7,
librosa 0.11.0, scikit-learn 1.7.2, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
pyarrow 24.0.0.

Speaker label = TIMIT speaker ID; sex label = leading directory letter (M/F),
used for the within-sex analyses. Corpus tree treated as read-only.

**Self-verification (passed before report):** row count = 6,300; unique utt = 6,300;
speakers = 630; decode failures = 0; the set of all-NaN feature columns equals
exactly the 10 deliberately not-attempted features (no unexpected all-NaN
columns); measured-out-of-40 = 30; capacity-inversion check ran.

---

## 1. Coverage and measured-out-of-40

**Measured features: 30 / 40.** (VTLE is not in the list and was not computed as a
feature.) All 30 measured features have 100 % coverage except **VOT (90.14 %)**,
which is undefined for utterances containing no voiceless stop (p/t/k).

The 10 **NOT-MEASURED** features were deliberately *not attempted* (0 coverage,
excluded, never imputed): 7 glottal-source features requiring EGG / reliable
inverse filtering — **GCT, CQ, MFDR, SQ, NAQ, IHI, VFP**; **Nasality** (needs a
nasometer/nasal accelerometer); and two ambiguously-defined features —
**SSPF, BGD**.

Coverage (source `results/coverage.csv`):

| Feature | Cov. | Status | | Feature | Cov. | Status |
|---|---|---|---|---|---|---|
| F0 | 1.000 | MEASURED | | spectral_skewness | 1.000 | MEASURED |
| jitter | 1.000 | MEASURED | | spectral_kurtosis | 1.000 | MEASURED |
| shimmer | 1.000 | MEASURED | | spectral_entropy | 1.000 | MEASURED |
| GCT | 0.000 | **NOT-MEASURED** | | spectral_rolloff | 1.000 | MEASURED |
| CQ | 0.000 | **NOT-MEASURED** | | spectral_flux | 1.000 | MEASURED |
| MFDR | 0.000 | **NOT-MEASURED** | | alpha_ratio | 1.000 | MEASURED |
| SQ | 0.000 | **NOT-MEASURED** | | LHR | 1.000 | MEASURED |
| NAQ | 0.000 | **NOT-MEASURED** | | SPI | 1.000 | MEASURED |
| SHR | 1.000 | MEASURED | | GNE | 1.000 | MEASURED |
| IHI | 0.000 | **NOT-MEASURED** | | SSPF | 0.000 | **NOT-MEASURED** |
| VFP | 0.000 | **NOT-MEASURED** | | CPP | 1.000 | MEASURED |
| semitone_SD_F0 | 1.000 | MEASURED | | dCPP | 1.000 | MEASURED |
| F1 | 1.000 | MEASURED | | RMS | 1.000 | MEASURED |
| F2 | 1.000 | MEASURED | | AMD | 1.000 | MEASURED |
| F3 | 1.000 | MEASURED | | speech_rate | 1.000 | MEASURED |
| F4 | 1.000 | MEASURED | | VOT | 0.9014 | MEASURED |
| F5 | 1.000 | MEASURED | | BGD | 0.000 | **NOT-MEASURED** |
| B1–B5 | 1.000 | MEASURED | | Nasality | 0.000 | **NOT-MEASURED** |

---

## 2. Population distributions and quantile bins

For each of the 30 measured features the across-speaker distribution was formed
from the 630 per-speaker means and q-quantile edges computed for q ∈ {2,3,5,10}.
**No feature shows degenerate/collapsed bins:** for every measured feature the
realized bin count equals q at all of q = 2, 3, 5, 10 (source
`results/binning.csv`; all `*_degenerate` flags = False, n_speakers = 630). VOT,
despite 90.14 % per-utterance coverage, still yields a per-speaker mean for all
630 speakers, so its binning is non-degenerate.

---

## 3. F-ratios and usable resolution — pooled and within-sex

Definitions: `within_var` = mean over speakers of within-speaker (across-utterance)
variance; `between_var` = variance of per-speaker means; `F_ratio =
between_var / within_var`; `q_max` = largest q ∈ {2,3,5,10} with mean
bin-crossing rate < 0.20. Within-sex columns recompute the whole decomposition
within each sex; `F_ratio(within-sex)` is the mean of the male and female
F-ratios; `q_max(within-sex)` is the min of male/female q_max. Source
`results/f_ratio.csv` (sorted by pooled F-ratio).

| Feature | within_var | between_var | F(pooled) | q_max(p) | F(male) | F(female) | F(within-sex) | q_max(w-sex) |
|---|---|---|---|---|---|---|---|---|
| F0 | 90.76 | 1696.23 | **18.69** | 3 | 3.045 | 4.099 | 3.572 | 2 |
| F5 | 5697.1 | 22906.3 | 4.021 | 2 | 4.128 | 2.549 | 3.338 | 2 |
| spectral_flux | 506.87 | 1826.61 | 3.604 | 2 | 3.625 | 3.550 | 3.587 | 2 |
| RMS | 5611.9 | 18728.7 | 3.337 | 2 | 3.319 | 3.391 | 3.355 | 2 |
| F4 | 5905.5 | 15953.7 | 2.702 | 2 | 1.130 | 1.700 | 1.415 | 1 |
| CPP | 0.001702 | 0.004559 | 2.679 | 2 | 1.810 | 1.088 | 1.449 | 1 |
| dCPP | 9.69e-05 | 2.02e-04 | 2.087 | 2 | 1.748 | 1.602 | 1.675 | 2 |
| B5 | 8249.7 | 13800.7 | 1.673 | 2 | 2.144 | 0.758 | 1.451 | 1 |
| shimmer | 1.87e-04 | 2.87e-04 | 1.530 | 2 | 0.744 | 0.646 | 0.695 | 1 |
| B4 | 4292.9 | 6225.2 | 1.450 | 1 | 1.261 | 1.085 | 1.173 | 1 |
| alpha_ratio | 2.149 | 2.864 | 1.332 | 1 | 1.371 | 1.087 | 1.229 | 1 |
| B1 | 3219.5 | 4212.3 | 1.308 | 1 | 0.831 | 0.296 | 0.564 | 1 |
| spectral_entropy | 0.001014 | 0.001136 | 1.120 | 1 | 0.770 | 0.604 | 0.687 | 1 |
| SHR | 0.02739 | 0.02689 | 0.982 | 1 | 0.648 | 0.415 | 0.531 | 1 |
| jitter | 1.23e-05 | 1.13e-05 | 0.923 | 1 | 0.829 | 0.526 | 0.678 | 1 |
| spectral_skewness | 0.9066 | 0.7914 | 0.873 | 1 | 0.908 | 0.699 | 0.803 | 1 |
| B3 | 3993.0 | 3471.6 | 0.869 | 1 | 0.847 | 0.904 | 0.875 | 1 |
| SPI | 5.676 | 4.445 | 0.783 | 1 | 0.768 | 0.775 | 0.771 | 1 |
| spectral_kurtosis | 150.89 | 113.65 | 0.753 | 1 | 0.770 | 0.687 | 0.728 | 1 |
| speech_rate | 2.451 | 1.814 | 0.740 | 1 | 0.743 | 0.704 | 0.723 | 1 |
| LHR | 16.85 | 12.08 | 0.717 | 1 | 0.668 | 0.728 | 0.698 | 1 |
| GNE | 0.002485 | 0.001779 | 0.716 | 1 | 0.752 | 0.599 | 0.676 | 1 |
| F3 | 9954.2 | 6959.8 | 0.699 | 1 | 0.495 | 0.771 | 0.633 | 1 |
| spectral_rolloff | 380205.7 | 225883.8 | 0.594 | 1 | 0.673 | 0.419 | 0.546 | 1 |
| F2 | 9041.5 | 4664.9 | 0.516 | 1 | 0.343 | 0.460 | 0.402 | 1 |
| semitone_SD_F0 | 1.571 | 0.7723 | 0.492 | 1 | 0.440 | 0.578 | 0.509 | 1 |
| F1 | 3313.2 | 1615.2 | 0.487 | 1 | 0.491 | 0.347 | 0.419 | 1 |
| B2 | 4534.1 | 2086.0 | 0.460 | 1 | 0.361 | 0.734 | 0.547 | 1 |
| AMD | 0.024486 | 0.009137 | 0.373 | 1 | 0.357 | 0.387 | 0.372 | 1 |
| VOT | 3.18e-04 | 5.81e-05 | 0.183 | 1 | 0.180 | 0.183 | 0.181 | 1 |

(One-way ANOVA F = 10 × F_ratio with p ≈ 0 for nearly all features; full ANOVA
F/p in `results/f_ratio.csv`.)

**Pooled vs within-sex — within-sex F-ratios are systematically LOWER than pooled.**
The collapse is largest for **F0**: pooled F = **18.69** falls to a within-sex mean of
**3.57** (male 3.05, female 4.10) — a ≈5.2× reduction (q_max 3 → 2). This is the
expected signature that **sex accounts for most of F0's apparent between-speaker
separation**; once you stay within one sex, F0 is far less distinctive. The
formants drop too: F4 2.70 → 1.42, F5 4.02 → 3.34, F2 0.52 → 0.40, F1 0.49 → 0.42.
A handful of source/spectral features are nearly sex-invariant (spectral_flux
3.60 → 3.59, RMS 3.34 → 3.36) — their distinctiveness does **not** come from sex.
Net effect: removing the sex split removes a large, shared, low-rank chunk of the
between-speaker variance, especially for pitch.

**Caveat (stated as required):** TIMIT is **single-session**, so within-speaker
variance omits day-to-day, health, channel, and affective variation. All F-ratios
are therefore **OPTIMISTIC UPPER BOUNDS** and all q_max values are optimistic.

---

## 4. Per-feature usable bit depth (mutual information)

Each feature was quantized into 2^b cells (b ∈ 1..8); MI between cell and speaker
identity was Miller–Madow corrected and de-biased against a permutation null
(200 shuffles, seed 1234): `I_corrected = max(0, I_mm − I_null_mean)`. Reported at
the b* maximizing I_corrected. Source `results/usable_bits.csv` (sorted).

| Feature | b* | q_eff | I_mm | I_null_mean | **I_corrected** | norm_MI | perm_p |
|---|---|---|---|---|---|---|---|
| F0 | 3 | 8 | 1.822 | 0.252 | **1.569** | 0.1688 | 0.00498 |
| spectral_flux | 3 | 8 | 1.261 | 0.253 | 1.008 | 0.1084 | 0.00498 |
| F5 | 3 | 8 | 1.260 | 0.253 | 1.007 | 0.1083 | 0.00498 |
| RMS | 3 | 8 | 1.220 | 0.250 | 0.970 | 0.1043 | 0.00498 |
| CPP | 3 | 8 | 1.049 | 0.252 | 0.797 | 0.0857 | 0.00498 |
| F4 | 3 | 8 | 1.027 | 0.252 | 0.775 | 0.0833 | 0.00498 |
| dCPP | 3 | 8 | 0.938 | 0.252 | 0.686 | 0.0738 | 0.00498 |
| B5 | 3 | 8 | 0.854 | 0.252 | 0.602 | 0.0647 | 0.00498 |
| shimmer | 2 | 4 | 0.644 | 0.0454 | 0.599 | 0.0644 | 0.00498 |
| alpha_ratio | 3 | 8 | 0.827 | 0.251 | 0.576 | 0.0619 | 0.00498 |
| B4 | 3 | 8 | 0.780 | 0.253 | 0.527 | 0.0566 | 0.00498 |
| spectral_rolloff | 3 | 8 | 0.772 | 0.252 | 0.520 | 0.0560 | 0.00498 |
| B1 | 2 | 4 | 0.548 | 0.0460 | 0.502 | 0.0540 | 0.00498 |
| SHR | 3 | 8 | 0.721 | 0.254 | 0.467 | 0.0502 | 0.00498 |
| semitone_SD_F0 | 3 | 8 | 0.714 | 0.253 | 0.460 | 0.0495 | 0.00498 |
| spectral_entropy | 2 | 4 | 0.502 | 0.0457 | 0.457 | 0.0491 | 0.00498 |
| spectral_skewness | 3 | 8 | 0.659 | 0.252 | 0.407 | 0.0438 | 0.00498 |
| spectral_kurtosis | 3 | 8 | 0.641 | 0.252 | 0.389 | 0.0418 | 0.00498 |
| jitter | 3 | 8 | 0.641 | 0.252 | 0.388 | 0.0418 | 0.00498 |
| B3 | 3 | 8 | 0.639 | 0.254 | 0.386 | 0.0415 | 0.00498 |
| LHR | 3 | 8 | 0.617 | 0.251 | 0.366 | 0.0394 | 0.00498 |
| SPI | 2 | 4 | 0.411 | 0.0477 | 0.363 | 0.0391 | 0.00498 |
| GNE | 3 | 8 | 0.590 | 0.252 | 0.338 | 0.0363 | 0.00498 |
| speech_rate | 3 | 8 | 0.581 | 0.252 | 0.329 | 0.0354 | 0.00498 |
| F3 | 3 | 8 | 0.537 | 0.254 | 0.283 | 0.0304 | 0.00498 |
| F2 | 2 | 4 | 0.302 | 0.0461 | 0.256 | 0.0275 | 0.00498 |
| F1 | 3 | 8 | 0.470 | 0.253 | 0.217 | 0.0233 | 0.00498 |
| B2 | 3 | 8 | 0.454 | 0.251 | 0.203 | 0.0218 | 0.00498 |
| AMD | 2 | 4 | 0.233 | 0.0457 | 0.187 | 0.0201 | 0.00498 |
| VOT | 1 | 2 | 0.0568 | 0.00622 | 0.0506 | 0.00544 | 0.00498 |

**Total summed usable bits across all 30 features = 15.685 bits** (perm p ≈
0.00498 for every feature — all significantly above the null). This sum is an
**optimistic over-count**: the features are correlated, so their per-feature MI
cannot simply be added. F0 alone supplies 1.569 bits (10 % of H(speaker)).

---

## 5. Effective dimensionality (participation ratio)

PR = (Σλ)² / Σλ² of the eigenvalues of the z-scored per-speaker feature
covariance over the 30 measured features; 95 % CIs from 1,000 speaker-level
bootstraps (seed 1234). Parent-residual = PR of the residuals after regressing
each per-speaker feature mean on [intercept, sex, height_cm, age_yr] (parents
available in TIMIT `DOC/SPKRINFO.TXT`). Source `results/effective_dim.json`.

| Condition | PR | 95 % CI | N speakers |
|---|---|---|---|
| **Pooled** | **8.39** | [8.00, 8.59] | 630 |
| Within male | 8.57 | [7.88, 9.05] | 438 |
| Within female | 9.89 | [8.78, 10.16] | 192 |
| **Within-sex (mean)** | **9.23** | — | — |
| **Parent-residual** (sex+height+age) | **9.42** | [8.81, 9.82] | 629 |

**Rise across the three:** PR climbs from **8.39 (pooled) → 9.23 (within-sex mean)
→ 9.42 (parent-residual)**, a total increase of **≈1.03 PR units (+12.3 %)** from
pooled to parent-residual. The direction confirms the hypothesis — effective
dimensionality **survives, and slightly increases, after the dominant shared-parent
confounders (sex, body size, age) are removed** — but the magnitude is modest, not
a large jump. Interpretation: the shared-parent variables (chiefly sex) load on a
small number of correlated directions; removing them flattens the top eigenvalues
and raises the participation ratio, yet ~9 effective dimensions of speaker-specific
structure remain across the 30-feature space. The within-female PR (9.89) is
notably higher than within-male (8.57), but with overlapping CIs and a much
smaller female N (192).

---

## 6. Joint usable speaker bits — held-out classifier lower bound

Features with ≥90 % coverage kept (all 30 measured features qualify);
incomplete rows listwise-deleted. **Retained: 5,679 / 6,300 utterances, 630
speakers, 30 features.** Chance = 1/630 = 0.001587; H(speaker) = log₂(630) =
**9.299 bits**. Utterance-disjoint stratified 5-fold CV (seed 1234), z-scored on
train-fold statistics only. Source `results/classifier.json`.

| Classifier | top-1 acc | 95 % CI | per-fold mean±std | log-loss (bits) | Fano (bits) | x-ent (bits) |
|---|---|---|---|---|---|---|
| Logistic regression (L2) | 0.5564 | [0.5435, 0.5694] | 0.5564 ± 0.0152 | 2.946 | 4.185 | 6.353 |
| MLP (256 hidden) | 0.4802 | [0.4672, 0.4932] | 0.4802 ± 0.0093 | 3.402 | 3.468 | 5.897 |
| Shrinkage LDA | **0.6151** | [0.6024, 0.6277] | 0.6151 ± 0.0116 | 2.952 | **4.759** | 6.347 |

Bit lower bounds — Fano: `I ≥ H(speaker) − [H_b(P_e) + P_e·log₂(S−1)]`;
cross-entropy: `H(speaker) − mean test log-loss (bits)`.

- **Headline Fano lower bound = 4.759 bits** (LDA, the strongest model).
- Cross-entropy lower bounds are higher (LDA 6.347, logreg 6.353 bits) but the
  contract headline is the larger **Fano** bound = **4.759 bits**.
- All bounds are **floors**: a stronger classifier can only raise them, and every
  bound stays below the H(speaker) = 9.299-bit sample ceiling.

**Capacity-inversion check: TRUE.** The higher-capacity MLP (0.4802) underperforms
**both** linear models (LDA 0.6151, logreg 0.5564) — the expected data-starvation
signature at ~8 training utterances/speaker (the MLP cannot fit a 630-way head
from so few examples per class).

---

## 7. Collision-metric sanity cross-check (optional)

Plugging the measured operating points into the collision formulae at n = 10¹⁰.
Source `results/collision.json`. Pooled F0 q_max = 3 was used as the
representative q (with a conservative q = 5).

| d source | PR (d) | q | log₁₀(m) | P(E) | P(M) | P(B) |
|---|---|---|---|---|---|---|
| PR_pooled | 8.389 | 3 | 4.003 | 1.0 | 9.94e-05 | 1.0 |
| PR_pooled | 8.389 | 5 | 5.864 | 1.0 | 1.37e-06 | 1.0 |
| PR_parent_residual | 9.422 | 3 | 4.495 | 1.0 | 3.20e-05 | 1.0 |
| PR_parent_residual | 9.422 | 5 | 6.586 | 1.0 | 2.60e-07 | 1.0 |

At the measured effective dimensionality (~8–9) and per-feature resolution
(q = 3–5), the code space m = q^d is only 10⁴–10⁶⁶, far below n = 10¹⁰, so the
expected number of collisions is large: **P(E) = 1 and P(B) = 1** (a birthday
collision among 10¹⁰ draws is essentially certain). The per-pair match
probability P(M) stays small (≈10⁻⁵–10⁻⁷). This is a sanity cross-check only,
not a headline, and it reinforces §5: ~9 effective, modestly-resolved dimensions
do not provide enough capacity to keep 10¹⁰ voices collision-free.

---

## Headline numbers (for direct quotation)

- **Measured features: 30 / 40** (10 not-measured: GCT, CQ, MFDR, SQ, NAQ, IHI,
  VFP, Nasality, SSPF, BGD — all 0 coverage, excluded, never imputed).
- **Decode failures: 0** (6,300/6,300 utterances decoded; extraction 4,949 s).
- **F0 F-ratio: pooled 18.69, within-sex 3.57** (male 3.05 / female 4.10);
  **F0 q_max = 3 pooled, 2 within-sex.**
- Top per-feature usable bit depth: **F0 = 1.569 bits**; total summed (optimistic,
  correlated) = **15.685 bits**.
- **PR(pooled) = 8.39** [8.00, 8.59]; **PR(within-sex) = 9.23** (M 8.57 / F 9.89);
  **PR(parent-residual) = 9.42** [8.81, 9.82] — rise of +12.3 % pooled→residual.
- **Classifier top-1: LDA 0.6151, logreg 0.5564, MLP 0.4802** (630-way,
  chance 0.00159); **capacity inversion = TRUE**.
- **Fano lower bound = 4.759 bits** (LDA, headline); cross-entropy lower bound =
  6.347 bits (LDA); both floors, both below H(speaker) = 9.299 bits.

*All F-ratios and q_max values are OPTIMISTIC UPPER BOUNDS because TIMIT is
single-session (no day-to-day/health/channel/affective within-speaker variation).*
