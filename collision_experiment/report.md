# Human Voice Is Unique — Empirical Replication on Mozilla Common Voice 17

_Generated 2026-06-12 18:32. Fixed seed **1234**._

## Bottom line

We measured 40 of the paper's 41 features on **1755 speakers / 18861 clips** of multi-session Common Voice audio and stress-tested the paper's two load-bearing assumptions. Both fail empirically on realistic data:

1. **Features are far from independent.** The participation-ratio effective dimensionality is **d_eff ≈ 12**, not 41 — and not even the paper's conservative floor of 27. Forty measured axes carry ~12 independent ones.

2. **Usable per-feature resolution is q ≤ 2, not 5–10.** No feature's bin-crossing rate stays under 20% at q≥3; 22 of 40 fail even at q=2. Multi-session within-speaker variability moves speakers across the equiprobable bins the paper assumes are stable.

**Consequence (n=10¹⁰):** plugging the *measured* d_eff and q_max into the paper's own formulae moves every collision metric from 'astronomically unique' to 'collisions certain' — P(B)=1 and P(E) up to ~10⁻² even at q=10 (Step 5). The paper's 'one-in-a-septillion' figures are an artifact of the independence + high-q assumptions, which our measurements do not support.

**The honest caveat in the other direction:** at *sample* scale the 1,736 real speakers are perfectly separable on these features (zero collisions at q=2,3, Step 6), and the participation ratio is a *linear* redundancy measure, so it is a conservative (collision-pessimistic) summary. Voice clearly carries strong individuating information; what these data refute is the *specific* astronomically small collision probability, not the qualitative claim that voices are highly distinctive. A definitive population-scale verdict needs far more speakers and cleaner (uncompressed, single-channel-controlled) audio.

## 0. Data, provenance, and honesty notes

**Data source.** The brief specified `mozilla-foundation/common_voice_17_0` (HF, config `en`, split `validated`). As of October 2025 Mozilla **emptied** that repository (only `README.md` + `.gitattributes` remain) and moved Common Voice exclusively to the Mozilla Data Collective (account + terms required); additionally `datasets>=5.0` removed script-based loaders. The official MODE-B download is therefore **blocked**. To still run the experiment on the *identical* CV 17.0 English data, we used the public, non-gated parquet mirror **`fixie-ai/common_voice_17_0`**, which preserves the full official schema (`client_id, path, audio, sentence, up_votes, down_votes, age, gender, accent, locale, segment, variant`). If you require the official source, the manual step is: create a Mozilla Data Collective account, accept the CV terms, download the English `validated` tarball, and re-run in MODE A pointing at the local release dir.

**Subset.** Pooled the first 4 `en/validated` parquet shards (18861 kept clips spanning ~1755 qualifying speakers; ~19.2k distinct client_ids were scanned). Audio decoded via soundfile (libsndfile MP3), resampled to 16 kHz mono. **client_id is treated as the speaker label.**

**Speaker filter.** Kept client_ids with **>= 5 validated clips**; capped at **30 clips/speaker** (seeded random sample). Final: **1755 speakers / 18861 clips**.

Clips/speaker (kept): min 5, median 8, mean 10.7, max 30.

Sex: {'male_masculine': 1009, 'NaN': 389, 'female_feminine': 357}

Age: {'twenties': 584, 'NaN': 354, 'thirties': 306, 'fourties': 168, 'teens': 144, 'fifties': 104, 'sixties': 68, 'seventies': 23, 'eighties': 4}

Top accents: {'NaN': 631, 'United States English': 557, 'England English': 191, 'India and South Asia (India, Pakistan, Sri Lanka)': 111, 'Canadian English': 79, 'Australian English': 54, 'Southern African (South Africa, Zimbabwe, Namibia)': 25, 'New Zealand English': 19, 'Scottish English': 13, 'Irish English': 12}

> **Within-speaker variance is MULTI-SESSION and varied-channel here** (crowd-sourced, different devices/rooms/days). Unlike TIMIT (single session, read sentences), this makes within-speaker variance **realistic rather than optimistic** — so the F-ratios below are *not* the optimistic upper bound that single-session corpora produce; if anything they are conservative.

## 1. Feature coverage (Step 1)

Total utterances: **18861**. Coverage = fraction of utterances for which each canonical feature was successfully computed. Features are **never imputed**; failures are NaN and reported as missing.

**Measured** (coverage ≥ 80%, used downstream): **40** of 41.

**Excluded** (low/zero coverage): ['VOT']


| feature | group | coverage | status |
|---|---|---:|---|
| F0 | canonical41 | 1.00 | measured |
| jitter | canonical41 | 1.00 | measured |
| shimmer | canonical41 | 1.00 | measured |
| GCT | canonical41 | 1.00 | measured |
| CQ | canonical41 | 1.00 | measured |
| MFDR | canonical41 | 1.00 | measured |
| SQ | canonical41 | 1.00 | measured |
| NAQ | canonical41 | 1.00 | measured |
| SHR | canonical41 | 1.00 | measured |
| IHI | canonical41 | 1.00 | measured |
| VFI | canonical41 | 1.00 | measured |
| CPP | canonical41 | 1.00 | measured |
| F1 | canonical41 | 1.00 | measured |
| F2 | canonical41 | 1.00 | measured |
| F3 | canonical41 | 1.00 | measured |
| F4 | canonical41 | 1.00 | measured |
| F5 | canonical41 | 1.00 | measured |
| B1 | canonical41 | 1.00 | measured |
| B2 | canonical41 | 1.00 | measured |
| B3 | canonical41 | 1.00 | measured |
| B4 | canonical41 | 1.00 | measured |
| B5 | canonical41 | 1.00 | measured |
| VTLE | canonical41 | 1.00 | measured |
| Nasality | canonical41 | 1.00 | measured |
| SpectralSkewness | canonical41 | 1.00 | measured |
| SpectralKurtosis | canonical41 | 1.00 | measured |
| SpectralEntropy | canonical41 | 1.00 | measured |
| SpectralRolloff | canonical41 | 1.00 | measured |
| SpectralFlux | canonical41 | 1.00 | measured |
| AlphaRatio | canonical41 | 1.00 | measured |
| LHR | canonical41 | 1.00 | measured |
| SPI | canonical41 | 1.00 | measured |
| GNE | canonical41 | 1.00 | measured |
| dCPP | canonical41 | 1.00 | measured |
| VOT | canonical41 | 0.00 | NOT MEASURED |
| SpeechRate | canonical41 | 1.00 | measured |
| BGD | canonical41 | 1.00 | measured |
| SemitoneSDF0 | canonical41 | 1.00 | measured |
| AMD | canonical41 | 1.00 | measured |
| SSPF | canonical41 | 0.89 | measured |
| RMS | canonical41 | 1.00 | measured |
| HNR | aux_HNR | 1.00 | measured |

**Tiering / honesty.** F0, jitter, shimmer, HNR(aux), F1–F5, B1–B5, the spectral moments/rolloff/flux, AlphaRatio, LHR, RMS, AMD, SemitoneSDF0, VTLE(estimated from formants), SpeechRate and BGD (Praat/librosa, well-established) are **Tier A**. SHR, IHI, GNE, SPI, SSPF, VFI, Nasality are **Tier B** best-effort DSP — computed, but absolute calibration is approximate. NAQ/CQ/GCT/SQ/MFDR come from a custom **IAIF** glottal inverse filter (**Tier C**): they are *computed* (so coverage is high) but on 16 kHz MP3 crowd audio their reliability is the weakest in the set — treat their contribution to d_eff with caution. **VOT is NOT MEASURED** (requires phoneme-level forced alignment + stop-burst detection, unavailable here).

## 2. Population distributions & quantile bins (Step 2)

Per-speaker means computed for each measured feature; q-quantile bin boundaries for q ∈ {2,3,5,10} saved to `bins.json` (equiprobable-by-construction, matching the paper's §3.6 binning). Histograms in `figs/dist_<feature>.png`.

## 3. F-ratios and empirical q_max (Step 3)

One-way ANOVA across speakers per feature. within_var = mean over speakers of within-speaker variance (across that speaker's utterances); between_var = variance of per-speaker means; F_ratio = between/within. q_max = largest q∈{2,3,5,10} whose mean bin-crossing rate < 0.20. **Because within-speaker variance is multi-session, these F-ratios are realistic, not an optimistic upper bound.**

| feature | within_var | between_var | F_ratio | ANOVA_F | p | q_max |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 511 | 1.71e+03 | 3.35 | 37.1 | 0.0e+00 | 2 |
| CPP | 0.00934 | 0.0238 | 2.55 | 26.6 | 0.0e+00 | 2 |
| RMS | 0.000304 | 0.000684 | 2.25 | 20.6 | 0.0e+00 | 2 |
| dCPP | 0.000515 | 0.00112 | 2.18 | 22.6 | 0.0e+00 | 2 |
| AlphaRatio | 8.01 | 15.5 | 1.93 | 18.0 | 0.0e+00 | 2 |
| VTLE | 0.163 | 0.305 | 1.87 | 18.0 | 0.0e+00 | 2 |
| SPI | 10.3 | 19.1 | 1.86 | 17.8 | 0.0e+00 | 2 |
| LHR | 17 | 31.3 | 1.84 | 18.0 | 0.0e+00 | 2 |
| GCT | 0.582 | 1.06 | 1.82 | 18.0 | 0.0e+00 | 2 |
| SpectralRolloff | 2.24e+05 | 3.93e+05 | 1.75 | 16.1 | 0.0e+00 | 2 |
| SpectralEntropy | 0.000875 | 0.00147 | 1.68 | 16.4 | 0.0e+00 | 2 |
| F5 | 1.95e+04 | 3.05e+04 | 1.56 | 15.2 | 0.0e+00 | 2 |
| SpectralFlux | 0.000684 | 0.00105 | 1.54 | 15.7 | 0.0e+00 | 2 |
| F4 | 1.71e+04 | 2.59e+04 | 1.52 | 14.7 | 0.0e+00 | 2 |
| shimmer | 0.000404 | 0.000588 | 1.46 | 14.1 | 0.0e+00 | 2 |
| B4 | 5.14e+03 | 7.02e+03 | 1.37 | 12.9 | 0.0e+00 | 1 |
| B1 | 6.39e+03 | 7.98e+03 | 1.25 | 13.0 | 0.0e+00 | 2 |
| CQ | 0.00426 | 0.00517 | 1.22 | 11.6 | 0.0e+00 | 1 |
| F3 | 1.66e+04 | 2e+04 | 1.20 | 12.0 | 0.0e+00 | 1 |
| B2 | 5.86e+03 | 7.01e+03 | 1.20 | 11.6 | 0.0e+00 | 1 |
| B3 | 5.5e+03 | 6.49e+03 | 1.18 | 11.1 | 0.0e+00 | 1 |
| B5 | 1.14e+04 | 1.31e+04 | 1.16 | 10.9 | 0.0e+00 | 1 |
| F2 | 1.36e+04 | 1.57e+04 | 1.15 | 11.1 | 0.0e+00 | 1 |
| F1 | 1.49e+04 | 1.53e+04 | 1.03 | 10.5 | 0.0e+00 | 1 |
| jitter | 3.88e-05 | 3.85e-05 | 0.99 | 10.2 | 0.0e+00 | 1 |
| Nasality | 22.1 | 20.4 | 0.92 | 8.9 | 0.0e+00 | 1 |
| VFI | 0.00365 | 0.00329 | 0.90 | 9.6 | 0.0e+00 | 1 |
| NAQ | 0.00214 | 0.00164 | 0.76 | 7.5 | 0.0e+00 | 1 |
| SemitoneSDF0 | 2.88 | 2.06 | 0.72 | 7.1 | 0.0e+00 | 1 |
| IHI | 4.92e-05 | 3.28e-05 | 0.67 | 7.0 | 0.0e+00 | 1 |
| AMD | 0.0376 | 0.0228 | 0.61 | 6.1 | 0.0e+00 | 1 |
| SQ | 4.15 | 2.46 | 0.59 | 5.4 | 0.0e+00 | 1 |
| SpeechRate | 0.401 | 0.237 | 0.59 | 5.8 | 0.0e+00 | 1 |
| SHR | 0.066 | 0.037 | 0.56 | 5.7 | 0.0e+00 | 1 |
| SSPF | 1.21e+06 | 6.66e+05 | 0.55 | 4.8 | 0.0e+00 | 1 |
| MFDR | 0.00407 | 0.00206 | 0.51 | 5.0 | 0.0e+00 | 1 |
| GNE | 0.00042 | 0.000202 | 0.48 | 4.5 | 0.0e+00 | 1 |
| BGD | 1.44 | 0.628 | 0.44 | 4.0 | 0.0e+00 | 1 |
| SpectralSkewness | 2.2 | 0.658 | 0.30 | 2.8 | 2.0e-235 | 2 |
| SpectralKurtosis | 3.95e+07 | 6.01e+06 | 0.15 | 1.4 | 3.1e-25 | 2 |

**Key finding.** Only **24 of 40** measured features have F_ratio > 1 (between-speaker variance exceeds within-speaker), and only 4 exceed F_ratio = 2. The most individuating features are F0, CPP, RMS, dCPP and the spectral-balance/formant measures; many source and prosodic features (SpeechRate, SHR, SSPF, MFDR, GNE, BGD) have F_ratio < 1 — on multi-session crowd audio they are *not* speaker-discriminative.

**q_max distribution across measured features:** {1: 22, 2: 18}. **No feature supports q ≥ 3** on this data, and 22 features cannot hold even q = 2 (q_max = 1). This is a sharp empirical contrast with the paper, which adopts q = 10 as its finest setting and q = 5 as 'conservative': **on realistic multi-session audio the usable per-feature resolution is q ≤ 2**, because within-speaker spread routinely crosses bin boundaries. This is the single biggest driver of the measured-vs-assumed gap in Step 5.

## 4. Effective dimensionality d_eff (Step 4)

Per-speaker mean-feature matrix; three estimators, bootstrap 95% CIs over speakers (1000 reps for PR; 200 for occupancy).

**Pooled:**

| estimator | d_eff | 95% CI |
|---|---:|---|
| PR_pearson | 12.42 | [12.05, 12.60] |
| PR_spearman | 11.39 | [11.09, 11.60] |
| cell_occupancy_q2 | 10.76 | [10.06, 10.13] |
| cell_occupancy_q3 | 6.79 | [6.35, 6.39] |

- Cell-occupancy q=2: d_eff(full 40-feature set)=10.76, which equals the sample-size ceiling log(n)/log(q)=10.76 — i.e. all 1736 speakers already occupy distinct cells. Distinct cells first exceed 95% of n at a subset of only **20** random features. **Estimator (c) is therefore censored by n: it is a *lower bound* on the true d_eff** (with 40 features and q≥2 the cell space q^k ≫ n, so occupancy cannot grow past n). Its bootstrap CI even sits *below* the point estimate because resampling speakers with replacement leaves only ~63% distinct, mechanically lowering the unique-cell count — another reason to read (c) as a floor, not an estimate.

- Cell-occupancy q=3: d_eff(full 40-feature set)=6.79, which equals the sample-size ceiling log(n)/log(q)=6.79 — i.e. all 1736 speakers already occupy distinct cells. Distinct cells first exceed 95% of n at a subset of only **11** random features. **Estimator (c) is therefore censored by n: it is a *lower bound* on the true d_eff** (with 40 features and q≥2 the cell space q^k ≫ n, so occupancy cannot grow past n). Its bootstrap CI even sits *below* the point estimate because resampling speakers with replacement leaves only ~63% distinct, mechanically lowering the unique-cell count — another reason to read (c) as a floor, not an estimate.

Out of k=40 measured features, the participation ratio collapses the effective dimensionality to ~12 — i.e. the measured voice features are substantially correlated, well below nominal independence.

**Stratified d_eff (PR estimators):**

| stratum | n | estimator | d_eff | 95% CI |
|---|---:|---|---:|---|
| pooled | 1736 | PR_pearson | 12.42 | [12.05, 12.60] |
| pooled | 1736 | PR_spearman | 11.39 | [11.09, 11.60] |
| sex=male_masculine | 998 | PR_pearson | 12.59 | [11.99, 12.80] |
| sex=male_masculine | 998 | PR_spearman | 11.64 | [11.18, 11.96] |
| sex=female_feminine | 353 | PR_pearson | 13.14 | [12.09, 13.24] |
| sex=female_feminine | 353 | PR_spearman | 12.58 | [11.53, 12.85] |
| accent=United States English | 552 | PR_pearson | 11.67 | [11.06, 11.85] |
| accent=United States English | 552 | PR_spearman | 11.06 | [10.47, 11.30] |
| accent=England English | 191 | PR_pearson | 12.20 | [10.62, 12.31] |
| accent=England English | 191 | PR_spearman | 11.41 | [10.15, 11.68] |
| accent=India and South Asia (India, Pakistan, Sri Lanka) | 108 | PR_pearson | 11.72 | [9.70, 11.49] |
| accent=India and South Asia (India, Pakistan, Sri Lanka) | 108 | PR_spearman | 11.10 | [9.43, 11.04] |
| accent=Canadian English | 79 | PR_pearson | 10.18 | [8.26, 10.00] |
| accent=Canadian English | 79 | PR_spearman | 9.29 | [7.69, 9.30] |
| age=twenties | 576 | PR_pearson | 11.95 | [11.30, 12.13] |
| age=twenties | 576 | PR_spearman | 11.10 | [10.52, 11.38] |
| age=thirties | 305 | PR_pearson | 11.74 | [10.73, 11.94] |
| age=thirties | 305 | PR_spearman | 11.04 | [10.19, 11.30] |
| age=fourties | 166 | PR_pearson | 11.91 | [10.02, 11.88] |
| age=fourties | 166 | PR_spearman | 11.31 | [9.91, 11.53] |
| age=teens | 144 | PR_pearson | 10.74 | [9.28, 10.99] |
| age=teens | 144 | PR_spearman | 10.42 | [9.04, 10.64] |
| age=fifties | 101 | PR_pearson | 11.57 | [8.81, 11.29] |
| age=fifties | 101 | PR_spearman | 10.83 | [9.01, 10.92] |
| age=sixties | 67 | PR_pearson | 10.46 | [8.15, 10.27] |
| age=sixties | 67 | PR_spearman | 9.73 | [7.67, 9.81] |

> **Reading the strata (a key result — and a nuanced one).** Pooled d_eff = 12.42.

> - **Sex strata are *higher* than pooled** (male 12.59, female 13.14). This is expected and instructive: **sex is itself a correlation-inducing axis** — it jointly drives F0, the formants and VTLE, so *pooling across sexes inflates the feature correlations and lowers pooled d_eff*. Removing the sex axis de-correlates the features and raises d_eff. This is the mirror image of the paper's §3.7 point that sex/size correlations shrink the effective dimensionality.

> - **Homogeneous accent / age cohorts trend *lower* than pooled** (e.g. Canadian English d_eff=10.18; the US-English cohort 11.67 with a CI that does not overlap pooled). Restricting to an anatomically/­experientially more homogeneous group removes between-group spread along several axes at once, collapsing the effective dimensionality — this is the empirical signature of the paper's **'low-d_eff regime'** for homogeneous cohorts (§7).

> Net: the demographic axes move d_eff in *both* directions depending on whether they de-correlate (sex) or homogenize (accent/age) the feature set — a more honest picture than a single monotone 'drop'. Scree plots: `figs/scree_pooled.png`, `figs/scree_cohort.png`.

## 5. Collision metrics — measured vs assumed (Step 5)

n = 1e+10, p = 1e-09. Exact formulae P(E)=1−(1−1/m)^(n−1); S=⌈log(1−p)/log(1−1/m)⌉; P(M)=1/m; P(B)=1−∏(1−i/m) in log space; m=q^d. (Our `collision.py` reproduces the paper's Table 1 at d=41 exactly.)

| config | q | d_used | m=q^d | P(E) | S | P(M) | P(B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| a_full_independence | 2.0 | 40.00 | 1.10e+12 | 9.05e-03 | 1.10e+03 | 9.09e-13 | 1.00e+00 |
| b_deff_point | 2.0 | 12.42 | 5.48e+03 | 1.00e+00 | 1.00e+00 | 1.83e-04 | 1.00e+00 |
| b_deff_ci_lo | 2.0 | 12.05 | 4.25e+03 | 1.00e+00 | 1.00e+00 | 2.35e-04 | 1.00e+00 |
| b_deff_ci_hi | 2.0 | 12.60 | 6.19e+03 | 1.00e+00 | 1.00e+00 | 1.61e-04 | 1.00e+00 |
| a_full_independence | 3.0 | 40.00 | 1.22e+19 | 8.23e-10 | 1.22e+10 | 8.23e-20 | 9.84e-01 |
| b_deff_point | 3.0 | 12.42 | 8.43e+05 | 1.00e+00 | 1.00e+00 | 1.19e-06 | 1.00e+00 |
| b_deff_ci_lo | 3.0 | 12.05 | 5.64e+05 | 1.00e+00 | 1.00e+00 | 1.77e-06 | 1.00e+00 |
| b_deff_ci_hi | 3.0 | 12.60 | 1.02e+06 | 1.00e+00 | 1.00e+00 | 9.77e-07 | 1.00e+00 |
| a_full_independence | 5.0 | 40.00 | 9.09e+27 | 1.10e-18 | 9.09e+18 | 1.10e-28 | 5.50e-09 |
| b_deff_point | 5.0 | 12.42 | 4.80e+08 | 1.00e+00 | 1.00e+00 | 2.08e-09 | 1.00e+00 |
| b_deff_ci_lo | 5.0 | 12.05 | 2.66e+08 | 1.00e+00 | 1.00e+00 | 3.75e-09 | 1.00e+00 |
| b_deff_ci_hi | 5.0 | 12.60 | 6.38e+08 | 1.00e+00 | 1.00e+00 | 1.57e-09 | 1.00e+00 |
| a_full_independence | 10.0 | 40.00 | 1.00e+40 | 1.00e-30 | 1.00e+31 | 1.00e-40 | 5.00e-21 |
| b_deff_point | 10.0 | 12.42 | 2.63e+12 | 3.80e-03 | 2.63e+03 | 3.80e-13 | 1.00e+00 |
| b_deff_ci_lo | 10.0 | 12.05 | 1.13e+12 | 8.79e-03 | 1.13e+03 | 8.83e-13 | 1.00e+00 |
| b_deff_ci_hi | 10.0 | 12.60 | 3.95e+12 | 2.53e-03 | 3.95e+03 | 2.53e-13 | 1.00e+00 |
| c_qmaxcap_deff | 1.366 | 12.42 | 4.81e+01 | 1.00e+00 | 1.00e+00 | 2.08e-02 | 1.00e+00 |

- **(a) full independence** at k = measured features reproduces the paper's regime: P(B) falls from 1.0 (q=2) and 0.98 (q=3) to ~5e-9 (q=5) and ~5e-21 (q=10) — i.e. voices are effectively unique for q ≥ 5, exactly the paper's qualitative conclusion (here at k=40 rather than 41).

- **(b) measured d_eff ≈ 12** (with CI) sharply *raises* collision probabilities versus (a): the correlation correction the paper brackets as d_eff∈[27,41] is, on these measured features, **far more severe** (d_eff is a single-digit-to-low-double-digit number, not 27–41), because we measure only 40 features and they are heavily correlated.

- **(c) q capped at empirical q_max (geo-mean q≈1.37) + measured d_eff** is the most conservative reading and gives the highest collision probabilities — this is where the framework's optimism is most exposed.

## 6. Direct empirical collision check (Step 6)

Bin the real speakers over all measured features at q=2,3 and count actual shared-cell collisions; compare to predictions.

| stratum | q | n | occupied cells | collision cells | speakers in collisions | observed pairs | pred(full-indep) | pred(d_eff) | obs/full | obs/d_eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | 2 | 1736 | 1736 | 0 | 0 | 0 | 1.37e-06 | 2.75e+02 | 0.00e+00 | 0.00 |
| pooled | 3 | 1736 | 1736 | 0 | 0 | 0 | 1.24e-13 | 1.79e+00 | 0.00e+00 | 0.00 |

**What actually happened (and the honest reading).** Observed collisions = **0** at q=2 and **0** at q=3: all 1736 real speakers fall into 1736 *distinct* cells. This is because the nominal cell count q^k (2^40≈1.1e12, 3^40≈1.2e19) dwarfs the sample size, so even the full-independence model predicts ≈0 collisions and we observe 0 — i.e. **obs ≈ pred(full-independence), and obs ≪ pred(d_eff)** (the PR-d_eff model would predict ~275 colliding pairs at q=2).

Three things follow, stated plainly:

1. **The test is under-powered at population scale.** With only 1736 speakers and q^k cells, no collisions can occur regardless of correlation; this direct count therefore cannot confirm or refute what happens at n=10^10. What it *does* establish is that the real speakers are **fully separable** on the 40 measured features at q≥2 — consistent with voice uniqueness *at sample scale*.

2. **PR-d_eff is a conservative (collision-pessimistic) summary.** The participation ratio measures *linear* redundancy; the speakers' discrete cell occupancy retains more separating information than q^d_eff cells would imply, so the 275-pair prediction does not materialise. The true discrete uniqueness sits *above* the linear d_eff.

3. **The empirical occupied-cell dimension is censored by n.** log(occupied)/log(q) = log(1736)/log(2) ≈ 10.8 at q=2 (every speaker its own cell), matching the Step-4(c) saturation result — the dataset is simply too small to *observe* the q^d_eff cell collapse directly. The population-scale verdict must come from Steps 4–5 (d_eff + q_max extrapolated to n=10^10), not from this sample-level count.

## 7. Homogeneous-cohort sub-analysis

Largest accent cohort with ≥200 speakers: **United States English** (n=552). Re-ran Steps 4–6 within it.

- Cohort d_eff(PR-pearson) = **11.67** [11.04, 11.86] vs pooled **12.42**.

Cohort collision band (selected):

| config | q | d_used | m | P(E) | P(B) |
|---|---:|---:|---:|---:|---:|
| a_full_independence | 2.0 | 40.00 | 1.10e+12 | 9.05e-03 | 1.00e+00 |
| b_deff_point | 2.0 | 11.67 | 3.27e+03 | 1.00e+00 | 1.00e+00 |
| a_full_independence | 3.0 | 40.00 | 1.22e+19 | 8.23e-10 | 9.84e-01 |
| b_deff_point | 3.0 | 11.67 | 3.71e+05 | 1.00e+00 | 1.00e+00 |
| a_full_independence | 5.0 | 40.00 | 9.09e+27 | 1.10e-18 | 5.50e-09 |
| b_deff_point | 5.0 | 11.67 | 1.44e+08 | 1.00e+00 | 1.00e+00 |
| a_full_independence | 10.0 | 40.00 | 1.00e+40 | 1.00e-30 | 5.00e-21 |
| b_deff_point | 10.0 | 11.67 | 4.71e+11 | 2.10e-02 | 1.00e+00 |
| c_qmaxcap_deff | 1.366 | 11.67 | 3.81e+01 | 1.00e+00 | 1.00e+00 |

> Within a single accent the effective dimensionality drops further and the collision band inflates relative to the pooled population — an empirical demonstration of the paper's **'low-d_eff regime'** (ethnically/­anatomically homogeneous cohorts) without needing a twins corpus.

## 8. Limitations (honest)

- **Feature coverage:** 40/41 features measured; **VOT not measured** (needs forced alignment). Tier-C glottal features (NAQ/CQ/GCT/SQ/MFDR) come from IAIF on compressed audio and are the least reliable.

- **MP3 compression** removes/colours high-frequency and source detail, biasing spectral-balance, GNE, SSPF and the glottal-source features specifically.

- **client_id = speaker** assumption: Common Voice client_ids are accounts, not verified individuals; a shared account or one person with two accounts adds noise.

- **Subset size & shard pooling:** 4 of 138 shards; clips are shuffled across shards so a speaker's clips are a random subset of their recordings — within-speaker variance is well sampled, but speakers are not the full CV population.

- **16 kHz resampling** caps analysis at 8 kHz Nyquist, truncating sibilant energy (SSPF) and high-band ratios.

- **d_eff caveat:** estimated on the *measured* feature subset, so it is the effective dimensionality *of what we could measure*, not of the full 41-feature construct the paper posits.
