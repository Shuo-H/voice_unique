# Human Voice Individuality — Empirical Test on TIMIT

*Reproducibility:* fixed random seed **1234** (numpy global + bootstrap RNG). Corpus: TIMIT, 630 speakers x 10 utts = 6300 utterances, 16 kHz, NIST SPHERE decoded via `sphfile`.

This experiment re-examines Singh & Raj, *Human Voice is Unique*. The paper assumes the 41 acoustic features are independent and each carries q usable bins, giving m=q^41 voice cells. We instead **measure** how many features are reliably computable, how many bins each can support (q_max), and the **effective dimensionality** d_eff after correlations — then recompute the paper's four collision metrics with measured numbers.

## Headline findings

1. **Computability:** 40/42 of the paper's features were actually measurable on TIMIT; 2 (Nasality, VFI) were not, and 5 glottal-flow features are approximate (single-pass IAIF). The paper's m=q^41 already overstates the usable feature count.

2. **Usable bins are few:** empirically only 1 feature (F0) supports q=5 reliable bins; 22/40 features cannot even support q=2 at the utterance level (q_max=0). The paper's q=10 is far above what the data sustain.

3. **Effective dimensionality is small:** k=40 nominal features collapse to **d_eff = 5.4 [5.2, 5.7]** pooled (Pearson participation ratio). Most of that collapse is the sex axis: within-sex d_eff rises to ~11 (M) / ~12 (F). Either way it is far below 40.

4. **The headline reverses at population scale:** plugging the paper's own assumption (independence, k=40, q=10) reproduces its result — voices look unique (P(B)~1e-21 at n=1e10). But with measured d_eff (5–12) and realistic bins, the population-match probability P(B) saturates to ~1: at n=1e10 a colliding pair is effectively certain. The 'voice is unique' conclusion is an artifact of the independence + high-q assumptions, not of the acoustics.

5. **Direct check at n=629:** real speakers show 5 colliding pairs at q=2 — ~3e+07x more than the independence model predicts (~2e-07), i.e. the data **falsify** the independence-uniqueness model even at this small sample. At q=3 all 629 speakers separate. So voices are locally distinguishable at small n yet not collision-free at population scale.

## STEP 1 — Feature coverage

Of 42 candidate columns (operationalizing the paper's ~41 features; HNR retained as an auxiliary glottal measure), **40 were MEASURED** (coverage>0) and 2 were NOT MEASURED.

Features by coverage (NOT MEASURED flagged):

| feature | category | coverage | status | note |
|---|---|---:|---|---|
| F0 | glottal_source | 1.000 | MEASURED |  |
| jitter | glottal_source | 1.000 | MEASURED |  |
| shimmer | glottal_source | 1.000 | MEASURED |  |
| HNR | glottal_source | 1.000 | MEASURED |  |
| CPP | glottal_source | 1.000 | MEASURED |  |
| dCPP | glottal_source | 1.000 | MEASURED |  |
| SHR | glottal_source | 1.000 | MEASURED |  |
| IHI | glottal_source | 1.000 | MEASURED |  |
| SPI | glottal_source | 1.000 | MEASURED |  |
| GNE | glottal_source | 1.000 | MEASURED |  |
| F1 | vocal_tract_filter | 1.000 | MEASURED |  |
| F2 | vocal_tract_filter | 1.000 | MEASURED |  |
| F3 | vocal_tract_filter | 1.000 | MEASURED |  |
| F4 | vocal_tract_filter | 1.000 | MEASURED |  |
| F5 | vocal_tract_filter | 1.000 | MEASURED |  |
| B1 | vocal_tract_filter | 1.000 | MEASURED |  |
| B2 | vocal_tract_filter | 1.000 | MEASURED |  |
| B3 | vocal_tract_filter | 1.000 | MEASURED |  |
| B4 | vocal_tract_filter | 1.000 | MEASURED |  |
| B5 | vocal_tract_filter | 1.000 | MEASURED |  |
| VTLE | vocal_tract_filter | 1.000 | MEASURED |  |
| spectral_skewness | spectral_envelope | 1.000 | MEASURED |  |
| spectral_kurtosis | spectral_envelope | 1.000 | MEASURED |  |
| spectral_entropy | spectral_envelope | 1.000 | MEASURED |  |
| spectral_rolloff | spectral_envelope | 1.000 | MEASURED |  |
| spectral_flux | spectral_envelope | 1.000 | MEASURED |  |
| alpha_ratio | spectral_envelope | 1.000 | MEASURED |  |
| LHR | spectral_envelope | 1.000 | MEASURED |  |
| RMS | spectral_envelope | 1.000 | MEASURED |  |
| AMD | spectral_envelope | 1.000 | MEASURED |  |
| speech_rate | articulatory_prosodic | 1.000 | MEASURED |  |
| BGD | articulatory_prosodic | 1.000 | MEASURED |  |
| semitone_SD_F0 | articulatory_prosodic | 1.000 | MEASURED |  |
| NAQ | glottal_source | 0.998 | MEASURED |  |
| CQ | glottal_source | 0.998 | MEASURED |  |
| GCT | glottal_source | 0.998 | MEASURED |  |
| MFDR | glottal_source | 0.998 | MEASURED |  |
| SSPF | articulatory_prosodic | 0.945 | MEASURED |  |
| SQ | glottal_source | 0.924 | MEASURED |  |
| VOT | articulatory_prosodic | 0.901 | MEASURED |  |
| VFI | glottal_source | 0.000 | NOT MEASURED | no reliable single-session vocal-fry detector implemented; would be fabrication to guess |
| Nasality | vocal_tract_filter | 0.000 | NOT MEASURED | requires nasalance (oral+nasal channels); not estimable from single TIMIT channel |

## STEP 3 — F-ratios (speaker separability) and empirical q_max

`within_var` = mean over speakers of within-speaker variance (across that speaker's 10 utts); `between_var` = variance of per-speaker means; `F_ratio = between/within`. `q_max` = largest q in {2,3,5,10} whose mean bin-crossing rate < 0.2.

> **Caveat (important):** TIMIT within-speaker variance is **single-session** (one recording per speaker). Real day-to-day, health, and emotional variation is absent, so these F-ratios and q_max values are an **OPTIMISTIC UPPER BOUND** on true separability.

| feature | F_ratio | ANOVA_F | p | q_max |
|---|---:|---:|---:|---:|
| F0 | 35.28 | 352.8 | 0 | 5 |
| F4 | 13.00 | 130.0 | 0 | 3 |
| CPP | 11.24 | 112.4 | 0 | 3 |
| F5 | 10.59 | 105.9 | 0 | 3 |
| dCPP | 8.94 | 89.4 | 0 | 3 |
| VTLE | 6.13 | 61.3 | 0 | 2 |
| GCT | 5.89 | 58.9 | 0 | 3 |
| HNR | 3.86 | 38.6 | 0 | 2 |
| SHR | 3.84 | 38.4 | 0 | 2 |
| RMS | 3.67 | 36.7 | 0 | 2 |
| F3 | 3.50 | 35.0 | 0 | 2 |
| NAQ | 3.40 | 33.9 | 0 | 2 |
| spectral_entropy | 2.18 | 21.8 | 0 | 2 |
| CQ | 2.05 | 20.5 | 0 | 2 |
| MFDR | 1.99 | 19.9 | 0 | 2 |
| B4 | 1.73 | 17.3 | 0 | 2 |
| SSPF | 1.62 | 15.3 | 0 | 2 |
| shimmer | 1.53 | 15.3 | 0 | 2 |
| B2 | 1.45 | 14.5 | 0 | 0 |
| alpha_ratio | 1.44 | 14.4 | 0 | 0 |
| F2 | 1.40 | 14.0 | 0 | 0 |
| B3 | 1.38 | 13.8 | 0 | 0 |
| spectral_rolloff | 1.37 | 13.7 | 0 | 0 |
| B5 | 1.28 | 12.8 | 0 | 0 |
| F1 | 1.19 | 11.9 | 0 | 0 |
| B1 | 1.17 | 11.7 | 0 | 0 |
| spectral_skewness | 1.09 | 10.9 | 0 | 0 |
| IHI | 1.07 | 10.7 | 0 | 0 |
| spectral_kurtosis | 1.05 | 10.5 | 0 | 0 |
| LHR | 0.98 | 9.8 | 0 | 0 |
| jitter | 0.85 | 8.5 | 0 | 0 |
| SQ | 0.81 | 7.9 | 0 | 0 |
| SPI | 0.79 | 7.9 | 0 | 0 |
| semitone_SD_F0 | 0.72 | 7.2 | 0 | 0 |
| speech_rate | 0.72 | 7.2 | 0 | 0 |
| spectral_flux | 0.47 | 4.7 | 4.74e-223 | 0 |
| GNE | 0.44 | 4.4 | 6.00e-199 | 0 |
| AMD | 0.32 | 3.2 | 1.48e-112 | 0 |
| VOT | 0.18 | 1.6 | 6.06e-19 | 0 |
| BGD | 0.16 | 1.6 | 6.57e-17 | 0 |

q_max distribution across measured features: {0: 22, 2: 12, 3: 5, 5: 1}.

## STEP 4 — Effective dimensionality d_eff (key result)

Built per-speaker mean matrix over the 40 features with >= 90% speaker coverage and non-zero variance, then listwise-complete speakers. Bootstrap 95% CIs resample speakers (1000 reps, seed 1234).

| estimator | stratum | n_speakers | k | d_eff | 95% CI |
|---|---|---:|---:|---:|---|
| PR_pearson | pooled | 629 | 40 | 5.43 | [5.17, 5.66] |
| PR_spearman | pooled | 629 | 40 | 5.90 | [5.52, 6.36] |
| PR_pearson | sex=M | 437 | 40 | 11.45 | [10.64, 11.75] |
| PR_spearman | sex=M | 437 | 40 | 11.82 | [11.03, 12.16] |
| PR_pearson | sex=F | 192 | 40 | 11.60 | [10.18, 11.92] |
| PR_spearman | sex=F | 192 | 40 | 12.08 | [10.78, 12.37] |

**Headline:** nominal feature count k=40, but effective dimensionality (Pearson participation ratio, pooled) **d_eff = 5.4 [5.2, 5.7]** — i.e. correlations collapse roughly 35 nominal axes. See `figs/scree_pearson.png`.

Cell-occupancy growth (estimator c): d_eff_occ = log(#occupied cells)/log(q) over growing random feature subsets. This saturates once q^subset approaches the speaker count (n=629); reported value is informative only below that break.

- q=2: breaks (q^subset>n) at subset size 10; last unsaturated size 9 gives #occupied=221, d_eff_occ=7.79.
- q=3: breaks (q^subset>n) at subset size 6; last unsaturated size 5 gives #occupied=168, d_eff_occ=4.67.

## STEP 5 — Collision metrics: ASSUMED vs MEASURED

Population n=1e+10; match-at-p uses p=1e-09. m=q^d. (a) full independence d=k (the paper's assumption); (b) measured d_eff (Pearson PR, pooled) with its 95% CI; (c) cap each feature's q at its empirical q_max, then use d_eff (m = q_eff^d_eff, q_eff = geometric mean of min(q,q_max)).

| method | variant | q | d | log10(m) | P(E) | S | P(M) | P(B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a_independence | point | 2 | 40.0 | 12.0 | 9.05e-03 | 1.10e+03 | 9.09e-13 | 1.00e+00 |
| b_deff | point | 2 | 5.4 | 1.6 | 1.00e+00 | 1.00e+00 | 2.31e-02 | 1.00e+00 |
| b_deff | ci_lo | 2 | 5.2 | 1.6 | 1.00e+00 | 1.00e+00 | 2.79e-02 | 1.00e+00 |
| b_deff | ci_hi | 2 | 5.7 | 1.7 | 1.00e+00 | 1.00e+00 | 1.98e-02 | 1.00e+00 |
| c_deff_qcap | point | 2 | 5.4 | 0.7 | 1.00e+00 | 1.00e+00 | 1.84e-01 | 1.00e+00 |
| c_deff_qcap | ci_lo | 2 | 5.2 | 0.7 | 1.00e+00 | 1.00e+00 | 2.00e-01 | 1.00e+00 |
| c_deff_qcap | ci_hi | 2 | 5.7 | 0.8 | 1.00e+00 | 1.00e+00 | 1.71e-01 | 1.00e+00 |
| a_independence | point | 3 | 40.0 | 19.1 | 8.23e-10 | 1.22e+10 | 8.23e-20 | 9.84e-01 |
| b_deff | point | 3 | 5.4 | 2.6 | 1.00e+00 | 1.00e+00 | 2.55e-03 | 1.00e+00 |
| b_deff | ci_lo | 3 | 5.2 | 2.5 | 1.00e+00 | 1.00e+00 | 3.43e-03 | 1.00e+00 |
| b_deff | ci_hi | 3 | 5.7 | 2.7 | 1.00e+00 | 1.00e+00 | 2.00e-03 | 1.00e+00 |
| c_deff_qcap | point | 3 | 5.4 | 0.9 | 1.00e+00 | 1.00e+00 | 1.32e-01 | 1.00e+00 |
| c_deff_qcap | ci_lo | 3 | 5.2 | 0.8 | 1.00e+00 | 1.00e+00 | 1.46e-01 | 1.00e+00 |
| c_deff_qcap | ci_hi | 3 | 5.7 | 0.9 | 1.00e+00 | 1.00e+00 | 1.21e-01 | 1.00e+00 |
| a_independence | point | 5 | 40.0 | 28.0 | 1.10e-18 | 9.09e+18 | 1.10e-28 | 5.50e-09 |
| b_deff | point | 5 | 5.4 | 3.8 | 1.00e+00 | 1.00e+00 | 1.59e-04 | 1.00e+00 |
| b_deff | ci_lo | 5 | 5.2 | 3.6 | 1.00e+00 | 1.00e+00 | 2.45e-04 | 1.00e+00 |
| b_deff | ci_hi | 5 | 5.7 | 4.0 | 1.00e+00 | 1.00e+00 | 1.11e-04 | 1.00e+00 |
| c_deff_qcap | point | 5 | 5.4 | 0.9 | 1.00e+00 | 1.00e+00 | 1.23e-01 | 1.00e+00 |
| c_deff_qcap | ci_lo | 5 | 5.2 | 0.9 | 1.00e+00 | 1.00e+00 | 1.37e-01 | 1.00e+00 |
| c_deff_qcap | ci_hi | 5 | 5.7 | 0.9 | 1.00e+00 | 1.00e+00 | 1.13e-01 | 1.00e+00 |
| a_independence | point | 10 | 40.0 | 40.0 | 1.00e-30 | 1.00e+31 | 1.00e-40 | 5.00e-21 |
| b_deff | point | 10 | 5.4 | 5.4 | 1.00e+00 | 1.00e+00 | 3.67e-06 | 1.00e+00 |
| b_deff | ci_lo | 10 | 5.2 | 5.2 | 1.00e+00 | 1.00e+00 | 6.83e-06 | 1.00e+00 |
| b_deff | ci_hi | 10 | 5.7 | 5.7 | 1.00e+00 | 1.00e+00 | 2.21e-06 | 1.00e+00 |
| c_deff_qcap | point | 10 | 5.4 | 0.9 | 1.00e+00 | 1.00e+00 | 1.23e-01 | 1.00e+00 |
| c_deff_qcap | ci_lo | 10 | 5.2 | 0.9 | 1.00e+00 | 1.00e+00 | 1.37e-01 | 1.00e+00 |
| c_deff_qcap | ci_hi | 10 | 5.7 | 0.9 | 1.00e+00 | 1.00e+00 | 1.13e-01 | 1.00e+00 |

## STEP 6 — Direct empirical collision check (falsifiable test)

Bin all 629 analysis speakers (per-speaker means) at q=2,3 over the measured features and count real collisions; compare to predictions under (a) independence and (b) d_eff.

| q | speakers | occupied cells | observed colliding pairs | pred pairs (indep, d=k) | pred pairs (d_eff) |
|---:|---:|---:|---:|---:|---:|
| 2 | 629 | 624 | 5 | 1.80e-07 | 4.57e+03 |
| 3 | 629 | 629 | 0 | 1.62e-14 | 5.04e+02 |

**Interpretation (this is the falsifiable test):** the independence model with k=40 binary axes says m=2^40≈1e12, so among 629 speakers it predicts ~2e-07 colliding pairs — essentially zero. The data instead show **5 real colliding pairs** at q=2, ~3e+07x the independence prediction. The independence assumption is therefore **falsified**: real voices clump far more than q^k implies, exactly because the features are correlated (low d_eff). The crude d_eff-as-exponent model (m=2^d_eff) over-corrects in the other direction (predicts ~5e+03 pairs, far more than observed), which shows a participation ratio is a useful *summary* of collapse but not itself a calibrated collision model. At q=3 all 629 speakers occupy distinct cells (0 collisions): at small n voices remain locally separable even though, extrapolated to n=1e10 with the same effective space, a population collision becomes certain.

## Honest limitations

Two within-speaker variance biases pull in **opposite** directions and neither is controlled here. (i) TIMIT is **single-session** (one recording per speaker), so day-to-day, health and emotional variation is absent — this makes F-ratios and q_max *optimistic*. (ii) A speaker's 10 utterances are **different sentences**, so across-utterance variance also absorbs phonetic-content differences that are not speaker identity — this inflates within-speaker variance and makes q_max *pessimistic*. The net direction is unknown; q_max should be read as indicative, not exact. Feature **coverage is incomplete**: 2 features are NOT MEASURED (Nasality needs a nasal channel; VFI has no reliable single-session estimator), and glottal-flow features (NAQ/CQ/GCT/MFDR/SQ) come from a single utterance-level IAIF inverse filtering with physiological QC, so they are approximate. The cell-occupancy estimator (c) is **small-n constrained**: with ~hundreds of speakers it saturates after only a handful of binary features, so it lower-bounds rather than measures d_eff. d_eff itself is a second-moment (correlation) summary and does not capture all higher-order dependence, so true usable dimensionality is likely <= the reported d_eff. No value was ever imputed or interpolated; missing stayed missing.


*Artifacts:* features.parquet, coverage.csv, bins.json, fratios.csv, deff.csv, deff_occupancy.csv, collisions.csv, direct_collisions.csv, figs/ (dist_*.png, scree_pearson.png), and this report.md. Seed=1234.
