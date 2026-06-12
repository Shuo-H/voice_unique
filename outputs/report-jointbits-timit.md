# TIMIT — Joint Usable Speaker Bits (Classifier Lower Bound)

*Reproducibility:* `numpy.default_rng(1234)` for folds, 1000-rep bootstraps, and 200 permutations; sklearn `random_state=1234`. Corpus label: **TIMIT** (single-session, 630 speakers x 10 utts).

**The headline is a LOWER BOUND on joint speaker information, classifier-dependent.** It comes from held-out speaker-identification error via Fano's inequality. A stronger classifier can only raise it. The binned plug-in MI curve (Step 4) is a **censored sanity check only** — its flattening past the censor point is a sampling artifact, not saturation.

## Data and feature set

Reused the TIMIT per-utterance `features.parquet`. Features with >= 90% coverage were kept: **40 features**. Listwise-complete rows for the joint analysis: **5035 / 6300 utterances** (1265 dropped, 20.1%, because they were missing at least one feature — chiefly VOT/SQ/SSPF). After listwise deletion **S = 629 speakers** remain (one speaker had no complete utterance); chance accuracy = 1/629 = 0.0016; H(speaker)=log2(629)=9.297 bits.

Feature set: AMD, B1, B2, B3, B4, B5, BGD, CPP, CQ, F0, F1, F2, F3, F4, F5, GCT, GNE, HNR, IHI, LHR, MFDR, NAQ, RMS, SHR, SPI, SQ, SSPF, VOT, VTLE, alpha_ratio, dCPP, jitter, semitone_SD_F0, shimmer, spectral_entropy, spectral_flux, spectral_kurtosis, spectral_rolloff, spectral_skewness, speech_rate.

No measured feature fell below 90% coverage.

## STEP 1 — Held-out speaker identification

Utterance-disjoint stratified 5-fold CV (each speaker split across folds; no utterance shared train/test). Features z-scored on train-fold statistics only.

| classifier | top-1 acc | 95% CI | log-loss (bits) | log-loss (nats) | per-fold acc (mean±std) |
|---|---:|---|---:|---:|---|
| Logistic regression (A, linear, weak) | 0.648 | [0.636, 0.661] | 2.536 | 1.758 | 0.650±0.020 |
| MLP (B, nonlinear, higher-capacity) | 0.540 | [0.527, 0.554] | 2.968 | 2.057 | 0.542±0.012 |
| Shrinkage-LDA (C, strong linear reference) | 0.759 | [0.747, 0.771] | 2.246 | 1.557 | 0.761±0.014 |

All accuracies are vastly above chance (1/629=0.0016). Note the **capacity inversion**: the higher-capacity MLP (0.540) *underperforms* linear logistic regression (0.648) because with ~8 utterances per speaker the nonlinear model overfits. The shrinkage-LDA reference is strongest (0.759).

## STEP 2 — Fano and cross-entropy lower bounds on joint bits

Fano: I_lower = H(speaker) - [H_b(P_error) + P_error*log2(S-1)]. Cross-entropy: I_xent_lower = H(speaker) - mean test log-loss (bits).

| classifier | Fano I_lower (bits) | Fano 95% CI | cross-entropy I_lower (bits) | xent 95% CI |
|---|---:|---|---:|---|
| Logistic regression (A, linear, weak) | 5.090 | [4.963, 5.222] | 6.760 | [6.697, 6.829] |
| MLP (B, nonlinear, higher-capacity) | 4.028 | [3.900, 4.159] | 6.329 | [6.227, 6.432] |
| Shrinkage-LDA (C, strong linear reference) | 6.261 | [6.131, 6.392] | 7.051 | [6.903, 7.213] |

### HEADLINE (LOWER BOUND, classifier-dependent)

Per spec, the headline is the **larger Fano I_lower of the two required classifiers (A=logreg, B=MLP)**: **5.090 bits** (from logreg). The weaker spec classifier gives 4.028 bits. The cross-entropy bound is tighter (higher): 6.760 bits for the same model. The shrinkage-LDA reference pushes the Fano bound to 6.261 bits (xent 7.051) — concrete evidence that **a stronger model only raises the bound**, so all of these are floors, not estimates.

## STEP 3 — Incremental joint bits (classifier-driven)

Greedy forward selection driven by held-out Fano I_lower. Engine = the strongest available bound model (**lda**); per the spec's intent ('add the feature that most increases held-out I_lower') we use the model that yields the best bound rather than the nominally-stronger-but-overfitting MLP. Each candidate is scored under the same utterance-disjoint 5-fold CV.

Maximum cumulative I_lower = **6.324 bits**; reaches 95% of that (**6.008 bits**) at **25 features**. Full path in `jointbits_greedy_timit.csv`; curve in `figs/joint_bits_curve_timit.png`.

First selections (feature : cumulative I_lower bits):
  F0 0.046, F5 0.277, RMS 0.790, F4 1.439, dCPP 1.939, B5 2.435, B4 2.885, F3 3.248 ...

This classifier-driven curve is the information-theoretic effective dimensionality in bits. Any plateau is partly the sample ceiling (log2(S)=9.30 bits), so the flattening is a floor on usable bits, not a ceiling on the acoustics.

## STEP 4 — Binned greedy curve (CENSORED sanity check only)

Traditional binned plug-in greedy MI (b=1 binary per feature, Miller-Madow + 200x permutation null). **Censor point k* = 13**: the first step where the number of occupied joint cells exceeds N/5 = 1007. In the figure (`figs/binned_greedy_censored_timit.png`) the curve is SOLID up to k* and DASHED beyond, labeled 'joint cells > N/5, estimator unreliable'.

**Any flattening or negative gain beyond k*=13 is a sample-size artifact, not true saturation.** Within the reliable region the binned estimate reaches only ~1.341 bits — far below the classifier bound — because binary per-feature quantization throws away within-feature resolution. The Step-3 classifier curve supersedes this binned curve entirely.

## STEP 5 — Reconciliation with sample-scale separability

- Headline I_lower (Fano, logreg) = **5.090 bits** => 2^I_lower = **34 distinguishable speaker classes** implied.

- Actual speakers S = 629; sample ceiling log2(S) = 9.30 bits.

The headline Fano bound (5.090 bits, ~34 classes) sits **below** the log2(S)=9.30-bit ceiling, so the measurement is **classifier-limited, not sample-ceilinged**: with 629 speakers we are not yet saturating the sample, and the gap to 9.3 bits reflects that these summary features (plus this classifier) cannot perfectly separate all 629 speakers. The tighter cross-entropy bound (6.760 bits, ~108 classes) and the LDA reference (6.261 bits) move toward — but still under — the ceiling.

## Limitations

1. **Classifier lower bound is model-dependent.** Every number here is a floor: a stronger classifier can only increase I_lower. We directly see this — shrinkage-LDA (6.261 bits) exceeds the spec headline (5.090 bits). Do not read the headline as the true joint information.

2. **Sample ceiling.** All bounds are capped by H(speaker)=log2(629)=9.30 bits; no held-out experiment on 629 speakers can demonstrate more than that, regardless of how informative the voice truly is.

3. **Cross-entropy bound calibration.** The cross-entropy bound assumes the classifier's predicted probabilities are well-calibrated; if it is over-confident, its log-loss is inflated and the xent bound is loosened (made smaller). We report it as a secondary, calibration-sensitive bound, not the headline.

4. **Single-session TIMIT.** One recording session per speaker means within-speaker variability (day-to-day, health, channel, emotion) is absent. Identification is therefore easier than in the real world and the bits here are an **OPTIMISTIC upper bound on a lower bound**: cross-session data would lower accuracy, b*, and I_lower.

5. **Uneven speaker priors.** After listwise deletion speakers have unequal complete-utterance counts (1-10, median 8); 7 speakers have a single complete utterance and are unidentifiable when that utterance is held out, slightly depressing accuracy. The Fano bound assumes a near-uniform speaker prior, a mild approximation given this spread.


*Artifacts:* jointbits_classifiers_timit.csv, jointbits_greedy_timit.csv, binned_greedy_censored_timit.csv, figs/joint_bits_curve_timit.png, figs/binned_greedy_censored_tim