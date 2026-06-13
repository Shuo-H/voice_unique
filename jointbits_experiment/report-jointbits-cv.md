# Joint usable speaker-information (bits) on Common Voice — a classifier-based lower bound

**One-line claim.** Using only measured acoustic features and a held-out speaker-identification classifier, the *joint usable speaker information* on Common Voice is **at least 6.92 bits** (cross-entropy bound, strongest classifier = logreg; ultra-conservative Fano floor = 4.88 bits). Every number here is a **LOWER BOUND**: classifier- and sample-dependent, and can only *rise* with a stronger classifier (e.g. a speaker-embedding net) or a larger corpus. The binned plug-in MI curve (Step 4) is a **censored sanity check only**.

**Headline cross-corpus finding (Step 6).** Run through the *identical* pipeline at matched S=630 and the same 28 features, multi-session mp3 Common Voice and single-session studio TIMIT are **statistically tied on speaker-ID accuracy** (best top-1 ≈ 0.626 vs 0.627), and TIMIT's usable-bits lower bound is only **~0.3 bits higher** (6.64 vs 6.37). The large single-vs-multi-session degradation one might expect a priori **does not appear** on this session-stable acoustic feature set — the robust claim is a *small, classifier-dependent* gap, not a large drop.

Seed **1234** everywhere (numpy default_rng, sklearn random_state, all folds, bootstraps, permutations). `speaker_id` (= Common Voice `client_id`) is taken as the speaker label — stated assumption. Identification among known speakers (not verification), utterance-disjoint 5-fold CV, balanced to **10 clips/speaker** to match TIMIT.

## Data, balancing, and feature-coverage handling

**Source.** Reused the Common Voice per-utterance `features.parquet` from the prior CV MI run (`mi_experiment/features.parquet`, long format: `speaker_id, utt_id, feature, value` + `sex/accent/age`). No re-extraction. The prior run measured **28** acoustic features and logged the 14-member glottal/inverse-filtering family (GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE, Nasality, SSPF, VOT, BGD) as **NOT MEASURED (0 coverage)** — so the anticipated sparse features (VOT, SSPF, …) are excluded *a priori*, not by the coverage filter.

**Coverage filter (drop sparse FEATURES, never utterances).** Of the 28 measured features, per-feature coverage ranged 0.9999–1.0000; **all ≥ 0.95**, so **28 features kept, 0 dropped** (none). Keeping listwise-complete utterances over the kept features: **19187 utts kept, 1 dropped** (a single utt with a NaN F0).

**Balancing.** Kept speakers with ≥10 complete clips and sampled **exactly 10** per speaker (seed 1234). Final **S_full = 1599 speakers, N = 15990 clips**, uniform speaker prior by construction, ceiling **H(speaker)=log2(S)=10.643 bits**.

**Sensitivity — drop-sparse-features vs TIMIT-style listwise-delete-utterances.** Because CV coverage is ~100%, the two policies are nearly identical here:

| policy | #features | #utts | S | top-1 acc (LDA) |
|---|---|---|---|---|
| drop sparse features (ours) | 28 | 15990 | 1599 | 0.518 |
| keep all features, listwise-delete utts (TIMIT-style) | 28 | 15990 | 1599 | 0.518 |

Cost of the TIMIT-style choice on CV: Δ top-1 acc = **0.0000** (negligible — no sparse features to force a trade-off). On a corpus with genuinely sparse features the listwise choice would discard utterances and shrink S; here it does not, so the CV↔TIMIT comparison is not distorted by the missing-data policy.

## Step 1 — Held-out speaker identification (3 classifiers)

_Common Voice, full balanced set  (S=1599, chance=1/S=6.25e-04, ceiling H=log2(S)=10.643 bits)_

| classifier | top-1 acc [95% CI] | per-fold acc (mean±std) | log-loss (bits / nats) | Fano I_lower (bits) [CI] | xent I_lower (bits) [CI] |
|---|---|---|---|---|---|
| A · multinomial logreg (mild L2) **(strongest)** | 0.551 [0.544, 0.559] | 0.551±0.007 | 3.727 / 2.583 | 4.877 [4.799, 4.960] | 6.916 [6.864, 6.970] |
| B · small MLP | 0.510 [0.503, 0.518] | 0.510±0.010 | 4.550 / 3.154 | 4.432 [4.355, 4.513] | 6.093 [5.988, 6.197] |
| C · shrinkage-LDA (Ledoit-Wolf) | 0.518 [0.510, 0.526] | 0.518±0.006 | 5.114 / 3.545 | 4.516 [4.431, 4.596] | 5.528 [5.401, 5.662] |

**Capacity inversion:** MLP top-1 (0.510) < logreg top-1 (0.551) — **inversion present**, as on TIMIT. The higher-capacity nonlinear model does *worse* than the weak linear one: attributable to only ~10 clips/speaker (8 train) starving a 1599-way nonlinear classifier. The strongest model by the (tighter) cross-entropy bound is **logreg**.

## Step 2 — Fano + cross-entropy lower bounds

With uniform prior, H(speaker)=log2(S)=10.643 bits.

- **Headline (primary): cross-entropy bound, strongest classifier (logreg) = 6.916 bits** [95% CI 6.864, 6.970]. Label: *lower bound, classifier+sample dependent.*
- **Ultra-conservative floor-of-floors: Fano bound, logreg = 4.877 bits** [4.799, 4.960]. Fano is worst-case (only the error *rate* enters), so it is necessarily looser than xent. *"Floor-of-floors" here means the worst-case bound TYPE (Fano) on the standard weak linear baseline (logreg), not the numerically smallest across all models: the MLP's Fano (4.432) is lower still, but only because the MLP is the capacity-inverted, starved classifier — a worse model, not a tighter floor. Among the non-degenerate models, logreg is the right conservative reference.*

**Calibration check (xent validity).** Reliability of the strongest classifier (logreg) top-1 confidence on held-out clips: **ECE = 0.2526**. 10-bin reliability table:

| conf bin | n | mean confidence | accuracy | gap (acc−conf) |
|---|---|---|---|---|
| [0.0, 0.1) | 3203 | 0.070 | 0.211 | 0.142 |
| [0.1, 0.2) | 4276 | 0.146 | 0.391 | 0.245 |
| [0.2, 0.3) | 2472 | 0.246 | 0.583 | 0.337 |
| [0.3, 0.4) | 1672 | 0.346 | 0.702 | 0.355 |
| [0.4, 0.5) | 1218 | 0.448 | 0.811 | 0.363 |
| [0.5, 0.6) | 961 | 0.548 | 0.836 | 0.288 |
| [0.6, 0.7) | 755 | 0.649 | 0.922 | 0.273 |
| [0.7, 0.8) | 625 | 0.750 | 0.949 | 0.199 |
| [0.8, 0.9) | 500 | 0.847 | 0.958 | 0.111 |
| [0.9, 1.0) | 308 | 0.941 | 0.961 | 0.020 |

The strongest classifier is systematically **UNDER-confident** here (weighted mean gap acc−conf = 0.253; accuracy exceeds stated confidence in every bin). **Any** miscalibration — over- or under-confident — inflates log-loss relative to the true posterior, which only **loosens** the cross-entropy bound. So the bound remains a valid, *conservative* floor: a temperature-/Platt-calibrated version of the same model would have lower log-loss and would **raise** the bound. The reported xent bound is therefore pessimistic on this axis too.

## Step 3 — Incremental joint bits (classifier-driven, LDA)

Greedy forward selection driven by held-out cross-entropy I_lower under shrinkage-LDA (the designated, fast incremental bound model), same utterance-disjoint CV, all features added (no early stop). **Max I_lower = 5.676 bits**; **18 features reach 95% of it.**

Selection order (first 12): F0 → AlphaRatio → RMS → SpectralRolloff → dCPP → SpectralFlux → B4 → F4 → HNR → B1 → LHR → B3 …

This is **usable joint bits extracted by this model on this corpus**, *not* a dimensionality count. Any plateau is partly the log2(S)=10.643-bit **sample ceiling**, not a property of the features. See `figs/joint_bits_curve_cv.png`.

## Step 4 — Binned plug-in greedy MI (CENSORED sanity check only)

Binary (median-split) per-feature greedy plug-in MI, Miller-Madow corrected, 200× permutation null. **Censor point k\* = 2** — the first step where occupied joint cells exceed N/5 = 3198. Plotted SOLID to k\*, DASHED beyond (`figs/binned_greedy_censored_cv.png`).

The permutation-null-corrected MI peaks at ~1.08 bits by step 4 and then **declines** as more binary features are added — the joint cell count outruns the sample, the null rises to meet the plug-in estimate, and the estimator becomes unreliable. This flattening/decline beyond k\* is a **sampling artifact, not saturation**. **Step 3 (classifier-driven) supersedes this curve** as the real bound; Step 4 only confirms the binned estimator censors itself exactly where finite-sample bias takes over.

## Step 5 — Reconciliation (the bound is a floor; three sources push the truth higher)

Headline I_lower = **6.92 bits** → 2^I_lower ≈ **121 implied distinguishable classes**, against S_full = 1599 speakers and a ceiling log2(S) = 10.643 bits (1,599 = S by definition).

**Do NOT read 2^I_lower as 'the number of distinguishable voices.'** Three independent sources push the *true* usable information **above** this headline:

1. **Bound looseness.** Fano is worst-case; cross-entropy is tighter but still a floor (equality only for a perfect posterior). Both are lower bounds by construction.
2. **Classifier limitation.** ~10 clips/speaker (8 train) starves the models — the capacity inversion is the symptom. A proper speaker-embedding system (x-vector/ECAPA) trained on far more data would raise accuracy and the bound.
3. **Sample ceiling.** I_lower can never exceed log2(S) = 10.643 bits with 1599 speakers, regardless of how separable the voices truly are.

Headline xent is 65.0% of the ceiling → **below the ceiling (classifier/bound-limited)**. The gap to the ceiling is driven by bound looseness + the weak classifier, **not** by the features failing to separate speakers — do not claim 'the features cannot separate the speakers.'

## Step 6 — Cross-corpus contrast with TIMIT (the key result)

TIMIT was run through the **identical** pipeline: the same 28-feature extractor on all 6,300 TIMIT wavs (630 speakers × 10 utts, 16 kHz), the same balancing, folds, classifiers, and bounds. Common feature set (CV ∩ TIMIT) = **28 features**. The **matched rows** (same S, same features, same 10 clips/speaker) isolate the *only* remaining difference: **single-session clean TIMIT vs multi-session mp3 Common Voice.**

| corpus | session | S | log2(S) | top-1 acc (A/B/C) | Fano I_low (logreg) | xent I_low (best) | #feat→95% | regime |
|---|---|---|---|---|---|---|---|---|
| CV (full) | multi-session+mp3 | 1599 | 10.64 | 0.551/0.510/0.518 | 4.88 | 6.92 (logreg) | 18 | bound/classifier-limited |
| CV (matched to TIMIT) | multi-session+mp3 | 630 | 9.30 | 0.626/0.567/0.608 | 4.87 | 6.37 (logreg) | 17 | bound/classifier-limited |
| TIMIT (matched) | single-session clean | 630 | 9.30 | 0.611/0.535/0.627 | 4.71 | 6.64 (lda) | 17 | bound/classifier-limited |

**Key matched contrast (S=630, 28 common features, 10 clips/speaker, TIMIT−CV deltas):**

| metric | CV matched | TIMIT matched | TIMIT − CV |
|---|---|---|---|
| top-1 acc, logreg | 0.626 | 0.611 | -0.016 |
| top-1 acc, MLP | 0.567 | 0.535 | -0.032 |
| top-1 acc, LDA | 0.608 | 0.627 | 0.019 |
| **best top-1 acc** | **0.626** | **0.627** | **0.001** |
| **best xent I_lower (bits)** | **6.37** | **6.64** | **0.28** |

**Robustness of the CV-matched row** (best top-1 / best xent over **8 independent random 630-speaker draws**): CV acc = 0.623 ± 0.0110 [0.610, 0.644]; CV xent = 6.37 ± 0.07 bits [6.30, 6.51]. TIMIT (fixed) acc = 0.627, xent = 6.64 bits. The single-draw result is representative — not a lucky subsample.

**Honest reading — the expected large single-vs-multi-session drop does NOT materialise.** At matched S and features the two corpora are **essentially tied on accuracy** (best top-1 0.626 CV vs 0.627 TIMIT; CV is actually *higher* on logreg and MLP, TIMIT higher only on LDA), and TIMIT's headline usable-bits lower bound is only **0.28 bits higher** (~4% relative). The direction of the *headline* (xent) is weakly TIMIT-favouring, but the magnitude is small and the sign flips by classifier — so the robust claim is a **small, classifier-dependent gap, not a large degradation.**

**A second confound, disclosed.** The matched rows equalise S, features, and clips/speaker, but TIMIT and CV still differ in *utterance content*: TIMIT's 10 prompts per speaker are phonetically-controlled read sentences (2 SA sentences are identical text across *all* speakers, plus 3 SI + 5 SX), whereas CV clips are free, mostly-unique volunteer reads. So the contrast bundles **session (single vs multi) + channel (clean vs mp3@16 kHz) + content (controlled vs free)**; it is not a pure session experiment. The identical SA text could make TIMIT speakers marginally easier to compare on matched phonetic content — a small effect that, if anything, *inflates* TIMIT's side, so the true session-only gap is if anything even smaller than the ~0.3 bits reported. We therefore claim only the **direction and small magnitude**, not a clean single-vs-multi-session decomposition.

**Why so small?** The 28 features are dominated by low-frequency, session-stable descriptors (F0, formants F1–F5, bandwidths, spectral shape, CPP/HNR) that mp3@16 kHz and across-session variability leave largely intact — the high-frequency / glottal-source detail that codecs and channels destroy was never in this feature set (and the glottal family was unmeasured). So on *this* feature set, multi-session mp3 audio carries about as much usable speaker information as single-session studio audio. Absolute bits remain **not** comparable across corpora (mp3/16 kHz still biases CV's absolute values low); the robust statement is that the **contrast is small**. The LDA flip (TIMIT's best, CV's worst) reflects LDA's Gaussian assumption fitting TIMIT's cleaner within-speaker spread better than CV's heavier-tailed multi-session spread. See `figs/crosscorpus_matched.png`.

## Step 7 — Homogeneous cohort (US-English)

US-English self-reported cohort: **S = 521** balanced speakers (≥300 ✓). Re-ran Steps 1–2 within it, plus a **matched-S random control** (same S drawn from the pooled corpus) to de-confound the lower ceiling.

- within-cohort (strongest, logreg): xent I_lower = **6.22 bits**, acc 0.638
- matched random control (S=521): xent I_lower = **6.25 bits**, acc 0.631

The two metrics disagree in sign: xent is **0.028 bits lower** in the cohort (the predicted direction — similar speakers harder to separate), but accuracy is **0.007 higher** in the cohort (the opposite direction). Crucially, the xent gap (0.028 bits) is **smaller than the random-subsample noise floor (±0.07 bits, from the 8-draw robustness check)**, so it is **within noise** — this bound-based estimator at S=521 is too coarse to confirm or refute the homogeneous-cohort prediction. The prior MI/NMI experiment, with a finer ceiling-normalised estimator, did detect a consistent cohort drop; the classifier-bound metric here lacks that sensitivity. CV accent labels are **self-reported and coarse** — suggestive only, not confirmatory.

## Limitations

1. **The bound is a floor.** Every headline is a lower bound — model- and sample-dependent; the true usable speaker information is higher. A stronger classifier or more data raises it.
2. **mp3 + 16 kHz degradation.** Common Voice clips are lossy mp3 resampled to 16 kHz, which attenuates high-frequency and glottal-source detail, biasing the *absolute* bits low and **breaking cross-corpus absolute comparability** — only the CV↔TIMIT *contrast* is robust.
3. **`client_id` = speaker assumption.** We treat one Common Voice `client_id` as one speaker.
4. **log2(S) ceiling.** No bound can exceed 10.643 bits (CV) / 9.299 bits (TIMIT) at these speaker counts.
5. **Multi-session variance makes CV *realistic*, not pessimistic.** TIMIT's single-session recording is optimistic (no day/device/room variability); CV's multi-session mp3 is the real-world condition. The headline empirical finding is that, at matched S and on this 28-feature set, this realistic condition costs **little** — accuracy is statistically tied and the usable-bits lower bound is only ~0.3 bits below TIMIT (a gap that is itself partly LDA-calibration, not separability). The robust prior expectation of a *large* multi-session penalty is **not** supported here; the session-stable low-frequency features absorb most of the variability.
6. **Self-reported demographics.** Sex/age/accent are self-reported and coarse; the Step-7 cohort analysis is suggestive only.
7. **Cross-corpus content confound.** The matched contrast equalises S/features/clips but not utterance content (TIMIT = controlled prompts incl. 2 speaker-shared SA sentences; CV = free reads), so it bundles session+channel+content, not session alone — only the direction and small magnitude of the gap are claimed.

---

**Artifacts.** `results.json`, `jointbits_classifiers_cv.csv`, `jointbits_classifiers_timit.csv`, `crosscorpus_table.csv`, `calibration_cv.csv`, `cumulative_bits_cv.csv`, `binned_greedy_censored_cv.csv`, `figs/joint_bits_curve_cv.png`, `figs/binned_greedy_censored_cv.png`, `figs/crosscorpus_matched.png`. Reproduce: `python jointbits_experiment/jb_run.py` (seed 1234).