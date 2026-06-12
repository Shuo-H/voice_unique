# voice_unique

Empirical re-examination of the claim that the **human voice is unique**, on the TIMIT
corpus. The repository contains the analysis code, the derived per-utterance feature
tables, and all result files (tables, figures, reports) supporting the findings.

The study revisits Singh & Raj, *Human Voice is Unique*, which assumes 41 acoustic features
are statistically independent and each carries `q` usable bins, giving `m = q^41` voice
"cells." We instead **measure** what is actually computable, how many bins each feature can
support, how many effective dimensions survive feature correlations, and how much speaker
information the features jointly carry — and we recompute the paper's collision metrics with
those measured quantities.

Everything is deterministic given a single fixed seed (**1234**): feature extraction, the
cross-validation folds, all bootstraps, all permutation nulls, and the classifier
`random_state`. An independent re-run reproduces the numbers bit-for-bit.

## Data availability

TIMIT (LDC93S1) audio is **not** redistributable and is **not** included here. Only the
**derived feature tables** computed from it are deposited (`outputs/features.parquet` and
`outputs/_features_wide.parquet`). To regenerate the feature tables from scratch you need
your own licensed copy of TIMIT (see Reproduction). The `.gitignore` blocks raw audio
(`*.WAV`, `*.sph`, `*.tgz`, `timit*/`) from ever being committed.

## Repository layout

```
voice_unique/
  code/                         analysis code (runnable, env-configurable, seed 1234)
    vu_extract.py               STEP 1: per-utterance feature extraction from TIMIT audio
    vu_analyze.py               F-ratios, effective dimensionality, collision metrics
    vu_quant.py                 quantization + Miller-Madow mutual information
    vu_jointbits.py             held-out speaker ID + Fano lower bound on joint bits
    run_experiment.py           orchestrator (extract -> analyze)
  outputs/                      derived tables + all result files (single VU_OUT folder)
    features.parquet            long-format per-utterance features (speaker_id,sex,utt_id,feature,value)
    _features_wide.parquet      one-row-per-utterance wide version
    coverage.csv                per-feature fraction of utterances successfully computed
    bins.json                   q-quantile bin edges per feature
    figs/                       all figures (84 PNGs)
    ... result CSVs + reports (see "File manifest" below) ...
  requirements.txt              pinned dependencies (Python 3.10)
```

## The three experiments

All run on TIMIT (630 speakers x 10 utterances = 6300 single-session utterances, 16 kHz).
40 of the paper's ~41 features were measurable; VFI and Nasality were not (no single-channel
estimator) and are recorded as NOT MEASURED — never imputed.

**1. Feature coverage, F-ratios, effective dimensionality, collisions** (`vu_extract.py`,
`vu_analyze.py` -> `outputs/report.md`). Nominal `k = 40` features collapse to an effective
dimensionality of **d_eff ≈ 5.4 [5.2, 5.7]** (pooled Pearson participation ratio; ~11-12
within each sex). Plugging the paper's own assumption (independence, k=40, q=10) reproduces
its result (population-match probability ~1e-21 at n=1e10), but with measured d_eff the
population match becomes effectively certain. A direct count finds **5 colliding speaker
pairs at q=2** versus ~0 predicted under independence — falsifying the independence
assumption.

**2. Quantization + bias-corrected mutual information** (`vu_quant.py` ->
`outputs/report-quant.md`). Per-feature usable speaker bits via MI with Miller-Madow
correction and a 200x permutation null; headline is always the bias-corrected
`I_corrected = max(0, I_mm - I_null_mean)`. F0 is most informative at **1.44 corrected bits**
(usable depth b*=3). F-ratio and corrected bits agree strongly (Spearman **rho = 0.973**),
with SQ the one notable divergence. A binned greedy join is reported as a censored sanity
check (sample-limited).

**3. Fano lower bound on joint speaker bits** (`vu_jointbits.py` ->
`outputs/report-jointbits-timit.md`). Held-out 629-way speaker identification under
utterance-disjoint 5-fold CV converts identification error into a **lower bound** on joint
information via Fano's inequality. Headline (the larger of the two required classifiers,
logreg vs MLP): **Fano I_lower = 5.09 bits** (LOWER BOUND, classifier-dependent); a
shrinkage-LDA reference reaches 6.26 bits, and the cross-entropy bound is 6.76 bits — all
floors, since a stronger model can only raise them. The classifier-driven greedy reaches
**6.32 bits (95% of max by 25 features)**; the binned plug-in curve peaks at only 1.34 bits
before censoring, demonstrating why binned MI must not be read as saturation. At ~5-6 bits
the measurement is classifier-limited, not yet sample-ceilinged (ceiling log2(629) = 9.30
bits).

A recurring, honestly-stated caveat: TIMIT is **single-session**, so within-speaker
variability is understated and every separability/bits number here is an **optimistic upper
bound**; cross-session data would lower them.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10
pip install -r requirements.txt
```

Point `VU_OUT` at the `outputs/` folder (it holds `features.parquet` and is where results are
written). The analysis scripts also auto-locate `features.parquet` next to themselves or in
the parent folder if `VU_OUT` is unset.

```bash
export VU_OUT="$(pwd)/outputs"

# Re-derive the feature tables from raw audio (needs a licensed TIMIT copy):
export VU_TIMIT_ROOT="/path/to/timit_LDC93S1/timit/TIMIT"   # dir containing TRAIN/ and TEST/
python code/vu_extract.py            # writes features.parquet, _features_wide.parquet, coverage.csv

# Re-run the analyses from the deposited feature tables (no audio needed):
python code/vu_analyze.py            # experiment 1
python code/vu_quant.py              # experiment 2  (needs fratios.csv from experiment 1)
python code/vu_jointbits.py          # experiment 3  (~30 min: held-out greedy speaker ID)
```

On Windows (cmd): use `set VU_OUT=...\outputs` and `python code\vu_jointbits.py`.

## File manifest

| file (in `outputs/`) | experiment | contents |
|---|---|---|
| `features.parquet`, `_features_wide.parquet` | (input) | derived per-utterance feature tables |
| `coverage.csv`, `bins.json` | (input) | per-feature coverage; q-quantile bin edges |
| `report.md` | 1 | coverage, F-ratios, d_eff, collision metrics |
| `fratios.csv` | 1 | within/between variance, F-ratio, ANOVA, q_max |
| `deff.csv`, `deff_occupancy.csv` | 1 | effective-dimensionality estimates + CIs |
| `collisions.csv`, `direct_collisions.csv` | 1 | collision metrics, assumed vs measured; direct count |
| `report-quant.md` | 2 | usable bits, F-ratio vs bits, joint-bits |
| `mi_per_feature_full.csv`, `usable_bits.csv` | 2 | MI grid; usable bit depth per feature |
| `fratio_vs_bits.csv`, `joint_greedy.csv` | 2 | variance-vs-information comparison; binned greedy |
| `report-jointbits-timit.md` | 3 | Fano/cross-entropy lower bounds |
| `jointbits_classifiers_timit.csv` | 3 | held-out accuracy, log-loss, Fano + xent bounds + CIs |
| `jointbits_greedy_timit.csv` | 3 | classifier-driven cumulative I_lower curve |
| `binned_greedy_censored_timit.csv` | 3 | censored binned-MI sanity check |
| `figs/` | 1-3 | distributions, scree, MI curves, joint-bits and censored curves |

## Notes

- The raw TIMIT corpus requires an LDC license; this repo deposits only derived features and
  results, consistent with that license.
- Reference: R. Singh and B. Raj, *Human Voice is Unique*, Center for Voice Intelligence and
  Security, Carnegie Mellon University.
- Seed = 1234 everywhere.
