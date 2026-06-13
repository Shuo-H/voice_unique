# Common Voice 17 — 40-feature distinctiveness battery (v2)

This folder is the **Common Voice** branch of an empirical test of the central claim of
**Singh & Raj, _"Human Voice is Unique"_** (CMU Center for Voice Intelligence and Security;
[arXiv:2506.18182](https://arxiv.org/abs/2506.18182)). That paper argues a voice reduces to ~41
independent, quantizable acoustic features, making the chance of two people sharing a voice in a
10-billion-person world "one in a few thousand to one in a septillion."

This is the **scaled-up v2 battery**: it runs the whole probe suite — feature coverage, F-ratios /
usable resolution, per-feature mutual information, effective dimensionality, a held-out classifier
lower bound, and a collision cross-check — on **10,676 speakers / 137,675 multi-session clips** from
Common Voice 17, over the **40-feature** list (VTLE removed; **39/40 measured**, only VOT
unmeasurable). All randomized steps use seed **1234**.

> **Branch note.** This work lives on the **`common-voice-experiments`** branch — the Common Voice
> (crowd-sourced, multi-session MP3) corpus. The parallel single-session studio analysis is the
> **TIMIT** battery (separate `timit_battery/`), and the `main` branch carries the combined
> multi-experiment layout (`collision_experiment/`, `mi_experiment/`, `jointbits_experiment/`,
> `cv17_v2_experiment/`). The files here are the flattened contents of that `cv17_v2_experiment/`.

## Folder structure

```
.                                  ← this folder (CV17 v2 battery)
├── README.md                      ← you are here
├── report.md                      ← the full human-readable writeup (read this for detail)
├── prompt_and_model_log.md        ← provenance / build log
│
├── features.py                    feature & DSP definitions (the canonical 40 + HNR aux)
├── common.py                      shared loaders: coverage, measured set, per-speaker means, bins
│
├── extract_v2.py                  Step 0/1: pool CV shards → extract features → features.parquet
├── step3_fratios.py               Step 2/3: quantile bins + F-ratios & usable resolution (pooled + within-sex)
├── step4_mi.py                    Step 4: per-feature usable bit depth (Miller–Madow MI + perm null)
├── step5_pr.py                    Step 5: effective dimensionality (PR) pooled / within-sex / parent-residual
├── step6_classifier.py            Step 6: joint usable bits — held-out speaker-ID classifier lower bound
├── report_v2.py                   Step 7: collision cross-check + assembles report.md
│
├── bins.json                      equiprobable q-quantile bin edges (q ∈ {2,3,5,10})
├── coverage.csv                   per-feature fraction of utterances successfully computed
├── fratios.csv                    F-ratios, q_max, ANOVA — pooled and within-sex
├── usable_bits.csv                per-feature usable bit depth (b*, I_corrected, NMI, perm-p)
├── mi_by_feature_bit.csv          full MI sweep over bit depths b ∈ {1..8}
├── pr_effective_dim.csv           participation-ratio d_eff (pooled / within-sex / parent-residual)
├── classifiers.csv                classifier accuracy + Fano / cross-entropy bounds
├── classifier_results.json        classifier run summary
├── collision_crosscheck.csv       measured d_eff + q_max plugged into the paper's collision formulae
│
├── artifacts/                     intermediate results & provenance
│   ├── dataset_summary.json         scale + metadata distributions (from extract)
│   ├── speaker_manifest.csv         speaker → n_clips kept, sex, accent, age
│   ├── bin_degeneracy.csv           collapsed/degenerate quantile bins
│   ├── fratio_summary.json          F-ratio headline numbers
│   ├── mi_summary.json              MI headline numbers
│   ├── pr_summary.json              d_eff headline numbers
│   ├── parent_R2.csv                per-feature R² on sex + age + accent (parent-residual step)
│   └── calibration_lda.csv          LDA probability calibration table
│
├── extract.log, step6.log         run logs
└── features.parquet               per-utterance feature table (~101 MB, git-ignored; regenerate via extract_v2.py)
```

## Method (shared across the whole battery)

| Element | Detail |
|---|---|
| **Paper under test** | Singh & Raj, _Human Voice is Unique_ — same 41-feature construct, same collision framework |
| **Data** | Common Voice 17, English, `validated` split. Mozilla emptied the official HF repo (Oct 2025), so we stream the public **`fixie-ai/common_voice_17_0`** parquet mirror, cached once in `../cv_cache/`. MP3 decoded via libsndfile → **16 kHz mono**. |
| **Speaker label** | `client_id` is treated as one speaker (stated assumption / limitation). |
| **Feature set** | The paper's canonical features minus VTLE (40 total). **39 measured**; **VOT is NOT MEASURED** (needs forced alignment). Features are **never imputed** — failures stay NaN. |
| **Reproducibility** | Fixed seed **1234** for every shuffle, bootstrap, fold, and subsample. |

## Data & where `features.parquet` comes from

There are two artifacts to keep straight:

1. **Raw audio shards** — the Common Voice 17 English `validated` split, downloaded as parquet
   shards from the public Hugging Face mirror **[`fixie-ai/common_voice_17_0`](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)**
   (the official `mozilla-foundation/common_voice_17_0` repo was emptied in Oct 2025). Each shard is
   `en/validated-{i:05d}-of-00138.parquet` (138 shards available; v2 used **24** → 10,676 speakers).
   They are cached **one level up**, at `../cv_cache/en/`, and are **not** part of this repo.
2. **`features.parquet`** — the per-utterance feature table. `extract_v2.py` decodes each shard's MP3
   bytes (libsndfile → 16 kHz mono), extracts the 40 features per clip, and writes this file. It is
   **~101 MB and git-ignored**, so it is **not committed** — you regenerate it locally.

What **is** committed and ready to use: every downstream result — `coverage.csv`, `fratios.csv`,
`usable_bits.csv`, `pr_effective_dim.csv`, `classifiers.csv`, everything in `artifacts/`, and the
assembled [`report.md`](report.md).

### Option A — directly use the provided results (no download, no extraction)

The committed CSV/JSON files and `report.md` are the precomputed outputs (seed 1234). Just read
`report.md` for the full writeup, or load the CSVs directly. Nothing to download. Note that the
*step scripts* still can't re-run without `features.parquet`, since they all load it — so Option A is
for **reading** results, not recomputing them.

### Option B — redo feature extraction from scratch

1. **Download the shards** into `../cv_cache/` (creates `../cv_cache/en/validated-*.parquet`):

   ```python
   from huggingface_hub import hf_hub_download
   REPO = "fixie-ai/common_voice_17_0"
   for i in range(24):                       # 24 shards ≈ 10,676 speakers; 138 available
       hf_hub_download(REPO, f"en/validated-{i:05d}-of-00138.parquet",
                       repo_type="dataset", local_dir="../cv_cache")
   ```

   (The full `common-voice-experiments` repo also ships an `ensure_shards()` helper in
   `collision_experiment/run_experiment.py` that does the same thing.)

2. **Extract**, which regenerates `features.parquet` + `coverage.csv` + `artifacts/`:

   ```bash
   python extract_v2.py
   ```

3. **Run the analysis steps** in order (each writes its CSV/JSON; `report_v2.py` assembles `report.md`):

   ```bash
   python step3_fratios.py     # Step 2/3
   python step4_mi.py          # Step 4
   python step5_pr.py          # Step 5
   python step6_classifier.py  # Step 6
   python report_v2.py         # Step 7 + report.md
   ```

> **Heads-up for re-running here:** `step4_mi.py`, `step6_classifier.py`, and `report_v2.py` import
> shared analysis modules (`mi_core`, `jb_core`, `collision`) from the *sibling* experiment folders of
> the multi-experiment layout. Those steps therefore need the full `common-voice-experiments` checkout
> (where `mi_experiment/`, `jointbits_experiment/`, `collision_experiment/` sit next to this folder),
> not this flattened copy alone. `extract_v2.py`, `step3_fratios.py`, and `step5_pr.py` are
> self-contained.

## Headline findings

Every prior (smaller-scale) finding holds at 6× scale, and the new parent-residual analysis sharpens
the independence story (full numbers in [`report.md`](report.md)):

- **`d_eff` stays low — and the demographic parents are not the source of the redundancy.** PR(pooled)
  **12.95** → within-sex **13.33** → **parent-residual 13.57**. Regressing out sex + age + accent
  *raises* `d_eff` only ~0.6: 39 features still collapse to **~14 independent axes**, far below the
  paper's nominal independence.
- **Usable resolution is still `q ≤ 2`:** 38/39 features fail q≥3 (only F0 reaches q_max=3 at scale).
- **A bigger classifier lower bound, same shape:** with ~5,095 speakers the floor rises to
  **Fano ≥ 5.75 bits / cross-entropy ≥ 8.45 bits**, and the **capacity inversion persists** — the
  regularized linear model still beats the MLP (0.5475 vs 0.5343).
- **Collision cross-check** at the measured `d_eff` (~13) and `q ≤ 2` flips P(B)=P(E)=1 at n=10¹⁰:
  the paper's astronomical figure is an **artifact of the independence + high-q assumptions**, which
  these measurements do not support.

Caveats: MP3@16 kHz biases the absolute numbers low (only *contrasts* are robust), `client_id =
speaker` is assumed, and the glottal-source family is the least reliable. 10,676 speakers is still far
from a 10-billion-person population, so the collision verdicts are extrapolations from `d_eff` +
`q_max`, not direct population counts.

## Reference

Rita Singh and Bhiksha Raj. *Human Voice is Unique.* Center for Voice Intelligence and Security,
Carnegie Mellon University, 2025. arXiv:2506.18182. <https://arxiv.org/abs/2506.18182>
