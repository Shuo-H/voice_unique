# Is the Human Voice Unique? — Four Empirical Probes

This repo empirically tests the central claim of **Singh & Raj, _"Human Voice is Unique"_**
(CMU Center for Voice Intelligence and Security; [arXiv:2506.18182](https://arxiv.org/abs/2506.18182)).
That paper argues that a voice can be reduced to ~41
independent, quantizable acoustic features, and that under those assumptions the chance of
two people sharing a voice in a 10-billion-person world is "one in a few thousand to one in
a septillion."

We don't take that on faith. Four independent experiments measure the real features on real
crowd-sourced speech and ask, from four different angles, **how much speaker identity a voice
actually carries** — and where the paper's astronomical numbers come from.

```
voice_unique/                     ← repo root
├── README.md                     ← you are here
│
├── collision_experiment/         ← Experiment 1: replicate & stress-test the collision math
├── mi_experiment/                ← Experiment 2: mutual information (bits) per feature
├── jointbits_experiment/         ← Experiment 3: classifier lower bound on joint bits (+ TIMIT)
├── cv17_v2_experiment/           ← Experiment 4: the full battery at 6× scale (10,676 speakers) + parent-residual d_eff
│
└── cv_cache/                     ← SHARED: cached Common Voice 17 parquet shards
                                    (~6.4 GB, git-ignored — created on first run)
```

## What the experiments have in common

They are deliberately built on one shared foundation so the results are comparable:

| Shared element | Detail |
|---|---|
| **Paper under test** | Singh & Raj, _Human Voice is Unique_ — same 41-feature construct, same collision framework |
| **Data** | Common Voice 17, English, `validated` split. Mozilla emptied the official HF repo (Oct 2025), so all of them stream the public **`fixie-ai/common_voice_17_0`** parquet mirror, cached once in `cv_cache/`. MP3 decoded via libsndfile → **16 kHz mono**. |
| **Speaker label** | `client_id` is treated as one speaker (stated assumption / limitation everywhere). |
| **Feature set** | The paper's canonical 41 acoustic features (F0, formants F1–F5 + bandwidths, jitter/shimmer, spectral shape, CPP/HNR, glottal-source family, …). |
| **Honesty rule** | Features are **never imputed or faked**. Anything not computable (the glottal/inverse-filter family, VOT — needs forced alignment) is logged as **NOT MEASURED** with 0 coverage, not approximated. This is why each experiment measures a slightly different subset (40 / 28 / 28 / 39). |
| **Reproducibility** | Fixed seed **1234** for every shuffle, bootstrap, fold, and subsample. |

Each experiment folder also follows the **same internal layout**:

```
<experiment>/
├── *features*.py     feature/DSP definitions          (what to measure)
├── *extract*.py      Step 1: extraction → parquet      (measure it)
├── *core*.py         the analysis methods
├── *analyze*.py      analysis driver                   (compute results)
├── *report*.py       assembles the human-readable report
├── run_*.py          one-command end-to-end runner
├── features.parquet  per-utterance features (long format)
├── figs/             plots
├── artifacts/, *.csv, *.json   intermediate results
├── report*.md        ← the writeup (read this for full detail)
└── prompt_and_model_log.md     provenance / build log
```

Run any one end-to-end with its driver, e.g. `python mi_experiment/run_mi.py`.

---

## Experiment 1 — `collision_experiment/`  ·  Replicate the collision math

**What it does.** A direct replication and stress-test of the paper's own collision-probability
formulae. It measures the paper's **two load-bearing assumptions** on 1,755 speakers / 18,861
multi-session clips: (a) feature *independence*, via the effective dimensionality `d_eff`
(participation ratio); and (b) per-feature *resolution* `q`, via how often a speaker's repeated
clips stay inside the same quantile bin (`q_max`). It then plugs the **measured** `d_eff` and
`q_max` back into the paper's exact equations. (`collision.py` reproduces the paper's Table 1 at
d=41 exactly, so the engine is validated.)

**Core conclusion.** Both assumptions fail on realistic audio:
- Features are **far from independent** — `d_eff ≈ 12`, not 41 (and below the paper's own floor of 27).
- Usable resolution is **`q ≤ 2`, not 5–10** — multi-session variability pushes speakers across bins.
- Feeding the measured values into the paper's formulae flips every metric from "astronomically
  unique" to **collisions certain** at n=10¹⁰. The one-in-a-septillion figure is an **artifact of the
  independence + high-q assumptions**.
- **Honest counterpoint:** at *sample* scale the 1,736 real speakers are perfectly separable
  (0 collisions at q=2,3). Voice is genuinely highly distinctive — what's refuted is the *specific*
  astronomical probability, not the qualitative claim. A population-scale verdict needs far more
  speakers and cleaner audio.

→ Full detail: [`collision_experiment/report.md`](collision_experiment/report.md)

## Experiment 2 — `mi_experiment/`  ·  How many *bits* of identity per feature

**What it does.** Reframes "uniqueness" information-theoretically: the bias-corrected **mutual
information (in bits)** between speaker identity and each feature's quantization bin, on a
balanced design (1,599 speakers × exactly 12 clips each). Bias is handled with Miller–Madow
**plus** a 200× permutation null, so the headline bits are conservative and significance-tested.
It reports per-feature usable bit-depth, a greedy **joint/cumulative** bits curve, and a
size-matched cohort analysis.

**Core conclusion.**
- Every measured feature carries significant speaker information (perm-p < 0.005). The best single
  feature is **F0 at 0.887 bits**, then RMS (0.732), CPP (0.679) — against a `log2(1599) = 10.64`-bit
  ceiling, so F0 alone ≈ 8% of the maximum.
- Joint information **saturates at ~1.12 bits using just 2 features**, then declines — but this peak
  is **sample-capped** (the joint permutation null rises as cells outrun the data), an *estimate* of
  usable joint bits, not a population constant.
- **Homogeneous cohorts carry fewer usable bits than size-matched random controls** (sex −24%,
  US-accent −8%) — a de-confounded confirmation of the paper's "low effective dimensionality among
  acoustically similar speakers."

→ Full detail: [`mi_experiment/report-cv-quant.md`](mi_experiment/report-cv-quant.md)

## Experiment 3 — `jointbits_experiment/`  ·  Classifier lower bound + TIMIT contrast

**What it does.** Where Exp. 2 bins each feature, this puts a **held-out speaker-ID classifier**
(logreg / MLP / shrinkage-LDA, utterance-disjoint 5-fold CV) on the joint feature vector and turns
its accuracy and log-loss into **Fano + cross-entropy lower bounds** on joint usable speaker
information. Crucially, it runs the **identical pipeline on TIMIT** (single-session studio audio) to
ask whether Common Voice's multi-session MP3 condition actually costs identity information.

**Core conclusion.**
- Joint usable speaker information is **≥ 6.92 bits** (cross-entropy bound, strongest classifier;
  ultra-conservative Fano floor 4.88 bits). Every number is a **lower bound** — a stronger classifier
  (e.g. an x-vector/ECAPA embedding net) or more data can only raise it.
- **Key cross-corpus result:** at matched S=630 and the same 28 features, multi-session MP3 Common
  Voice and single-session studio **TIMIT are statistically tied** on speaker-ID accuracy
  (≈0.626 vs 0.627) and within **~0.3 bits**. The large single-vs-multi-session penalty one would
  expect a priori **does not appear** — the session-stable low-frequency features (F0, formants,
  spectral shape, CPP/HNR) absorb most of the channel/session variability.

→ Full detail: [`jointbits_experiment/report-jointbits-cv.md`](jointbits_experiment/report-jointbits-cv.md)

## Experiment 4 — `cv17_v2_experiment/`  ·  The full battery at 6× scale + parent-residual d_eff

**What it does.** A scaled-up **v2** that runs all three probes' methods together on a much larger
speaker set — **10,676 speakers / 137,675 multi-session clips** (24 shards, ~6× the prior runs) —
over the **40-feature** list (VTLE removed; **39/40 measured**, only VOT unmeasurable). It
re-measures F-ratios and `q_max` **pooled *and* within-sex**, per-feature usable bits (Miller–Madow
MI), the participation-ratio `d_eff`, and the classifier Fano/cross-entropy bounds — and adds one
analysis the smaller runs couldn't support: a **parent-residual `d_eff`** that regresses every
feature on the shared parents **sex + age + accent** (CV uniquely has all three) *before* measuring
effective dimensionality. All randomized steps use seed 1234.

**Core conclusion.** Every prior finding holds at scale, and the new parent-residual analysis
sharpens the independence story:
- **`d_eff` stays low — and the parents are not the source of the redundancy.** PR(pooled) **12.95**
  [12.78, 13.07] → within-sex **13.33** → **parent-residual 13.57** [13.39, 13.71] (CIs
  non-overlapping). Regressing out sex/age/accent *raises* `d_eff` only ~0.6: 39 features still
  collapse to **~14 independent axes** even after the dominant confounders are stripped. The shared
  parents are themselves correlation-inducing (that is *why* pooled `d_eff` is slightly lower), not
  the origin of the redundancy.
- **Usable resolution is still `q ≤ 2`:** 38/39 features fail q≥3 (only F0 reaches q_max=3 at this
  scale). Within-sex F-ratios are *lower* than pooled for 38/39 features — pooling across sexes
  inflates the apparent separation of the sex-linked source/filter features (F0: 3.47 pooled → 1.45
  male / 2.18 female).
- **A bigger lower bound, same shape.** With ~5,095 speakers the classifier floor rises to
  **Fano ≥ 5.75 bits / cross-entropy ≥ 8.45 bits** (vs 4.88 / 6.92 in Exp. 3), and the **capacity
  inversion persists** — the regularized linear model still beats the MLP (0.5475 vs 0.5343).
- **Collision cross-check** at the measured PR (~13) and q ≤ 2 again flips P(B)=P(E)=1 at n=10¹⁰.

→ Full detail: [`cv17_v2_experiment/report.md`](cv17_v2_experiment/report.md)

---

## The four results together

The paper says: *voice = many independent high-resolution features ⇒ astronomically unique.*
The experiments converge on a more nuanced picture:

1. **The astronomical number is an artifact** of the independence + high-q assumptions; measured
   features are correlated (`d_eff ≈ 12`) and low-resolution (`q ≤ 2`) on real audio (Exp. 1) —
   and this **holds at 6× scale** (10,676 speakers, Exp. 4), where even regressing out sex + age +
   accent leaves `d_eff ≈ 14`, so the redundancy is intrinsic to the features, not just the
   demographic confounders.
2. **But voice genuinely carries strong identity information** — real speakers are perfectly
   separable at sample scale (Exp. 1), individual features carry significant bits (Exp. 2), and a
   simple linear classifier already extracts ≥6.9 bits (Exp. 3), rising to **≥8.4 bits** on the
   larger set (Exp. 4).
3. **The discriminating information is robust**, sitting in session-stable low-frequency features —
   so much so that lossy multi-session crowd audio nearly matches clean studio audio (Exp. 3), and
   the linear>nonlinear "capacity inversion" persists as the speaker set grows (Exp. 4).

Shared caveats across all four: MP3@16 kHz biases the absolute numbers low (only *contrasts* are
robust), `client_id = speaker` is assumed, and the glottal-source family is the least reliable. The
first three experiments use ~1.6k speakers; Exp. 4 scales to **10,676**, but even that is far from a
10-billion-person population, so the collision verdicts remain extrapolations from `d_eff` + `q_max`,
not direct population counts.

---

## Reference

Rita Singh and Bhiksha Raj. *Human Voice is Unique.* Center for Voice Intelligence and Security,
Carnegie Mellon University, 2025. arXiv:2506.18182. <https://arxiv.org/abs/2506.18182>
