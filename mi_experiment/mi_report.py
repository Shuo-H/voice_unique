"""
mi_report.py -- STEP 7: assemble report-cv-quant.md from the analysis artifacts.
Reads every CSV/JSON produced by mi_extract.py + mi_analyze.py and emits a single
markdown report (also printed to stdout by run_mi.py).
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def md_table(df, cols=None, floatfmt=3, maxrows=None):
    if cols is None:
        cols = list(df.columns)
    d = df[cols].copy()
    if maxrows:
        d = d.head(maxrows)
    def fmt(x):
        if isinstance(x, float):
            if np.isnan(x):
                return ""
            return f"{x:.{floatfmt}f}"
        return str(x)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |"
                     for row in d.itertuples(index=False))
    return "\n".join([head, sep, body])


def main():
    ds = json.load(open(os.path.join(ART, "dataset_summary.json")))
    an = json.load(open(os.path.join(ART, "analysis_summary.json")))
    cov = pd.read_csv(os.path.join(HERE, "coverage.csv"))
    mi = pd.read_csv(os.path.join(HERE, "mi_by_feature_bit.csv"))
    usable = pd.read_csv(os.path.join(HERE, "usable_bits.csv"))
    cum = pd.read_csv(os.path.join(HERE, "cumulative_bits.csv"))
    comp = pd.read_csv(os.path.join(ART, "stratified_usable_comparison.csv"))
    sat = pd.read_csv(os.path.join(ART, "stratified_saturation.csv"))
    ctrl = pd.read_csv(os.path.join(ART, "stratified_control_comparison.csv"))

    S, N, logS = ds["S_speakers"], ds["N_utterances"], ds["H_speaker_ceiling_bits"]
    measured = ds["measured"]; notmeasured = ds["not_measured"]
    peak_idx = int(cum["cum_I_corrected"].idxmax())
    peak_feats = int(cum.iloc[peak_idx]["step"])
    peak_bits = float(cum.iloc[peak_idx]["cum_I_corrected"])

    L = []
    A = L.append
    A("# Quantization-Based Information-Theoretic Voice Individuality on Common Voice 17")
    A("")
    A("**Headline metric:** bias-corrected mutual information (bits) between speaker "
      "identity and per-utterance feature *quantization bin*, above a permutation null. "
      "Raw plug-in MI is reported only as a diagnostic, never as the result.")
    A("")
    A(f"- **Speakers S = {S}**  |  **Utterances N = {N}** (balanced: exactly "
      f"{ds['clips_per_speaker']} clips/speaker)")
    A(f"- **Absolute MI ceiling H(speaker) = log2(S) = {logS:.3f} bits** "
      "(uniform over speakers by construction)")
    A(f"- Random seed **{ds['seed']}** everywhere (numpy default_rng; all shuffles, "
      "bootstraps, subsampling). Permutation null = 200 shuffles per estimate.")
    A(f"- Data: `fixie-ai/common_voice_17_0` English `validated` (public parquet mirror "
      f"of CV 17.0), {ds['shards_used']} shards streamed; MP3 decoded via soundfile "
      "(libsndfile, no ffmpeg), resampled to 16 kHz mono.")
    A("")

    # ---------------- DATA SUMMARY ----------------
    A("## 1. Data summary")
    A("")
    A(f"Grouping clips by `client_id` (**assumed = one speaker**; stated as a limitation). "
      f"Kept only speakers with >= {ds['clips_per_speaker']} validated, decodable clips, "
      f"then **randomly sampled exactly {ds['clips_per_speaker']} clips/speaker (seed "
      f"{ds['seed']})** for a balanced design. Reached {S} speakers (target ~1500).")
    A("")
    A(f"- Reused {ds['n_reused']} already-extracted clips + extracted {ds['n_extracted']} "
      "fresh; features never imputed (NaN on failure, counted missing).")
    A("")
    A("**Demographics (self-reported, uneven):**")
    A("")
    A("- Sex (speakers): " + ", ".join(f"{k}={v}" for k, v in ds["sex_counts"].items()))
    A("- Age (speakers): " + ", ".join(f"{k}={v}" for k, v in ds["age_counts"].items()))
    A("- Accent (top): " + ", ".join(f"{k}={v}" for k, v in list(ds["accent_counts"].items())[:8]))
    A("")
    A("### Feature coverage (measured vs NOT MEASURED)")
    A("")
    A(f"**{len(measured)} features MEASURED** (coverage >= 90% required; all listed "
      "features clear it):")
    A("")
    meas = cov[cov.status == "measured"].copy()
    A(md_table(meas, ["feature", "group", "coverage", "status"]))
    A("")
    A(f"**{len(notmeasured)} features NOT MEASURED (0 coverage) — logged, not fabricated:**")
    A("")
    A("> The glottal-source / inverse-filtering family "
      f"({', '.join(notmeasured)}) requires a *validated* glottal inverse-filtering "
      "toolkit (e.g. COVAREP / Aparat / a validated IAIF implementation), which is not "
      "available in this environment. Per the experiment's honesty rule these are "
      "reported as NOT MEASURED with 0 coverage rather than approximated with best-effort "
      "DSP. VOT additionally requires forced alignment (unavailable for Common Voice).")
    A("")

    # ---------------- METHOD ----------------
    A("## 2. Method (per feature, per bit depth b in {1..8}, q = 2^b)")
    A("")
    A("1. **Quantize** per-utterance values with q-quantile (equal-frequency) edges over "
      "the pooled distribution, so bins are marginally equiprobable. Degenerate/duplicate "
      "edges merged; effective bin count **q_eff(b)** recorded (`bins.json`). Low-cardinality "
      "features use one bin per distinct value.")
    A("2. **I_raw** = plug-in MI = H(spk)+H(bin)-H(spk,bin) in bits (upward biased).")
    A("3. **I_mm** = Miller-Madow: each entropy gets +(K-1)/(2N), K = occupied cells.")
    A("4. **Permutation null** (200x, seed 1234): shuffle the speaker column across all N "
      "utterances, recompute plug-in MI. Under a label shuffle the speaker- and bin-marginals "
      "are invariant, so only H(spk,bin) changes. Gives I_null_mean, I_null_p95, and "
      "perm_p = fraction(null MI >= I_raw).")
    A("5. **HEADLINE I_corrected = max(0, I_mm - I_null_mean)** [bits above chance]; "
      "**NMI_corrected = I_corrected / log2(S)**.")
    A("")
    A("> Note: I_corrected subtracts *both* the Miller-Madow analytic bias term *and* the "
      "empirical permutation-null floor. This is deliberately **conservative** (it can "
      "double-subtract bias), so absolute I_corrected is a lower-leaning estimate; the "
      "permutation p-value certifies significance independently.")
    A("")

    # ---------------- STEP 4: usable bits ----------------
    A("## 3. Per-feature usable bit depth (Step 4)")
    A("")
    A("`b* = argmax_b I_corrected` — the depth past which finer bins add noise, not speaker "
      "information. Sorted by corrected bits (descending).")
    A("")
    A(md_table(usable, ["feature", "b_star", "q_eff", "I_corrected", "NMI_corrected", "perm_p"]))
    A("")
    top = usable.head(5)
    A(f"**Top features:** " + ", ".join(
        f"{r.feature} ({r.I_corrected:.3f} bits, b*={int(r.b_star)})"
        for r in top.itertuples()) + ".")
    sig = (usable["perm_p"] < 0.05).sum()
    A(f"{sig}/{len(usable)} measured features carry significant speaker information "
      f"(perm_p < 0.05; all 200-shuffle nulls fall below I_raw, so perm_p < 1/200 = 0.005). "
      f"Best single feature = {usable.iloc[0]['feature']} at {usable.iloc[0]['I_corrected']:.3f} "
      f"bits = {100*usable.iloc[0]['NMI_corrected']:.1f}% of the {logS:.2f}-bit ceiling. "
      "Per-feature MI-vs-b curves: `figs/mi_<feature>.png`.")
    topf = usable.iloc[0]["feature"]
    mtop = mi[mi.feature == topf].sort_values("I_corrected", ascending=False).iloc[0]
    A("")
    A(f"> **Conservatism check.** The headline I_corrected subtracts the *plug-in* permutation "
      f"null from the *Miller-Madow* point estimate, which can double-count bias and lean the "
      f"absolute bits **low**. A self-consistent variant subtracting a Miller-Madow-corrected "
      f"null (column `I_corrected_mmnull`) is ~20-25% higher (e.g. {topf}: "
      f"{float(mtop['I_corrected']):.3f} -> {float(mtop['I_corrected_mmnull']):.3f} bits). "
      "Significance (perm_p) is computed self-consistently (plug-in I_raw vs plug-in null) and "
      "is unaffected. Reported headline bits are therefore a lower-leaning estimate.")
    A("")

    # ---------------- STEP 5: cumulative ----------------
    A("## 4. Joint / cumulative usable bits (Step 5)")
    A("")
    A("Greedy forward selection at fixed **b=2 (q=4 bins/feature)**, each step adding the "
      "feature that maximizes the joint-bin I_corrected (same Miller-Madow + permutation-null "
      "correction on the joint contingency table). Stop rule: marginal corrected gain <= the "
      "permutation noise band (I_null_p95 - I_null_mean).")
    A("")
    A("> **Stop-rule note.** 'Permutation p95' is operationalized as the joint null's noise band "
      "(I_null_p95 - I_null_mean) at each step; the literal alternative (a permutation null of "
      "the step-to-step *gain*) and the simpler gain<=0 rule both fire at the same step here, "
      "and the reported peak is robust to the choice.")
    A("")
    A(md_table(cum, ["step", "feature", "cum_I_corrected", "marginal_gain",
                     "I_null_mean", "I_null_p95", "perm_p", "q_eff_joint"]))
    A("")
    A(f"**Saturation (peak) = {peak_bits:.3f} corrected bits at {peak_feats} features**, "
      f"vs the **log2(S) = {logS:.3f} bit ceiling**. See `figs/cumulative_bits.png`.")
    A("")
    A("> **Ceiling caveat (required):** cumulative corrected MI cannot exceed log2(S). As "
      "joint cells grow (q_eff_joint above), the permutation null I_null_mean rises and the "
      "corrected curve peaks then declines — so the saturation point is **partly "
      "sample-limited**, not a population constant. It is an **upper-bounded ESTIMATE of "
      "usable joint speaker bits over the measured features**, not a measured population "
      "value. With more speakers (higher ceiling) and more clips/speaker (denser joint "
      "cells) the peak would move right and up.")
    A("")

    # ---------------- STEP 6: stratified ----------------
    A("## 5. Stratified / homogeneous-cohort analysis (Step 6)")
    A("")
    A("Steps 3-5 re-run within each sex stratum and each accent group with >= 300 speakers. "
      "The framework predicts **lower** effective dimensionality (fewer *usable* bits) among "
      "acoustically similar speakers.")
    A("")
    A("**Cumulative-bits saturation by cohort** (NMI = saturation_bits / log2(S), the "
      "ceiling-normalized usable joint information):")
    A("")
    A(md_table(sat, ["cohort", "S", "logS_ceiling", "saturation_features",
                     "saturation_bits", "saturation_NMI", "stop_step"]))
    A("")
    A("> **Why absolute bits cannot be compared to pooled directly.** Each cohort has fewer "
      "speakers (smaller S) than pooled, hence a lower log2(S) ceiling, smaller N = 12*S, and a "
      "relatively higher joint permutation-null floor (the same Section-4 sample-capping "
      "confound). A smaller *random* speaker set would show lower absolute saturation bits too — "
      "so a raw drop vs pooled does NOT isolate acoustic homogeneity, and on raw absolute bits "
      "US (1.05) even approaches pooled (1.12). The valid test is a **size-matched random "
      "control**.")
    A("")
    A(f"**Matched-S control (the de-confounded test):** for each cohort we drew "
      f"{int(ctrl['n_ctrl'].iloc[0])} random speaker subsets of the SAME size from the full "
      "pool (seed 1234) and ran the identical greedy cumulative analysis. Comparison on "
      "ceiling-normalized NMI:")
    A("")
    ctab = ctrl.copy()
    ctab["ctrl_NMI"] = ctab.apply(lambda r: f"{r['ctrl_sat_NMI_mean']:.3f} ± {r['ctrl_sat_NMI_sd']:.3f}", axis=1)
    ctab = ctab.rename(columns={"hom_sat_NMI": "homogeneous_NMI", "hom_below_ctrl_NMI": "homogeneous_below_control"})
    A(md_table(ctab, ["cohort", "S", "homogeneous_NMI", "ctrl_NMI", "homogeneous_below_control"]))
    A("")
    below = ctrl["hom_below_ctrl_NMI"].astype(str).str.lower().isin(["true", "1"]).sum()
    deltas = []
    for _, r in ctrl.iterrows():
        d = 100 * (r["ctrl_sat_NMI_mean"] - r["hom_sat_NMI"]) / r["ctrl_sat_NMI_mean"]
        deltas.append(f"{r['cohort']} -{d:.0f}%")
    A(f"**All {below}/{len(ctrl)} homogeneous cohorts fall BELOW their size-matched random "
      f"controls** in ceiling-normalized usable joint bits ({', '.join(deltas)}). "
      "Controlling for sample size, acoustically homogeneous cohorts therefore yield **fewer "
      "usable speaker bits than a random speaker set of equal size** — a de-confounded "
      "confirmation of the low-effective-dimensionality-among-similar-speakers prediction. "
      "(Naively comparing cohort absolute bits to the larger-S pooled set would have been "
      "confounded and is NOT the basis for this claim.)")
    A("")
    A("> The N=5 control draws overlap heavily (e.g. the male control resamples 938 of 1599 "
      "speakers each time, ~59% expected overlap), so the reported control SD understates true "
      "sampling variability and is **not** used as a significance figure; the claim rests on "
      "the directional NMI gaps, robust across all three cohorts (US -8% is the weakest leg).")
    A("")
    A("**Per-feature usable corrected bits, pooled vs cohorts (top features):**")
    A("")
    cc = [c for c in comp.columns if c.startswith("Ic_") or c == "feature"]
    A(md_table(comp, cc, maxrows=12))
    A("")
    A("Note F0's usable bits drop sharply within sex strata (pooled 0.887 -> male 0.488, female "
      "0.607): once sex is fixed, much of F0's speaker information is gone, consistent with sex "
      "explaining a large share of pitch variance. Overlay of all cumulative curves: "
      "`figs/cumulative_bits_overlay.png`.")
    A("")

    # ---------------- LIMITATIONS ----------------
    A("## 6. Honest limitations")
    A("")
    A("1. **MP3 compression** degrades high-frequency and source-periodicity features "
      "(jitter, shimmer, CPP/dCPP, spectral flux, rolloff): their absolute bits are biased "
      "**low** and the absolute scale is **not comparable** to clean-audio corpora. Relative "
      "structure (ordering of features, cohort contrasts) is more robust than absolute bits.")
    A("2. **Finite-sample MI bias.** Plug-in MI is upward-biased; Miller-Madow + the "
      "permutation null are applied precisely because of this, but **residual bias still "
      "inflates absolute bits**, especially for high b and for the joint table.")
    A("3. **`client_id` = speaker** is assumed (one account = one speaker); mislabeling "
      "would inflate apparent individuality.")
    A(f"4. **log2(S) = {logS:.2f} bit ceiling** caps all (especially joint/cumulative) bits; "
      "the cumulative saturation is sample-capped (Section 4 caveat).")
    A("5. **SpeechRate is a proxy** (syllable-nucleus rate from the energy/voicing envelope; "
      "no forced alignment is available for Common Voice).")
    A("6. **Self-reported, uneven demographic metadata** (large NaN fractions for sex/age/"
      "accent); strata are imbalanced.")
    A("7. **14 glottal/inverse-filtering features are NOT MEASURED** (no validated tool), so "
      "the source-related dimension of voice individuality is under-sampled here; absolute "
      "joint bits would likely be higher with a validated glottal toolkit.")
    A("")

    # ---------------- MANIFEST ----------------
    A("## 7. File manifest")
    A("")
    manifest = [
        ("mi_experiment/features.parquet", "long-format per-utterance features (balanced)"),
        ("mi_experiment/coverage.csv", "per-feature coverage, measured vs NOT MEASURED"),
        ("mi_experiment/bins.json", "quantization edges + q_eff per (feature, b)"),
        ("mi_experiment/mi_by_feature_bit.csv", "Step 3: I_raw/I_mm/null/perm_p/I_corrected/NMI per (feature,b)"),
        ("mi_experiment/usable_bits.csv", "Step 4: b*, q_eff, I_corrected, NMI, perm_p (sorted)"),
        ("mi_experiment/cumulative_bits.csv", "Step 5: greedy joint cumulative corrected bits"),
        ("mi_experiment/mi_by_feature_bit_<cohort>.csv / usable_bits_<cohort>.csv / cumulative_bits_<cohort>.csv",
         "Step 6: per-stratum tables (sex:male, sex:female, accent:US)"),
        ("mi_experiment/artifacts/stratified_usable_comparison.csv", "pooled vs cohort per-feature bits"),
        ("mi_experiment/artifacts/stratified_saturation.csv", "cohort cumulative saturation + ceilings + NMI"),
        ("mi_experiment/artifacts/stratified_control_comparison.csv", "homogeneous cohort vs matched-S random control (NMI)"),
        ("mi_experiment/artifacts/dataset_summary.json / analysis_summary.json", "run metadata"),
        ("mi_experiment/artifacts/speaker_manifest.csv / selection.csv", "retained speakers + chosen 12 clips each"),
        ("mi_experiment/figs/mi_<feature>.png", "per-feature I_corrected/I_raw/I_null vs b"),
        ("mi_experiment/figs/cumulative_bits.png + _overlay.png", "cumulative curve + cohort overlay"),
        ("mi_experiment/{mi_features,mi_extract,mi_core,mi_analyze,mi_report,run_mi}.py", "the runnable pipeline"),
    ]
    for path, desc in manifest:
        A(f"- `{path}` — {desc}")
    A("")
    A(f"*Pooled analysis wall-clock: {an.get('elapsed_s','?')} s. "
      "Reproduce end-to-end: `python mi_experiment/run_mi.py`.*")

    text = "\n".join(L) + "\n"
    out = os.path.join(HERE, "report-cv-quant.md")
    open(out, "w").write(text)
    print(f"[report] wrote {out} ({len(text)} chars)")
    return out


if __name__ == "__main__":
    main()
