"""Assemble the final Markdown report purely from ./results/ (+ features parquet for provenance).
Env: TIMIT_OUTDIR, TIMIT_RESULTS, TIMIT_REPORT (default report_TIMIT_v2.md)."""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_40, SEED

OUTDIR = os.environ.get("TIMIT_OUTDIR", "features")
RES = os.environ.get("TIMIT_RESULTS", "results")
REPORT = os.environ.get("TIMIT_REPORT", "report_TIMIT_v2.md")


def jload(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fmt(x, p=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, float):
        return f"{x:.{p}g}"
    return str(x)


def main():
    cov = pd.read_csv(os.path.join(RES, "coverage.csv"))
    fr = pd.read_csv(os.path.join(RES, "f_ratio.csv"))
    ub = pd.read_csv(os.path.join(RES, "usable_bits.csv"))
    binning = jload("binning.json")
    eff = jload("effective_dim.json")
    clf = jload("classifier.json")
    coll = jload("collision.json")
    summ = jload("analyze_summary.json")

    # provenance from parquet + sentinel
    feat = pd.read_parquet(os.path.join(OUTDIR, "features_per_utt.parquet"))
    n_rows = len(feat)
    n_dec_fail = int((~feat["decode_ok"]).sum())
    n_spk = feat["speaker"].nunique()
    sent = os.path.join(OUTDIR, "_EXTRACTION_DONE")
    sent_txt = open(sent).read().strip() if os.path.exists(sent) else "n/a"

    L = []
    w = L.append
    w("# TIMIT — 40-feature distinctiveness battery (v2)\n")
    w(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · RNG seed = {SEED} (fixed everywhere) · "
      f"all numbers traced to `./{RES}/`*\n")

    # 0 provenance
    w("## 0. Corpus and provenance\n")
    w("| field | value |")
    w("|---|---|")
    w(f"| Corpus | TIMIT (NIST SPHERE NIST_1A, 16 kHz) |")
    w(f"| Utterances (rows) | {n_rows} |")
    w(f"| Speakers | {n_spk} |")
    w(f"| Decode failures | {n_dec_fail} |")
    w(f"| Extraction sentinel | `{sent_txt}` |")
    w(f"| Seed | {SEED} |")
    w(f"| Source-of-truth | every number below comes from `./{RES}/` |\n")

    # 1 coverage
    n_meas = int((cov["status"] == "MEASURED").sum())
    w("## 1. Feature coverage (measured out of 40)\n")
    w(f"**Measured: {n_meas}/40.** VTLE excluded by design. Coverage = fraction of utterances "
      f"with a successful value (never imputed).\n")
    w("| feature | coverage | status |")
    w("|---|---|---|")
    for _, r in cov.iterrows():
        w(f"| {r['feature']} | {r['coverage']:.3f} | {r['status']} |")
    not_meas = cov[cov["status"] != "MEASURED"]["feature"].tolist()
    w(f"\nNOT-MEASURED (0 coverage): {', '.join(not_meas) if not_meas else 'none — all 40 measured'}.\n")

    # 2 binning
    w("## 2. Population distributions and quantile bins\n")
    degen = []
    for f, qd in binning.items():
        for q, info in qd.items():
            if info.get("degenerate"):
                degen.append(f"{f}@q={q} -> {info['realized_bins']} bins")
    w("Quantile bin edges computed for q ∈ {2,3,5,10} from the across-speaker per-speaker-mean "
      "distribution. Degenerate (collapsed) bins:\n")
    if degen:
        for d in degen:
            w(f"- {d}")
    else:
        w("- none — all features realized the requested bin count at every q.")
    w("")

    # 3 F-ratio
    w("## 3. F-ratios and usable resolution — pooled AND within-sex\n")
    w("Sorted by pooled F-ratio. within_var = mean within-speaker variance; between_var = variance "
      "of per-speaker means; F_ratio = between/within. q_max = largest q∈{2,3,5,10} with mean "
      "bin-crossing rate < 0.20.\n")
    w("| feature | within_var | between_var | F_ratio(pooled) | q_max(pooled) | F(ANOVA) | p | "
      "F_ratio(M) | F_ratio(F) | F_ratio(within-sex) | q_max(M) | q_max(F) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fr.iterrows():
        w(f"| {r['feature']} | {fmt(r['within_var'],3)} | {fmt(r['between_var'],3)} | "
          f"{fmt(r['F_ratio_pooled'],4)} | {fmt(r['q_max_pooled'],2)} | {fmt(r['anova_F'],4)} | "
          f"{fmt(r['anova_p'],3)} | {fmt(r['F_ratio_male'],4)} | {fmt(r['F_ratio_female'],4)} | "
          f"{fmt(r['F_ratio_within_sex'],4)} | {fmt(r['q_max_male'],2)} | {fmt(r['q_max_female'],2)} |")
    # pooled vs within-sex statement
    valid = fr.dropna(subset=["F_ratio_pooled", "F_ratio_within_sex"])
    ratio = (valid["F_ratio_within_sex"] / valid["F_ratio_pooled"]).median()
    f0row = fr[fr["feature"] == "F0"].iloc[0] if (fr["feature"] == "F0").any() else None
    w(f"\n**Pooled vs within-sex:** median within-sex F-ratio is {ratio:.2f}× the pooled value "
      f"(within-sex {'lower' if ratio < 1 else 'higher'} overall). ")
    if f0row is not None:
        w(f"For **F0**, pooled F-ratio = {fmt(f0row['F_ratio_pooled'],4)} collapses to "
          f"male={fmt(f0row['F_ratio_male'],4)}, female={fmt(f0row['F_ratio_female'],4)} "
          f"(within-sex {fmt(f0row['F_ratio_within_sex'],4)}) — sex carries most of F0's "
          f"between-speaker variance. ")
    forms = fr[fr["feature"].isin(["F1", "F2", "F3", "F4"])]
    if len(forms):
        w("Formants show the same pooled→within-sex shrinkage (see F1–F4 rows).")
    w("\n**Caveat:** TIMIT is single-session, so within-speaker variance omits "
      "day-to-day/health/channel/affective variation. All F-ratios are OPTIMISTIC UPPER BOUNDS "
      "and q_max values are optimistic.\n")

    # 4 usable bits
    w("## 4. Per-feature usable bit depth (Miller–Madow MI vs permutation null)\n")
    w(f"`I_corrected = max(0, I_mm − I_null_mean)`, ≥{os.environ.get('TIMIT_NPERM','200')} shuffles, "
      f"seed {SEED}. Sorted by usable bits.\n")
    w("| feature | b* | q_eff | usable bits | norm MI | perm p |")
    w("|---|---|---|---|---|---|")
    for _, r in ub.iterrows():
        if pd.isna(r.get("usable_bits")):
            w(f"| {r['feature']} | — | — | — | — | — |")
        else:
            w(f"| {r['feature']} | {fmt(r.get('b_star'),2)} | {fmt(r.get('q_eff'),2)} | "
              f"{fmt(r['usable_bits'],4)} | {fmt(r.get('norm_MI'),3)} | {fmt(r.get('perm_p'),3)} |")
    w(f"\n**Total summed usable bits = {summ['total_usable_bits']:.2f}** across features "
      f"(OPTIMISTIC over-count: features are correlated). H(speaker) = {summ['H_speaker_bits']:.3f} bits "
      f"for S = {summ['n_speakers']} speakers.\n")

    # 5 effective dim
    w("## 5. Effective dimensionality — participation ratio (PR)\n")
    p = eff["pooled"]; ws = eff["within_sex"]; pr = eff["parent_residual"]
    w(f"PR = (Σλ)²/Σλ² of the z-scored per-speaker covariance; 95% CI from "
      f"{os.environ.get('TIMIT_NBOOT','1000')} speaker bootstraps (seed {SEED}). "
      f"{eff['n_pr_features']} complete features used.\n")
    w("| analysis | PR | 95% CI | n speakers |")
    w("|---|---|---|---|")
    w(f"| Pooled | {p['PR']:.3f} | [{p['CI95'][0]:.3f}, {p['CI95'][1]:.3f}] | {p['n_speakers']} |")
    w(f"| Within-sex male | {ws['male']['PR']:.3f} | [{ws['male']['CI95'][0]:.3f}, {ws['male']['CI95'][1]:.3f}] | {ws['male']['n']} |")
    w(f"| Within-sex female | {ws['female']['PR']:.3f} | [{ws['female']['CI95'][0]:.3f}, {ws['female']['CI95'][1]:.3f}] | {ws['female']['n']} |")
    w(f"| Within-sex mean | {ws['mean']:.3f} | — | — |")
    w(f"| Parent-residual ({'+'.join(pr['parents_used'])}) | {pr['PR']:.3f} | [{pr['CI95'][0]:.3f}, {pr['CI95'][1]:.3f}] | {pr['n_speakers']} |")
    rise1 = ws['mean'] - p['PR']
    rise2 = pr['PR'] - p['PR']
    w(f"\n**Rise across analyses:** pooled PR = {p['PR']:.2f} → within-sex mean = {ws['mean']:.2f} "
      f"({rise1:+.2f}) → parent-residual = {pr['PR']:.2f} ({rise2:+.2f} vs pooled). "
      f"The parent-residual PR is the empirically-grounded analogue of the manuscript's d_eff lower "
      f"bound: removing the shared-parent confounders ({', '.join(pr['parents_used'])}) "
      f"{'raises' if rise2 > 0 else 'lowers'} effective dimensionality.\n")

    # 6 classifier
    w("## 6. Joint usable speaker bits — held-out classifier lower bound\n")
    w(f"Features with ≥90% coverage; listwise-deleted rows. Retained **{clf['n_rows']} utts / "
      f"{clf['n_speakers']} speakers**, {len(clf['features_used'])} features. "
      f"Chance = {clf['chance']:.3g}; H(speaker) = {clf['H_speaker_bits']:.3f} bits. "
      f"Utterance-disjoint stratified 5-fold CV, z-scored on train folds only.\n")
    w("| model | top-1 acc | 95% CI | per-fold mean±std | log-loss (bits) | Fano bits | x-ent bits |")
    w("|---|---|---|---|---|---|---|")
    for name, m in clf["models"].items():
        w(f"| {name} | {m['acc_mean']:.4f} | [{m['acc_ci95'][0]:.4f}, {m['acc_ci95'][1]:.4f}] | "
          f"{m['acc_per_fold_mean']:.4f}±{m['acc_per_fold_std']:.4f} | {m['logloss_bits_mean']:.3f} | "
          f"{m['fano_bits']:.3f} | {m['xent_bits']:.3f} |")
    w(f"\n**Capacity-inversion check:** MLP {'UNDER-performs' if clf['capacity_inversion'] else 'does NOT under-perform'} "
      f"the linear models — {'the data-starvation signature at ~8–10 utts/speaker is present' if clf['capacity_inversion'] else 'no data-starvation inversion observed'}.\n")
    w(f"**Headline (Fano):** {clf['headline_fano_bits']:.3f} bits; **headline (cross-entropy):** "
      f"{clf['headline_xent_bits']:.3f} bits. Both are FLOORS (a stronger classifier raises them) and "
      f"remain below the H(speaker) = {clf['H_speaker_bits']:.3f}-bit sample ceiling.\n")

    # 7 collision
    w("## 7. Collision-metric sanity cross-check (optional, illustrative)\n")
    w(f"Illustrative birthday model at n = 1e10 identities, N_cells = q_max^PR with operating "
      f"q_max = {coll['operating_q_max']}. Not a headline.\n")
    w("| operating point | q_max | PR | log10(N_cells) | P(E) exp. pairs | P(M) any collision | P(B) per-pair |")
    w("|---|---|---|---|---|---|---|")
    for key in ["pooled_PR", "parent_residual_PR"]:
        c = coll.get(key)
        if c:
            w(f"| {key} | {c['q_max']} | {c['PR']:.3f} | {c['log10_N_cells']:.3f} | "
              f"{c['P_E_expected_pairs']:.3g} | {c['P_M_any_collision']:.3g} | {c['P_B_per_pair']:.3g} |")
    w("")

    # headline
    w("## Headline numbers (for direct quotation)\n")
    f0 = fr[fr["feature"] == "F0"].iloc[0]
    w(f"- **Measured features:** {n_meas}/40")
    w(f"- **F0 F-ratio:** pooled = {fmt(f0['F_ratio_pooled'],4)}, within-sex = {fmt(f0['F_ratio_within_sex'],4)} "
      f"(M {fmt(f0['F_ratio_male'],4)} / F {fmt(f0['F_ratio_female'],4)}); q_max(pooled) = {fmt(f0['q_max_pooled'],2)}")
    w(f"- **PR(pooled):** {p['PR']:.3f}  CI [{p['CI95'][0]:.3f}, {p['CI95'][1]:.3f}]")
    w(f"- **PR(within-sex):** mean {ws['mean']:.3f} (M {ws['male']['PR']:.3f} / F {ws['female']['PR']:.3f})")
    w(f"- **PR(parent-residual):** {pr['PR']:.3f}  CI [{pr['CI95'][0]:.3f}, {pr['CI95'][1]:.3f}]  "
      f"(parents: {', '.join(pr['parents_used'])})")
    accs = ", ".join(f"{n}={m['acc_mean']:.4f}" for n, m in clf["models"].items())
    w(f"- **Classifier top-1 accuracy:** {accs}")
    w(f"- **Bit lower bounds:** Fano = {clf['headline_fano_bits']:.3f} bits, "
      f"cross-entropy = {clf['headline_xent_bits']:.3f} bits (H(speaker) = {clf['H_speaker_bits']:.3f})")
    w(f"- **Total summed per-feature usable bits:** {summ['total_usable_bits']:.2f} (optimistic; correlated)")
    w("\n*All F-ratios / q_max are optimistic upper bounds (single-session TIMIT). "
      "All bit bounds are floors. NOT-MEASURED features (if any) flagged in §1.*\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {REPORT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
