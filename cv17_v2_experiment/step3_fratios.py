"""
step3_fratios.py -- STEP 2 (quantile bins) + STEP 3 (F-ratios & usable resolution),
computed POOLED and WITHIN-SEX (male / female).

within_var = mean over speakers of within-speaker variance (across that speaker's
             utterances)
between_var = variance of per-speaker means
F_ratio    = between/within ; ANOVA F/p one-way across speakers
q_max      = largest q in {2,3,5,10} whose mean bin-crossing rate < 0.20, using
             equiprobable bins re-estimated within the relevant population.

Within-sex uses CV `gender` (male_masculine / female_feminine); NaN-gender
speakers are dropped for the within-sex computation only.  Seed 1234.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.stats as sstats
import common as C
import features as F

CROSS_THRESH = 0.20


def _equiprob_edges(values, q):
    e = np.quantile(values, np.linspace(0, 1, q + 1))
    e[0] = -np.inf; e[-1] = np.inf
    return e


def _qmax_for(groups_vals, all_vals):
    """groups_vals: list of per-speaker utt-value arrays; all_vals: pooled values
    (per-speaker means) used to set equiprobable edges. Returns (q_max, {q:rate})."""
    q_max, rates = 1, {}
    for q in C.QS:
        edges = _equiprob_edges(all_vals, q)
        rr = []
        for v in groups_vals:
            if v.size < 2:
                continue
            cells = np.digitize(v, edges[1:-1])
            modal = np.bincount(cells, minlength=q).argmax()
            rr.append(np.mean(cells != modal))
        mcr = float(np.mean(rr)) if rr else 1.0
        rates[q] = mcr
        if mcr < CROSS_THRESH:
            q_max = q
    return q_max, rates


def fratio_block(wide, feats, spk_means_for_edges):
    """Return per-feature dict of within/between/F/ANOVA/q_max for one population.
    `wide` is utt-level (rows=utts) restricted to the population; edges come from
    that population's per-speaker means (spk_means_for_edges[feat] -> array)."""
    out = {}
    for f in feats:
        groups = [g[f].dropna().values for _, g in wide.groupby("speaker_id")
                  if g[f].dropna().size >= 2]
        if len(groups) < 5:
            continue
        within = float(np.mean([np.var(g, ddof=1) for g in groups]))
        means = np.array([g.mean() for g in groups])
        between = float(np.var(means, ddof=1))
        F_ratio = between / within if within > 0 else np.nan
        try:
            aF, ap = sstats.f_oneway(*groups)
        except Exception:
            aF, ap = np.nan, np.nan
        qmax, rates = _qmax_for(groups, spk_means_for_edges[f])
        out[f] = dict(within_var=within, between_var=between, F_ratio=F_ratio,
                      ANOVA_F=float(aF), p=float(ap), q_max=qmax,
                      **{f"crossrate_q{q}": rates[q] for q in C.QS})
    return out


def main():
    os.chdir(C.HERE)
    df = C.load_long()
    cov = C.coverage_table(df)
    cov.to_csv("coverage.csv", index=False)
    feats = C.measured_features(cov)
    print(f"[load] {df.speaker_id.nunique()} speakers, {df.utt_id.nunique()} utts, "
          f"{len(feats)} measured features")

    wide = C.wide_utt(df, feats)
    spk, ndrop = C.speaker_means(wide, feats)
    print(f"[matrix] {len(spk)} complete speakers ({ndrop} dropped for NaN)")

    # STEP 2: bins + degenerate report (from pooled per-speaker means)
    bins, degen = C.quantile_bins(spk, feats)
    json.dump(bins, open("bins.json", "w"), indent=1)
    deg_rows = [dict(feature=f, display=F.disp(f),
                     **{f"collapsed_edges_q{q}": degen[f][str(q)] for q in C.QS})
                for f in feats]
    pd.DataFrame(deg_rows).to_csv("artifacts/bin_degeneracy.csv", index=False)
    n_deg = sum(1 for f in feats for q in C.QS if degen[f][str(q)] > 0)
    print(f"[step2] bins.json written; {n_deg} (feature,q) cells have collapsed bins")

    # pooled per-speaker means as edge source
    edges_pool = {f: spk[f].dropna().values for f in feats}

    # ---- POOLED ----
    pooled = fratio_block(wide, feats, edges_pool)

    # ---- WITHIN-SEX ----
    sex_blocks = {}
    for sx, key in [("male", "male_masculine"), ("female", "female_feminine")]:
        w_sx = wide[wide.sex == key]
        sp_sx = spk[spk.sex == key]
        n_spk_sx = w_sx.speaker_id.nunique()
        edges_sx = {f: sp_sx[f].dropna().values for f in feats}
        sex_blocks[sx] = (fratio_block(w_sx, feats, edges_sx), n_spk_sx)
        print(f"[within-sex] {sx}: {n_spk_sx} speakers")

    # assemble table
    rows = []
    for f in feats:
        if f not in pooled:
            continue
        p = pooled[f]
        m = sex_blocks["male"][0].get(f, {})
        fm = sex_blocks["female"][0].get(f, {})
        qm_male = m.get("q_max", np.nan); qm_fem = fm.get("q_max", np.nan)
        qmax_within = (int(min(qm_male, qm_fem))
                       if np.isfinite(qm_male) and np.isfinite(qm_fem) else np.nan)
        rows.append(dict(
            feature=f, display=F.disp(f), group=F.V2_GROUP.get(f),
            within_var=p["within_var"], between_var=p["between_var"],
            F_ratio_pooled=p["F_ratio"], ANOVA_F=p["ANOVA_F"], p=p["p"],
            q_max_pooled=p["q_max"],
            F_ratio_male=m.get("F_ratio", np.nan),
            F_ratio_female=fm.get("F_ratio", np.nan),
            q_max_male=qm_male, q_max_female=qm_fem, q_max_within=qmax_within,
            crossrate_q2_pooled=p["crossrate_q2"], crossrate_q3_pooled=p["crossrate_q3"],
        ))
    fr = pd.DataFrame(rows).sort_values("F_ratio_pooled", ascending=False)
    fr.to_csv("fratios.csv", index=False)

    # summary stats
    qd_pool = {int(k): int(v) for k, v in fr.q_max_pooled.value_counts().sort_index().items()}
    n_q_ge3 = int((fr.q_max_pooled >= 3).sum())
    n_q_eq2 = int((fr.q_max_pooled == 2).sum())
    n_q_eq1 = int((fr.q_max_pooled == 1).sum())
    # within-sex vs pooled (does within-sex EXCEED pooled?)
    fr["mean_within_sex_F"] = fr[["F_ratio_male", "F_ratio_female"]].mean(axis=1)
    fr["within_exceeds_pooled"] = fr["mean_within_sex_F"] > fr["F_ratio_pooled"]
    summ = dict(
        n_features=len(fr), n_complete_speakers=int(len(spk)),
        qmax_pooled_dist=qd_pool,
        n_qmax_ge3=n_q_ge3, n_qmax_eq2=n_q_eq2, n_qmax_eq1=n_q_eq1,
        n_F_gt1=int((fr.F_ratio_pooled > 1).sum()),
        n_F_gt2=int((fr.F_ratio_pooled > 2).sum()),
        n_within_exceeds_pooled=int(fr["within_exceeds_pooled"].sum()),
        top5_pooled=fr.head(5)[["feature", "F_ratio_pooled", "F_ratio_male",
                                "F_ratio_female", "q_max_pooled"]].to_dict("records"),
    )
    json.dump(summ, open("artifacts/fratio_summary.json", "w"), indent=2, default=str)
    print(f"[step3] fratios.csv written. q_max(pooled) dist: {qd_pool}; "
          f"q>=3 fails (q_max<3): {len(fr)-n_q_ge3}; "
          f"within-sex>pooled in {summ['n_within_exceeds_pooled']}/{len(fr)} feats")
    # quick look at F0
    if "F0" in set(fr.feature):
        r = fr[fr.feature == "F0"].iloc[0]
        print(f"[step3] F0: F_pooled={r.F_ratio_pooled:.1f} F_male={r.F_ratio_male:.1f} "
              f"F_female={r.F_ratio_female:.1f} q_max_pooled={int(r.q_max_pooled)}")


if __name__ == "__main__":
    main()
