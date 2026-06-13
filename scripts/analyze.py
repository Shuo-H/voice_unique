"""Analysis stages 1-5 and 7. Reads features_per_utt.parquet, writes results/*.
Env: TIMIT_OUTDIR (features dir), TIMIT_RESULTS (results dir). Seed 1234 everywhere."""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_40, SEED

rng = np.random.default_rng(SEED)
QS = [2, 3, 5, 10]
LN2 = np.log(2)


def log(msg):
    with open("run.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)


def per_speaker_tables(df, feats):
    """Return per-speaker mean df and within-speaker variance df (index=speaker)."""
    means = df.groupby("speaker")[feats].mean()
    var = df.groupby("speaker")[feats].var(ddof=1)
    sex = df.groupby("speaker")["sex"].first()
    return means, var, sex


# ---------- 2/3 binning + F-ratio ----------

def bin_edges(values, q):
    v = values[~np.isnan(values)]
    if v.size < q:
        return None, 0
    edges = np.quantile(v, np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    realized = len(edges) - 1
    return edges, realized


def assign_bins(values, edges):
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def crossing_rate(df, feat, edges, speakers_means_index):
    """mean over speakers of (1 - frac in modal bin) using utterance-level assignments."""
    rates = []
    for spk, grp in df.groupby("speaker"):
        vals = grp[feat].values
        vals = vals[~np.isnan(vals)]
        if vals.size < 2:
            continue
        b = assign_bins(vals, edges)
        modal = np.bincount(b).max()
        rates.append(1 - modal / b.size)
    return float(np.mean(rates)) if rates else np.nan


def fratio_block(df, feats):
    """Return dict feat-> {within_var, between_var, F_ratio, anova_F, anova_p, q_max, realized_bins{q}}."""
    from scipy.stats import f_oneway
    means, var, _ = per_speaker_tables(df, feats)
    out = {}
    for f in feats:
        col_means = means[f].values
        within = np.nanmean(var[f].values)
        between = np.nanvar(col_means, ddof=1)
        fr = between / within if within and within > 0 else np.nan
        # anova across speakers (utterance level)
        groups = [g[f].dropna().values for _, g in df.groupby("speaker") if g[f].dropna().size > 1]
        try:
            F, p = f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        # q_max via crossing rate using across-speaker mean distribution edges
        qmax = None
        for q in QS:
            edges, realized = bin_edges(col_means, q)
            if edges is None:
                continue
            cr = crossing_rate(df, f, edges, means.index)
            if not np.isnan(cr) and cr < 0.20:
                qmax = q
        out[f] = {"within_var": float(within), "between_var": float(between),
                  "F_ratio": float(fr) if not np.isnan(fr) else None,
                  "anova_F": float(F) if not np.isnan(F) else None,
                  "anova_p": float(p) if not np.isnan(p) else None,
                  "q_max": qmax}
    return out


# ---------- 4 usable bits (MI, Miller-Madow, permutation null) ----------

def mi_mm_bits(cells, spk_codes, mult, m_y, m_x):
    """Miller-Madow corrected MI in bits. spk_codes are int; mult>max(spk_code) so the
    joint code cells*mult+spk is unique per (cell,speaker). Vectorized, no pandas."""
    N = len(spk_codes)
    mi_nats = mutual_info_score(cells, spk_codes)
    joint = cells.astype(np.int64) * mult + spk_codes
    m_xy = len(np.unique(joint))
    corr = (m_x + m_y - m_xy - 1) / (2.0 * N)  # nats
    return (mi_nats + corr) / LN2


def usable_bits(df, feats, n_perm=200):
    speaker_codes_full = pd.Categorical(df["speaker"].values).codes.astype(np.int64)
    S = int(speaker_codes_full.max()) + 1
    mult = S + 1
    H_spk = np.log2(S)
    out = {}
    for f in feats:
        col = df[f].values
        valid = ~np.isnan(col)
        v = col[valid]; spk = speaker_codes_full[valid]
        m_y = len(np.unique(spk))
        if v.size < 10 or m_y < 2:
            out[f] = None
            continue
        best = None
        for b in range(1, 9):
            nb = 2 ** b
            try:
                cells = pd.qcut(v, nb, labels=False, duplicates="drop")
            except Exception:
                continue
            cells = np.asarray(cells)
            q_eff = len(np.unique(cells))
            if q_eff < 2:
                continue
            m_x = q_eff
            i_mm = mi_mm_bits(cells, spk, mult, m_y, m_x)
            # permutation null (cells fixed -> m_x, m_y constant across shuffles)
            nulls = np.empty(n_perm)
            for k in range(n_perm):
                sh = rng.permutation(spk)
                nulls[k] = mi_mm_bits(cells, sh, mult, m_y, m_x)
            i_null = float(np.mean(nulls))
            i_corr = max(0.0, i_mm - i_null)
            p = (1 + np.sum(nulls >= i_mm)) / (n_perm + 1)
            cand = {"b": b, "q_eff": int(q_eff), "I_mm_bits": float(i_mm),
                    "I_null_bits": i_null, "I_corrected_bits": float(i_corr),
                    "norm_MI": float(i_corr / H_spk), "perm_p": float(p)}
            if best is None or cand["I_corrected_bits"] > best["I_corrected_bits"]:
                best = cand
        out[f] = best
    return out, H_spk, S


# ---------- 5 effective dimensionality (participation ratio) ----------

def participation_ratio(Z):
    """PR of covariance eigenvalues of z-scored matrix Z (rows=speakers, cols=feats)."""
    C = np.cov(Z, rowvar=False)
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-12]
    return (w.sum() ** 2) / (w ** 2).sum()


def zscore(M):
    mu = M.mean(axis=0); sd = M.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    return (M - mu) / sd


def pr_with_ci(M, n_boot=1000):
    pr = participation_ratio(zscore(M))
    boots = np.empty(n_boot)
    n = M.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = participation_ratio(zscore(M[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(pr), [float(lo), float(hi)]


def main():
    outdir = os.environ.get("TIMIT_OUTDIR", "features")
    results = os.environ.get("TIMIT_RESULTS", "results")
    os.makedirs(results, exist_ok=True)
    df = pd.read_parquet(os.path.join(outdir, "features_per_utt.parquet"))
    df = df[df["decode_ok"]].copy()
    log(f"analyze: loaded {len(df)} decoded rows, {df['speaker'].nunique()} speakers")

    # ---- 1 coverage ----
    cov_rows = []
    for f in FEATURES_40:
        c = df[f].notna().mean()
        cov_rows.append({"feature": f, "coverage": round(float(c), 4),
                         "status": "MEASURED" if c > 0 else "NOT-MEASURED"})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(results, "coverage.csv"), index=False)
    measured = [r["feature"] for r in cov_rows if r["status"] == "MEASURED"]
    n_measured = len(measured)
    log(f"measured features: {n_measured}/40")

    # ---- 2 binning ----
    means_all, _, sex_all = per_speaker_tables(df, measured)
    binning = {}
    for f in measured:
        bn = {}
        for q in QS:
            edges, realized = bin_edges(means_all[f].values, q)
            bn[str(q)] = {"requested": q, "realized_bins": int(realized),
                          "degenerate": bool(realized < q),
                          "edges": [round(float(e), 6) for e in edges] if edges is not None else None}
        binning[f] = bn
    with open(os.path.join(results, "binning.json"), "w") as fh:
        json.dump(binning, fh, indent=2)

    # ---- 3 F-ratios pooled + within-sex ----
    pooled = fratio_block(df, measured)
    male = fratio_block(df[df["sex"] == "M"], measured)
    female = fratio_block(df[df["sex"] == "F"], measured)
    fr_rows = []
    for f in measured:
        p, m, fe = pooled[f], male[f], female[f]
        # combined within-sex F_ratio: average of male & female between/within decompositions
        wm, wf = m["F_ratio"], fe["F_ratio"]
        comb = np.nanmean([x for x in [wm, wf] if x is not None]) if (wm is not None or wf is not None) else None
        fr_rows.append({
            "feature": f,
            "within_var": p["within_var"], "between_var": p["between_var"],
            "F_ratio_pooled": p["F_ratio"], "q_max_pooled": p["q_max"],
            "anova_F": p["anova_F"], "anova_p": p["anova_p"],
            "F_ratio_male": m["F_ratio"], "F_ratio_female": fe["F_ratio"],
            "F_ratio_within_sex": float(comb) if comb is not None else None,
            "q_max_male": m["q_max"], "q_max_female": fe["q_max"],
        })
    fr_df = pd.DataFrame(fr_rows).sort_values("F_ratio_pooled", ascending=False, na_position="last")
    fr_df.to_csv(os.path.join(results, "f_ratio.csv"), index=False)

    # ---- 4 usable bits ----
    ub, H_spk, S = usable_bits(df, measured, n_perm=int(os.environ.get("TIMIT_NPERM", 200)))
    ub_rows = []
    for f in measured:
        b = ub[f]
        if b is None:
            ub_rows.append({"feature": f, "usable_bits": None})
        else:
            ub_rows.append({"feature": f, "b_star": b["b"], "q_eff": b["q_eff"],
                            "usable_bits": b["I_corrected_bits"], "norm_MI": b["norm_MI"],
                            "perm_p": b["perm_p"], "I_mm_bits": b["I_mm_bits"]})
    ub_df = pd.DataFrame(ub_rows).sort_values("usable_bits", ascending=False, na_position="last")
    ub_df.to_csv(os.path.join(results, "usable_bits.csv"), index=False)
    total_bits = float(np.nansum([r.get("usable_bits") or 0 for r in ub_rows]))

    # ---- 5 effective dimensionality ----
    # use measured features whose per-speaker means are complete across all speakers
    pr_feats = [f for f in measured if means_all[f].notna().all()]
    M_all = means_all[pr_feats].values
    pr_pooled, ci_pooled = pr_with_ci(M_all, n_boot=int(os.environ.get("TIMIT_NBOOT", 1000)))

    males_idx = sex_all.values == "M"
    fem_idx = sex_all.values == "F"
    pr_m, ci_m = pr_with_ci(M_all[males_idx], n_boot=int(os.environ.get("TIMIT_NBOOT", 1000)))
    pr_f, ci_f = pr_with_ci(M_all[fem_idx], n_boot=int(os.environ.get("TIMIT_NBOOT", 1000)))

    # parent-residual: regress each feature on sex(+age+height if available) then PR on residuals
    spk_meta = df.groupby("speaker").agg(sex=("sex", "first"),
                                         age=("age_years", "first"),
                                         height=("height_cm", "first"))
    spk_meta = spk_meta.loc[means_all.index]
    design_cols = [(spk_meta["sex"] == "M").astype(float).values]
    used_parents = ["sex"]
    keep = np.ones(len(spk_meta), dtype=bool)
    if spk_meta["age"].notna().mean() > 0.9:
        design_cols.append(spk_meta["age"].values.astype(float)); used_parents.append("age")
        keep &= spk_meta["age"].notna().values
    if spk_meta["height"].notna().mean() > 0.9:
        design_cols.append(spk_meta["height"].values.astype(float)); used_parents.append("height")
        keep &= spk_meta["height"].notna().values
    X = np.column_stack([np.ones(len(spk_meta))] + design_cols)[keep]
    Mp = M_all[keep]
    beta, *_ = np.linalg.lstsq(X, Mp, rcond=None)
    resid = Mp - X @ beta
    pr_resid, ci_resid = pr_with_ci(resid, n_boot=int(os.environ.get("TIMIT_NBOOT", 1000)))

    effdim = {
        "pr_features_used": pr_feats, "n_pr_features": len(pr_feats),
        "pooled": {"PR": pr_pooled, "CI95": ci_pooled, "n_speakers": int(M_all.shape[0])},
        "within_sex": {"male": {"PR": pr_m, "CI95": ci_m, "n": int(males_idx.sum())},
                       "female": {"PR": pr_f, "CI95": ci_f, "n": int(fem_idx.sum())},
                       "mean": float(np.mean([pr_m, pr_f]))},
        "parent_residual": {"PR": pr_resid, "CI95": ci_resid, "parents_used": used_parents,
                            "n_speakers": int(keep.sum())},
    }
    with open(os.path.join(results, "effective_dim.json"), "w") as fh:
        json.dump(effdim, fh, indent=2)

    # ---- 7 collision sanity (illustrative birthday model) ----
    def collision(qmax, pr, n=1e10):
        if qmax is None or qmax < 2:
            return None
        N_cells = qmax ** pr
        pe = n * n / (2.0 * N_cells)             # expected colliding pairs (approx)
        pm = 1.0 - np.exp(-min(n * n / (2.0 * N_cells), 700))  # P(>=1 collision)
        pb = 1.0 / N_cells                        # per-pair collision prob
        return {"q_max": qmax, "PR": pr, "log10_N_cells": float(np.log10(N_cells)),
                "P_E_expected_pairs": float(pe), "P_M_any_collision": float(pm),
                "P_B_per_pair": float(pb)}
    # use median q_max over top features as the operating q_max
    qmaxes = [r["q_max_pooled"] for r in fr_rows if r["q_max_pooled"]]
    q_op = int(np.median(qmaxes)) if qmaxes else 2
    collision_out = {
        "operating_q_max": q_op,
        "pooled_PR": collision(q_op, pr_pooled),
        "parent_residual_PR": collision(q_op, pr_resid),
    }
    with open(os.path.join(results, "collision.json"), "w") as fh:
        json.dump(collision_out, fh, indent=2)

    summary = {"n_rows": int(len(df)), "n_speakers": int(S), "H_speaker_bits": float(H_spk),
               "n_measured": n_measured, "measured": measured,
               "total_usable_bits": total_bits, "operating_q_max": q_op}
    with open(os.path.join(results, "analyze_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"analyze DONE: PR pooled={pr_pooled:.3f} within-sex mean={effdim['within_sex']['mean']:.3f} "
        f"parent-resid={pr_resid:.3f}; total usable bits={total_bits:.2f}")


if __name__ == "__main__":
    main()
