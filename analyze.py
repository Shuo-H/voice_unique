"""Sections 1-5 and 7 of the TIMIT battery. Reads features/features_per_utt.parquet
and results/speaker_meta.csv; writes machine-readable tables to results/.
Single fixed RNG seed 1234 everywhere."""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

SEED = 1234
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

from feat_lib import FEATURES_40, NOT_ATTEMPTED

QSET = [2, 3, 5, 10]
COV_MEASURED = 0.50      # threshold to call a feature MEASURED
COV_PR = 0.90            # coverage to enter PR / matrix analyses
N_PERM = 200
N_BOOT = 1000

df = pd.read_parquet("features/features_per_utt.parquet")
meta = pd.read_csv("results/speaker_meta.csv")
os.makedirs("results", exist_ok=True)

N_utt = len(df)
speakers = sorted(df["speaker"].unique())
N_spk = len(speakers)
sex_of = df.groupby("speaker")["sex"].first().to_dict()

# ---------------- Section 1: coverage ----------------
cov_rows = []
for f in FEATURES_40:
    vals = df[f].to_numpy(dtype=float)
    cov = float(np.mean(np.isfinite(vals)))
    attempted = f not in NOT_ATTEMPTED
    measured = attempted and cov >= COV_MEASURED
    cov_rows.append({"feature": f, "coverage": round(cov, 4),
                     "attempted": attempted,
                     "status": "MEASURED" if measured else "NOT-MEASURED"})
cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv("results/coverage.csv", index=False)
MEAS = [r["feature"] for r in cov_rows if r["status"] == "MEASURED"]
n_meas = len(MEAS)
print(f"[1] measured {n_meas}/40")

# ---------------- per-speaker aggregation ----------------
def per_speaker(feature, spk_list):
    means, withinvars, utt_by_spk = {}, {}, {}
    g = df[df["speaker"].isin(spk_list)].groupby("speaker")[feature]
    for spk, series in g:
        v = series.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size >= 1:
            means[spk] = float(np.mean(v))
            utt_by_spk[spk] = v
        if v.size >= 2:
            withinvars[spk] = float(np.var(v, ddof=1))
    return means, withinvars, utt_by_spk

ALL = speakers
MALE = [s for s in speakers if sex_of[s] == "M"]
FEM = [s for s in speakers if sex_of[s] == "F"]

# ---------------- Section 2: quantile bins ----------------
bin_rows = []
for f in MEAS:
    means, _, _ = per_speaker(f, ALL)
    x = np.array(sorted(means.values()))
    row = {"feature": f, "n_speakers": len(x)}
    for q in QSET:
        edges = np.quantile(x, np.linspace(0, 1, q + 1))
        interior = edges[1:-1]
        uniq = np.unique(np.round(interior, 10))
        realized = len(uniq) + 1 if interior.size else 1
        # realized bins actually occupied:
        binidx = np.digitize(x, uniq)
        realized_occ = len(np.unique(binidx))
        row[f"q{q}_realized"] = int(realized_occ)
        row[f"q{q}_degenerate"] = bool(realized_occ < q)
    bin_rows.append(row)
pd.DataFrame(bin_rows).to_csv("results/binning.csv", index=False)
print(f"[2] binning done for {len(bin_rows)} features")

# ---------------- Section 3: F-ratios + q_max ----------------
def bin_crossing_qmax(means, utt_by_spk):
    x = np.array(sorted(means.values()))
    qmax = 1
    rates = {}
    for q in QSET:
        edges = np.quantile(x, np.linspace(0, 1, q + 1))[1:-1]
        edges = np.unique(edges)
        spk_rates = []
        for spk, uv in utt_by_spk.items():
            if uv.size < 2:
                continue
            home = np.digitize(means[spk], edges)
            bins = np.digitize(uv, edges)
            spk_rates.append(np.mean(bins != home))
        mr = float(np.mean(spk_rates)) if spk_rates else 1.0
        rates[q] = mr
        if mr < 0.20:
            qmax = q
    return qmax, rates

def decomp(feature, spk_list):
    means, wv, ubs = per_speaker(feature, spk_list)
    if len(means) < 2:
        return None
    within_var = float(np.mean(list(wv.values()))) if wv else np.nan
    between_var = float(np.var(list(means.values()), ddof=1))
    F = between_var / within_var if (within_var and within_var > 0) else np.nan
    # one-way ANOVA on utt-level values
    groups = [ubs[s] for s in ubs if ubs[s].size >= 1]
    try:
        anF, anp = stats.f_oneway(*groups)
    except Exception:
        anF, anp = np.nan, np.nan
    qmax, rates = bin_crossing_qmax(means, ubs)
    return dict(within_var=within_var, between_var=between_var, F_ratio=F,
                anova_F=float(anF), anova_p=float(anp), q_max=qmax,
                cross_rates=rates, n_spk=len(means))

fr_rows = []
for f in MEAS:
    p = decomp(f, ALL)
    m = decomp(f, MALE)
    w = decomp(f, FEM)
    fr_rows.append({
        "feature": f,
        "within_var": p["within_var"], "between_var": p["between_var"],
        "F_ratio_pooled": p["F_ratio"], "q_max_pooled": p["q_max"],
        "anova_F": p["anova_F"], "anova_p": p["anova_p"],
        "F_ratio_male": m["F_ratio"] if m else np.nan,
        "F_ratio_female": w["F_ratio"] if w else np.nan,
        "F_ratio_within_sex": np.nanmean([m["F_ratio"] if m else np.nan,
                                          w["F_ratio"] if w else np.nan]),
        "q_max_male": m["q_max"] if m else np.nan,
        "q_max_female": w["q_max"] if w else np.nan,
        "q_max_within_sex": int(min(m["q_max"] if m else 1,
                                    w["q_max"] if w else 1)),
    })
fr_df = pd.DataFrame(fr_rows).sort_values("F_ratio_pooled", ascending=False)
fr_df.to_csv("results/f_ratio.csv", index=False)
print(f"[3] f-ratios done; top feature {fr_df.iloc[0]['feature']} "
      f"F={fr_df.iloc[0]['F_ratio_pooled']:.2f}")

# ---------------- Section 4: usable bit depth (MI) ----------------
def entropy_counts(counts):
    n = counts.sum()
    p = counts[counts > 0] / n
    H = -np.sum(p * np.log2(p))
    K = np.count_nonzero(counts)
    H_mm = H + (K - 1) / (2 * n) / np.log(2)  # Miller-Madow (bits)
    return H_mm

def mi_mm(xbin, yidx, q, S):
    n = len(xbin)
    joint = np.bincount(xbin * S + yidx, minlength=q * S).astype(float)
    Hxy = entropy_counts(joint)
    Hx = entropy_counts(np.bincount(xbin, minlength=q).astype(float))
    Hy = entropy_counts(np.bincount(yidx, minlength=S).astype(float))
    return Hx + Hy - Hxy

spk_to_i = {s: i for i, s in enumerate(speakers)}
ub_rows = []
H_speaker = np.log2(N_spk)
for f in MEAS:
    vals = df[f].to_numpy(dtype=float)
    yall = df["speaker"].map(spk_to_i).to_numpy()
    mask = np.isfinite(vals)
    x = vals[mask]; y = yall[mask]
    S = N_spk
    best = None
    for b in range(1, 9):
        q = 2 ** b
        edges = np.quantile(x, np.linspace(0, 1, q + 1))[1:-1]
        edges = np.unique(edges)
        xb = np.digitize(x, edges)
        qeff = len(np.unique(xb))
        obs = mi_mm(xb, y, xb.max() + 1, S)
        nulls = np.empty(N_PERM)
        yp = y.copy()
        for k in range(N_PERM):
            rng.shuffle(yp)
            nulls[k] = mi_mm(xb, yp, xb.max() + 1, S)
        null_mean = float(np.mean(nulls))
        I_corr = max(0.0, obs - null_mean)
        pval = float((np.sum(nulls >= obs) + 1) / (N_PERM + 1))
        cand = dict(b=b, q_eff=int(qeff), I_mm=float(obs),
                    I_null_mean=null_mean, I_corrected=I_corr,
                    norm_MI=I_corr / H_speaker, perm_p=pval)
        if best is None or I_corr > best["I_corrected"]:
            best = cand
    best["feature"] = f
    ub_rows.append(best)
ub_df = pd.DataFrame(ub_rows).sort_values("I_corrected", ascending=False)
ub_df = ub_df[["feature", "b", "q_eff", "I_mm", "I_null_mean",
               "I_corrected", "norm_MI", "perm_p"]]
ub_df.to_csv("results/usable_bits.csv", index=False)
total_bits = float(ub_df["I_corrected"].sum())
print(f"[4] usable bits done; total summed={total_bits:.3f} bits "
      f"(top {ub_df.iloc[0]['feature']}={ub_df.iloc[0]['I_corrected']:.3f})")

# ---------------- Section 5: effective dimensionality (PR) ----------------
PR_FEATS = [r["feature"] for r in cov_rows
            if r["status"] == "MEASURED" and r["coverage"] >= COV_PR]

def speaker_matrix(spk_list, feats):
    rows, kept = [], []
    for s in spk_list:
        means = {}
        sub = df[df["speaker"] == s]
        ok = True
        for f in feats:
            v = sub[f].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                ok = False; break
            means[f] = np.mean(v)
        if ok:
            rows.append([means[f] for f in feats]); kept.append(s)
    return np.array(rows), kept

def participation_ratio(M):
    if M.shape[0] < 3:
        return np.nan
    Z = (M - M.mean(0)) / (M.std(0, ddof=1) + 1e-12)
    C = np.cov(Z, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-10]
    return float((ev.sum() ** 2) / np.sum(ev ** 2))

def pr_with_ci(M):
    pr = participation_ratio(M)
    n = M.shape[0]
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(participation_ratio(M[idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pr, float(lo), float(hi)

Mall, kept_all = speaker_matrix(ALL, PR_FEATS)
pr_pool, pl, ph = pr_with_ci(Mall)
Mm, _ = speaker_matrix(MALE, PR_FEATS)
Mf, _ = speaker_matrix(FEM, PR_FEATS)
pr_m, ml, mh = pr_with_ci(Mm)
pr_f, fl, fh = pr_with_ci(Mf)

# parent-residual: regress each feature on [1, sex, height, age]
meta_idx = meta.set_index("speaker")
def design_for(spk_list):
    keep = []
    X = []
    for s in spk_list:
        if s not in meta_idx.index:
            continue
        r = meta_idx.loc[s]
        if r["height_cm"] == "" or pd.isna(r["height_cm"]) or \
           r["age_yr"] == "" or pd.isna(r["age_yr"]):
            continue
        sex_d = 1.0 if r["sex"] == "M" else 0.0
        X.append([1.0, sex_d, float(r["height_cm"]), float(r["age_yr"])])
        keep.append(s)
    return np.array(X), keep

Xd, keep_meta = design_for(ALL)
Mr, kept_r = speaker_matrix(keep_meta, PR_FEATS)
# align design to kept_r
pos = {s: i for i, s in enumerate(keep_meta)}
Xd2 = np.array([Xd[pos[s]] for s in kept_r])
# residualize each column
beta, *_ = np.linalg.lstsq(Xd2, Mr, rcond=None)
resid = Mr - Xd2 @ beta
pr_res, rl, rh = pr_with_ci(resid)

eff = {
    "pr_features_used": PR_FEATS, "n_pr_features": len(PR_FEATS),
    "pooled": {"PR": pr_pool, "ci95": [pl, ph], "n_speakers": Mall.shape[0]},
    "within_sex": {"male": {"PR": pr_m, "ci95": [ml, mh], "n": Mm.shape[0]},
                   "female": {"PR": pr_f, "ci95": [fl, fh], "n": Mf.shape[0]},
                   "mean_PR": float(np.mean([pr_m, pr_f]))},
    "parent_residual": {"PR": pr_res, "ci95": [rl, rh],
                        "n_speakers": resid.shape[0],
                        "parents": ["sex", "height_cm", "age_yr"]},
}
with open("results/effective_dim.json", "w") as fh:
    json.dump(eff, fh, indent=2)
print(f"[5] PR pooled={pr_pool:.2f} within-sex mean={eff['within_sex']['mean_PR']:.2f} "
      f"parent-resid={pr_res:.2f}")

# ---------------- Section 7: collision cross-check ----------------
def collisions(d, q, n=1e10):
    m = q ** d
    P_E = 1 - np.exp(-(n - 1) / m)
    P_M = 1.0 / m
    logPBbar = -n * (n - 1) / (2 * m)
    P_B = 1 - np.exp(logPBbar) if logPBbar > -700 else 1.0
    return {"d": d, "q": q, "log10_m": float(d * np.log10(q)),
            "P_E": P_E, "P_M": P_M, "P_B": P_B}

# representative q: pooled F0 q_max and a conservative q=5
f0_qmax = int(fr_df[fr_df["feature"] == "F0"]["q_max_pooled"].iloc[0]) \
    if "F0" in set(fr_df["feature"]) else 5
coll = {"n": 1e10,
        "operating_points": []}
for (label, d) in [("PR_pooled", pr_pool), ("PR_parent_residual", pr_res)]:
    for q in sorted(set([f0_qmax, 5])):
        c = collisions(d, q)
        c["d_source"] = label
        coll["operating_points"].append(c)
with open("results/collision.json", "w") as fh:
    json.dump(coll, fh, indent=2)
print(f"[7] collision cross-check done (F0 q_max={f0_qmax})")

# summary of provenance / counts
summary = {"n_utt": N_utt, "n_speakers": N_spk, "n_measured": n_meas,
           "measured_features": MEAS, "total_usable_bits": total_bits,
           "pr_features": PR_FEATS}
with open("results/analyze_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("ANALYZE_DONE")
