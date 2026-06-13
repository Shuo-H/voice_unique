"""
step5_pr.py -- STEP 5: effective dimensionality (participation ratio) of the
per-speaker mean-feature matrix, computed three ways:

  (a) POOLED            : all complete speakers, z-scored.
  (b) WITHIN-SEX        : male and female separately + mean.
  (c) PARENT-RESIDUAL   : regress each feature (per-speaker mean) on the shared
                          parents sex + age-bucket + accent (categorical dummies;
                          missing -> 'unknown' level, rare accents -> 'other'),
                          then PR on the residuals.  This is the effective
                          dimensionality SURVIVING after the dominant confounders
                          are removed.

PR = (Σλ_i)^2 / Σλ_i^2 of the z-scored (correlation) matrix.  95% CIs from 1000
speaker-level bootstraps (seed 1234); the parent-residual regression is refit
inside every bootstrap resample.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import common as C
import features as F

SEED = 1234
N_BOOT = 1000
RARE_ACCENT_MIN = 10        # accents with fewer speakers -> 'other'


def participation_ratio_corr(M):
    """PR of the correlation matrix of columns of M (rows=speakers)."""
    Mz = C.zscore(M)
    Cc = np.corrcoef(Mz, rowvar=False)
    if not np.all(np.isfinite(Cc)):
        return np.nan
    lam = np.linalg.eigvalsh(Cc)
    lam = lam[lam > 0]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def design_parents(meta):
    """Build the parent design matrix (intercept + dummies for sex, age, accent).
    meta: DataFrame with columns sex, accent, age (NaN allowed)."""
    d = meta.copy()
    d["sex"] = d["sex"].fillna("unknown")
    d["age"] = d["age"].fillna("unknown")
    acc = d["accent"].fillna("unknown").astype(str)
    vc = acc.value_counts()
    rare = set(vc[vc < RARE_ACCENT_MIN].index)
    d["accent"] = acc.where(~acc.isin(rare), "other")
    dummies = pd.get_dummies(d[["sex", "age", "accent"]], drop_first=True)
    X = np.column_stack([np.ones(len(d)), dummies.to_numpy(dtype=float)])
    return X, dummies.shape[1]


def residualize(Y, X):
    """OLS residuals of each column of Y on design X (least squares)."""
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ B


def boot_pr(func, n, seed=SEED, nboot=N_BOOT):
    """Bootstrap PR via `func(idx)` which returns PR for a speaker-index set."""
    rng = np.random.default_rng(seed)
    pt = func(np.arange(n))
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        v = func(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = (np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return float(pt), float(lo), float(hi)


def main():
    os.chdir(C.HERE)
    t0 = time.time()
    df = C.load_long()
    cov = C.coverage_table(df)
    feats = C.measured_features(cov)
    wide = C.wide_utt(df, feats)
    spk, ndrop = C.speaker_means(wide, feats)
    print(f"[step5] {len(spk)} complete speakers x {len(feats)} features "
          f"({ndrop} dropped)", flush=True)

    rows = []

    # ---- (a) POOLED ----
    Yp = spk[feats].to_numpy(dtype=float)
    pr, lo, hi = boot_pr(lambda idx: participation_ratio_corr(Yp[idx]), len(Yp))
    rows.append(dict(analysis="pooled", n_speakers=len(Yp), k_features=len(feats),
                     PR=pr, ci_lo=lo, ci_hi=hi))
    print(f"[step5] (a) pooled PR={pr:.2f} [{lo:.2f},{hi:.2f}]", flush=True)

    # ---- (b) WITHIN-SEX ----
    sex_prs = []
    for sx, key in [("male", "male_masculine"), ("female", "female_feminine")]:
        sub = spk[spk.sex == key]
        Ys = sub[feats].to_numpy(dtype=float)
        pr, lo, hi = boot_pr(lambda idx: participation_ratio_corr(Ys[idx]), len(Ys),
                             seed=SEED + (1 if sx == "male" else 2))
        rows.append(dict(analysis=f"within_sex_{sx}", n_speakers=len(Ys),
                         k_features=len(feats), PR=pr, ci_lo=lo, ci_hi=hi))
        sex_prs.append(pr)
        print(f"[step5] (b) {sx} PR={pr:.2f} [{lo:.2f},{hi:.2f}] (n={len(Ys)})",
              flush=True)
    rows.append(dict(analysis="within_sex_mean", n_speakers=np.nan,
                     k_features=len(feats), PR=float(np.mean(sex_prs)),
                     ci_lo=np.nan, ci_hi=np.nan))

    # ---- (c) PARENT-RESIDUAL (sex + age + accent) ----
    meta = spk[["sex", "accent", "age"]]
    Xfull, n_dummies = design_parents(meta)

    def pr_resid(idx):
        Xi, Yi = Xfull[idx], Yp[idx]
        R = residualize(Yi, Xi)
        return participation_ratio_corr(R)

    pr, lo, hi = boot_pr(pr_resid, len(Yp), seed=SEED + 3)
    rows.append(dict(analysis="parent_residual", n_speakers=len(Yp),
                     k_features=len(feats), PR=pr, ci_lo=lo, ci_hi=hi,
                     n_parent_dummies=n_dummies))
    print(f"[step5] (c) parent-residual PR={pr:.2f} [{lo:.2f},{hi:.2f}] "
          f"({n_dummies} parent dummies)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("pr_effective_dim.csv", index=False)

    # variance explained by parents (avg R^2 across features) for context
    R = residualize(Yp, Xfull)
    ss_tot = ((Yp - Yp.mean(0)) ** 2).sum(0)
    ss_res = (R ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    parent_r2 = pd.DataFrame(dict(feature=feats, display=[F.disp(f) for f in feats],
                                  parent_R2=r2)).sort_values("parent_R2",
                                                             ascending=False)
    parent_r2.to_csv("artifacts/parent_R2.csv", index=False)

    pooled_pr = out[out.analysis == "pooled"].PR.iloc[0]
    sex_mean = out[out.analysis == "within_sex_mean"].PR.iloc[0]
    resid_pr = out[out.analysis == "parent_residual"].PR.iloc[0]
    summ = dict(k_features=len(feats), n_speakers=int(len(Yp)),
                PR_pooled=float(pooled_pr), PR_within_sex_mean=float(sex_mean),
                PR_parent_residual=float(resid_pr),
                rise_pooled_to_within_sex=float(sex_mean - pooled_pr),
                rise_pooled_to_residual=float(resid_pr - pooled_pr),
                mean_parent_R2=float(r2.mean()),
                top_parent_R2=parent_r2.head(6).to_dict("records"))
    json.dump(summ, open("artifacts/pr_summary.json", "w"), indent=2, default=str)
    print(f"[step5] PR rise: pooled {pooled_pr:.2f} -> within-sex {sex_mean:.2f} "
          f"-> parent-residual {resid_pr:.2f}; mean parent R^2={r2.mean():.3f}; "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
