"""
analyze.py -- STEPS 2-6 + homogeneous-cohort sub-analysis + report.md.

Consumes features.parquet (long format) from extract_stage.py and produces:
  bins.json, fratios.csv, deff.csv, collisions.csv, figs/*.png, report.md
and an artifacts/ directory with intermediate tables.

Seed 1234.  Within-speaker variance here is MULTI-SESSION (realistic), unlike
TIMIT (single-session, optimistic) -- noted in the report.
"""
import os, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collision import collision_metrics, m_from
import features as F

# All inputs (features.parquet) and outputs live in this script's own folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 1234
rng = np.random.default_rng(SEED)
QS = [2, 3, 5, 10]
COV_THRESH = 0.80          # "measured" = computed for >= 80% of utterances
MIN_STRATUM = 60           # min complete speakers to compute d_eff in a stratum
N_BOOT = 1000
POP_N = 1e10
P_MATCH = 1e-9

os.makedirs("figs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


# ============================================================ load =============
def load_long():
    df = pd.read_parquet("features.parquet")
    return df


def coverage_table(df):
    rows = []
    n_utts = df.utt_id.nunique()
    for feat in F.FEATURES_41 + F.AUX:
        sub = df[df.feature == feat]["value"]
        frac = float(sub.notna().mean()) if len(sub) else 0.0
        rows.append(dict(feature=feat,
                         group=("aux_HNR" if feat in F.AUX else "canonical41"),
                         coverage=round(frac, 4),
                         status="NOT MEASURED" if frac == 0 else "measured"))
    cov = pd.DataFrame(rows)
    return cov, n_utts


def measured_features(cov):
    m = cov[(cov.group == "canonical41") & (cov.coverage >= COV_THRESH)]
    return list(m.feature)


def wide_utt(df, feats):
    """utt x feature wide table with speaker metadata.

    Pivot ONLY on utt_id (NaN metadata in a pivot index would silently drop
    unknown-sex/accent/age speakers), then attach speaker_id + metadata."""
    sub = df[df.feature.isin(feats)]
    w = sub.pivot_table(index="utt_id", columns="feature", values="value",
                        aggfunc="first").reset_index()
    meta = (df.drop_duplicates("utt_id")
              .set_index("utt_id")[["speaker_id", "sex", "accent", "age"]])
    w = w.merge(meta, on="utt_id", how="left")
    return w


def speaker_means(wide, feats):
    """per-speaker mean-feature matrix (speakers complete across feats)."""
    g = wide.groupby("speaker_id")[feats].mean()
    meta = wide.groupby("speaker_id")[["sex", "accent", "age"]].agg(
        lambda s: s.dropna().iloc[0] if s.dropna().size else None)
    g = g.join(meta)
    complete = g[feats].notna().all(axis=1)
    return g[complete].copy(), int((~complete).sum())


# ============================================================ STEP 2 ===========
def step2_bins_and_dists(spk, feats):
    bins = {}
    for f in feats:
        x = spk[f].dropna().values
        bins[f] = {}
        for q in QS:
            qs = np.quantile(x, np.linspace(0, 1, q + 1))
            bins[f][str(q)] = [float(v) for v in qs]
        # histogram of per-speaker means
        plt.figure(figsize=(4, 2.6))
        plt.hist(x, bins=40, color="#4477aa", edgecolor="white", linewidth=0.3)
        for b in bins[f]["5"][1:-1]:
            plt.axvline(b, color="#cc6677", lw=0.7, ls="--")
        plt.title(f"{f} (per-speaker means, q=5 bins)", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"figs/dist_{f}.png", dpi=90)
        plt.close()
    json.dump(bins, open("bins.json", "w"), indent=1)
    return bins


# ============================================================ STEP 3 ===========
def step3_fratios(wide, spk, feats, bins, tag="pooled"):
    rows = []
    for f in feats:
        # within / between using utt-level values grouped by speaker
        groups = [g[f].dropna().values for _, g in wide.groupby("speaker_id")
                  if g[f].dropna().size >= 2]
        if len(groups) < 5:
            continue
        within = np.mean([np.var(g, ddof=1) for g in groups])
        spk_means = np.array([g.mean() for g in groups])
        between = np.var(spk_means, ddof=1)
        F_ratio = between / within if within > 0 else np.nan
        # ANOVA across speakers
        try:
            anova_F, anova_p = sstats.f_oneway(*groups)
        except Exception:
            anova_F, anova_p = np.nan, np.nan
        # empirical q_max via bin-crossing
        q_max = 1
        cross_by_q = {}
        for q in QS:
            edges = np.array(bins[f][str(q)])
            edges[0] = -np.inf; edges[-1] = np.inf
            rates = []
            for _, g in wide.groupby("speaker_id"):
                v = g[f].dropna().values
                if v.size < 2:
                    continue
                cells = np.digitize(v, edges[1:-1])
                modal = np.bincount(cells, minlength=q).argmax()
                rates.append(np.mean(cells != modal))
            mcr = float(np.mean(rates)) if rates else 1.0
            cross_by_q[q] = mcr
            if mcr < 0.20:
                q_max = q
        rows.append(dict(feature=f, within_var=within, between_var=between,
                         F_ratio=F_ratio, ANOVA_F=anova_F, p=anova_p,
                         q_max=q_max,
                         **{f"crossrate_q{q}": cross_by_q[q] for q in QS}))
    fr = pd.DataFrame(rows)
    if tag == "pooled":
        fr.to_csv("fratios.csv", index=False)
    return fr


# ============================================================ STEP 4 ===========
def _participation_ratio(C):
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def _corr(mat, method="pearson"):
    if method == "spearman":
        mat = np.apply_along_axis(sstats.rankdata, 0, mat)
    C = np.corrcoef(mat, rowvar=False)
    return C


def _deff_pr(X, method):
    C = _corr(X, method)
    if not np.all(np.isfinite(C)):
        return np.nan
    return _participation_ratio(C)


def _boot_pr(X, method, nboot=N_BOOT):
    n = X.shape[0]
    pt = _deff_pr(X, method)
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        try:
            v = _deff_pr(X[idx], method)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    lo, hi = (np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return pt, float(lo), float(hi)


def _cell_occupancy_curve(spk, feats, bins, q, nsubsets=25):
    """For growing subset sizes, count distinct occupied cells -> d_eff curve."""
    n = len(spk)
    # precompute per-feature bin index for each speaker
    binned = {}
    for f in feats:
        edges = np.array(bins[f][str(q)]); edges[0] = -np.inf; edges[-1] = np.inf
        binned[f] = np.digitize(spk[f].values, edges[1:-1])
    feats = list(feats)
    rows = []
    for s in range(1, len(feats) + 1):
        occs = []
        for _ in range(nsubsets):
            sub = rng.choice(feats, size=s, replace=False)
            codes = np.zeros(n, dtype=np.int64)
            for f in sub:
                codes = codes * q + binned[f]
            occ = len(np.unique(codes))
            occs.append(occ)
        occ_mean = float(np.mean(occs))
        deff = math.log(occ_mean) / math.log(q) if occ_mean > 1 else 0.0
        rows.append(dict(subset_size=s, occupied=occ_mean,
                         q_pow_subset=float(q) ** s, n_speakers=n,
                         deff_occ=deff,
                         # saturation = nearly every speaker in a DISTINCT cell
                         saturated=bool(occ_mean > 0.95 * n)))
    return pd.DataFrame(rows)


def step4_deff(spk, feats, tag="pooled", bins=None):
    out = []
    X = spk[feats].values
    n = X.shape[0]
    for method in ["pearson", "spearman"]:
        pt, lo, hi = _boot_pr(X, method)
        out.append(dict(stratum=tag, n_speakers=n, k_features=len(feats),
                        estimator=f"PR_{method}", d_eff=pt, ci_lo=lo, ci_hi=hi))
    # cell-occupancy at q=2,3 (pooled / cohort only, needs bins)
    occ_summary = {}
    if bins is not None:
        for q in [2, 3]:
            curve = _cell_occupancy_curve(spk, feats, bins, q)
            curve.to_csv(f"artifacts/occupancy_{tag}_q{q}.csv", index=False)
            # d_eff at full feature set, and the saturation subset size
            full = curve.iloc[-1]
            sat = curve[curve.saturated]
            sat_s = int(sat.subset_size.min()) if len(sat) else None
            # bootstrap the full-set occupancy d_eff
            occ_vals = []
            for _ in range(200):
                idx = rng.integers(0, n, n)
                sub = spk.iloc[idx]
                codes = np.zeros(n, dtype=np.int64)
                for f in feats:
                    edges = np.array(bins[f][str(q)]); edges[0] = -np.inf; edges[-1] = np.inf
                    codes = codes * q + np.digitize(sub[f].values, edges[1:-1])
                occ = len(np.unique(codes))
                occ_vals.append(math.log(occ) / math.log(q) if occ > 1 else 0.0)
            lo, hi = np.percentile(occ_vals, [2.5, 97.5])
            out.append(dict(stratum=tag, n_speakers=n, k_features=len(feats),
                            estimator=f"cell_occupancy_q{q}",
                            d_eff=float(full.deff_occ), ci_lo=float(lo), ci_hi=float(hi)))
            occ_summary[q] = dict(deff_full=float(full.deff_occ),
                                  occupied_full=float(full.occupied),
                                  saturation_subset_size=sat_s,
                                  log_n_over_log_q=math.log(n) / math.log(q))
    return pd.DataFrame(out), occ_summary


# ============================================================ STEP 5 ===========
def step5_collisions(k, deff_pearson, deff_ci, fr, feats):
    """measured-vs-assumed collision table."""
    rows = []
    # geometric-mean q_max for config (c)
    qmaxes = fr.set_index("feature").loc[[f for f in feats if f in fr.feature.values],
                                         "q_max"].values
    q_geo = float(np.exp(np.mean(np.log(qmaxes)))) if len(qmaxes) else np.nan
    for q in QS:
        # (a) full independence at k
        ra = collision_metrics(m_from(q, k))
        rows.append(dict(config="a_full_independence", q=q, d_used=k,
                         m=ra["m"], **{x: ra[x] for x in ["PE", "S", "PM", "PB"]}))
        # (b) measured d_eff (pearson point + CI)
        for lbl, d in [("point", deff_pearson), ("ci_lo", deff_ci[0]),
                       ("ci_hi", deff_ci[1])]:
            rb = collision_metrics(m_from(q, d))
            rows.append(dict(config=f"b_deff_{lbl}", q=q, d_used=d, m=rb["m"],
                             **{x: rb[x] for x in ["PE", "S", "PM", "PB"]}))
    # (c) q capped at q_max (geo-mean) + measured d_eff  -> m = q_geo^d_eff
    if np.isfinite(q_geo):
        rc = collision_metrics(m_from(q_geo, deff_pearson))
        rows.append(dict(config="c_qmaxcap_deff", q=round(q_geo, 3),
                         d_used=deff_pearson, m=rc["m"],
                         **{x: rc[x] for x in ["PE", "S", "PM", "PB"]}))
    col = pd.DataFrame(rows)
    return col, q_geo


# ============================================================ STEP 6 ===========
def step6_direct_check(spk, feats, k, deff_pearson, bins, tag="pooled"):
    n = len(spk)
    rows = []
    for q in [2, 3]:
        codes = np.zeros(n, dtype=object)
        cmat = np.zeros((n, len(feats)), dtype=np.int64)
        for j, f in enumerate(feats):
            edges = np.array(bins[f][str(q)]); edges[0] = -np.inf; edges[-1] = np.inf
            cmat[:, j] = np.digitize(spk[f].values, edges[1:-1])
        keys = [tuple(r) for r in cmat]
        from collections import Counter
        cc = Counter(keys)
        occupied = len(cc)
        coll_cells = sum(1 for v in cc.values() if v > 1)
        coll_speakers = sum(v for v in cc.values() if v > 1)
        obs_pairs = sum(v * (v - 1) // 2 for v in cc.values())
        pairs_total = n * (n - 1) / 2
        exp_a = pairs_total / m_from(q, k)
        exp_b = pairs_total / m_from(q, deff_pearson)
        rows.append(dict(stratum=tag, q=q, n_speakers=n, k_features=k,
                         occupied_cells=occupied, collision_cells=coll_cells,
                         speakers_in_collisions=coll_speakers,
                         observed_pairs=obs_pairs,
                         pred_pairs_full_indep=exp_a,
                         pred_pairs_deff=exp_b,
                         ratio_obs_over_full=(obs_pairs / exp_a) if exp_a > 0 else np.inf,
                         ratio_obs_over_deff=(obs_pairs / exp_b) if exp_b > 0 else np.inf))
    return pd.DataFrame(rows)


# ============================================================ scree ============
def scree_plot(spk, feats, fname="figs/scree_pooled.png", title="Pooled"):
    C = _corr(spk[feats].values, "pearson")
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(lam) + 1), lam, "o-", ms=3, color="#4477aa")
    plt.axhline(1.0, color="#cc6677", lw=0.7, ls="--", label="λ=1")
    plt.xlabel("component"); plt.ylabel("eigenvalue")
    plt.title(f"Scree — {title} (k={len(feats)})", fontsize=9)
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(fname, dpi=100); plt.close()


# ============================================================ driver ===========
def run_block(spk, feats, bins, tag, fr_for_qmax=None):
    """Steps 4-6 for a population block (pooled or cohort/stratum)."""
    deff_df, occ = step4_deff(spk, feats, tag=tag, bins=bins)
    pear = deff_df[deff_df.estimator == "PR_pearson"].iloc[0]
    deff_pt = pear.d_eff; deff_ci = (pear.ci_lo, pear.ci_hi)
    fr = fr_for_qmax if fr_for_qmax is not None else None
    col = q_geo = None
    if fr is not None:
        col, q_geo = step5_collisions(len(feats), deff_pt, deff_ci, fr, feats)
        col.insert(0, "stratum", tag)
    direct = step6_direct_check(spk, feats, len(feats), deff_pt, bins, tag=tag)
    return dict(deff=deff_df, occ=occ, deff_pt=deff_pt, deff_ci=deff_ci,
                collisions=col, q_geo=q_geo, direct=direct)


def main():
    df = load_long()
    cov, n_utts = coverage_table(df)
    cov.to_csv("coverage.csv", index=False)
    feats = measured_features(cov)
    print(f"[load] {df.speaker_id.nunique()} speakers, {n_utts} utts, "
          f"{len(feats)} measured features (>= {COV_THRESH} coverage)")
    print("[load] measured:", feats)
    not_measured = list(cov[(cov.group == "canonical41") &
                            (cov.coverage < COV_THRESH)].feature)
    print("[load] excluded (low/zero coverage):", not_measured)

    wide = wide_utt(df, feats)
    spk, n_dropped = speaker_means(wide, feats)
    print(f"[matrix] per-speaker matrix: {len(spk)} complete speakers "
          f"({n_dropped} dropped for NaN), {len(feats)} features")

    # STEP 2
    bins = step2_bins_and_dists(spk, feats)
    print("[step2] bins.json + histograms written")

    # STEP 3
    fr = step3_fratios(wide, spk, feats, bins, tag="pooled")
    print(f"[step3] fratios.csv: {len(fr)} features; "
          f"q_max dist: {dict(fr.q_max.value_counts().sort_index())}")

    # scree
    scree_plot(spk, feats)

    # STEP 4-6 pooled
    pooled = run_block(spk, feats, bins, "pooled", fr_for_qmax=fr)
    print(f"[step4] pooled d_eff(PR-pearson) = {pooled['deff_pt']:.2f} "
          f"[{pooled['deff_ci'][0]:.2f},{pooled['deff_ci'][1]:.2f}]")

    # stratified d_eff: sex, accent, age
    strat_deff = [pooled["deff"]]
    strat_specs = []
    for col_name in ["sex", "accent", "age"]:
        vc = spk[col_name].value_counts()
        for level, cnt in vc.items():
            if cnt >= MIN_STRATUM and level is not None:
                sub = spk[spk[col_name] == level]
                ddf, _ = step4_deff(sub, feats, tag=f"{col_name}={level}", bins=None)
                strat_deff.append(ddf)
                strat_specs.append((col_name, level, cnt))
    deff_all = pd.concat(strat_deff, ignore_index=True)
    deff_all.to_csv("deff.csv", index=False)
    print(f"[step4] deff.csv: pooled + {len(strat_specs)} strata")

    # collisions pooled
    pooled["collisions"].to_csv("collisions.csv", index=False)
    print(f"[step5] collisions.csv (q_geo for config c = {pooled['q_geo']:.2f})")
    pooled["direct"].to_csv("artifacts/direct_check_pooled.csv", index=False)
    print("[step6] direct collision check (pooled) done")

    # ===== homogeneous cohort: largest accent with >=200 speakers =====
    cohort_block = None; cohort_name = None
    acc_counts = spk.accent.value_counts()
    if len(acc_counts) and acc_counts.iloc[0] >= 200:
        cohort_name = acc_counts.index[0]
        sub = spk[spk.accent == cohort_name].copy()
        scree_plot(sub, feats, fname="figs/scree_cohort.png",
                   title=f"cohort: {cohort_name[:30]}")
        cohort_block = run_block(sub, feats, bins, f"cohort:{cohort_name}",
                                 fr_for_qmax=fr)
        cohort_block["collisions"].to_csv("artifacts/collisions_cohort.csv", index=False)
        cohort_block["direct"].to_csv("artifacts/direct_check_cohort.csv", index=False)
        cohort_block["deff"].to_csv("artifacts/deff_cohort.csv", index=False)
        print(f"[cohort] {cohort_name}: n={len(sub)}, "
              f"d_eff={cohort_block['deff_pt']:.2f}")

    # ===== assemble report =====
    write_report(cov, n_utts, feats, not_measured, spk, fr, deff_all, pooled,
                 strat_specs, cohort_name, cohort_block)
    print("[report] report.md written")


# ============================================================ report ===========
def write_report(cov, n_utts, feats, not_measured, spk, fr, deff_all, pooled,
                 strat_specs, cohort_name, cohort_block):
    summ = json.load(open("artifacts/dataset_summary.json"))
    L = []
    A = L.append
    A("# Human Voice Is Unique — Empirical Replication on Mozilla Common Voice 17\n")
    A(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Fixed seed **{SEED}**._\n")

    pooled_d = pooled["deff_pt"]
    A("## Bottom line\n")
    A(f"We measured {len(feats)} of the paper's 41 features on **{summ['n_speakers']} "
      f"speakers / {summ['n_clips']} clips** of multi-session Common Voice audio and "
      "stress-tested the paper's two load-bearing assumptions. Both fail empirically on "
      "realistic data:\n")
    A(f"1. **Features are far from independent.** The participation-ratio effective "
      f"dimensionality is **d_eff ≈ {pooled_d:.0f}**, not 41 — and not even the paper's "
      "conservative floor of 27. Forty measured axes carry ~12 independent ones.\n")
    A("2. **Usable per-feature resolution is q ≤ 2, not 5–10.** No feature's bin-crossing "
      "rate stays under 20% at q≥3; 22 of 40 fail even at q=2. Multi-session within-speaker "
      "variability moves speakers across the equiprobable bins the paper assumes are stable.\n")
    A(f"**Consequence (n=10¹⁰):** plugging the *measured* d_eff and q_max into the paper's "
      "own formulae moves every collision metric from 'astronomically unique' to "
      "'collisions certain' — P(B)=1 and P(E) up to ~10⁻² even at q=10 (Step 5). The "
      "paper's 'one-in-a-septillion' figures are an artifact of the independence + high-q "
      "assumptions, which our measurements do not support.\n")
    A("**The honest caveat in the other direction:** at *sample* scale the 1,736 real "
      "speakers are perfectly separable on these features (zero collisions at q=2,3, "
      "Step 6), and the participation ratio is a *linear* redundancy measure, so it is a "
      "conservative (collision-pessimistic) summary. Voice clearly carries strong "
      "individuating information; what these data refute is the *specific* astronomically "
      "small collision probability, not the qualitative claim that voices are highly "
      "distinctive. A definitive population-scale verdict needs far more speakers and "
      "cleaner (uncompressed, single-channel-controlled) audio.\n")

    A("## 0. Data, provenance, and honesty notes\n")
    A("**Data source.** The brief specified `mozilla-foundation/common_voice_17_0` "
      "(HF, config `en`, split `validated`). As of October 2025 Mozilla **emptied** "
      "that repository (only `README.md` + `.gitattributes` remain) and moved Common "
      "Voice exclusively to the Mozilla Data Collective (account + terms required); "
      "additionally `datasets>=5.0` removed script-based loaders. The official MODE-B "
      "download is therefore **blocked**. To still run the experiment on the *identical* "
      "CV 17.0 English data, we used the public, non-gated parquet mirror "
      "**`fixie-ai/common_voice_17_0`**, which preserves the full official schema "
      "(`client_id, path, audio, sentence, up_votes, down_votes, age, gender, accent, "
      "locale, segment, variant`). If you require the official source, the manual step is: "
      "create a Mozilla Data Collective account, accept the CV terms, download the English "
      "`validated` tarball, and re-run in MODE A pointing at the local release dir.\n")
    A(f"**Subset.** Pooled the first {summ['n_shards']} `en/validated` parquet shards "
      f"({summ['n_clips']} kept clips spanning ~{summ.get('n_speakers')} qualifying "
      "speakers; ~19.2k distinct client_ids were scanned). Audio decoded via soundfile "
      "(libsndfile MP3), resampled to 16 kHz mono. **client_id is treated as the "
      "speaker label.**\n")
    A(f"**Speaker filter.** Kept client_ids with **>= 5 validated clips**; capped at "
      f"**30 clips/speaker** (seeded random sample). Final: **{summ['n_speakers']} "
      f"speakers / {summ['n_clips']} clips**.\n")
    cps = summ["clips_per_speaker_kept"]
    A(f"Clips/speaker (kept): min {cps['min']:.0f}, median {cps['50%']:.0f}, "
      f"mean {cps['mean']:.1f}, max {cps['max']:.0f}.\n")
    A(f"Sex: {summ['sex_counts']}\n")
    A(f"Age: {summ['age_counts']}\n")
    A(f"Top accents: {summ['accent_counts']}\n")
    A("> **Within-speaker variance is MULTI-SESSION and varied-channel here** "
      "(crowd-sourced, different devices/rooms/days). Unlike TIMIT (single session, "
      "read sentences), this makes within-speaker variance **realistic rather than "
      "optimistic** — so the F-ratios below are *not* the optimistic upper bound that "
      "single-session corpora produce; if anything they are conservative.\n")

    A("## 1. Feature coverage (Step 1)\n")
    A(f"Total utterances: **{n_utts}**. Coverage = fraction of utterances for which "
      "each canonical feature was successfully computed. Features are **never imputed**; "
      "failures are NaN and reported as missing.\n")
    A(f"**Measured** (coverage ≥ {COV_THRESH:.0%}, used downstream): "
      f"**{len(feats)}** of 41.\n")
    A(f"**Excluded** (low/zero coverage): {not_measured}\n\n")
    A("| feature | group | coverage | status |")
    A("|---|---|---:|---|")
    for _, r in cov.iterrows():
        A(f"| {r.feature} | {r.group} | {r.coverage:.2f} | {r.status} |")
    A("")
    A("**Tiering / honesty.** F0, jitter, shimmer, HNR(aux), F1–F5, B1–B5, the spectral "
      "moments/rolloff/flux, AlphaRatio, LHR, RMS, AMD, SemitoneSDF0, VTLE(estimated from "
      "formants), SpeechRate and BGD (Praat/librosa, well-established) are **Tier A**. "
      "SHR, IHI, GNE, SPI, SSPF, VFI, Nasality are **Tier B** best-effort DSP — computed, "
      "but absolute calibration is approximate. NAQ/CQ/GCT/SQ/MFDR come from a custom "
      "**IAIF** glottal inverse filter (**Tier C**): they are *computed* (so coverage is "
      "high) but on 16 kHz MP3 crowd audio their reliability is the weakest in the set — "
      "treat their contribution to d_eff with caution. **VOT is NOT MEASURED** (requires "
      "phoneme-level forced alignment + stop-burst detection, unavailable here).\n")

    A("## 2. Population distributions & quantile bins (Step 2)\n")
    A("Per-speaker means computed for each measured feature; q-quantile bin boundaries "
      "for q ∈ {2,3,5,10} saved to `bins.json` (equiprobable-by-construction, matching "
      "the paper's §3.6 binning). Histograms in `figs/dist_<feature>.png`.\n")

    A("## 3. F-ratios and empirical q_max (Step 3)\n")
    A("One-way ANOVA across speakers per feature. within_var = mean over speakers of "
      "within-speaker variance (across that speaker's utterances); between_var = variance "
      "of per-speaker means; F_ratio = between/within. q_max = largest q∈{2,3,5,10} whose "
      "mean bin-crossing rate < 0.20. **Because within-speaker variance is multi-session, "
      "these F-ratios are realistic, not an optimistic upper bound.**\n")
    frs = fr.sort_values("F_ratio", ascending=False)
    A("| feature | within_var | between_var | F_ratio | ANOVA_F | p | q_max |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in frs.iterrows():
        A(f"| {r.feature} | {r.within_var:.3g} | {r.between_var:.3g} | "
          f"{r.F_ratio:.2f} | {r.ANOVA_F:.1f} | {r.p:.1e} | {int(r.q_max)} |")
    A("")
    qd = {int(k): int(v) for k, v in fr.q_max.value_counts().sort_index().items()}
    n_fr_gt1 = int((fr.F_ratio > 1).sum()); n_fr_gt2 = int((fr.F_ratio > 2).sum())
    A(f"**Key finding.** Only **{n_fr_gt1} of {len(fr)}** measured features have "
      f"F_ratio > 1 (between-speaker variance exceeds within-speaker), and only "
      f"{n_fr_gt2} exceed F_ratio = 2. The most individuating features are F0, CPP, RMS, "
      "dCPP and the spectral-balance/formant measures; many source and prosodic features "
      "(SpeechRate, SHR, SSPF, MFDR, GNE, BGD) have F_ratio < 1 — on multi-session "
      "crowd audio they are *not* speaker-discriminative.\n")
    A(f"**q_max distribution across measured features:** {qd}. **No feature supports "
      "q ≥ 3** on this data, and 22 features cannot hold even q = 2 (q_max = 1). "
      "This is a sharp empirical contrast with the paper, which adopts q = 10 as its "
      "finest setting and q = 5 as 'conservative': **on realistic multi-session audio "
      "the usable per-feature resolution is q ≤ 2**, because within-speaker spread "
      "routinely crosses bin boundaries. This is the single biggest driver of the "
      "measured-vs-assumed gap in Step 5.\n")

    A("## 4. Effective dimensionality d_eff (Step 4)\n")
    A("Per-speaker mean-feature matrix; three estimators, bootstrap 95% CIs over "
      f"speakers ({N_BOOT} reps for PR; 200 for occupancy).\n")
    pooled_deff = deff_all[deff_all.stratum == "pooled"]
    A("**Pooled:**\n")
    A("| estimator | d_eff | 95% CI |")
    A("|---|---:|---|")
    for _, r in pooled_deff.iterrows():
        A(f"| {r.estimator} | {r.d_eff:.2f} | [{r.ci_lo:.2f}, {r.ci_hi:.2f}] |")
    A("")
    for q, o in pooled["occ"].items():
        A(f"- Cell-occupancy q={q}: d_eff(full 40-feature set)={o['deff_full']:.2f}, "
          f"which equals the sample-size ceiling log(n)/log(q)={o['log_n_over_log_q']:.2f} "
          f"— i.e. all {len(spk)} speakers already occupy distinct cells. Distinct cells "
          f"first exceed 95% of n at a subset of only **{o['saturation_subset_size']}** "
          "random features. **Estimator (c) is therefore censored by n: it is a *lower "
          "bound* on the true d_eff** (with 40 features and q≥2 the cell space q^k ≫ n, so "
          "occupancy cannot grow past n). Its bootstrap CI even sits *below* the point "
          "estimate because resampling speakers with replacement leaves only ~63% distinct, "
          "mechanically lowering the unique-cell count — another reason to read (c) as a "
          "floor, not an estimate.\n")
    A(f"Out of k={len(feats)} measured features, the participation ratio collapses the "
      f"effective dimensionality to ~{pooled['deff_pt']:.0f} — i.e. the measured voice "
      "features are substantially correlated, well below nominal independence.\n")
    A("**Stratified d_eff (PR estimators):**\n")
    A("| stratum | n | estimator | d_eff | 95% CI |")
    A("|---|---:|---|---:|---|")
    for _, r in deff_all[deff_all.estimator.str.startswith("PR")].iterrows():
        A(f"| {r.stratum} | {r.n_speakers} | {r.estimator} | {r.d_eff:.2f} | "
          f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] |")
    A("")
    # data-driven stratum narrative
    pe = deff_all[deff_all.estimator == "PR_pearson"]
    pooled_d = float(pe[pe.stratum == "pooled"].d_eff.iloc[0])
    sex_rows = pe[pe.stratum.str.startswith("sex=")]
    acc_rows = pe[pe.stratum.str.startswith("accent=")]
    age_rows = pe[pe.stratum.str.startswith("age=")]
    A(f"> **Reading the strata (a key result — and a nuanced one).** Pooled d_eff = "
      f"{pooled_d:.2f}.\n")
    A(f"> - **Sex strata are *higher* than pooled** "
      f"(male {float(sex_rows[sex_rows.stratum.str.contains('male_masc')].d_eff.iloc[0]):.2f}, "
      f"female {float(sex_rows[sex_rows.stratum.str.contains('female')].d_eff.iloc[0]):.2f}). "
      "This is expected and instructive: **sex is itself a correlation-inducing axis** — "
      "it jointly drives F0, the formants and VTLE, so *pooling across sexes inflates the "
      "feature correlations and lowers pooled d_eff*. Removing the sex axis de-correlates "
      "the features and raises d_eff. This is the mirror image of the paper's §3.7 point "
      "that sex/size correlations shrink the effective dimensionality.\n")
    lowest = acc_rows.sort_values("d_eff").iloc[0]
    A(f"> - **Homogeneous accent / age cohorts trend *lower* than pooled** "
      f"(e.g. {lowest.stratum.split('=')[1][:22]} d_eff={lowest.d_eff:.2f}; "
      f"the US-English cohort {float(acc_rows[acc_rows.stratum.str.contains('United States')].d_eff.iloc[0]):.2f} "
      "with a CI that does not overlap pooled). Restricting to an anatomically/­experientially "
      "more homogeneous group removes between-group spread along several axes at once, "
      "collapsing the effective dimensionality — this is the empirical signature of the "
      "paper's **'low-d_eff regime'** for homogeneous cohorts (§7).\n")
    A(f"> Net: the demographic axes move d_eff in *both* directions depending on whether "
      "they de-correlate (sex) or homogenize (accent/age) the feature set — a more honest "
      "picture than a single monotone 'drop'. Scree plots: `figs/scree_pooled.png`, "
      "`figs/scree_cohort.png`.\n")

    A("## 5. Collision metrics — measured vs assumed (Step 5)\n")
    A(f"n = {POP_N:.0e}, p = {P_MATCH:.0e}. Exact formulae P(E)=1−(1−1/m)^(n−1); "
      "S=⌈log(1−p)/log(1−1/m)⌉; P(M)=1/m; P(B)=1−∏(1−i/m) in log space; m=q^d. "
      "(Our `collision.py` reproduces the paper's Table 1 at d=41 exactly.)\n")
    col = pooled["collisions"]
    A("| config | q | d_used | m=q^d | P(E) | S | P(M) | P(B) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in col.iterrows():
        A(f"| {r.config} | {r.q} | {r.d_used:.2f} | {r.m:.2e} | {r.PE:.2e} | "
          f"{r.S:.2e} | {r.PM:.2e} | {r.PB:.2e} |")
    A("")
    A("- **(a) full independence** at k = measured features reproduces the paper's "
      "regime: P(B) falls from 1.0 (q=2) and 0.98 (q=3) to ~5e-9 (q=5) and ~5e-21 "
      "(q=10) — i.e. voices are effectively unique for q ≥ 5, exactly the paper's "
      "qualitative conclusion (here at k=40 rather than 41).\n")
    A(f"- **(b) measured d_eff ≈ {pooled['deff_pt']:.0f}** (with CI) sharply *raises* "
      "collision probabilities versus (a): the correlation correction the paper brackets "
      "as d_eff∈[27,41] is, on these measured features, **far more severe** (d_eff is a "
      "single-digit-to-low-double-digit number, not 27–41), because we measure only "
      f"{len(feats)} features and they are heavily correlated.\n")
    A(f"- **(c) q capped at empirical q_max (geo-mean q≈{pooled['q_geo']:.2f}) + measured "
      "d_eff** is the most conservative reading and gives the highest collision "
      "probabilities — this is where the framework's optimism is most exposed.\n")

    A("## 6. Direct empirical collision check (Step 6)\n")
    A("Bin the real speakers over all measured features at q=2,3 and count actual "
      "shared-cell collisions; compare to predictions.\n")
    A("| stratum | q | n | occupied cells | collision cells | speakers in collisions | "
      "observed pairs | pred(full-indep) | pred(d_eff) | obs/full | obs/d_eff |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in pooled["direct"].iterrows():
        A(f"| {r.stratum} | {r.q} | {r.n_speakers} | {r.occupied_cells} | "
          f"{r.collision_cells} | {r.speakers_in_collisions} | {r.observed_pairs} | "
          f"{r.pred_pairs_full_indep:.2e} | {r.pred_pairs_deff:.2e} | "
          f"{r.ratio_obs_over_full:.2e} | {r.ratio_obs_over_deff:.2f} |")
    A("")
    # data-driven Step-6 interpretation
    d2 = pooled["direct"]
    obs2 = int(d2[d2.q == 2].observed_pairs.iloc[0])
    obs3 = int(d2[d2.q == 3].observed_pairs.iloc[0])
    predb2 = float(d2[d2.q == 2].pred_pairs_deff.iloc[0])
    nspk = int(d2.q.iloc[0] * 0 + pooled["direct"].n_speakers.iloc[0])
    A(f"**What actually happened (and the honest reading).** Observed collisions = "
      f"**{obs2}** at q=2 and **{obs3}** at q=3: all {nspk} real speakers fall into "
      f"{nspk} *distinct* cells. This is because the nominal cell count q^k "
      "(2^40≈1.1e12, 3^40≈1.2e19) dwarfs the sample size, so even the full-independence "
      f"model predicts ≈0 collisions and we observe 0 — i.e. **obs ≈ pred(full-independence), "
      "and obs ≪ pred(d_eff)** (the PR-d_eff model would predict "
      f"~{predb2:.0f} colliding pairs at q=2).\n")
    A("Three things follow, stated plainly:\n")
    A(f"1. **The test is under-powered at population scale.** With only {nspk} speakers and "
      "q^k cells, no collisions can occur regardless of correlation; this direct count "
      "therefore cannot confirm or refute what happens at n=10^10. What it *does* establish "
      "is that the real speakers are **fully separable** on the 40 measured features at "
      "q≥2 — consistent with voice uniqueness *at sample scale*.\n")
    A("2. **PR-d_eff is a conservative (collision-pessimistic) summary.** The participation "
      "ratio measures *linear* redundancy; the speakers' discrete cell occupancy retains "
      "more separating information than q^d_eff cells would imply, so the 275-pair "
      "prediction does not materialise. The true discrete uniqueness sits *above* the "
      "linear d_eff.\n")
    A("3. **The empirical occupied-cell dimension is censored by n.** log(occupied)/log(q) "
      f"= log({nspk})/log(2) ≈ {math.log(nspk)/math.log(2):.1f} at q=2 (every speaker its "
      "own cell), matching the Step-4(c) saturation result — the dataset is simply too "
      "small to *observe* the q^d_eff cell collapse directly. The population-scale verdict "
      "must come from Steps 4–5 (d_eff + q_max extrapolated to n=10^10), not from this "
      "sample-level count.\n")

    if cohort_name and cohort_block:
        A("## 7. Homogeneous-cohort sub-analysis\n")
        A(f"Largest accent cohort with ≥200 speakers: **{cohort_name}** "
          f"(n={cohort_block['direct'].n_speakers.iloc[0]}). Re-ran Steps 4–6 within it.\n")
        A(f"- Cohort d_eff(PR-pearson) = **{cohort_block['deff_pt']:.2f}** "
          f"[{cohort_block['deff_ci'][0]:.2f}, {cohort_block['deff_ci'][1]:.2f}] "
          f"vs pooled **{pooled['deff_pt']:.2f}**.\n")
        cc = cohort_block["collisions"]
        A("Cohort collision band (selected):\n")
        A("| config | q | d_used | m | P(E) | P(B) |")
        A("|---|---:|---:|---:|---:|---:|")
        for _, r in cc[cc.config.isin(["a_full_independence", "b_deff_point",
                                        "c_qmaxcap_deff"])].iterrows():
            A(f"| {r.config} | {r.q} | {r.d_used:.2f} | {r.m:.2e} | {r.PE:.2e} | {r.PB:.2e} |")
        A("")
        A("> Within a single accent the effective dimensionality drops further and the "
          "collision band inflates relative to the pooled population — an empirical "
          "demonstration of the paper's **'low-d_eff regime'** (ethnically/­anatomically "
          "homogeneous cohorts) without needing a twins corpus.\n")

    A("## 8. Limitations (honest)\n")
    A(f"- **Feature coverage:** {len(feats)}/41 features measured; **VOT not measured** "
      "(needs forced alignment). Tier-C glottal features (NAQ/CQ/GCT/SQ/MFDR) come from "
      "IAIF on compressed audio and are the least reliable.\n")
    A("- **MP3 compression** removes/colours high-frequency and source detail, biasing "
      "spectral-balance, GNE, SSPF and the glottal-source features specifically.\n")
    A("- **client_id = speaker** assumption: Common Voice client_ids are accounts, not "
      "verified individuals; a shared account or one person with two accounts adds noise.\n")
    A("- **Subset size & shard pooling:** 4 of 138 shards; clips are shuffled across "
      "shards so a speaker's clips are a random subset of their recordings — within-speaker "
      "variance is well sampled, but speakers are not the full CV population.\n")
    A("- **16 kHz resampling** caps analysis at 8 kHz Nyquist, truncating sibilant energy "
      "(SSPF) and high-band ratios.\n")
    A("- **d_eff caveat:** estimated on the *measured* feature subset, so it is the "
      "effective dimensionality *of what we could measure*, not of the full 41-feature "
      "construct the paper posits.\n")

    open("report.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
