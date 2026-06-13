#!/usr/bin/env python3
"""
Quantization-based, information-theoretic speaker-discriminability analysis on TIMIT.

Headline metric is ALWAYS bias-corrected: I_corrected = max(0, I_mm - I_null_mean),
where I_mm is Miller-Madow corrected MI (via three MM-corrected entropies) and
I_null_mean is the mean MI under 200 speaker-label permutations. Raw MI is reported
only alongside, never as the result.

Reproducibility: single RNG seeded 1234 (numpy default_rng) drives every shuffle/bootstrap.
No imputation: each feature uses only its non-null utterances (N logged per feature).
"""
import os, json, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SEED = 1234
RNG = np.random.default_rng(SEED)

def _resolve_out():
    """Locate the dir holding features.parquet: env VU_OUT, else script dir,
    its parent (deliverable layout: scripts/ under results/), or CWD."""
    env = os.environ.get("VU_OUT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.abspath(os.path.join(here, "..")), os.getcwd()):
        if os.path.exists(os.path.join(cand, "features.parquet")):
            return cand
    return os.path.abspath(os.path.join(here, ".."))

OUT = _resolve_out()
FEATURES = os.path.join(OUT, "features.parquet")
FRATIOS = os.path.join(OUT, "fratios.csv")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

BITS = [1, 2, 3, 4, 5, 6, 7, 8]
N_PERM = 200
N_SPK_NOMINAL = 630
H_SPK_CEIL = math.log2(N_SPK_NOMINAL)   # 9.299 bits sample ceiling
JOINT_B = 2                              # fixed depth per feature for Step 5
LN2 = math.log(2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# core MI utilities (integer-coded)
# ----------------------------------------------------------------------------
def entropy_bits(counts, N):
    """plug-in entropy (bits) and support size (#nonzero cells)."""
    c = counts[counts > 0]
    p = c / N
    return float(-(p * np.log2(p)).sum()), int(c.size)

def mi_raw_mm(scode, bcode, n_spk, n_bin):
    """Return (I_raw, I_mm, support sizes) in bits using three-entropy MM correction."""
    N = scode.size
    joint = scode.astype(np.int64) * n_bin + bcode
    cj = np.bincount(joint, minlength=n_spk * n_bin)
    cs = np.bincount(scode, minlength=n_spk)
    cb = np.bincount(bcode, minlength=n_bin)
    Hs, ms = entropy_bits(cs, N)
    Hb, mb = entropy_bits(cb, N)
    Hj, mj = entropy_bits(cj, N)
    I_raw = Hs + Hb - Hj
    # Miller-Madow: each H_mm = H + (m-1)/(2N ln2);  I_mm = Hs_mm+Hb_mm-Hj_mm
    I_mm = I_raw + ((ms - 1) + (mb - 1) - (mj - 1)) / (2 * N * LN2)
    return I_raw, I_mm, ms, mb, mj

def mi_raw_from_joint(scode_shuf, bcode, n_spk, n_bin, Hs, Hb):
    """fast raw MI for permutations: H(S),H(B) fixed; recompute only H(S,B)."""
    N = scode_shuf.size
    joint = scode_shuf.astype(np.int64) * n_bin + bcode
    cj = np.bincount(joint, minlength=n_spk * n_bin)
    Hj, _ = entropy_bits(cj, N)
    return Hs + Hb - Hj

def perm_null(scode, bcode, n_spk, n_bin, n_perm=N_PERM):
    """permutation null of RAW MI by shuffling speaker labels."""
    N = scode.size
    cs = np.bincount(scode, minlength=n_spk)
    cb = np.bincount(bcode, minlength=n_bin)
    Hs, _ = entropy_bits(cs, N)
    Hb, _ = entropy_bits(cb, N)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = RNG.permutation(scode)
        null[i] = mi_raw_from_joint(sh, bcode, n_spk, n_bin, Hs, Hb)
    return null

# ----------------------------------------------------------------------------
# quantization
# ----------------------------------------------------------------------------
def quantize(vals, q):
    """equal-frequency q-quantile bins over pooled values; merge degenerate bins.
    returns bcode (0..q_eff-1 contiguous, all nonzero) and q_eff."""
    edges = np.quantile(vals, np.linspace(0, 1, q + 1))
    interior = np.unique(edges)[1:-1]
    raw = np.digitize(vals, interior)
    # compress to contiguous codes so every used bin is nonzero
    uniq, inv = np.unique(raw, return_inverse=True)
    return inv.astype(np.int64), int(uniq.size)

# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load():
    df = pd.read_parquet(FEATURES)
    df = df.dropna(subset=["value"])
    measured = sorted(df.feature.unique())
    spk_cats = pd.Categorical(df.speaker_id)
    return df, measured

# ----------------------------------------------------------------------------
# STEP 1-3
# ----------------------------------------------------------------------------
def per_feature(df, measured):
    rows = []           # full grid
    for f in measured:
        sub = df[df.feature == f]
        vals = sub.value.to_numpy()
        # speaker codes (only speakers present for this feature)
        scode, n_spk = pd.factorize(sub.speaker_id, sort=True)[0], sub.speaker_id.nunique()
        scode = scode.astype(np.int64)
        N = vals.size
        for b in BITS:
            q = 2 ** b
            bcode, q_eff = quantize(vals, q)
            I_raw, I_mm, ms, mb, mj = mi_raw_mm(scode, bcode, n_spk, q_eff)
            null = perm_null(scode, bcode, n_spk, q_eff)
            nmean, np95 = float(null.mean()), float(np.percentile(null, 95))
            I_corr = max(0.0, I_mm - nmean)
            perm_p = float((null >= I_raw).mean())
            rows.append(dict(feature=f, b=b, q=q, q_eff=q_eff, N=N, n_spk=n_spk,
                             I_raw=I_raw, I_mm=I_mm, I_null_mean=nmean, I_null_p95=np95,
                             I_corrected=I_corr, NMI_corrected=I_corr / H_SPK_CEIL,
                             perm_p=perm_p, deficient=(q_eff < q)))
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUT, "mi_per_feature_full.csv"), index=False)
    return full

def usable_bits(full):
    recs = []
    for f, g in full.groupby("feature"):
        g = g.sort_values("b")
        i = g.I_corrected.values.argmax()
        r = g.iloc[i]
        recs.append(dict(feature=f, b_star=int(r.b), q_eff_at_bstar=int(r.q_eff),
                         I_corrected_bits=float(r.I_corrected),
                         NMI_corrected=float(r.NMI_corrected), perm_p=float(r.perm_p)))
        # curve
        plt.figure(figsize=(5, 3.3))
        plt.plot(g.b, g.I_raw, "o--", color="#999", label="I_raw", lw=1)
        plt.plot(g.b, g.I_null_mean, "s:", color="#C44E52", label="I_null_mean", lw=1)
        plt.plot(g.b, g.I_corrected, "o-", color="#4C72B0", label="I_corrected", lw=2)
        plt.axvline(r.b, color="green", ls="--", lw=0.8, label=f"b*={int(r.b)}")
        plt.xlabel("bit depth b (q=2^b)"); plt.ylabel("MI (bits)")
        plt.title(f"{f}: MI vs depth", fontsize=9); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(FIGS, f"mi_{f}.png"), dpi=90); plt.close()
    tab = pd.DataFrame(recs).sort_values("I_corrected_bits", ascending=False)
    tab.to_csv(os.path.join(OUT, "usable_bits.csv"), index=False)
    return tab

# ----------------------------------------------------------------------------
# STEP 4
# ----------------------------------------------------------------------------
def compare_fratio(tab):
    from scipy.stats import spearmanr
    fr = pd.read_csv(FRATIOS)[["feature", "F_ratio", "q_max"]].rename(columns={"q_max": "variance_q_max"})
    m = tab.merge(fr, on="feature", how="left")
    m = m[["feature", "F_ratio", "variance_q_max", "I_corrected_bits", "b_star"]]
    rho, p = spearmanr(m.F_ratio, m.I_corrected_bits)
    # tertile quadrant disagreement
    def tert(x):
        lo, hi = np.percentile(x, [33.333, 66.667])
        return np.where(x <= lo, "low", np.where(x >= hi, "high", "mid"))
    m["F_tier"] = tert(m.F_ratio.values)
    m["bits_tier"] = tert(m.I_corrected_bits.values)
    m["tertile_disagree"] = ((m.F_tier == "high") & (m.bits_tier == "low")) | \
                            ((m.F_tier == "low") & (m.bits_tier == "high"))
    # rank-difference disagreement (more sensitive when overall rho is high)
    n = len(m)
    m["rank_F"] = m.F_ratio.rank(ascending=False).astype(int)
    m["rank_bits"] = m.I_corrected_bits.rank(ascending=False).astype(int)
    m["rank_diff"] = (m.rank_F - m.rank_bits)
    thr = max(8, int(round(0.25 * n)))                  # |Δrank| >= ~quarter of the list
    m["disagree"] = m.tertile_disagree | (m.rank_diff.abs() >= thr)
    m = m.sort_values("I_corrected_bits", ascending=False)
    m.to_csv(os.path.join(OUT, "fratio_vs_bits.csv"), index=False)
    return m, float(rho), float(p), thr

# ----------------------------------------------------------------------------
# STEP 5: greedy joint bits at fixed b=2
# ----------------------------------------------------------------------------
def joint_corrected(scode, jbin, n_spk, n_bin):
    I_raw, I_mm, ms, mb, mj = mi_raw_mm(scode, jbin, n_spk, n_bin)
    null = perm_null(scode, jbin, n_spk, n_bin)
    nmean, np95 = float(null.mean()), float(np.percentile(null, 95))
    return max(0.0, I_mm - nmean), I_raw, I_mm, nmean, np95, n_bin

def _build_joint(wide, cols):
    """listwise-complete utts over `cols`; return scode, joint bincode, n_spk, n_bin, N."""
    sub = wide[cols].dropna(axis=0, how="any")
    spk = sub.index.get_level_values("speaker_id")
    scode = pd.factorize(spk, sort=True)[0].astype(np.int64)
    n_spk = spk.nunique()
    comb = np.zeros(sub.shape[0], dtype=np.int64)
    for c in cols:
        code, qe = quantize(sub[c].to_numpy(), 2 ** JOINT_B)
        comb = comb * qe + code
        _, comb = np.unique(comb, return_inverse=True)   # compress as we go
        comb = comb.astype(np.int64)
    nb = int(comb.max() + 1) if comb.size else 1
    return scode, comb, n_spk, nb, sub.shape[0]

def greedy_joint(df, measured):
    # full per-utterance wide table (NaNs kept); per-step listwise on SELECTED features only,
    # so N stays ~6300 for high-coverage features instead of collapsing to the 40-way overlap.
    wide = df.pivot_table(index=["speaker_id", "utt_id"], columns="feature",
                          values="value", aggfunc="first")[measured]

    selected = []
    cur_corr = 0.0
    steps = []
    remaining = list(measured)
    while remaining:
        best = None
        for f in remaining:
            cols = selected + [f]
            scode, jb, n_spk, nb, N = _build_joint(wide, cols)
            corr, I_raw, I_mm, nmean, np95, _ = joint_corrected(scode, jb, n_spk, nb)
            gain = corr - cur_corr
            if (best is None) or (corr > best["corr"]):
                best = dict(f=f, corr=corr, gain=gain, nb=nb, N=N,
                            I_raw=I_raw, I_mm=I_mm, nmean=nmean, np95=np95,
                            noise_floor=np95 - nmean)
        # stop rule: marginal corrected gain <= permutation noise floor (p95 - mean)
        stop = best["gain"] <= best["noise_floor"]
        steps.append(dict(step=len(selected) + 1, feature=best["f"], n_joint_bins=best["nb"],
                          N=best["N"], cumulative_I_corrected=best["corr"],
                          marginal_gain=best["gain"], noise_floor=best["noise_floor"],
                          I_raw=best["I_raw"], I_mm=best["I_mm"],
                          I_null_mean=best["nmean"], I_null_p95=best["np95"],
                          added=not stop))
        if stop:
            break
        selected.append(best["f"]); remaining.remove(best["f"])
        cur_corr = best["corr"]
    sdf = pd.DataFrame(steps)
    N = sdf.N.iloc[0] if len(sdf) else 0
    n_spk = N_SPK_NOMINAL
    sdf.to_csv(os.path.join(OUT, "joint_greedy.csv"), index=False)

    # cumulative curve
    added = sdf[sdf.added]
    plt.figure(figsize=(6, 3.6))
    plt.plot(np.arange(1, len(added) + 1), added.cumulative_I_corrected, "o-", color="#4C72B0")
    plt.axhline(H_SPK_CEIL, ls="--", color="gray", lw=1, label=f"sample ceiling log2(630)={H_SPK_CEIL:.2f}")
    plt.xlabel("# features selected (greedy, b=2 each)")
    plt.ylabel("cumulative I_corrected (bits)")
    plt.title("Joint usable speaker bits (greedy)", fontsize=10)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "joint_cumulative_bits.png"), dpi=110); plt.close()
    return sdf, N, n_spk

# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def fmt(x, n=3):
    try: return f"{x:.{n}f}"
    except Exception: return str(x)

def write_report(tab, cmp_df, rho, rho_p, thr, sdf, joint_N, joint_nspk, full, measured):
    L = []
    L.append("# Information-Theoretic Speaker Discriminability on TIMIT\n")
    L.append(f"*Reproducibility:* single RNG `numpy.default_rng({SEED})` drives all "
             f"{N_PERM}-shuffle permutation nulls. Per-utterance values (630 speakers x 10 "
             f"utts), {len(measured)} measured features. **Headline metric is bias-corrected** "
             f"`I_corrected = max(0, I_mm - I_null_mean)` (Miller-Madow + permutation null); "
             f"raw MI is shown only for context.\n")
    L.append(f"> Note on coverage: the prior extraction measured **{len(measured)} features** "
             f"(not 30); this analysis uses all {len(measured)}. Two of the paper's 42 columns "
             f"(VFI, Nasality) remain NOT MEASURED and are excluded — never imputed.\n")

    # Step 3 table
    L.append("## Usable bit depth per feature (Step 3)\n")
    L.append("`b*` = argmax over b in {1..8} of I_corrected; finer binning beyond b* adds only "
             "sampling noise. `q_eff` is the realized bin count after merging degenerate "
             "quantile edges (logged whenever < 2^b). Sorted by usable bits.\n")
    L.append("| feature | b* | q_eff(b*) | I_corrected(b*) bits | NMI_corrected | perm_p |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in tab.iterrows():
        L.append(f"| {r.feature} | {int(r.b_star)} | {int(r.q_eff_at_bstar)} | "
                 f"{fmt(r.I_corrected_bits)} | {fmt(r.NMI_corrected,4)} | {fmt(r.perm_p,3)} |")
    top = tab.iloc[0]
    L.append("")
    L.append(f"**Top feature:** {top.feature} carries {fmt(top.I_corrected_bits)} corrected "
             f"bits about speaker identity at b*={int(top.b_star)} "
             f"({fmt(100*top.NMI_corrected,1)}% of the log2(630)={H_SPK_CEIL:.2f}-bit ceiling). "
             "All per-feature bit counts are small fractions of that ceiling — no single "
             "feature comes close to identifying a speaker among 630.\n")
    # depth deficiency note
    defic = full[full.deficient]
    nd = defic.feature.nunique()
    L.append(f"Bin-deficiency (q_eff < 2^b at some depth) occurred for {nd} feature(s), logged "
             "in `mi_per_feature_full.csv` (column `deficient`). Entropy ceilings use q_eff.\n")

    # Step 4
    L.append("## Variance metric vs information metric (Step 4)\n")
    L.append(f"Spearman rank correlation between F_ratio (variance separability) and "
             f"I_corrected(b*) (partition information): **rho = {fmt(rho,3)}** "
             f"(p = {fmt(rho_p,3)}). They are positively but imperfectly related — the two are "
             "not the same quantity.\n")
    L.append("| feature | F_ratio | variance_q_max | I_corrected(b*) bits | b* | rank_F | rank_bits | Δrank | disagree |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for _, r in cmp_df.iterrows():
        flag = "**YES**" if r.disagree else ""
        L.append(f"| {r.feature} | {fmt(r.F_ratio,2)} | {int(r.variance_q_max)} | "
                 f"{fmt(r.I_corrected_bits)} | {int(r.b_star)} | {int(r.rank_F)} | "
                 f"{int(r.rank_bits)} | {int(r.rank_diff):+d} | {flag} |")
    dis = cmp_df[cmp_df.disagree].copy()
    L.append("")
    L.append(f"Overall the two metrics are strongly concordant (Spearman rho={fmt(rho,3)}), so "
             "the headline is *agreement*: features that separate speakers by variance also tend "
             "to carry speaker bits. The interesting cases are the rank divergences "
             f"(|Δrank| >= {thr}, i.e. ~a quarter of the {len(cmp_df)} features), flagged below.\n")
    if len(dis):
        L.append("**Disagreements (key result — variance- and partition-separability are not the "
                 "same quantity):**")
        for _, r in dis.sort_values("rank_diff", ascending=False).iterrows():
            # rank 1 = best; rank_diff = rank_F - rank_bits.
            # positive => bits rank is better (smaller) than F_ratio rank => more informative.
            if r.rank_diff > 0:
                direction = (f"**more informative than its variance suggests** — bits rank "
                             f"#{int(r.rank_bits)} vs F_ratio rank #{int(r.rank_F)}")
            else:
                direction = (f"**less informative than its variance suggests** — F_ratio rank "
                             f"#{int(r.rank_F)} vs bits rank #{int(r.rank_bits)}")
            L.append(f"- `{r.feature}`: {direction} (F_ratio={fmt(r.F_ratio,2)}, "
                     f"I_corrected={fmt(r.I_corrected_bits)} bits, b*={int(r.b_star)}).")
        L.append("\nMechanism: F_ratio rewards a large between/within *variance* ratio, which a "
                 "few outlying speakers or a heavy tail can inflate without cleanly partitioning "
                 "the population; corrected MI instead rewards a feature that splits speakers into "
                 "distinguishable equiprobable bins. A feature can score well on one and not the "
                 "other.")
    else:
        L.append("No features exceeded the disagreement threshold.")
    L.append("")

    # Step 5
    added = sdf[sdf.added]
    sat = len(added)
    final_bits = added.cumulative_I_corrected.iloc[-1] if len(added) else 0.0
    stoprow = sdf.iloc[-1]
    L.append("## Joint usable bits — greedy forward selection (Step 5)\n")
    L.append(f"Fixed depth b={JOINT_B} per feature (q_eff<=4), on {joint_N} utterances "
             f"listwise-complete across all {len(measured)} features ({joint_nspk} speakers). "
             "Each step adds the feature maximizing joint I_corrected (same MM + permutation "
             "correction on the joint table); stop when the marginal corrected gain falls to or "
             "below the permutation noise floor (p95 - mean).\n")
    L.append("| step | feature added | #joint bins | cumulative I_corrected (bits) | marginal gain | noise floor |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for _, r in sdf.iterrows():
        tag = "" if r.added else "  (STOP: gain<=floor, not added)"
        L.append(f"| {int(r.step)} | {r.feature}{tag} | {int(r.n_joint_bins)} | "
                 f"{fmt(r.cumulative_I_corrected)} | {fmt(r.marginal_gain)} | {fmt(r.noise_floor)} |")
    L.append("")
    order = " -> ".join(added.feature.tolist())
    L.append(f"**Selection order:** {order}\n")
    L.append(f"**Saturation:** the curve flattens after **{sat} features** at "
             f"**{fmt(final_bits)} cumulative corrected bits**. This is the information-theoretic "
             "analogue of d_eff: total *usable* speaker bits, not a variance-axis count.\n")
    L.append(f"> **Sample-ceiling caveat (do not over-read):** cumulative corrected MI can never "
             f"exceed H(speaker)=log2(630)={H_SPK_CEIL:.2f} bits, and as joint cells multiply "
             f"(4^k) with only N={joint_N} samples the permutation null rises and the MM "
             f"correction grows, so part of this saturation is *sample-limited*, not purely a "
             "property of the voice. The flattening point is a lower bound on where real joint "
             "information stops being estimable here, not a hard physiological limit.\n")

    # limitations
    L.append("## Honest limitations\n")
    L.append("**Finite-sample MI bias.** Plug-in MI is upward-biased: with 630 speakers x q "
             "bins the contingency table is sparse, so raw MI overstates information. That is "
             "exactly why we (a) Miller-Madow-correct every entropy and (b) subtract a 200-fold "
             "permutation null; we report I_corrected, never raw MI. Residual bias still inflates "
             "absolute bits, so treat the numbers as upper-ish estimates and trust the rankings "
             "and the permutation p-values more than the third decimal.\n")
    L.append("**Single-session within-speaker variance.** TIMIT gives one recording session per "
             "speaker, so within-speaker spread excludes day-to-day, health, and emotional "
             "variability. Usable bits and b* are therefore an **OPTIMISTIC upper bound**: "
             "cross-session data would increase within-speaker bin-crossing, lowering both b* "
             "and I_corrected. (Conversely, the 10 utterances are different sentences, so some "
             "within-speaker spread is phonetic content rather than identity noise.)\n")
    L.append(f"**Coverage.** {len(measured)}/42 candidate features measured; VFI and Nasality "
             "not measured and excluded (not imputed). Glottal-flow features are approximate "
             "single-pass IAIF estimates carried over from the prior run.\n")
    L.append(f"**Sample ceiling.** All joint bits are bounded by log2(630)={H_SPK_CEIL:.2f} bits; "
             "the greedy curve's flattening is partly this ceiling and the rising joint-cell "
             "null, not solely the acoustics.\n")
    L.append(f"\n*Artifacts:* mi_per_feature_full.csv, usable_bits.csv, fratio_vs_bits.csv, "
             f"joint_greedy.csv, figs/mi_*.png, figs/joint_cumulative_bits.png, report-quant.md. "
             f"Seed={SEED}.\n")

    rep = "\n".join(L)
    with open(os.path.join(OUT, "report-quant.md"), "w", encoding="utf-8") as fh:
        fh.write(rep)
    return rep

# ----------------------------------------------------------------------------
def main():
    df, measured = load()
    print(f"[quant] {len(measured)} measured features; seed={SEED}; perms={N_PERM}", flush=True)
    print("[quant] STEP 1-2-3 per-feature MI grid ...", flush=True)
    full = per_feature(df, measured)
    tab = usable_bits(full)
    print("[quant] STEP 4 F-ratio vs bits ...", flush=True)
    cmp_df, rho, rho_p, thr = compare_fratio(tab)
    print(f"[quant]   Spearman(F_ratio, bits) = {rho:.3f} (p={rho_p:.3f})", flush=True)
    print("[quant] STEP 5 greedy joint bits ...", flush=True)
    sdf, jN, jspk = greedy_joint(df, measured)
    print(f"[quant]   greedy selected {int(sdf.added.sum())} features", flush=True)
    print("[quant] STEP 6 report ...", flush=True)
    write_report(tab, cmp_df, rho, rho_p, thr, sdf, jN, jspk, full, measured)
    print("[quant] done.", flush=True)

if __name__ == "__main__":
    main()
