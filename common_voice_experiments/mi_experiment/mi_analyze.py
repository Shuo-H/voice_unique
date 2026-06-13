"""
mi_analyze.py -- STEPS 2-6 of the information-theoretic voice-individuality
experiment.  Operates on mi_experiment/features.parquet (balanced 12 clips/spk).

  STEP 2  quantization grid (q = 2^b, b in 1..8), equal-frequency edges, q_eff -> bins.json
  STEP 3  per (feature,b): I_raw, I_mm (Miller-Madow), permutation null (200x, seed 1234),
          perm_p, I_corrected = max(0, I_mm - I_null_mean), NMI_corrected -> mi_by_feature_bit.csv
  STEP 4  b* = argmax_b I_corrected; figs/mi_<feature>.png; usable_bits.csv
  STEP 5  greedy forward selection at b=2 (q=4): cumulative I_corrected on the joint
          contingency table (MM + permutation null) -> cumulative_bits.csv, figs/cumulative_bits.png
  STEP 6  re-run 3-5 within sex strata and any accent group with >=300 speakers;
          stratified tables + overlay figure + comparison.

Seed 1234 everywhere (per-unit deterministic seeds -> parallel-invariant results).
"""
import os, sys, json, time, hashlib, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mi_core as MC
import mi_features as MF

SEED = 1234
NPERM = 200
BITS = [1, 2, 3, 4, 5, 6, 7, 8]
JOINT_B = 2                 # q = 2^2 = 4 bins/feature for the joint analysis
MAXJOINT = 20              # cap greedy depth (curve flattens well before this)
N_WORKERS = max(2, (os.cpu_count() or 4) - 2)
FIGS = os.path.join(HERE, "figs")
ART = os.path.join(HERE, "artifacts")


def shash(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")


# ============================================================= STEP 3 worker ==
def _step3_feature(payload):
    feat, vals, spk, S, nperm, seed0 = payload
    rows, binsrec = [], {}
    occ = int(np.unique(spk).size)                  # speakers actually present (NaN-drop aware)
    logS = np.log2(occ) if occ > 1 else 1.0
    for b in BITS:
        labels, q_eff, edges = MC.quantize(vals, 2 ** b)
        m = MC.mi_metrics(spk, labels, S, nperm=nperm, seed=seed0 + b)
        rows.append(dict(feature=feat, b=b, q_eff=q_eff,
                         I_raw=m["I_raw"], I_mm=m["I_mm"],
                         I_null_mean=m["I_null_mean"], I_null_p95=m["I_null_p95"],
                         perm_p=m["perm_p"], I_corrected=m["I_corrected"],
                         NMI_corrected=m["I_corrected"] / logS,
                         I_corrected_mmnull=m["I_corrected_mmnull"],
                         I_null_mm_mean=m["I_null_mm_mean"]))
        binsrec[b] = dict(q_nominal=2 ** b, q_eff=int(q_eff),
                          edges=[float(x) for x in edges])
    return feat, rows, binsrec


# ============================================================= STEP 5 worker ==
def _step5_candidate(payload):
    selected_dense, qeff_cand, cand_labels, cand_feat, spk, S, nperm, seed = payload
    new_raw = selected_dense.astype(np.int64) * np.int64(qeff_cand) + cand_labels.astype(np.int64)
    _, new_dense = np.unique(new_raw, return_inverse=True)
    m = MC.mi_metrics(spk, new_dense.astype(np.int64), S, nperm=nperm, seed=seed)
    return cand_feat, m


def greedy_cumulative(cc, spk_cc, S_cc, tag, ex, suffix=None, save_fig=False, outdir=None):
    """Greedy forward selection at b=2 (q=4) maximizing joint-bin I_corrected.
    cc: complete-case wide subset; spk_cc: dense speaker codes; ex: executor.
    Returns (cum_df, sat_row, stop_step, logS_cc)."""
    outdir = outdir or HERE
    feats = list(MF.MEASURED)
    logS_cc = float(np.log2(S_cc)) if S_cc > 1 else 1.0
    binmap = {}
    for f in feats:
        labels, q_eff, _ = MC.quantize(cc[f].to_numpy(), 2 ** JOINT_B)
        binmap[f] = (labels.astype(np.int64), int(q_eff))
    selected, remaining = [], list(feats)
    selected_dense = np.zeros(len(cc), dtype=np.int64)
    prev_ic, stop_step, curve = 0.0, None, []
    while remaining and len(selected) < MAXJOINT:
        payloads = [(selected_dense, binmap[f][1], binmap[f][0], f, spk_cc, S_cc, NPERM,
                     shash(f"{tag}|joint|{len(selected)}|{f}")) for f in remaining]
        results = list(ex.map(_step5_candidate, payloads))
        bf, bm = max(results, key=lambda x: x[1]["I_corrected"])
        gain = bm["I_corrected"] - prev_ic
        noise_band = bm["I_null_p95"] - bm["I_null_mean"]
        curve.append(dict(step=len(selected) + 1, feature=bf,
                          cum_I_corrected=bm["I_corrected"], marginal_gain=gain,
                          I_mm=bm["I_mm"], I_null_mean=bm["I_null_mean"],
                          I_null_p95=bm["I_null_p95"], perm_p=bm["perm_p"],
                          q_eff_joint=bm["q_eff"], cum_NMI=bm["I_corrected"] / logS_cc,
                          cum_I_corrected_mmnull=bm["I_corrected_mmnull"]))
        if stop_step is None and gain <= noise_band:
            stop_step = len(selected) + 1
        selected.append(bf); remaining.remove(bf)
        nr = selected_dense * np.int64(binmap[bf][1]) + binmap[bf][0]
        _, selected_dense = np.unique(nr, return_inverse=True)
        selected_dense = selected_dense.astype(np.int64)
        prev_ic = bm["I_corrected"]
        if stop_step is not None and len(selected) >= stop_step + 3:
            break
        if selected_dense.max() + 1 >= len(cc):
            break
    cum_df = pd.DataFrame(curve)
    peak_idx = int(cum_df["cum_I_corrected"].idxmax())
    sat = cum_df.iloc[peak_idx]
    if suffix is not None:
        cum_df.to_csv(os.path.join(outdir, f"cumulative_bits{suffix}.csv"), index=False)
    if save_fig:
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(cum_df["step"], cum_df["I_mm"], "^--", color="#999999", lw=1.3,
                 label="joint I_mm (uncorrected)")
        plt.plot(cum_df["step"], cum_df["I_null_mean"], "s:", color="#e07b39",
                 label="joint null mean (bias floor)")
        plt.plot(cum_df["step"], cum_df["cum_I_corrected"], "o-", color="#2c6fbb", lw=2.2,
                 label="cumulative I_corrected (headline)")
        plt.axhline(logS_cc, color="red", ls="--", label=f"log2(S)={logS_cc:.2f} ceiling")
        plt.axvline(int(sat["step"]), color="purple", ls="-", alpha=.5,
                    label=f"peak @ {int(sat['step'])} ({sat['cum_I_corrected']:.2f} bits)")
        if stop_step:
            plt.axvline(stop_step, color="green", ls="--", alpha=.6, label=f"stop @ {stop_step}")
        plt.xlabel("# features (greedy order, b=2/q=4)"); plt.ylabel("cumulative MI (bits)")
        plt.title(f"Cumulative usable speaker bits -- {tag} (S={S_cc}, N={len(cc)})")
        plt.legend(fontsize=8); plt.tight_layout()
        figname = "cumulative_bits.png" if tag == "pooled" else f"cumulative_bits_{tag}.png"
        plt.savefig(os.path.join(FIGS, figname), dpi=120); plt.close()
    return cum_df, sat, stop_step, logS_cc


# ================================================================= cohort ======
def run_cohort(wide, spk_str, tag, save_bins=False, make_feature_figs=False,
               outdir=HERE):
    """wide: DataFrame index=utt, columns=MEASURED (float, may contain NaN).
    spk_str: pandas Series utt -> speaker_id string (aligned with wide.index)."""
    t0 = time.time()
    feats = list(MF.MEASURED)
    # dense speaker codes
    spk_codes_all, uniq_spk = pd.factorize(spk_str.loc[wide.index], sort=True)
    S = len(uniq_spk)
    logS = float(np.log2(S))
    print(f"\n=== cohort '{tag}': S={S} speakers, N={len(wide)} utts, "
          f"log2(S)={logS:.3f} bits ===", flush=True)

    # ---------------- STEP 3 (parallel across features) ----------------
    payloads = []
    for f in feats:
        v = wide[f].to_numpy()
        ok = np.isfinite(v)
        payloads.append((f, v[ok], spk_codes_all[ok], S, NPERM, shash(tag + "|" + f)))
    rows_all, bins_all = [], {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for feat, rows, binsrec in ex.map(_step3_feature, payloads):
            rows_all.extend(rows)
            bins_all[feat] = binsrec
    mi_df = pd.DataFrame(rows_all).sort_values(["feature", "b"]).reset_index(drop=True)
    suffix = "" if tag == "pooled" else f"_{tag}"
    mi_df.to_csv(os.path.join(outdir, f"mi_by_feature_bit{suffix}.csv"), index=False)
    print(f"[step3] mi_by_feature_bit{suffix}.csv ({len(mi_df)} rows)", flush=True)

    if save_bins:
        json.dump(bins_all, open(os.path.join(outdir, "bins.json"), "w"), indent=1)
        print("[step2] bins.json", flush=True)

    # ---------------- STEP 4: b*, usable bits, per-feature figs ----------------
    usable = []
    for f in feats:
        sub = mi_df[mi_df.feature == f].sort_values("b")
        ic = sub["I_corrected"].to_numpy()
        bstar_i = int(np.argmax(ic))            # ties -> smallest b
        r = sub.iloc[bstar_i]
        usable.append(dict(feature=f, b_star=int(r["b"]), q_eff=int(r["q_eff"]),
                           I_corrected=float(r["I_corrected"]),
                           NMI_corrected=float(r["NMI_corrected"]),
                           perm_p=float(r["perm_p"])))
        if make_feature_figs:
            plt.figure(figsize=(5, 3.2))
            plt.plot(sub["b"], sub["I_raw"], "o--", color="#bbbbbb", label="I_raw (plug-in)")
            plt.plot(sub["b"], sub["I_null_mean"], "s:", color="#e07b39", label="I_null mean")
            plt.plot(sub["b"], sub["I_corrected"], "o-", color="#2c6fbb", lw=2, label="I_corrected")
            plt.axvline(int(r["b"]), color="green", ls="--", alpha=.6, label=f"b*={int(r['b'])}")
            plt.xlabel("bit depth b (q=2^b bins)"); plt.ylabel("MI (bits)")
            plt.title(f"{f}  (S={S})"); plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(os.path.join(FIGS, f"mi_{f}.png"), dpi=110); plt.close()
    usable_df = pd.DataFrame(usable).sort_values("I_corrected", ascending=False).reset_index(drop=True)
    usable_df.to_csv(os.path.join(outdir, f"usable_bits{suffix}.csv"), index=False)
    print(f"[step4] usable_bits{suffix}.csv  top: "
          f"{', '.join(usable_df.head(5)['feature'])}", flush=True)

    # ---------------- STEP 5: greedy joint at b=2 (q=4) ----------------
    cc = wide.dropna(subset=feats)
    spk_cc, _ = pd.factorize(spk_str.loc[cc.index], sort=True)
    S_cc = int(spk_cc.max() + 1)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        cum_df, sat, stop_step, logS_cc = greedy_cumulative(
            cc, spk_cc, S_cc, tag, ex, suffix=suffix, save_fig=True, outdir=outdir)
    print(f"[step5] cumulative_bits{suffix}.csv: peak @ {int(sat['step'])} features = "
          f"{sat['cum_I_corrected']:.3f} bits, stop_step={stop_step} "
          f"(ceiling log2(S)={logS_cc:.3f})", flush=True)

    print(f"[cohort '{tag}'] done in {time.time()-t0:.0f}s", flush=True)
    return dict(tag=tag, S=S, N=len(wide), logS=logS, S_cc=S_cc, N_cc=len(cc),
                logS_cc=logS_cc, usable=usable_df, cum=cum_df, stop_step=stop_step,
                sat_bits=float(sat["cum_I_corrected"]),
                sat_NMI=float(sat["cum_I_corrected"]) / logS_cc,
                sat_features=int(sat["step"]),
                mi=mi_df)


# ================================================================= main ========
def main():
    t0 = time.time()
    os.makedirs(FIGS, exist_ok=True); os.makedirs(ART, exist_ok=True)
    df = pd.read_parquet(os.path.join(HERE, "features.parquet"))
    # wide table + per-utt metadata
    wide = df.pivot(index="utt_id", columns="feature", values="value")[MF.MEASURED]
    meta = df.drop_duplicates("utt_id").set_index("utt_id")[["speaker_id", "sex", "accent", "age"]]
    wide = wide.loc[meta.index]
    spk_str = meta["speaker_id"]
    print(f"[load] wide {wide.shape}, {spk_str.nunique()} speakers", flush=True)

    results = {}
    # pooled (also writes bins.json + per-feature figures)
    results["pooled"] = run_cohort(wide, spk_str, "pooled",
                                   save_bins=True, make_feature_figs=True)

    # ---- STEP 6: strata ----
    cohorts = []
    sex_counts = meta.groupby("sex")["speaker_id"].nunique()
    for sx in ["male_masculine", "female_feminine"]:
        if sex_counts.get(sx, 0) >= 1:
            cohorts.append(("sex:" + sx.split("_")[0], meta["sex"] == sx))
    acc_counts = meta.groupby("accent")["speaker_id"].nunique()
    for acc, n in acc_counts.items():
        if isinstance(acc, str) and n >= 300:
            short = "US" if "United States" in acc else acc.split()[0]
            cohorts.append(("accent:" + short, meta["accent"] == acc))
    for tag, mask in cohorts:
        utts = meta.index[mask.values]
        results[tag] = run_cohort(wide.loc[utts], spk_str, tag)

    # ---- matched-S random-subsample CONTROLS ----
    # Each homogeneous cohort is compared to N_CTRL random speaker subsets of the
    # SAME size drawn from the full pool. This de-confounds acoustic homogeneity
    # from the lower S / lower log2(S) ceiling / higher relative null floor that a
    # smaller cohort has regardless of homogeneity. If the homogeneous cohort's
    # ceiling-normalized usable bits sit BELOW the size-matched random control, the
    # low-effective-dimensionality-among-similar-speakers prediction is supported.
    N_CTRL = 5
    all_spk = np.array(sorted(spk_str.unique()))
    rng = np.random.default_rng(SEED)
    control_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for tag in [t for t in results if t != "pooled"]:
            S_target = results[tag]["S_cc"]
            cb, cn = [], []
            for d in range(N_CTRL):
                draw = rng.choice(all_spk, size=S_target, replace=False)
                utts = meta.index[meta["speaker_id"].isin(draw).values]
                ccd = wide.loc[utts].dropna(subset=list(MF.MEASURED))
                spk_d, _ = pd.factorize(spk_str.loc[ccd.index], sort=True)
                S_d = int(spk_d.max() + 1)
                _, sat_d, _, logS_d = greedy_cumulative(
                    ccd, spk_d, S_d, f"ctrl:{tag}:{d}", ex, suffix=None, save_fig=False)
                cb.append(float(sat_d["cum_I_corrected"]))
                cn.append(float(sat_d["cum_I_corrected"]) / logS_d)
            control_rows.append(dict(
                cohort=tag, S=S_target,
                hom_sat_bits=results[tag]["sat_bits"], hom_sat_NMI=results[tag]["sat_NMI"],
                ctrl_sat_bits_mean=float(np.mean(cb)), ctrl_sat_bits_sd=float(np.std(cb)),
                ctrl_sat_NMI_mean=float(np.mean(cn)), ctrl_sat_NMI_sd=float(np.std(cn)),
                n_ctrl=N_CTRL,
                hom_below_ctrl_NMI=bool(results[tag]["sat_NMI"] < np.mean(cn))))
    pd.DataFrame(control_rows).to_csv(
        os.path.join(ART, "stratified_control_comparison.csv"), index=False)
    print("[step6] matched-S random-subsample controls written", flush=True)

    # ---- stratified comparison: per-feature usable bits & cumulative saturation ----
    comp = results["pooled"]["usable"][["feature", "b_star", "I_corrected", "NMI_corrected", "perm_p"]].copy()
    comp = comp.rename(columns={"b_star": "b*_pooled", "I_corrected": "Ic_pooled",
                                "NMI_corrected": "NMI_pooled", "perm_p": "p_pooled"})
    for tag in results:
        if tag == "pooled":
            continue
        u = results[tag]["usable"][["feature", "b_star", "I_corrected"]].rename(
            columns={"b_star": f"b*_{tag}", "I_corrected": f"Ic_{tag}"})
        comp = comp.merge(u, on="feature", how="left")
    comp = comp.sort_values("Ic_pooled", ascending=False)
    comp.to_csv(os.path.join(ART, "stratified_usable_comparison.csv"), index=False)

    sat_rows = [dict(cohort=t, S=results[t]["S_cc"], logS_ceiling=results[t]["logS_cc"],
                     saturation_features=results[t]["sat_features"],
                     saturation_bits=results[t]["sat_bits"],
                     saturation_NMI=results[t]["sat_NMI"],
                     stop_step=results[t]["stop_step"]) for t in results]
    sat_df = pd.DataFrame(sat_rows)
    sat_df.to_csv(os.path.join(ART, "stratified_saturation.csv"), index=False)
    print("\n[step6] stratified comparison tables written", flush=True)

    # overlay cumulative figure
    plt.figure(figsize=(6.5, 4.3))
    for t in results:
        c = results[t]["cum"]
        plt.plot(c["step"], c["cum_I_corrected"], "o-", lw=1.8,
                 label=f"{t} (S={results[t]['S_cc']}, ceil={results[t]['logS_cc']:.1f})")
    plt.xlabel("# features (greedy, b=2/q=4)"); plt.ylabel("cumulative I_corrected (bits)")
    plt.title("Cumulative usable speaker bits: pooled vs homogeneous cohorts")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cumulative_bits_overlay.png"), dpi=120); plt.close()

    # persist a compact machine-readable summary for the report stage
    summ = dict(
        pooled=dict(S=results["pooled"]["S"], N=results["pooled"]["N"],
                    logS=results["pooled"]["logS"],
                    sat_features=results["pooled"]["sat_features"],
                    sat_bits=results["pooled"]["sat_bits"],
                    stop_step=results["pooled"]["stop_step"]),
        cohorts={t: dict(S=results[t]["S_cc"], logS=results[t]["logS_cc"],
                         sat_features=results[t]["sat_features"],
                         sat_bits=results[t]["sat_bits"],
                         sat_NMI=results[t]["sat_NMI"]) for t in results},
        elapsed_s=round(time.time() - t0, 1))
    json.dump(summ, open(os.path.join(ART, "analysis_summary.json"), "w"),
              indent=2, default=str)
    print(f"\n[done] analysis in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
