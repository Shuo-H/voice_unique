"""
jb_run.py -- run the full JOINT usable-speaker-information lower-bound analysis
(Steps 1-7) on Common Voice and TIMIT and write all artifacts.

Headline numbers are LOWER BOUNDS (classifier- and sample-dependent): a stronger
classifier or larger corpus can only raise them. The binned plug-in MI curve
(Step 4) is a CENSORED sanity check only.

Seed 1234 everywhere. One command:  python jointbits_experiment/jb_run.py
Outputs (in jointbits_experiment/): CSVs, figs/, results.json (consumed by jb_report.py).
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jb_core as jb

CV_PARQUET = os.path.join(os.path.dirname(HERE), "mi_experiment", "features.parquet")
TIMIT_PARQUET = os.path.join(HERE, "timit_features.parquet")
SEED = jb.SEED
np.random.seed(SEED)

CLF_LABEL = {"logreg": "A: multinomial logreg (L2)",
             "mlp": "B: small MLP",
             "lda": "C: shrinkage-LDA (Ledoit-Wolf)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- one corpus ---
def prepare(parquet, label, balance=True):
    """load -> coverage_drop -> balance(10/spk) -> folds -> X,y. Returns a bundle
    plus the coverage / sensitivity bookkeeping."""
    wide, feats = jb.load_wide(parquet)
    cd = jb.coverage_drop(wide, feats)
    # sensitivity: TIMIT-style keep-all-features listwise-delete utts
    lw_all, n_lw_keep, n_lw_drop = jb.listwise_all_features(wide, feats)
    bal, spk = jb.balance_speakers(cd["wide"])
    fold = jb.assign_folds(bal)
    X, y = jb.design_matrix(bal, cd["kept_features"])
    bundle = dict(label=label, wide=wide, all_features=feats, cd=cd,
                  bal=bal, spk=spk, fold=fold, X=X, y=y,
                  kept_features=cd["kept_features"], S=len(spk), N=len(bal),
                  lw_all_keep=n_lw_keep, lw_all_drop=n_lw_drop)
    return bundle


def step1_classifiers(bundle, corpus_tag):
    """Run all three classifiers; return summaries + raw cv results (for LDA/best
    calibration and greedy)."""
    log(f"  Step1 classifiers on {corpus_tag} (S={bundle['S']}, N={bundle['N']})")
    res, summ = {}, {}
    for name, fac in jb.CLASSIFIERS.items():
        t0 = time.time()
        r = jb.cv_evaluate(bundle["X"], bundle["y"], bundle["fold"],
                           bundle["S"], fac, SEED)
        s = jb.summarize(r, bundle["S"], n_boot=1000, seed=SEED)
        res[name] = r
        summ[name] = s
        log(f"    {name:7s} acc={s['top1_acc']:.4f} "
            f"logloss={s['logloss_bits']:.3f}b/{s['logloss_nats']:.3f}nats "
            f"Fano={s['fano_lower_bits']:.3f} xent={s['xent_lower_bits']:.3f} "
            f"({time.time()-t0:.0f}s)")
    # strongest by xent lower bound
    strongest = max(summ, key=lambda k: summ[k]["xent_lower_bits"])
    inversion = summ["mlp"]["top1_acc"] < summ["logreg"]["top1_acc"]
    return res, summ, strongest, inversion


def main():
    t_start = time.time()
    OUT = {}
    rng = np.random.default_rng(SEED)

    # ============================ CV (full) ==============================
    log("=== Common Voice: prepare ===")
    cv = prepare(CV_PARQUET, "Common Voice (multi-session, mp3@16k)")
    log(f"CV: features kept={cv['cd']['n_features_kept']} "
        f"dropped={cv['cd']['dropped_features']} | "
        f"utts kept={cv['cd']['n_utts_kept']} dropped={cv['cd']['n_utts_dropped']} | "
        f"S_full={cv['S']} N={cv['N']} ceiling=log2(S)={np.log2(cv['S']):.3f} bits")

    res_cv, summ_cv, strong_cv, inv_cv = step1_classifiers(cv, "CV full")

    # sensitivity: drop-features-keep-utts (ours) vs keep-all-listwise (TIMIT-style)
    # both yield identical feature set here when coverage~100%, but quantify acc.
    log("  Sensitivity: keep-all-features + listwise-delete utts (TIMIT-style)")
    wide_lw = cv["wide"].dropna(subset=cv["all_features"]).reset_index(drop=True)
    bal_lw, spk_lw = jb.balance_speakers(wide_lw)
    fold_lw = jb.assign_folds(bal_lw)
    X_lw, y_lw = jb.design_matrix(bal_lw, cv["all_features"])
    r_lw = jb.cv_evaluate(X_lw, y_lw, fold_lw, len(spk_lw), jb.make_lda, SEED)
    s_lw = jb.summarize(r_lw, len(spk_lw), n_boot=200, seed=SEED)
    sens = dict(
        drop_feat_S=cv["S"], drop_feat_acc=summ_cv["lda"]["top1_acc"],
        drop_feat_nfeat=cv["cd"]["n_features_kept"], drop_feat_nutt=cv["N"],
        listwise_S=len(spk_lw), listwise_acc=s_lw["top1_acc"],
        listwise_nfeat=len(cv["all_features"]), listwise_nutt=len(bal_lw),
    )
    log(f"    drop-features: S={sens['drop_feat_S']} acc(LDA)={sens['drop_feat_acc']:.4f}"
        f" | listwise-all: S={sens['listwise_S']} acc(LDA)={sens['listwise_acc']:.4f}")
    OUT["sensitivity"] = sens

    # calibration on strongest (CV)
    cal_cv = jb.calibration(res_cv[strong_cv], n_bins=10)
    cal_cv["table"].to_csv(os.path.join(HERE, "calibration_cv.csv"), index=False)
    log(f"  Calibration (strongest={strong_cv}) ECE={cal_cv['ece']:.4f}")

    # Step3 greedy (LDA) on CV full  ----> figure + 95% point
    log("  Step3 greedy forward selection (LDA), CV full ...")
    t0 = time.time()
    greedy_cv = jb.greedy_forward_lda(cv["X"], cv["y"], cv["fold"], cv["S"],
                                      cv["kept_features"], SEED, verbose=True)
    log(f"    greedy done ({time.time()-t0:.0f}s) max I_lower={greedy_cv['max_bits']:.3f} "
        f"#feat@95%={greedy_cv['n_features_95']}")
    pd.DataFrame({"step": range(1, len(greedy_cv["order"]) + 1),
                  "feature": greedy_cv["order"],
                  "cum_I_lower_bits": greedy_cv["cum_bits"]}
                 ).to_csv(os.path.join(HERE, "cumulative_bits_cv.csv"), index=False)

    # Step4 binned greedy censored on CV full
    log("  Step4 binned greedy (censored sanity check), CV full ...")
    t0 = time.time()
    binned_cv = jb.binned_greedy_censored(cv["X"], cv["y"], cv["S"],
                                          cv["kept_features"], nperm=200, seed=SEED)
    binned_cv["table"].to_csv(os.path.join(HERE, "binned_greedy_censored_cv.csv"),
                              index=False)
    log(f"    binned done ({time.time()-t0:.0f}s) k*={binned_cv['k_star']} "
        f"(censor cells>{binned_cv['censor_thresh']:.0f})")

    # ============================ TIMIT (full=matched) ====================
    log("=== TIMIT: prepare ===")
    tm = prepare(TIMIT_PARQUET, "TIMIT (single-session, clean 16k)")
    log(f"TIMIT: features kept={tm['cd']['n_features_kept']} "
        f"dropped={tm['cd']['dropped_features']} | "
        f"S={tm['S']} N={tm['N']} ceiling={np.log2(tm['S']):.3f} bits")
    res_tm, summ_tm, strong_tm, inv_tm = step1_classifiers(tm, "TIMIT")

    # common feature set (intersection of kept features in both corpora)
    common = sorted(set(cv["kept_features"]) & set(tm["kept_features"]))
    log(f"  common feature set (CV ∩ TIMIT): {len(common)} features")

    # TIMIT restricted to common features (for the matched row)
    Xtm_c, ytm = jb.design_matrix(tm["bal"], common)
    res_tm_c, summ_tm_c = {}, {}
    for name, fac in jb.CLASSIFIERS.items():
        r = jb.cv_evaluate(Xtm_c, ytm, tm["fold"], tm["S"], fac, SEED)
        res_tm_c[name] = r
        summ_tm_c[name] = jb.summarize(r, tm["S"], n_boot=1000, seed=SEED)
    strong_tm_c = max(summ_tm_c, key=lambda k: summ_tm_c[k]["xent_lower_bits"])
    greedy_tm = jb.greedy_forward_lda(Xtm_c, ytm, tm["fold"], tm["S"], common, SEED)
    log(f"  TIMIT(common) strongest={strong_tm_c} "
        f"acc(A/B/C)={summ_tm_c['logreg']['top1_acc']:.3f}/"
        f"{summ_tm_c['mlp']['top1_acc']:.3f}/{summ_tm_c['lda']['top1_acc']:.3f} "
        f"#feat@95%={greedy_tm['n_features_95']}")

    # ============================ CV matched to TIMIT S ===================
    S_match = tm["S"]
    log(f"=== CV matched: random subsample to S={S_match}, common features ===")
    cv_sub, spk_sub = jb.subsample_speakers(cv["bal"], S_match, seed=SEED)
    fold_sub = jb.assign_folds(cv_sub)
    Xsub, ysub = jb.design_matrix(cv_sub, common)
    res_sub, summ_sub = {}, {}
    for name, fac in jb.CLASSIFIERS.items():
        r = jb.cv_evaluate(Xsub, ysub, fold_sub, S_match, fac, SEED)
        res_sub[name] = r
        summ_sub[name] = jb.summarize(r, S_match, n_boot=1000, seed=SEED)
    strong_sub = max(summ_sub, key=lambda k: summ_sub[k]["xent_lower_bits"])
    greedy_sub = jb.greedy_forward_lda(Xsub, ysub, fold_sub, S_match, common, SEED)
    log(f"  CV(matched) strongest={strong_sub} "
        f"acc(A/B/C)={summ_sub['logreg']['top1_acc']:.3f}/"
        f"{summ_sub['mlp']['top1_acc']:.3f}/{summ_sub['lda']['top1_acc']:.3f} "
        f"#feat@95%={greedy_sub['n_features_95']}")

    # ============================ Step 7: US-English cohort ===============
    log("=== Step7: US-English cohort (CV) + matched-S control ===")
    cohort = None
    us_mask = cv["wide"]["accent"].astype(str).str.startswith("United States")
    n_us_spk = cv["wide"].loc[us_mask, "speaker_id"].nunique()
    log(f"  US-English speakers available (pre-balance): {n_us_spk}")
    if n_us_spk >= 300:
        us_wide = cv["wide"][us_mask].copy()
        us_cd = jb.coverage_drop(us_wide, cv["all_features"])
        us_bal, us_spk = jb.balance_speakers(us_cd["wide"])
        us_fold = jb.assign_folds(us_bal)
        Xus, yus = jb.design_matrix(us_bal, us_cd["kept_features"])
        res_us, summ_us = {}, {}
        for name, fac in jb.CLASSIFIERS.items():
            r = jb.cv_evaluate(Xus, yus, us_fold, len(us_spk), fac, SEED)
            summ_us[name] = jb.summarize(r, len(us_spk), n_boot=1000, seed=SEED)
        strong_us = max(summ_us, key=lambda k: summ_us[k]["xent_lower_bits"])
        # matched-S random control from the pooled CV (de-confound the ceiling)
        cv_ctrl, _ = jb.subsample_speakers(cv["bal"], len(us_spk), seed=SEED + 7)
        ctrl_fold = jb.assign_folds(cv_ctrl)
        Xc, yc = jb.design_matrix(cv_ctrl, cv["kept_features"])
        summ_ctrl = {}
        for name, fac in jb.CLASSIFIERS.items():
            r = jb.cv_evaluate(Xc, yc, ctrl_fold, len(us_spk), fac, SEED)
            summ_ctrl[name] = jb.summarize(r, len(us_spk), n_boot=1000, seed=SEED)
        strong_ctrl = max(summ_ctrl, key=lambda k: summ_ctrl[k]["xent_lower_bits"])
        cohort = dict(S=len(us_spk), strong=strong_us,
                      cohort=summ_us, control=summ_ctrl,
                      strong_ctrl=strong_ctrl)
        log(f"  US cohort S={len(us_spk)} strongest={strong_us} "
            f"xent={summ_us[strong_us]['xent_lower_bits']:.3f} | "
            f"matched random control xent={summ_ctrl[strong_ctrl]['xent_lower_bits']:.3f}")

    # ============================ figures =================================
    make_figures(greedy_cv, binned_cv, cv["S"],
                 summ_cv, summ_sub, summ_tm_c, S_match, tm["S"])

    # ============================ assemble + write ========================
    def pack(summ, strong, inv, S):
        return dict(S=int(S), ceiling_bits=float(np.log2(S)),
                    strongest=strong, capacity_inversion=bool(inv),
                    clf={k: serialize_summary(v) for k, v in summ.items()})

    OUT["seed"] = SEED
    OUT["clips_per_speaker"] = jb.CLIPS_PER_SPEAKER
    OUT["cv"] = dict(
        coverage=cv["cd"]["coverage"],
        kept_features=cv["kept_features"], dropped_features=cv["cd"]["dropped_features"],
        n_features_kept=cv["cd"]["n_features_kept"],
        n_utts_kept=cv["cd"]["n_utts_kept"], n_utts_dropped=cv["cd"]["n_utts_dropped"],
        **pack(summ_cv, strong_cv, inv_cv, cv["S"]),
        calibration_ece=cal_cv["ece"],
        greedy=dict(order=greedy_cv["order"], cum_bits=greedy_cv["cum_bits"],
                    max_bits=greedy_cv["max_bits"], n95=greedy_cv["n_features_95"]),
        binned_kstar=binned_cv["k_star"], binned_thresh=binned_cv["censor_thresh"],
    )
    OUT["timit"] = dict(
        coverage=tm["cd"]["coverage"], kept_features=tm["kept_features"],
        dropped_features=tm["cd"]["dropped_features"],
        **pack(summ_tm, strong_tm, inv_tm, tm["S"]),
    )
    OUT["common_features"] = common
    OUT["timit_matched"] = dict(**pack(summ_tm_c, strong_tm_c,
                                summ_tm_c["mlp"]["top1_acc"] < summ_tm_c["logreg"]["top1_acc"],
                                tm["S"]), greedy_n95=greedy_tm["n_features_95"],
                                greedy_max=greedy_tm["max_bits"])
    OUT["cv_matched"] = dict(**pack(summ_sub, strong_sub,
                             summ_sub["mlp"]["top1_acc"] < summ_sub["logreg"]["top1_acc"],
                             S_match), greedy_n95=greedy_sub["n_features_95"],
                             greedy_max=greedy_sub["max_bits"])
    if cohort:
        OUT["cohort"] = dict(
            S=cohort["S"], strong=cohort["strong"], strong_ctrl=cohort["strong_ctrl"],
            cohort={k: serialize_summary(v) for k, v in cohort["cohort"].items()},
            control={k: serialize_summary(v) for k, v in cohort["control"].items()},
        )

    # cross-corpus table CSV (Step6)
    write_crosscorpus_csv(OUT, common)
    # per-corpus classifier CSVs (incl. the spec-referenced TIMIT file)
    write_clf_csv(summ_cv, "jointbits_classifiers_cv.csv", cv["S"])
    write_clf_csv(summ_tm, "jointbits_classifiers_timit.csv", tm["S"])

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    log(f"=== DONE in {(time.time()-t_start)/60:.1f} min. results.json written. ===")


# ------------------------------------------------------------- helpers --------
def serialize_summary(s):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}


def write_clf_csv(summ, fname, S):
    rows = []
    for name, s in summ.items():
        rows.append(dict(
            classifier=name, label=CLF_LABEL[name], S=S, chance=1.0 / S,
            ceiling_bits=np.log2(S),
            top1_acc=s["top1_acc"], acc_ci_lo=s["acc_ci"][0], acc_ci_hi=s["acc_ci"][1],
            fold_acc_mean=s["fold_acc_mean"], fold_acc_std=s["fold_acc_std"],
            logloss_bits=s["logloss_bits"], logloss_nats=s["logloss_nats"],
            fold_logloss_bits_mean=s["fold_logloss_bits_mean"],
            fold_logloss_bits_std=s["fold_logloss_bits_std"],
            fano_lower_bits=s["fano_lower_bits"],
            fano_ci_lo=s["fano_lower_ci"][0], fano_ci_hi=s["fano_lower_ci"][1],
            xent_lower_bits=s["xent_lower_bits"],
            xent_ci_lo=s["xent_lower_ci"][0], xent_ci_hi=s["xent_lower_ci"][1],
        ))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, fname), index=False)


def write_crosscorpus_csv(OUT, common):
    def row(tag, session, d, n95):
        S = d["S"]
        best = d["strongest"]
        c = d["clf"]
        xent = c[best]["xent_lower_bits"]
        ceil = np.log2(S)
        limited = "sample-ceilinged" if xent >= 0.85 * ceil else "bound/classifier-limited"
        return dict(
            corpus=tag, session_type=session, S=S, log2S_ceiling=ceil,
            top1_A_logreg=c["logreg"]["top1_acc"],
            top1_B_mlp=c["mlp"]["top1_acc"],
            top1_C_lda=c["lda"]["top1_acc"],
            fano_lower_logreg=c["logreg"]["fano_lower_bits"],
            xent_lower_best=xent, xent_best_clf=best,
            n_features_95=n95, regime=limited)
    rows = [
        row("CV (full)", "multi-session+mp3", OUT["cv"], OUT["cv"]["greedy"]["n95"]),
        row("CV (matched to TIMIT)", "multi-session+mp3", OUT["cv_matched"],
            OUT["cv_matched"]["greedy_n95"]),
        row("TIMIT (matched)", "single-session clean", OUT["timit_matched"],
            OUT["timit_matched"]["greedy_n95"]),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "crosscorpus_table.csv"), index=False)


def make_figures(greedy_cv, binned_cv, S_cv, summ_cv, summ_sub, summ_tm_c,
                 S_match, S_tm):
    # ---- Step3: cumulative joint-bits curve (CV) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cum = greedy_cv["cum_bits"]
    xs = np.arange(1, len(cum) + 1)
    ax.plot(xs, cum, "-o", ms=4, color="#1f77b4", label="held-out xent I_lower (LDA)")
    ax.axhline(np.log2(S_cv), ls="--", color="gray",
               label=f"H(speaker)=log2(S)={np.log2(S_cv):.2f} bits (sample ceiling)")
    n95 = greedy_cv["n_features_95"]
    ax.axvline(n95, ls=":", color="red", label=f"95% of max at {n95} features")
    ax.set_xlabel("# features (greedy forward order)")
    ax.set_ylabel("Joint usable bits (lower bound)")
    ax.set_title("CV: cumulative usable joint speaker-bits (classifier-driven, LDA)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "joint_bits_curve_cv.png"), dpi=130)
    plt.close(fig)

    # ---- Step4: binned greedy censored (CV) ----
    t = binned_cv["table"]
    k_star = binned_cv["k_star"] or len(t)
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = t["step"].to_numpy()
    immi = t["I_mm"].to_numpy()
    icorr = t["I_corrected"].to_numpy()
    # solid up to k*, dashed beyond
    def split_plot(y, color, label):
        ax.plot(xs[:k_star], y[:k_star], "-", color=color, label=label)
        ax.plot(xs[k_star - 1:], y[k_star - 1:], "--", color=color)
    split_plot(immi, "#ff7f0e", "I_mm (Miller-Madow plug-in)")
    split_plot(icorr, "#2ca02c", "I_corrected (above perm null)")
    ax.axvline(k_star, ls=":", color="red",
               label=f"censor k*={k_star} (joint cells > N/5, unreliable)")
    ax.set_xlabel("# binary features (greedy)")
    ax.set_ylabel("Plug-in joint MI (bits)")
    ax.set_title("CV binned greedy MI — CENSORED sanity check (Step 3 supersedes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "binned_greedy_censored_cv.png"), dpi=130)
    plt.close(fig)

    # ---- Step6: cross-corpus bars (matched) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"CV matched\n(S={S_match})", f"TIMIT matched\n(S={S_tm})"]
    accs = [summ_sub[max(summ_sub, key=lambda k: summ_sub[k]['top1_acc'])]["top1_acc"],
            summ_tm_c[max(summ_tm_c, key=lambda k: summ_tm_c[k]['top1_acc'])]["top1_acc"]]
    # use best-accuracy clf per corpus for the bar
    accs = [max(summ_sub[c]["top1_acc"] for c in summ_sub),
            max(summ_tm_c[c]["top1_acc"] for c in summ_tm_c)]
    xents = [max(summ_sub[c]["xent_lower_bits"] for c in summ_sub),
             max(summ_tm_c[c]["xent_lower_bits"] for c in summ_tm_c)]
    axes[0].bar(labels, accs, color=["#1f77b4", "#d62728"])
    axes[0].set_ylabel("best top-1 accuracy")
    axes[0].set_title("Speaker-ID accuracy (matched S, common features)")
    for i, v in enumerate(accs):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    axes[1].bar(labels, xents, color=["#1f77b4", "#d62728"])
    axes[1].axhline(np.log2(S_match), ls="--", color="gray",
                    label=f"ceiling≈log2(S)={np.log2(S_match):.2f}")
    axes[1].set_ylabel("xent I_lower (bits)")
    axes[1].set_title("Usable joint bits (lower bound)")
    for i, v in enumerate(xents):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axes[1].legend(fontsize=8)
    fig.suptitle("Step 6: matched CV vs TIMIT — single- vs multi-session contrast")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "crosscorpus_matched.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
