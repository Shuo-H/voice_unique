"""
step6_classifier.py -- STEP 6: JOINT usable speaker bits, held-out classifier
LOWER BOUND.  Reuses the corpus-agnostic jb_core machinery on the v2 40-feature
features.parquet.

Pipeline: load_wide -> coverage_drop (>= 90%) -> listwise-complete -> balance
EXACTLY 10 clips/speaker (uniform prior) -> utterance-disjoint stratified 5-fold
CV, z-scored on train folds only -> three classifiers:
  A regularized multinomial logistic regression (L2)
  B small MLP (one hidden layer, nonlinear)
  C shrinkage-LDA (Ledoit-Wolf)
Report per classifier: top-1 acc + bootstrap CI + per-fold, mean log-loss (bits),
Fano lower bound + CI, cross-entropy lower bound + CI.  All bounds are FLOORS
below H(speaker)=log2(S).  Capacity-inversion check: MLP top-1 < logreg top-1.

Seed 1234 everywhere.
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "jointbits_experiment"))
import jb_core as jb
import features as F

SEED = jb.SEED
COVERAGE_THRESH = 0.90      # v2 brief: keep >= 90%-coverage features
CLF_LABEL = {"logreg": "A: multinomial logreg (L2)",
             "mlp": "B: small MLP",
             "lda": "C: shrinkage-LDA (Ledoit-Wolf)"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.chdir(HERE)
    t0 = time.time()
    wide, feats = jb.load_wide(os.path.join(HERE, "features.parquet"))
    # restrict to the canonical 40 that survived the pivot (drops all-NaN VOT and
    # excludes aux HNR); coverage_drop then removes any <90%-coverage feature (SSPF).
    feats = [f for f in F.FEATURES_40 if f in wide.columns]
    cd = jb.coverage_drop(wide, feats, thresh=COVERAGE_THRESH)
    bal, spk = jb.balance_speakers(cd["wide"])           # 10 clips/speaker
    fold = jb.assign_folds(bal)
    X, y = jb.design_matrix(bal, cd["kept_features"])
    S, N = len(spk), len(bal)
    ceiling = float(np.log2(S))
    log(f"features kept={cd['n_features_kept']} dropped={cd['dropped_features']}")
    log(f"S={S} speakers, N={N} clips (10/spk), ceiling log2(S)={ceiling:.3f} bits")

    summ = {}
    for name, fac in jb.CLASSIFIERS.items():
        tc = time.time()
        r = jb.cv_evaluate(X, y, fold, S, fac, SEED)
        s = jb.summarize(r, S, n_boot=1000, seed=SEED)
        summ[name] = s
        log(f"  {name:7s} acc={s['top1_acc']:.4f} "
            f"[{s['acc_ci'][0]:.4f},{s['acc_ci'][1]:.4f}] "
            f"logloss={s['logloss_bits']:.3f}b "
            f"Fano={s['fano_lower_bits']:.3f} xent={s['xent_lower_bits']:.3f} "
            f"({time.time()-tc:.0f}s)")
        if name == "lda":
            cal = jb.calibration(r, n_bins=10)
            cal["table"].to_csv("artifacts/calibration_lda.csv", index=False)

    strongest = max(summ, key=lambda k: summ[k]["xent_lower_bits"])
    inversion = summ["mlp"]["top1_acc"] < summ["logreg"]["top1_acc"]

    # write classifier table
    rows = []
    for name, s in summ.items():
        rows.append(dict(
            classifier=name, label=CLF_LABEL[name], S=S, N=N, chance=1.0 / S,
            ceiling_bits=ceiling,
            top1_acc=s["top1_acc"], acc_ci_lo=s["acc_ci"][0], acc_ci_hi=s["acc_ci"][1],
            fold_acc_mean=s["fold_acc_mean"], fold_acc_std=s["fold_acc_std"],
            logloss_bits=s["logloss_bits"], logloss_nats=s["logloss_nats"],
            fold_logloss_bits_mean=s["fold_logloss_bits_mean"],
            fold_logloss_bits_std=s["fold_logloss_bits_std"],
            fano_lower_bits=s["fano_lower_bits"],
            fano_ci_lo=s["fano_lower_ci"][0], fano_ci_hi=s["fano_lower_ci"][1],
            xent_lower_bits=s["xent_lower_bits"],
            xent_ci_lo=s["xent_lower_ci"][0], xent_ci_hi=s["xent_lower_ci"][1]))
    df = pd.DataFrame(rows)
    df.to_csv("classifiers.csv", index=False)
    # per-fold accuracies (recompute, store)
    perfold = {}
    for name, fac in jb.CLASSIFIERS.items():
        pass  # already have fold means/std in summ
    fano_best = max(summ[k]["fano_lower_bits"] for k in summ)
    xent_best = max(summ[k]["xent_lower_bits"] for k in summ)
    summary = dict(seed=SEED, S=S, N=N, clips_per_speaker=10, ceiling_bits=ceiling,
                   n_features_kept=cd["n_features_kept"],
                   kept_features=cd["kept_features"],
                   dropped_features=cd["dropped_features"],
                   n_utts_dropped_listwise=cd["n_utts_dropped"],
                   strongest=strongest, capacity_inversion=bool(inversion),
                   fano_lower_best=float(fano_best), xent_lower_best=float(xent_best),
                   classifiers={name: {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in s.items()}
                                for name, s in summ.items()},
                   elapsed_s=round(time.time() - t0, 1))
    json.dump(summary, open("classifier_results.json", "w"), indent=2, default=float)
    log(f"DONE {(time.time()-t0)/60:.1f} min. strongest={strongest} "
        f"Fano_best={fano_best:.3f} xent_best={xent_best:.3f} "
        f"inversion(MLP<logreg)={inversion}")


if __name__ == "__main__":
    main()
