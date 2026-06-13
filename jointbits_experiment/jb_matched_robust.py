"""
jb_matched_robust.py -- robustness of the Step-6 matched contrast.

The Step-6 CV-matched row uses ONE random 630-speaker subsample (seed 1234). To
confirm the surprising near-equality of CV and TIMIT at matched S is not a lucky
draw, re-draw the CV subsample over several seeds and report mean+/-std of the
best top-1 accuracy and best cross-entropy lower bound across draws. TIMIT is a
fixed corpus (no subsample variance) -> its single value is the reference.

Writes matched_robustness.csv and prints a summary.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jb_core as jb

CV_PARQUET = os.path.join(os.path.dirname(HERE), "mi_experiment", "features.parquet")
SEEDS = [1234, 1, 2, 3, 4, 5, 6, 7]   # 8 independent 630-speaker draws


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    common = R["common_features"]
    S_match = R["timit_matched"]["S"]
    # TIMIT reference (fixed)
    tmm = R["timit_matched"]["clf"]
    tm_best_acc = max(tmm[c]["top1_acc"] for c in tmm)
    tm_best_xent = max(tmm[c]["xent_lower_bits"] for c in tmm)

    wide, feats = jb.load_wide(CV_PARQUET)
    cd = jb.coverage_drop(wide, feats)
    bal, _ = jb.balance_speakers(cd["wide"])

    rows = []
    for sd in SEEDS:
        sub, spk = jb.subsample_speakers(bal, S_match, seed=sd)
        fold = jb.assign_folds(sub, seed=sd)
        X, y = jb.design_matrix(sub, common)
        accs, xents = {}, {}
        for name, fac in jb.CLASSIFIERS.items():
            r = jb.cv_evaluate(X, y, fold, S_match, fac, seed=jb.SEED)
            s = jb.summarize(r, S_match, n_boot=200, seed=jb.SEED)
            accs[name] = s["top1_acc"]; xents[name] = s["xent_lower_bits"]
        best_acc = max(accs.values()); best_xent = max(xents.values())
        rows.append(dict(seed=sd, S=S_match,
                         logreg_acc=accs["logreg"], mlp_acc=accs["mlp"], lda_acc=accs["lda"],
                         best_acc=best_acc, best_xent=best_xent))
        print(f"seed {sd}: best_acc={best_acc:.4f} best_xent={best_xent:.3f} "
              f"(lr/mlp/lda acc={accs['logreg']:.3f}/{accs['mlp']:.3f}/{accs['lda']:.3f})",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "matched_robustness.csv"), index=False)

    ba, bx = df["best_acc"], df["best_xent"]
    summ = dict(
        n_draws=len(SEEDS), S=S_match,
        cv_best_acc_mean=float(ba.mean()), cv_best_acc_std=float(ba.std(ddof=1)),
        cv_best_acc_min=float(ba.min()), cv_best_acc_max=float(ba.max()),
        cv_best_xent_mean=float(bx.mean()), cv_best_xent_std=float(bx.std(ddof=1)),
        cv_best_xent_min=float(bx.min()), cv_best_xent_max=float(bx.max()),
        timit_best_acc=tm_best_acc, timit_best_xent=tm_best_xent,
    )
    json.dump(summ, open(os.path.join(HERE, "matched_robustness_summary.json"), "w"),
              indent=2)
    print("\n=== MATCHED ROBUSTNESS (CV over %d draws vs fixed TIMIT) ===" % len(SEEDS))
    print(f"CV    best top-1 acc : {summ['cv_best_acc_mean']:.4f} +/- {summ['cv_best_acc_std']:.4f} "
          f"[{summ['cv_best_acc_min']:.4f}, {summ['cv_best_acc_max']:.4f}]")
    print(f"TIMIT best top-1 acc : {tm_best_acc:.4f}")
    print(f"CV    best xent bits : {summ['cv_best_xent_mean']:.3f} +/- {summ['cv_best_xent_std']:.3f} "
          f"[{summ['cv_best_xent_min']:.3f}, {summ['cv_best_xent_max']:.3f}]")
    print(f"TIMIT best xent bits : {tm_best_xent:.3f}")
    dz_acc = (tm_best_acc - summ['cv_best_acc_mean']) / summ['cv_best_acc_std'] if summ['cv_best_acc_std'] else float('nan')
    dz_x = (tm_best_xent - summ['cv_best_xent_mean']) / summ['cv_best_xent_std'] if summ['cv_best_xent_std'] else float('nan')
    print(f"TIMIT vs CV-draws: acc +{tm_best_acc-summ['cv_best_acc_mean']:+.4f} ({dz_acc:+.1f} sd), "
          f"xent {tm_best_xent-summ['cv_best_xent_mean']:+.3f} bits ({dz_x:+.1f} sd)")


if __name__ == "__main__":
    main()
