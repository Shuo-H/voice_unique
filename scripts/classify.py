"""Stage 6: held-out classifier lower bound on joint usable speaker bits.
Env: TIMIT_OUTDIR, TIMIT_RESULTS. Seed 1234."""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_40, SEED

LN2 = np.log(2)


def log(msg):
    with open("run.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)


def Hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ci95(vals):
    vals = np.asarray(vals, float)
    m = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals))
    return float(m), [float(m - 1.96 * se), float(m + 1.96 * se)]


def main():
    outdir = os.environ.get("TIMIT_OUTDIR", "features")
    results = os.environ.get("TIMIT_RESULTS", "results")
    os.makedirs(results, exist_ok=True)
    df = pd.read_parquet(os.path.join(outdir, "features_per_utt.parquet"))
    df = df[df["decode_ok"]].copy()

    # keep features with >=90% coverage
    keep_feats = [f for f in FEATURES_40 if df[f].notna().mean() >= 0.90]
    sub = df.dropna(subset=keep_feats).copy()
    # require speakers with >=2 utts so each can be split across folds
    vc = sub["speaker"].value_counts()
    sub = sub[sub["speaker"].isin(vc[vc >= 2].index)].copy()
    S = sub["speaker"].nunique()
    H_spk = np.log2(S)
    log(f"classify: {len(sub)} rows, {S} speakers, {len(keep_feats)} features (>=90% cov)")

    X = sub[keep_feats].values
    y = sub["speaker"].astype("category")
    classes = list(y.cat.categories)
    y_idx = y.cat.codes.values

    n_splits = int(os.environ.get("TIMIT_NFOLDS", 5))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial",
                                             random_state=SEED),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, alpha=1e-3,
                                     random_state=SEED),
        "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }

    out = {"n_rows": int(len(sub)), "n_speakers": int(S), "H_speaker_bits": float(H_spk),
           "chance": float(1.0 / S), "features_used": keep_feats, "models": {}}

    all_classes = np.arange(len(classes))
    for name, ctor in models.items():
        accs, lls = [], []
        for tr, te in skf.split(X, y_idx):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = ctor()
            clf.fit(Xtr, y_idx[tr])
            pred = clf.predict(Xte)
            accs.append(accuracy_score(y_idx[te], pred))
            proba = clf.predict_proba(Xte)
            ll = log_loss(y_idx[te], proba, labels=clf.classes_)  # nats
            lls.append(ll / LN2)  # bits
        acc_m, acc_ci = ci95(accs)
        ll_m, ll_ci = ci95(lls)
        Pe = 1 - acc_m
        fano = H_spk - (Hb(Pe) + Pe * np.log2(max(S - 1, 1)))
        xent = H_spk - ll_m
        out["models"][name] = {
            "acc_mean": acc_m, "acc_ci95": acc_ci,
            "acc_per_fold_mean": float(np.mean(accs)), "acc_per_fold_std": float(np.std(accs, ddof=1)),
            "acc_folds": [float(a) for a in accs],
            "logloss_bits_mean": ll_m, "logloss_bits_ci95": ll_ci,
            "Pe": float(Pe), "fano_bits": float(fano), "xent_bits": float(xent),
        }
        log(f"  {name}: acc={acc_m:.4f} fano={fano:.2f} bits xent={xent:.2f} bits")

    # capacity inversion: MLP vs linear best
    lin_best = max(out["models"]["logreg"]["acc_mean"], out["models"]["lda"]["acc_mean"])
    out["capacity_inversion"] = bool(out["models"]["mlp"]["acc_mean"] < lin_best)
    out["headline_fano_bits"] = float(max(m["fano_bits"] for m in out["models"].values()))
    out["headline_xent_bits"] = float(max(m["xent_bits"] for m in out["models"].values()))

    with open(os.path.join(results, "classifier.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"classify DONE: headline Fano={out['headline_fano_bits']:.2f} bits, "
        f"capacity_inversion={out['capacity_inversion']}")


if __name__ == "__main__":
    main()
