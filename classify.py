"""Section 6: held-out classifier lower bounds (Fano + cross-entropy).
Utterance-disjoint stratified 5-fold CV. Seed 1234."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

SEED = 1234
np.random.seed(SEED)

from feat_lib import FEATURES_40, NOT_ATTEMPTED

COV_CLF = 0.90
df = pd.read_parquet("features/features_per_utt.parquet")

# features with >=90% coverage
cov = {f: float(np.mean(np.isfinite(df[f].to_numpy(dtype=float))))
       for f in FEATURES_40}
feats = [f for f in FEATURES_40 if f not in NOT_ATTEMPTED and cov[f] >= COV_CLF]

# listwise-delete incomplete rows
X = df[feats].to_numpy(dtype=float)
mask = np.all(np.isfinite(X), axis=1)
Xc = X[mask]
spk = df["speaker"].to_numpy()[mask]
classes = sorted(np.unique(spk))
S = len(classes)
cls_to_i = {c: i for i, c in enumerate(classes)}
y = np.array([cls_to_i[s] for s in spk])
N = len(y)
H_speaker = np.log2(S)
print(f"[6] retained rows={N}/{len(df)} speakers={S} features={len(feats)}")
print(f"    H(speaker)=log2({S})={H_speaker:.3f} bits chance={1/S:.5f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def make_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0,
                                     multi_class="multinomial", random_state=SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                             early_stopping=True, random_state=SEED),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }

results = {}
for name in ["logreg", "mlp", "lda"]:
    accs, lls = [], []
    for tr, te in skf.split(Xc, y):
        sc = StandardScaler().fit(Xc[tr])
        Xtr, Xte = sc.transform(Xc[tr]), sc.transform(Xc[te])
        clf = make_models()[name]
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(np.mean(pred == y[te]))
        proba = clf.predict_proba(Xte)
        ll = log_loss(y[te], proba, labels=np.arange(S))
        lls.append(ll / np.log(2))  # bits
    accs = np.array(accs); lls = np.array(lls)
    acc = float(accs.mean())
    # 95% CI (normal approx on the N held-out predictions, all utts predicted once)
    se = np.sqrt(acc * (1 - acc) / N)
    ci = [float(acc - 1.96 * se), float(acc + 1.96 * se)]
    Pe = 1 - acc
    # Fano: I >= H - [Hb(Pe) + Pe*log2(S-1)]
    Hb = 0.0 if Pe in (0.0, 1.0) else -(Pe*np.log2(Pe) + (1-Pe)*np.log2(1-Pe))
    fano = H_speaker - (Hb + Pe * np.log2(S - 1))
    xent = H_speaker - float(lls.mean())
    # bootstrap CI for fano/xent via per-fold spread
    results[name] = {
        "top1_acc": acc, "acc_ci95": ci,
        "per_fold_acc_mean": float(accs.mean()), "per_fold_acc_std": float(accs.std()),
        "per_fold_acc": accs.tolist(),
        "logloss_bits_mean": float(lls.mean()), "logloss_bits_std": float(lls.std()),
        "fano_bits": float(fano), "xent_bits": float(xent),
    }
    print(f"    {name}: acc={acc:.4f} fano={fano:.3f}b xent={xent:.3f}b "
          f"logloss={lls.mean():.3f}b")

lin_best = max(results["logreg"]["top1_acc"], results["lda"]["top1_acc"])
cap_inv = results["mlp"]["top1_acc"] < lin_best
out = {
    "retained_utts": int(N), "retained_speakers": int(S),
    "n_features": len(feats), "features": feats,
    "chance": 1.0 / S, "H_speaker_bits": float(H_speaker),
    "classifiers": results,
    "capacity_inversion": bool(cap_inv),
    "headline_fano_bits": float(max(results[m]["fano_bits"] for m in results)),
    "headline_fano_model": max(results, key=lambda m: results[m]["fano_bits"]),
}
with open("results/classifier.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(f"    capacity_inversion={cap_inv} headline_fano={out['headline_fano_bits']:.3f}b")
print("CLASSIFY_DONE")
