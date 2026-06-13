"""
jb_core.py -- core machinery for the JOINT usable speaker-information lower bound
(classifier-based, Fano + cross-entropy) on long-format per-utterance features.

Everything here is corpus-agnostic: the same functions run on the Common Voice
features.parquet and on the TIMIT timit_features.parquet, so Step 6 is a true
apples-to-apples contrast.

Conventions
-----------
* speaker label  = the `speaker_id` column (CV: client_id hash; TIMIT: speaker code).
* All information quantities are BITS (log base 2).
* Every headline number produced downstream is a LOWER BOUND on the joint usable
  speaker information: classifier- and sample-dependent, can only rise with a
  stronger model or more data.
* Seed 1234 everywhere (numpy default_rng, sklearn random_state, folds, bootstrap).

Pipeline per corpus
-------------------
  load_wide -> coverage_drop -> balance_speakers(10/spk) -> assign_folds(5)
  -> cv_evaluate(clf) -> summarize (acc, logloss bits/nats, per-fold, bootstrap CI)
  -> fano/xent bounds -> calibration -> greedy_forward_lda -> binned_greedy_censored
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mi_experiment"))
import mi_core as mc  # noqa: E402  reuse Miller-Madow MI + permutation null

SEED = 1234
CLIPS_PER_SPEAKER = 10
N_FOLDS = 5
COVERAGE_THRESH = 0.95
EPS = 1e-15


# ----------------------------------------------------------------- load -------
def load_wide(parquet_path):
    """Long (speaker_id, utt_id, feature, value [+ sex/accent/age]) -> wide:
    one row per utt, one column per feature, plus metadata columns. Returns
    (wide_df, feature_cols)."""
    df = pd.read_parquet(parquet_path)
    meta_cols = [c for c in ("speaker_id", "utt_id", "sex", "accent", "age")
                 if c in df.columns]
    feats = sorted(df["feature"].unique().tolist())
    wide = df.pivot_table(index=["speaker_id", "utt_id"], columns="feature",
                          values="value", aggfunc="first")
    wide = wide.reset_index()
    # attach metadata (one value per utt)
    extra = [c for c in ("sex", "accent", "age") if c in df.columns]
    if extra:
        meta = (df[["speaker_id", "utt_id"] + extra]
                .drop_duplicates(["speaker_id", "utt_id"]))
        wide = wide.merge(meta, on=["speaker_id", "utt_id"], how="left")
    wide.columns.name = None
    return wide, feats


# ------------------------------------------------------- coverage handling ----
def coverage_table(wide, feature_cols):
    """Per-feature fraction of utterances with a non-NaN value (over ALL utts)."""
    return (wide[feature_cols].notna().mean()
            .sort_values()
            .rename("coverage").to_frame())


def coverage_drop(wide, feature_cols, thresh=COVERAGE_THRESH):
    """DROP SPARSE FEATURES (coverage < thresh), then keep utterances that are
    listwise-complete over the REMAINING features. Returns dict with the cleaned
    wide frame and bookkeeping."""
    cov = wide[feature_cols].notna().mean()
    kept = [f for f in feature_cols if cov[f] >= thresh]
    dropped = [f for f in feature_cols if cov[f] < thresh]
    before = len(wide)
    clean = wide.dropna(subset=kept).reset_index(drop=True)
    return dict(
        wide=clean, kept_features=kept, dropped_features=dropped,
        coverage=cov.to_dict(),
        n_features_kept=len(kept), n_utts_kept=len(clean),
        n_utts_dropped=before - len(clean), n_utts_before=before,
    )


def listwise_all_features(wide, feature_cols):
    """TIMIT-style alternative: keep ALL features, listwise-delete any utt with a
    NaN in any feature. Returns (clean_wide, n_utts_kept, n_utts_dropped)."""
    before = len(wide)
    clean = wide.dropna(subset=feature_cols).reset_index(drop=True)
    return clean, len(clean), before - len(clean)


# ------------------------------------------------------- speaker balancing ----
def balance_speakers(wide, clips=CLIPS_PER_SPEAKER, seed=SEED):
    """Keep speakers with >= `clips` rows; randomly sample EXACTLY `clips` rows
    per speaker. Adds a dense integer `speaker_idx`. Uniform prior by
    construction. Returns (balanced_wide, speaker_list)."""
    rng = np.random.default_rng(seed)
    counts = wide.groupby("speaker_id").size()
    eligible = sorted(counts[counts >= clips].index.tolist())
    parts = []
    for spk in eligible:
        idx = wide.index[wide["speaker_id"] == spk].to_numpy()
        pick = rng.permutation(idx)[:clips]
        parts.append(pick)
    sel = np.concatenate(parts)
    bal = wide.loc[sel].copy()
    # stable dense speaker index
    spk_list = eligible
    code = {s: i for i, s in enumerate(spk_list)}
    bal["speaker_idx"] = bal["speaker_id"].map(code).astype(int)
    bal = bal.sort_values(["speaker_idx", "utt_id"]).reset_index(drop=True)
    return bal, spk_list


def subsample_speakers(bal, n_speakers, seed=SEED):
    """Random subsample of EXACTLY n_speakers speakers (for the TIMIT-matched
    CV row). Re-densifies speaker_idx. Keeps each speaker's 10 clips intact."""
    rng = np.random.default_rng(seed)
    spk = np.array(sorted(bal["speaker_id"].unique()))
    pick = rng.permutation(spk)[:n_speakers]
    pick = set(pick.tolist())
    sub = bal[bal["speaker_id"].isin(pick)].copy()
    spk_list = sorted(sub["speaker_id"].unique())
    code = {s: i for i, s in enumerate(spk_list)}
    sub["speaker_idx"] = sub["speaker_id"].map(code).astype(int)
    sub = sub.sort_values(["speaker_idx", "utt_id"]).reset_index(drop=True)
    return sub, spk_list


# ---------------------------------------------------- utterance-disjoint CV ---
def assign_folds(bal, n_folds=N_FOLDS, seed=SEED):
    """Stratified within-speaker fold assignment: each speaker's `clips` rows are
    split evenly across `n_folds` folds (utterance-disjoint). Every speaker
    appears in train AND test of every fold; no utterance is shared. Returns an
    int array `fold` aligned to bal rows."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(bal), -1, dtype=int)
    for _, idx in bal.groupby("speaker_idx").groups.items():
        idx = np.array(sorted(idx))
        perm = rng.permutation(len(idx))
        # contiguous near-equal groups -> exactly clips/n_folds per fold
        groups = np.array_split(perm, n_folds)
        for f, g in enumerate(groups):
            fold[idx[g]] = f
    assert (fold >= 0).all()
    return fold


# ------------------------------------------------------------- classifiers ----
def make_logreg(seed=SEED):
    # multinomial logistic regression, mild L2 (C=1.0); weak linear reference
    return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)


def make_mlp(seed=SEED):
    # small MLP: one hidden layer, nonlinear, higher capacity
    return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                         max_iter=300, early_stopping=True,
                         n_iter_no_change=12, random_state=seed)


def make_lda(seed=SEED):
    # shrinkage-LDA (Ledoit-Wolf): strong regularized-linear reference
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


CLASSIFIERS = {"logreg": make_logreg, "mlp": make_mlp, "lda": make_lda}


# ------------------------------------------------------- CV evaluation --------
def _zscore_fit_apply(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cv_evaluate(X, y, fold, S, clf_factory, seed=SEED):
    """Utterance-disjoint K-fold CV. z-score with TRAIN-fold stats only (no
    leakage). Returns pooled out-of-fold per-clip arrays + per-fold metrics.

    Returns dict:
      y_true[N], pred[N] (argmax), p_true[N] (prob of true class),
      conf[N] (max prob), loss_bits[N] (-log2 p_true),
      fold_acc[K], fold_logloss_bits[K]
    """
    N = len(y)
    pred = np.full(N, -1, dtype=int)
    p_true = np.zeros(N)
    conf = np.zeros(N)
    folds = sorted(np.unique(fold))
    fold_acc, fold_ll = [], []
    classes = np.arange(S)
    for f in folds:
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        Xtr, Xte = _zscore_fit_apply(X[tr], X[te])
        clf = clf_factory(seed)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        # map clf.classes_ -> full S columns (every speaker is in train, so
        # clf.classes_ == all S, but be robust)
        if proba.shape[1] != S:
            full = np.zeros((proba.shape[0], S))
            full[:, clf.classes_] = proba
            proba = full
        pr = np.clip(proba, EPS, 1.0)
        pred[te] = np.argmax(proba, axis=1)
        conf[te] = proba.max(axis=1)
        p_true[te] = proba[np.arange(len(te)), y[te]]
        # per-fold metrics
        acc_f = float(np.mean(pred[te] == y[te]))
        ll_f = float(np.mean(-np.log2(np.clip(p_true[te], EPS, 1.0))))
        fold_acc.append(acc_f)
        fold_ll.append(ll_f)
    loss_bits = -np.log2(np.clip(p_true, EPS, 1.0))
    return dict(y_true=y.copy(), pred=pred, p_true=p_true, conf=conf,
                loss_bits=loss_bits,
                fold_acc=np.array(fold_acc),
                fold_logloss_bits=np.array(fold_ll))


# ------------------------------------------------------------- bounds ---------
def _binary_entropy_bits(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def fano_lower_bits(acc, S):
    """I >= H(spk) - [Hb(Perr) + Perr*log2(S-1)],  uniform prior H=log2(S)."""
    Perr = 1.0 - acc
    H = np.log2(S)
    if S <= 1:
        return 0.0
    return float(H - (_binary_entropy_bits(Perr) + Perr * np.log2(S - 1)))


def xent_lower_bits(logloss_bits, S):
    """I >= H(spk) - mean_test_logloss_bits, uniform prior."""
    return float(np.log2(S) - logloss_bits)


def summarize(result, S, n_boot=1000, seed=SEED):
    """Pooled top-1 acc, mean log-loss (bits & nats), per-fold mean+/-std, and a
    bootstrap 95% CI over test clips for: acc, logloss, Fano bound, xent bound."""
    correct = (result["pred"] == result["y_true"]).astype(float)
    loss_bits = result["loss_bits"]
    N = len(correct)
    acc = float(correct.mean())
    ll_bits = float(loss_bits.mean())
    ll_nats = ll_bits * np.log(2)
    out = dict(
        S=S, N=N, chance=1.0 / S, H_bits=float(np.log2(S)),
        top1_acc=acc,
        logloss_bits=ll_bits, logloss_nats=ll_nats,
        fold_acc_mean=float(result["fold_acc"].mean()),
        fold_acc_std=float(result["fold_acc"].std(ddof=0)),
        fold_logloss_bits_mean=float(result["fold_logloss_bits"].mean()),
        fold_logloss_bits_std=float(result["fold_logloss_bits"].std(ddof=0)),
        fano_lower_bits=fano_lower_bits(acc, S),
        xent_lower_bits=xent_lower_bits(ll_bits, S),
    )
    # bootstrap over test clips
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot); lls = np.empty(n_boot)
    fanos = np.empty(n_boot); xents = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.integers(0, N, N)
        a = float(correct[bi].mean())
        l = float(loss_bits[bi].mean())
        accs[b] = a; lls[b] = l
        fanos[b] = fano_lower_bits(a, S)
        xents[b] = xent_lower_bits(l, S)
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    out["acc_ci"] = ci(accs)
    out["logloss_bits_ci"] = ci(lls)
    out["fano_lower_ci"] = ci(fanos)
    out["xent_lower_ci"] = ci(xents)
    return out


# ----------------------------------------------------------- calibration ------
def calibration(result, n_bins=10):
    """Reliability of the top-1 confidence (max proba) on held-out clips +
    Expected Calibration Error (ECE). Over-confidence => xent bound loosened
    (conservative), so the bound stays a valid floor."""
    conf = result["conf"]
    correct = (result["pred"] == result["y_true"]).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin_lo=lo, bin_hi=hi, n=0,
                             mean_conf=np.nan, accuracy=np.nan, gap=np.nan))
            continue
        mc_ = float(conf[m].mean()); acc_ = float(correct[m].mean())
        rows.append(dict(bin_lo=lo, bin_hi=hi, n=n,
                         mean_conf=mc_, accuracy=acc_, gap=acc_ - mc_))
        ece += (n / N) * abs(acc_ - mc_)
    return dict(ece=float(ece), table=pd.DataFrame(rows))


# --------------------------------------------- Step 3: greedy forward (LDA) ---
def greedy_forward_lda(X, y, fold, S, feature_names, seed=SEED, verbose=False):
    """Greedy forward selection driven by held-out cross-entropy lower bound
    I_lower = log2(S) - mean_test_logloss_bits, using shrinkage-LDA. Adds ALL
    features (no early stop). Returns selection order + cumulative I_lower."""
    remaining = list(range(X.shape[1]))
    selected = []
    order, cum_bits = [], []
    H = np.log2(S)
    while remaining:
        best_feat, best_bits = None, -np.inf
        for j in remaining:
            cols = selected + [j]
            res = cv_evaluate(X[:, cols], y, fold, S, make_lda, seed)
            ll = float(res["loss_bits"].mean())
            bits = H - ll
            if bits > best_bits:
                best_bits, best_feat = bits, j
        selected.append(best_feat)
        remaining.remove(best_feat)
        order.append(feature_names[best_feat])
        cum_bits.append(best_bits)
        if verbose:
            print(f"  +{feature_names[best_feat]:16s} I_lower={best_bits:.3f} bits "
                  f"({len(selected)}/{X.shape[1]})")
    cum = np.array(cum_bits)
    max_bits = float(cum.max())
    # #features to reach 95% of max I_lower (first crossing)
    if max_bits <= 0:
        n95 = len(cum)
    else:
        hit = np.where(cum >= 0.95 * max_bits)[0]
        n95 = int(hit[0] + 1) if len(hit) else len(cum)
    return dict(order=order, cum_bits=cum.tolist(),
                max_bits=max_bits, n_features_95=n95)


# ------------------------------- Step 4: binned greedy censored (sanity) ------
def binned_greedy_censored(X, speaker_idx, S, feature_names, nperm=200, seed=SEED):
    """Binary-bin (median split) greedy plug-in MI with Miller-Madow correction +
    permutation null. Selection by I_mm (biased, plug-in MM); for the selected
    cumulative joint we ALSO compute the permutation-null-corrected I and the
    occupied joint-cell count. Censor point k* = first step where occupied joint
    cells > N/5 (estimator unreliable beyond). CENSORED SANITY CHECK ONLY."""
    N = X.shape[0]
    spk = np.asarray(speaker_idx)
    # binarize each feature at its median -> {0,1}
    B = np.zeros_like(X, dtype=np.int64)
    for j in range(X.shape[1]):
        med = np.median(X[:, j])
        B[:, j] = (X[:, j] > med).astype(np.int64)
    remaining = list(range(X.shape[1]))
    selected = []
    joint = np.zeros(N, dtype=np.int64)  # dense joint label of selected bits
    censor_thresh = N / 5.0
    rows = []
    k_star = None
    while remaining:
        # pick feature maximizing plug-in Miller-Madow joint MI (cheap, no perms)
        best_j, best_immi = None, -np.inf
        best_joint = None
        for j in remaining:
            cand = joint * 2 + B[:, j]
            _, cand_dense = np.unique(cand, return_inverse=True)
            Hx, Kx = mc.entropy_bits(np.bincount(spk, minlength=S))
            Hy, Ky = mc.entropy_bits(np.bincount(cand_dense))
            Hxy, Kxy = mc.joint_entropy_bits(spk, cand_dense)
            I_mm = ((Hx + (Kx - 1) / (2 * N)) + (Hy + (Ky - 1) / (2 * N))
                    - (Hxy + (Kxy - 1) / (2 * N)))
            if I_mm > best_immi:
                best_immi, best_j = I_mm, j
                best_joint = cand_dense
        selected.append(best_j)
        remaining.remove(best_j)
        joint = best_joint
        # full metrics (with permutation null) for the chosen cumulative joint
        m = mc.mi_metrics(spk, joint, S, nperm=nperm, seed=seed)
        occupied = m["Kxy"]
        censored = occupied > censor_thresh
        if censored and k_star is None:
            k_star = len(selected)
        rows.append(dict(step=len(selected), feature=feature_names[best_j],
                         I_mm=m["I_mm"], I_corrected=m["I_corrected"],
                         I_null_mean=m["I_null_mean"], occupied_cells=occupied,
                         censored=bool(censored)))
    return dict(table=pd.DataFrame(rows), k_star=k_star,
                censor_thresh=censor_thresh, N=N)


# --------------------------------------------------------- design matrix ------
def design_matrix(bal, feature_cols):
    """Return (X[N,F] float64, y[N] speaker_idx, feature_cols)."""
    X = bal[feature_cols].to_numpy(dtype=np.float64)
    y = bal["speaker_idx"].to_numpy(dtype=int)
    return X, y
