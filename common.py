"""
common.py -- shared loaders/utilities for the v2 (40-feature) battery.

Long-format features.parquet -> coverage, measured-feature set, utt-wide table,
per-speaker mean matrix, and q-quantile bin edges.  Seed 1234 everywhere.
"""
import os, json
import numpy as np, pandas as pd
import features as F

SEED = 1234
QS = [2, 3, 5, 10]
COV_THRESH = 0.80          # "measured" (used downstream) = coverage >= 80%
HERE = os.path.dirname(os.path.abspath(__file__))


def load_long(path=None):
    return pd.read_parquet(path or os.path.join(HERE, "features.parquet"))


def coverage_table(df):
    rows = []
    canon = [f for f in F.FEATURES_40]
    present = set(df.feature.unique())
    order = [f for f in (F.FEATURES_40 + F.AUX) if f in present]
    for feat in order:
        sub = df[df.feature == feat]["value"]
        frac = float(sub.notna().mean()) if len(sub) else 0.0
        rows.append(dict(feature=feat, display=F.disp(feat),
                         group=("aux_HNR" if feat in F.AUX else F.V2_GROUP.get(feat)),
                         coverage=round(frac, 4),
                         status="NOT MEASURED" if frac == 0 else "measured"))
    return pd.DataFrame(rows)


def measured_features(cov, thresh=COV_THRESH):
    m = cov[(cov.group != "aux_HNR") & (cov.coverage >= thresh)]
    # keep canonical order
    feats = [f for f in F.FEATURES_40 if f in set(m.feature)]
    return feats


def wide_utt(df, feats):
    """utt x feature wide table + speaker metadata (pivot on utt_id only)."""
    sub = df[df.feature.isin(feats)]
    w = sub.pivot_table(index="utt_id", columns="feature", values="value",
                        aggfunc="first").reset_index()
    meta = (df.drop_duplicates("utt_id")
              .set_index("utt_id")[["speaker_id", "sex", "accent", "age"]])
    w = w.merge(meta, on="utt_id", how="left")
    return w


def speaker_means(wide, feats):
    """per-speaker mean-feature matrix (speakers complete across feats) + meta."""
    g = wide.groupby("speaker_id")[feats].mean()
    meta = wide.groupby("speaker_id")[["sex", "accent", "age"]].agg(
        lambda s: s.dropna().iloc[0] if s.dropna().size else None)
    g = g.join(meta)
    complete = g[feats].notna().all(axis=1)
    return g[complete].copy(), int((~complete).sum())


def quantile_bins(spk, feats, qs=QS):
    """Equiprobable q-quantile bin edges from per-speaker means; flags degenerate
    (collapsed) bins where consecutive edges coincide."""
    bins, degenerate = {}, {}
    for f in feats:
        x = spk[f].dropna().values
        bins[f] = {}
        degenerate[f] = {}
        for q in qs:
            qs_e = np.quantile(x, np.linspace(0, 1, q + 1))
            uniq = np.unique(qs_e)
            bins[f][str(q)] = [float(v) for v in qs_e]
            degenerate[f][str(q)] = int((q + 1) - len(uniq))  # #collapsed edges
    return bins, degenerate


def zscore(M):
    mu = M.mean(axis=0); sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd
