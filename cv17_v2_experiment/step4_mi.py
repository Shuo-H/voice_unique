"""
step4_mi.py -- STEP 4: per-feature USABLE BIT DEPTH (information-theoretic).

Balanced sample of 5 clips/speaker (seed 1234) over ALL qualifying speakers
(uniform speaker prior; keeps the full scaled-up speaker set since every kept
speaker has >= 5 clips).  For each measured feature and bit depth b in 1..8
(q = 2^b equal-frequency bins -> q_eff realized bins):

  I_raw   plug-in MI ;  I_mm  Miller-Madow MI ;
  I_null  permutation null (>=200 speaker-label shuffles, seed 1234) ;
  I_corrected = max(0, I_mm - I_null_mean)   [bits above chance]  <- HEADLINE
  NMI = I_corrected / log2(S) ;  perm_p = frac(null >= I_raw)

b* = argmax_b I_corrected (ties -> smallest b).  Output usable_bits.csv sorted by
usable bits; total summed usable bits across features = OPTIMISTIC over-count
(ignores redundancy; the joint analysis in Step 6 supersedes it).
"""
import os, sys, json, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import common as C
import features as F

sys.path.insert(0, os.path.join(os.path.dirname(C.HERE), "mi_experiment"))
import mi_core as MC

SEED = 1234
NPERM = 200
BITS = [1, 2, 3, 4, 5, 6, 7, 8]
CLIPS = 5
N_WORKERS = max(2, (os.cpu_count() or 4) - 2)


def balance(wide, feats, clips=CLIPS, seed=SEED):
    """Sample exactly `clips` utts/speaker (all speakers have >= clips). Returns
    balanced wide rows + dense speaker codes + S."""
    rng = np.random.default_rng(seed)
    parts = []
    for spk, idx in wide.groupby("speaker_id").groups.items():
        idx = np.array(sorted(idx))
        parts.append(rng.permutation(idx)[:clips])
    sel = np.concatenate(parts)
    bal = wide.loc[sel].copy()
    codes, uniq = pd.factorize(bal["speaker_id"], sort=True)
    bal["spk"] = codes
    return bal, int(len(uniq))


def _feature_worker(payload):
    feat, vals, spk, S, logS = payload
    ok = np.isfinite(vals)
    vals, spk = vals[ok], spk[ok]
    rows = []
    for b in BITS:
        labels, q_eff, _ = MC.quantize(vals, 2 ** b)
        m = MC.mi_metrics(spk, labels, S, nperm=NPERM, seed=SEED + b)
        rows.append(dict(feature=feat, b=b, q_eff=q_eff, N=int(vals.size),
                         I_raw=m["I_raw"], I_mm=m["I_mm"],
                         I_null_mean=m["I_null_mean"], perm_p=m["perm_p"],
                         I_corrected=m["I_corrected"],
                         NMI_corrected=m["I_corrected"] / logS))
    return rows


def main():
    os.chdir(C.HERE)
    t0 = time.time()
    df = C.load_long()
    cov = C.coverage_table(df)
    feats = C.measured_features(cov)
    wide = C.wide_utt(df, feats)
    bal, S = balance(wide, feats)
    logS = float(np.log2(S))
    print(f"[step4] balanced {CLIPS} clips/spk: S={S} speakers, N={len(bal)} utts, "
          f"log2(S)={logS:.3f} bits ceiling; {len(feats)} features", flush=True)

    spk = bal["spk"].to_numpy()
    payloads = [(f, bal[f].to_numpy(dtype=float), spk, S, logS) for f in feats]
    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for rows in ex.map(_feature_worker, payloads):
            all_rows.extend(rows)
    perbit = pd.DataFrame(all_rows)
    perbit.to_csv("mi_by_feature_bit.csv", index=False)

    # b* per feature
    usable = []
    for f in feats:
        sub = perbit[perbit.feature == f].sort_values("b")
        i = int(np.argmax(sub["I_corrected"].to_numpy()))
        r = sub.iloc[i]
        usable.append(dict(feature=f, display=F.disp(f), group=F.V2_GROUP.get(f),
                           b_star=int(r.b), q_eff=int(r.q_eff),
                           I_corrected_bits=float(r.I_corrected),
                           NMI=float(r.NMI_corrected), perm_p=float(r.perm_p),
                           N=int(r.N)))
    ub = pd.DataFrame(usable).sort_values("I_corrected_bits", ascending=False)
    ub.to_csv("usable_bits.csv", index=False)
    total = float(ub["I_corrected_bits"].sum())
    n_sig = int((ub["perm_p"] < 0.05).sum())
    summ = dict(S=S, N=int(len(bal)), logS=logS, clips_per_speaker=CLIPS,
                n_features=len(feats), total_usable_bits_optimistic=total,
                n_perm_significant=n_sig,
                top5=ub.head(5)[["feature", "b_star", "I_corrected_bits", "NMI",
                                 "perm_p"]].to_dict("records"))
    json.dump(summ, open("artifacts/mi_summary.json", "w"), indent=2, default=str)
    print(f"[step4] usable_bits.csv written. total usable bits (optimistic sum) = "
          f"{total:.3f}; top: {ub.head(5)['feature'].tolist()}; "
          f"perm-sig features={n_sig}/{len(feats)}; {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
