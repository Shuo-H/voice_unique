"""
mi_core.py -- information-theoretic estimators (in BITS) for the
voice-individuality experiment.

All quantities are bits (log base 2).  The headline estimator is bias-corrected
mutual information above a permutation null:

    I_raw      plug-in MI  = H(spk) + H(bin) - H(spk,bin)         [upward biased]
    I_mm       Miller-Madow MI: each entropy gets +(K-1)/(2N),
               K = number of OCCUPIED cells, N = sample size.
    I_null     permutation null: shuffle the speaker labels across all N
               utterances, recompute PLUG-IN MI.  Under a label shuffle the
               speaker- and bin-marginals are invariant, so only H(spk,bin)
               changes -> I_null = H(spk)+H(bin)-H(spk,bin)_perm.
    perm_p     fraction of null MI >= I_raw.
    I_corrected = max(0, I_mm - I_null_mean)   [bits above chance]  <-- HEADLINE
    NMI_corrected = I_corrected / log2(S).

Seed 1234 everywhere (caller passes deterministic per-unit seeds so results are
identical regardless of parallel scheduling).
"""
import numpy as np


# ----------------------------------------------------------------- entropy ----
def entropy_bits(counts):
    """Plug-in Shannon entropy (bits) of a count vector; also returns the number
    of occupied cells K (>0 count)."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    if n <= 0:
        return 0.0, 0
    nz = counts[counts > 0]
    p = nz / n
    H = float(-np.sum(p * np.log2(p)))
    return H, int(nz.size)


def joint_entropy_bits(a, b):
    """Plug-in entropy (bits) of the joint distribution of two dense integer
    label arrays a in [0,Sa), b in [0,Sb).  O(N log N), O(N) memory (safe even
    when Sa*Sb is enormous).  Returns (H, K_occupied)."""
    if a.size == 0:
        return 0.0, 0
    Q = int(b.max()) + 1
    key = a.astype(np.int64) * np.int64(Q) + b.astype(np.int64)
    _, counts = np.unique(key, return_counts=True)
    return entropy_bits(counts)


# --------------------------------------------------------------- quantize -----
def quantize(values, q):
    """Equal-frequency (q-quantile) binning of a 1-D finite array into q bins.
    Degenerate/duplicate quantile edges are merged; the effective bin count
    q_eff (<= q) is returned.  Bins are marginally (near-)equiprobable.

    Low-cardinality fallback: if the feature has U <= q distinct values, use one
    bin per distinct value (equal-frequency is impossible there) so that a
    genuinely discrete/low-entropy feature is not collapsed to a single bin and
    its information is preserved.

    The returned labels are always densified (contiguous 0..q_eff-1) and q_eff is
    the number of REALIZED OCCUPIED bins, so q_eff never overstates the achieved
    resolution even when a point mass sitting on a quantile edge empties a bin.

    Returns (labels[int 0..q_eff-1], q_eff, edges)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros(0, dtype=np.int64), 1, np.array([np.nan])
    uniq_vals = np.unique(v)
    if uniq_vals.size <= 1:
        return np.zeros(v.size, dtype=np.int64), 1, uniq_vals
    if uniq_vals.size <= q:
        # one bin per distinct value (max resolution this feature supports)
        labels = np.searchsorted(uniq_vals, v, side="left").astype(np.int64)
        edges = uniq_vals
    else:
        # equal-frequency quantile binning
        edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, q + 1)))
        if len(edges) <= 2:
            labels = np.zeros(v.size, dtype=np.int64)
        else:
            labels = np.searchsorted(edges[1:-1], v, side="right").astype(np.int64)
    # densify -> q_eff = realized occupied bins (faithful to the actual partition)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64), int(labels.max()) + 1, edges


# ------------------------------------------------------------- MI metrics -----
def _null_result(N, q_eff):
    return dict(I_raw=0.0, I_mm=0.0, I_null_mean=0.0, I_null_p95=0.0, perm_p=1.0,
                I_corrected=0.0, I_null_mm_mean=0.0, I_corrected_mmnull=0.0,
                N=int(N), q_eff=int(q_eff), Kxy=0)


def mi_metrics(spk, bins, S, nperm=200, seed=1234):
    """Full MI panel for one (speaker-labels, bin-labels) pair on a balanced
    sample.  spk in [0,S) dense, bins in [0,q_eff) dense, same length N.

    Returns dict: I_raw, I_mm, I_null_mean, I_null_p95, perm_p, I_corrected
    (HEADLINE; plug-in null per spec), I_null_mm_mean + I_corrected_mmnull
    (diagnostic: null recomputed with the SAME Miller-Madow correction as the
    point estimate -> the less-conservative, self-consistent variant), plus
    N, q_eff, Kxy (occupied joint cells)."""
    spk = np.asarray(spk)
    bins = np.asarray(bins)
    N = spk.size
    q_eff = int(bins.max()) + 1 if N else 1
    if N == 0 or q_eff <= 1:               # fully-missing or constant feature -> 0 info
        return _null_result(N, q_eff)

    # marginals (invariant under speaker-label permutation)
    Hx, Kx = entropy_bits(np.bincount(spk, minlength=S))
    Hy, Ky = entropy_bits(np.bincount(bins, minlength=q_eff))
    Hxy, Kxy = joint_entropy_bits(spk, bins)

    I_raw = Hx + Hy - Hxy
    # Miller-Madow: +(K-1)/(2N) per entropy
    Hx_mm = Hx + (Kx - 1) / (2.0 * N)
    Hy_mm = Hy + (Ky - 1) / (2.0 * N)
    Hxy_mm = Hxy + (Kxy - 1) / (2.0 * N)
    I_mm = Hx_mm + Hy_mm - Hxy_mm

    # permutation null (only the joint entropy changes under a speaker-label shuffle)
    rng = np.random.default_rng(seed)
    nulls = np.empty(nperm, dtype=np.float64)      # plug-in null (matches I_raw)
    nulls_mm = np.empty(nperm, dtype=np.float64)   # MM-consistent null (matches I_mm)
    for i in range(nperm):
        perm = rng.permutation(N)
        Hxy_p, Kxy_p = joint_entropy_bits(spk[perm], bins)
        nulls[i] = Hx + Hy - Hxy_p
        nulls_mm[i] = Hx_mm + Hy_mm - (Hxy_p + (Kxy_p - 1) / (2.0 * N))
    I_null_mean = float(nulls.mean())
    I_null_p95 = float(np.percentile(nulls, 95))
    perm_p = float(np.mean(nulls >= I_raw))
    I_null_mm_mean = float(nulls_mm.mean())

    return dict(I_raw=float(I_raw), I_mm=float(I_mm),
                I_null_mean=I_null_mean, I_null_p95=I_null_p95, perm_p=perm_p,
                I_corrected=max(0.0, I_mm - I_null_mean),          # HEADLINE (spec)
                I_null_mm_mean=I_null_mm_mean,
                I_corrected_mmnull=max(0.0, I_mm - I_null_mm_mean),  # diagnostic
                N=int(N), q_eff=int(q_eff), Kxy=int(Kxy))


# ------------------------------------------------------------ self-test -------
if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    S = 200
    reps = 12
    spk = np.repeat(np.arange(S), reps)          # balanced 12/speaker
    N = spk.size

    # (1) INDEPENDENT bins -> I_corrected ~ 0, perm_p large
    bins_indep = rng.integers(0, 4, size=N)
    r1 = mi_metrics(spk, bins_indep, S, nperm=200, seed=1234)
    print("[indep ]  I_raw=%.4f I_mm=%.4f null=%.4f I_corr=%.4f perm_p=%.3f"
          % (r1["I_raw"], r1["I_mm"], r1["I_null_mean"], r1["I_corrected"], r1["perm_p"]))

    # (2) DETERMINISTIC bin = speaker mod 4 -> MI = H(bin) = 2 bits, perm_p=0
    bins_det = spk % 4
    r2 = mi_metrics(spk, bins_det, S, nperm=200, seed=1234)
    print("[det   ]  I_raw=%.4f I_mm=%.4f null=%.4f I_corr=%.4f perm_p=%.3f  (expect ~2 bits)"
          % (r2["I_raw"], r2["I_mm"], r2["I_null_mean"], r2["I_corrected"], r2["perm_p"]))

    # (3) NOISY-but-informative: bin = speaker-group + noise
    grp = (spk // 50)                            # 4 groups
    flip = rng.random(N) < 0.25
    bins_noisy = np.where(flip, rng.integers(0, 4, N), grp)
    r3 = mi_metrics(spk, bins_noisy, S, nperm=200, seed=1234)
    print("[noisy ]  I_raw=%.4f I_mm=%.4f null=%.4f I_corr=%.4f perm_p=%.3f"
          % (r3["I_raw"], r3["I_mm"], r3["I_null_mean"], r3["I_corrected"], r3["perm_p"]))

    # (4) quantize sanity
    vals = rng.normal(size=5000)
    lab, qeff, edg = quantize(vals, 8)
    occ = np.bincount(lab)
    print("[quant ]  q=8 q_eff=%d  bin counts min/max=%d/%d (near-equal => equal-freq OK)"
          % (qeff, occ.min(), occ.max()))
    # degenerate feature
    lab2, qeff2, _ = quantize(np.r_[np.zeros(900), np.ones(100)], 8)
    print("[quant ]  degenerate 2-value feature -> q_eff=%d (expect 2)" % qeff2)
