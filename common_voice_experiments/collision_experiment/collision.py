"""
collision.py -- the paper's four collision metrics, implemented with the exact
formulae from the experiment brief / Singh & Raj (2025):

  P(E) = 1 - (1 - 1/m)^(n-1)                       (Eq 1, exact match)
  S    = ceil( log(1-p) / log(1 - 1/m) )           (Eq 6, match-with-p)
  P(M) = 1/m                                       (Eq 9, pair match)
  P(B) = 1 - prod_{i=1}^{n-1}(1 - i/m)             (Eq 14, population/birthday match)

m = q^d  (d = #features under full independence, or d_eff).

Numerics:
- P(E), S use log1p/expm1 for accuracy at m >> n.
- P(B) is evaluated in log space.  Looping n=1e10 terms is infeasible and the
  log-gamma closed form cancels catastrophically at large m, so we use the exact
  power-series of the log-product,
        log P(Bbar) = - sum_{k>=1} S_k / (k m^k),  S_k = sum_{i=1}^{n-1} i^k,
  with closed-form power sums S_1..S_4.  In every regime where the result is not
  saturated, n/m is tiny and the leading term -n(n-1)/(2m) dominates (the paper's
  Eq 19); when m <= n-1 collisions are certain by pigeonhole so P(B)=1.
"""
import math

N_DEFAULT = 1e10
P_DEFAULT = 1e-9


def _pop_match(m, n=N_DEFAULT):
    N = n - 1.0
    if m <= N:                       # pigeonhole: more people than cells
        return 1.0
    Ni = int(round(N))
    S1 = Ni * (Ni + 1) // 2
    S2 = Ni * (Ni + 1) * (2 * Ni + 1) // 6
    S3 = (Ni * (Ni + 1) // 2) ** 2
    S4 = Ni * (Ni + 1) * (2 * Ni + 1) * (3 * Ni * Ni + 3 * Ni - 1) // 30
    Ss = [S1, S2, S3, S4]
    logPbar = 0.0
    for k, Sk in enumerate(Ss, start=1):
        try:
            term = Sk / (k * (m ** k))
        except OverflowError:
            term = 0.0
        logPbar -= term
        if logPbar < -745:           # exp underflows -> P(B)=1
            return 1.0
        if abs(term) < 1e-18:
            break
    return min(max(-math.expm1(logPbar), 0.0), 1.0)


def collision_metrics(m, n=N_DEFAULT, p=P_DEFAULT):
    """Return dict with PE, NE, S, PM, NM, PB for a given cell count m."""
    if m <= 1:
        return dict(m=m, PE=1.0, NE=1.0, S=1, PM=1.0, NM=1.0, PB=1.0)
    N = n - 1.0
    log1m = math.log1p(-1.0 / m)             # log(1 - 1/m)
    PE = -math.expm1(N * log1m)              # 1 - (1-1/m)^(n-1)
    PM = 1.0 / m
    S = math.ceil(math.log1p(-p) / log1m)    # ceil(log(1-p)/log(1-1/m))
    PB = _pop_match(m, n)
    NE = (1.0 / PE) if PE > 0 else float("inf")
    NM = 1.0 / PM
    return dict(m=m, PE=PE, NE=NE, S=S, PM=PM, NM=NM, PB=PB)


def m_from(q, d):
    """Cell count m = q^d (d may be non-integer for d_eff)."""
    return float(q) ** float(d)


if __name__ == "__main__":
    # sanity: reproduce the paper's Table 1 at d=41, n=1e10
    print("q   P(E)        S           P(M)        P(B)")
    for q in [10, 8, 5, 4, 3, 2]:
        r = collision_metrics(m_from(q, 41))
        print(f"{q:<3d} {r['PE']:.2e}  {r['S']:.2e}  {r['PM']:.2e}  {r['PB']:.2e}")
