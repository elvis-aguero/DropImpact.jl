#!/usr/bin/env python3
"""
CAS: is the continued-fraction SEED truncation justified, and does the recurrence forgive it?

src/reid.jl evaluates Q_l(q) = j_{l+1}(q)/j_l(q) by downward recurrence

    Q_{n-1} = 1 / ( (2n+1)/q - Q_n ),        seeded with   Q_{n0} ~ q/(2*n0+3).

Two things must be established, not assumed:
  (A) how big is the seed's dropped remainder, and
  (B) how fast does the downward recurrence contract a seed error,
so that (B) can be shown to dominate (A).

Run: python3 derivations/cas_bessel_ratio_seed.py
"""
import sympy as sp

q, n = sp.symbols('q n', positive=True)

# ---------------------------------------------------------------------------
# (A) Small-argument expansion of the exact ratio.
#     j_n(q) = q^n/(2n+1)!! * [1 - q^2/(2(2n+3)) + q^4/(8(2n+3)(2n+5)) - ...]
# ---------------------------------------------------------------------------
def jseries(order, z, k=3):
    """bracketed series factor of j_order(z), k terms"""
    terms = [sp.Integer(1)]
    if k > 1:
        terms.append(-z**2/(2*(2*order+3)))
    if k > 2:
        terms.append(z**4/(8*(2*order+3)*(2*order+5)))
    return sum(terms)

ratio_prefactor = q/(2*n+3)                      # from q^{n+1}/(2n+3)!! over q^n/(2n+1)!!
exact_bracket = sp.series(jseries(n+1, q)/jseries(n, q), q, 0, 5).removeO()
Q_exact = sp.simplify(ratio_prefactor*sp.expand(exact_bracket))
print("(A) Exact ratio, small-q:")
print("    Q_n =", sp.simplify(sp.expand(Q_exact)))
rel_rem = sp.simplify(sp.expand(Q_exact/ratio_prefactor - 1))
lead = sp.simplify(sp.series(rel_rem, q, 0, 3).removeO())
print("    seed Q_n ~ q/(2n+3) has RELATIVE remainder:", lead)
print("    i.e. O(q^2/n^2) -- NOT small when n0 is comparable to |q|, so the seed alone")
print("    is not enough; the contraction below is what carries the accuracy.\n")

# ---------------------------------------------------------------------------
# (B) Error contraction of the downward recurrence.
#     Q_{n-1} = f(Q_n) = 1/((2n+1)/q - Q_n)  =>  dQ_{n-1}/dQ_n = Q_{n-1}^2
# ---------------------------------------------------------------------------
Qn = sp.symbols('Q_n')
f = 1/((2*n+1)/q - Qn)
df = sp.simplify(sp.diff(f, Qn))
print("(B) Contraction:")
print("    dQ_{n-1}/dQ_n =", df, "= Q_{n-1}^2")
assert sp.simplify(df - f**2) == 0

# absolute error e_{n-1} = Q_{n-1}^2 e_n ; relative E = e/Q gives
#   E_{n-1} = e_{n-1}/Q_{n-1} = Q_{n-1} e_n = Q_{n-1} Q_n E_n
print("    => absolute:  e_{n-1} = Q_{n-1}^2 * e_n")
print("    => RELATIVE:  E_{n-1} = Q_{n-1} * Q_n * E_n")
# with Q_n ~ q/(2n+3) for 2n >> |q|:
factor = sp.simplify((q/(2*n+1))*(q/(2*n+3)))
print("    and with Q_n ~ q/(2n+3), the per-step relative factor is",
      sp.simplify(factor), "~ (q/2n)^2")
print("    So each downward step SHRINKS the relative seed error by ~(q/2n)^2, which is")
print("    << 1 precisely when 2n >> |q|. The seed's O(q^2/n0^2) remainder is therefore")
print("    annihilated provided n0 is taken well above both l and |q| -- which is the")
print("    Miller/continued-fraction premise, and is what src/reid.jl's")
print("    n0 = l + max(60, l/2 + |q|) + |q| enforces.\n")

# ---------------------------------------------------------------------------
# Cumulative bound: product of per-step factors from n0 down to l+1.
# ---------------------------------------------------------------------------
n0, L = sp.symbols('n_0 l', positive=True)
print("(C) Cumulative relative-error factor from n0 down to l+1 is")
print("    prod_{n=l+1}^{n0} (q/2n)^2, bounded above by (q/(2(l+1)))^(2(n0-l))")
print("    -- geometric in the number of steps, so accuracy is exponential in the padding.")
print()
print("CONCLUSION: the seed truncation is justified NOT because the dropped term is small")
print("(it is O(q^2/n0^2), which can be O(1)) but because the recurrence is contracting.")
print("The claim to TEST in source is therefore pad-independence, not seed accuracy.")
print()
print("ALL ASSERTIONS PASSED")
