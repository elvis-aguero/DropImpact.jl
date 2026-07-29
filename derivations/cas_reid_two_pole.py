#!/usr/bin/env python3
"""
CAS derivation: mapping a pair of Reid eigenvalues onto the coefficients of the
second-order drop-mode ODE, faithfully in BOTH the underdamped and overdamped cases.

The drop mode obeys (design doc eq:lambda-l, src/affine.jl)

    beta_ddot + 2*lam*beta_dot + om2*beta = F_l * b_l,

so its characteristic polynomial is s^2 + 2*lam*s + om2. Given the two least-damped
Reid eigenvalues we want (lam, om2) such that the ODE reproduces them exactly.

Run: python3 derivations/cas_reid_two_pole.py
"""
import sympy as sp

lam, om2 = sp.symbols('lambda omega2', real=True, positive=True)
s = sp.symbols('s')

char = s**2 + 2*lam*s + om2
print("Characteristic polynomial:", char)
print("Roots:", sp.solve(char, s))
print()

# ---------------------------------------------------------------------------
# Vieta: for monic s^2 + 2*lam*s + om2 with roots s1, s2:
#     s1 + s2 = -2*lam     and     s1*s2 = om2
# So the UNIQUE faithful mapping, valid for any root pair, is
#     lam = -(s1+s2)/2,    om2 = s1*s2.
# ---------------------------------------------------------------------------
s1, s2 = sp.symbols('s1 s2')
lam_map = -(s1 + s2)/2
om2_map = s1*s2
print("Proposed unified mapping:  lam = -(s1+s2)/2,  om2 = s1*s2")
recon = sp.expand((s - s1)*(s - s2))
target = sp.expand(s**2 + 2*lam_map*s + om2_map)
print("  reconstruct (s-s1)(s-s2) - [s^2+2*lam*s+om2] =", sp.simplify(recon - target))
assert sp.simplify(recon - target) == 0
print("  => exact for ANY pair, no case split needed.\n")

# ---------------------------------------------------------------------------
# CASE A: underdamped. Reid returns a conjugate pair sigma = -(g) +- i*w with
# decay rate g = Re, frequency w = |Im|. In our sign convention the eigenvalue is
# a DECAY rate, so the ODE roots are s = -g +- i*w.
# ---------------------------------------------------------------------------
g, w = sp.symbols('gamma omega', real=True, positive=True)
sA1, sA2 = -g + sp.I*w, -g - sp.I*w
lamA = sp.simplify(-(sA1 + sA2)/2)
om2A = sp.simplify(sp.expand(sA1*sA2))
print("CASE A (underdamped, s = -gamma +- i*omega):")
print("   lam  =", lamA)
print("   om2  =", om2A, "   <-- gamma^2 + omega^2, NOT omega^2")
assert sp.simplify(lamA - g) == 0
assert sp.simplify(om2A - (g**2 + w**2)) == 0
# sanity: the ODE's own oscillation frequency must come back out
w_back = sp.sqrt(om2A - lamA**2)
print("   sqrt(om2 - lam^2) =", sp.simplify(w_back), " (recovers omega)")
assert sp.simplify(w_back - w) == 0
print()

# ---------------------------------------------------------------------------
# CASE B: overdamped. Reid's pair has merged onto the real axis: two distinct
# real decay rates g1 < g2, i.e. ODE roots s = -g1, -g2.
# ---------------------------------------------------------------------------
g1, g2 = sp.symbols('gamma_1 gamma_2', real=True, positive=True)
lamB = sp.simplify(-((-g1) + (-g2))/2)
om2B = sp.simplify((-g1)*(-g2))
print("CASE B (overdamped, s = -gamma_1, -gamma_2):")
print("   lam  =", lamB, "   <-- arithmetic mean of the two rates")
print("   om2  =", om2B, "   <-- their product")
assert sp.simplify(lamB - (g1 + g2)/2) == 0
assert sp.simplify(om2B - g1*g2) == 0
# and the ODE is genuinely overdamped there: lam^2 > om2 iff g1 != g2
disc = sp.simplify(lamB**2 - om2B)
print("   lam^2 - om2 =", sp.factor(disc), " >= 0, zero only if the rates coincide")
assert sp.simplify(disc - ((g1 - g2)/2)**2) == 0
print()

# ---------------------------------------------------------------------------
# What the SUPERSEDED critically-damped choice did, and its error.
#   old: lam = g1 (least-damped rate only), om2 = lam^2  -> double root at -g1.
# It reproduces the slow rate and INVENTS the fast one, discarding g2 entirely.
# ---------------------------------------------------------------------------
print("SUPERSEDED choice (om2 = lam^2, double root at the least-damped rate):")
lam_old, om2_old = g1, g1**2
roots_old = sp.solve(s**2 + 2*lam_old*s + om2_old, s)
print("   roots:", roots_old, " -> second rate forced to", -roots_old[0],
      "instead of -gamma_2")
print("   relative error in the fast rate: (gamma_2 - gamma_1)/gamma_2")
print()

# ---------------------------------------------------------------------------
# Degenerate check: as g2 -> g1 the faithful mapping reduces to the old one, so
# the correction is a strict generalisation and nothing is lost at the merge point.
# ---------------------------------------------------------------------------
print("Limit g2 -> g1 of the faithful mapping:")
print("   lam ->", sp.limit(lamB, g2, g1), "   om2 ->", sp.limit(om2B, g2, g1))
assert sp.limit(lamB, g2, g1) == g1 and sp.limit(om2B, g2, g1) == g1**2
print("   == the critically damped case, recovered continuously.\n")

print("ALL ASSERTIONS PASSED")
