#!/usr/bin/env python3
"""
CAS for the tangency-based contact-edge selector.

TWO THINGS TO ESTABLISH, in the order they matter:

  (A) WHY THE CURRENT RULE FAILS.  src/timestepper.jl selects
          theta_c = inf{ theta : non-intersection and monotone-r hold }.
      Measured behaviour (derivations/debug_feasibility_collapse.jl): at t = 0.009 the
      non-intersection predicate is FALSE for all theta < 0.0492 and the infimum is a sensible
      0.0492; one step later it is TRUE even at theta = 1e-4, the infimum degenerates to 0, and
      the contact patch never recovers. This section shows the degeneracy is structural, not a
      numerical accident: the infimum rule is insensitive to the MAGNITUDE of the gap and
      therefore carries no information once the constraint stops binding.

  (B) THE ANALYTIC TANGENCY RESIDUAL.  eq:tangency-selector is T = d(C)/d(theta) at theta_c.
      src/residual.jl:149 currently evaluates it by a CENTRED FINITE DIFFERENCE with h = 1e-6.
      That is tolerable for a reported diagnostic and unacceptable for a quantity we intend to
      ROOT-FIND, since differencing a Bessel sum at h = 1e-6 loses ~half the available digits.
      This derives dC/dtheta in closed form and verifies it against symbolic differentiation.

Run: python3 derivations/cas_tangency_residual.py
"""
import sympy as sp

# ---------------------------------------------------------------------------------------------
# (A) The infimum rule degenerates when the constraint goes inactive.
# ---------------------------------------------------------------------------------------------
print("=" * 88)
print("(A) Degeneracy of theta_c = inf{feasible}")
print("=" * 88)

th, tc = sp.symbols('theta theta_c', positive=True)
# Near the contact edge the gap continues as C(theta) ~ a*(theta - theta_c) + O((.)^2), with
# a = dC/dtheta evaluated at the edge. Non-overlap beyond the edge requires C < 0 there, i.e.
# a <= 0. So the feasible set is {theta_c : a(theta_c) <= 0}.
a = sp.Function('a')
print("  feasible set  F = { theta_c : a(theta_c) <= 0 },   a := dC/dtheta at the edge")
print("  current rule  theta_c = inf F")
print("  tangency rule a(theta_c) = 0")
print()
print("  If a(theta_c) < 0 for every theta_c > 0 -- the measured situation at t >= 0.010 --")
print("  then F = (0, theta_max] and inf F = 0 REGARDLESS of the shape or size of a.")
print("  The rule returns the same answer whether the surfaces are barely separated or")
print("  wildly so: it uses only the SIGN of a, never its magnitude.")
print()
# Demonstrate on a concrete a < 0 everywhere: inf F = 0 while argmin|a| is interior.
a_expr = -(tc - sp.Rational(1, 20))**2 - sp.Rational(1, 1000)      # strictly negative
crit = sp.solve(sp.diff(sp.Abs(a_expr).rewrite(sp.Piecewise), tc), tc)
print(f"  example a(theta_c) = {a_expr}   (strictly negative for all theta_c)")
print(f"    inf F        = 0            (degenerate: no information)")
print(f"    argmin |a|   = {sp.nsimplify(sp.Rational(1,20))}         (interior, set by the shape of a)")
assert a_expr.subs(tc, sp.Rational(1, 20)) < 0
print("  => the tangency rule remains informative exactly where the infimum rule collapses.\n")

# ---------------------------------------------------------------------------------------------
# (B) Analytic dC/dtheta.
# ---------------------------------------------------------------------------------------------
print("=" * 88)
print("(B) Closed form for T = dC/dtheta")
print("=" * 88)

theta = sp.symbols('theta')
# Droplet shape and the forward map (eq:eta-xi, eq:forward-map):
#   xi(theta) = 1 + sum_l beta_l P_l(cos theta)
#   r(theta)  = xi sin(theta),      z_d(theta) = -xi cos(theta)
# Bath surface (eq:bath-modes):  eta(r) = sum_m a_m J0(k_m r)
# Gap:  C = eta(r(theta)) - z_cm + z_d(theta)
xi = sp.Function('xi')
eta = sp.Function('eta')
z_cm = sp.symbols('z_cm')

r_expr = xi(theta) * sp.sin(theta)
zd_expr = -xi(theta) * sp.cos(theta)
C = eta(r_expr) - z_cm + zd_expr

T_sym = sp.simplify(sp.diff(C, theta))
print("  symbolic dC/dtheta =")
sp.pprint(T_sym)

# The closed form we intend to implement:
#   T = eta'(r) * (xi' sin + xi cos)  +  (-xi' cos + xi sin)
xip = sp.Derivative(xi(theta), theta)
etap = sp.Subs(sp.Derivative(eta(sp.Symbol('u')), sp.Symbol('u')), sp.Symbol('u'), r_expr)
T_closed = etap * (xip * sp.sin(theta) + xi(theta) * sp.cos(theta)) \
           + (-xip * sp.cos(theta) + xi(theta) * sp.sin(theta))
diff = sp.simplify(sp.expand(T_sym.doit() - T_closed.doit()))
print(f"\n  closed form minus symbolic derivative = {diff}")
assert diff == 0
print("  => closed form VERIFIED against symbolic differentiation.\n")

# Concrete pieces, for the implementation:
#   xi'(theta)  = -sin(theta) * sum_l beta_l P_l'(cos theta)
#   eta'(r)     = -sum_m a_m k_m J1(k_m r)
l, m = sp.symbols('l m', integer=True, positive=True)
beta_l, a_m, k_m = sp.symbols('beta_l a_m k_m')
u = sp.symbols('u')
P = sp.Function('P')
xi_series = 1 + beta_l * P(sp.cos(theta))
print("  per-term derivatives needed by the implementation:")
print(f"    d/dtheta [ beta_l P_l(cos theta) ] = {sp.simplify(sp.diff(beta_l * P(sp.cos(theta)), theta))}")
print("      i.e.  xi'(theta) = -sin(theta) * sum_l beta_l P_l'(cos theta)")
print("    d/dr [ a_m J0(k_m r) ] = -a_m k_m J1(k_m r)")
print("      i.e.  eta'(r) = -sum_m a_m k_m J1(k_m r)")
print()

# ---------------------------------------------------------------------------------------------
# (C) The known degeneracy at the poles, reproduced from the closed form.
# ---------------------------------------------------------------------------------------------
print("=" * 88)
print("(C) T vanishes identically at theta = 0 and theta = pi (eq:tangency-degenerate)")
print("=" * 88)
# The abstract form is NOT enough: with eta left unspecified the CAS correctly refuses, returning
# T(0) = xi(0)*eta'(0) -- honest, since nothing so far says eta'(0) vanishes. It does, for a
# PHYSICAL reason: axisymmetry, via J1(0) = 0. So this section uses the CONCRETE model
# expressions rather than abstract functions, which is also closer to what the code evaluates.
th2 = sp.symbols('theta')
bl, am, km, zc = sp.symbols('beta_l a_m k_m z_cm')
for L_DEG in (2, 3):
    xi_c = 1 + bl * sp.legendre(L_DEG, sp.cos(th2))          # eq:eta-xi
    r_c_ = xi_c * sp.sin(th2)                                 # eq:forward-map
    zd_c = -xi_c * sp.cos(th2)
    eta_c = am * sp.besselj(0, km * r_c_)                     # eq:bath-modes
    C_c = eta_c - zc + zd_c
    T_c = sp.diff(C_c, th2)
    for val, name in ((0, "0"), (sp.pi, "pi")):
        v = sp.simplify(sp.limit(T_c, th2, val))
        print(f"  l = {L_DEG}:  T(theta = {name}) = {v}")
        assert v == 0, f"T did not vanish at theta={name} for l={L_DEG}: {v}"
print("  => vanishes identically at BOTH poles, for any beta_l, a_m, k_m, z_cm.")
print("     This is eq:tangency-degenerate. Two consequences for the implementation:")
print("       * a seed is required at onset (onset_theta_c already provides one);")
print("       * the residual must be root-found AWAY from theta = 0, never bracketed from it.")
print("     Note the CAS refused to grant this from the abstract form: the degeneracy follows")
print("     from physics (J1(0) = 0, axisymmetry), not from the chain rule.\n")

print("ALL ASSERTIONS PASSED")
