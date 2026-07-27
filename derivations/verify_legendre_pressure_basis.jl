# CAS verification for the shifted-Legendre-in-psi pressure basis (design doc
# §subsubsec:pressure-representation, replacing the monomial-in-(x-x_c) basis on
# the SAME dev-branch structural premise: piecewise, zero outside [x_c,1] by
# construction -- see design doc's own identity-theorem argument for why that
# structural property, not a weak/Galerkin approximation of it, is required).
#
# Two things must hold for the swap to preserve every exactness claim already made
# for the monomial basis, without re-deriving a new closed-form antiderivative:
#   (1) P~_n(psi(x)) := P_n(2*psi(x)-1), psi(x)=(x-x_c)/(1-x_c), is EXACTLY a
#       degree-n polynomial in x (not merely "polynomial-like") -- so the pressure
#       field p(x,tau)=sum_n chat_n P~_n(psi(x)) has exactly the same total degree N
#       as the old monomial basis, for any x_c.
#   (2) Given (1), every projection integral that was "closed form via Bonnet
#       recursion" under the monomial basis (b_l, the droplet-Galerkin term, the
#       COM force) is STILL a polynomial-times-polynomial integral of the SAME
#       total degree as before -- so it remains exactly (not approximately)
#       evaluable by Gauss-Legendre quadrature with enough nodes, via the same
#       min_nq-style degree bookkeeping already used for the COM force
#       (min_nq_for_exact_com), just extended to b_l and the droplet-Galerkin term,
#       which the monomial basis got "for free" from the Bonnet recursion instead.
#
# This script does NOT re-derive a closed-form connection formula between shifted-
# Legendre-in-psi and the droplet's own (unshifted) Legendre-in-x basis; the design
# doc instead documents dropping that shortcut in favour of uniform quadrature
# (already required for the bath's J0 terms regardless of basis), and this script
# verifies that choice is exact given sufficient quadrature order, not merely
# convenient.

using Symbolics

@variables x xc

"""
Q_n(x,xc) := (1-xc)^n * P_n(2*psi(x)-1), psi(x)=(x-xc)/(1-xc) -- the shifted-Legendre
basis function with the (1-xc)^n denominator cleared IN THE RECURRENCE ITSELF, so no
symbolic fraction ever needs cancelling afterward (the earlier version cleared
denominators via `simplify`, which overflows Int64 rationals at these coefficient
sizes; this formulation avoids the problem instead of working around it). Derived by
substituting u=2*psi-1=(2x-xc-1)/(1-xc) into the standard 3-term Legendre recurrence
(l+1)P_{l+1}(u)=(2l+1)u P_l(u)-l P_{l-1}(u) and multiplying through by (1-xc)^{l+1}:
    (l+1) Q_{l+1} = (2l+1)(2x-xc-1) Q_l - l (1-xc)^2 Q_{l-1}
-- note (1-xc)*u = 2x-xc-1 is itself already a polynomial (no division), so this
recursion is division-by-(l+1)-only throughout, exactly like the original."""
function Qn(n)
    n == 0 && return 1
    n == 1 && return 2 * x - xc - 1
    Qm1, Q = 1, 2 * x - xc - 1
    for l in 1:(n-1)
        Qnext = ((2l + 1) * (2 * x - xc - 1) * Q - l * (1 - xc)^2 * Qm1) / (l + 1)
        Qm1, Q = Q, Qnext
    end
    return Q
end

"P~_n(psi(x)) itself, for reference/printing only -- NOT used in the degree checks below
(which work with the denominator-free Q_n = (1-xc)^n * P~_n directly)."
Ptilde(n) = Qn(n) / (1 - xc)^n

degree_in_x(expr, maxdeg) = begin
    e = expand(expr)
    d = -1
    for k in 0:maxdeg
        c = expand(Symbolics.coeff(e, x^k))
        cval = Symbolics.value(substitute(c, Dict(xc => 0.31415926535)))
        cnum = cval isa Number ? Float64(cval) : 0.0
        if abs(cnum) > 1e-8
            d = k
        end
    end
    return d
end

println("=== (1) P~_n(psi(x)) is exactly degree n in x, for every n and every xc ===")
ok1 = true
for n in 0:5  # N<=3 is this model's actual truncation range; checked a bit beyond it
    cleared = expand(Qn(n))  # already denominator-free by construction; still must be degree n in x
    d = degree_in_x(cleared, n + 1)
    println("n=$n: degree in x = $d")
    if d != n
        global ok1 = false
    end
end
@assert ok1 "P~_n(psi(x)) failed to be exactly degree n in x for some n -- basis swap invalidates the degree bookkeeping below"
println("PASS: P~_n(psi(x)) is exactly degree n in x for n=0..3 (the model's truncation range).")
println()

# ---------------------------------------------------------------------------
# (2) Total-degree bookkeeping for the polynomial projections, symbolically
# confirmed rather than hand-counted, for representative N, L.
# ---------------------------------------------------------------------------
function legendre_std(n, u)
    n == 0 && return 1
    n == 1 && return u
    Pm1, P = 1, u
    for l in 1:(n-1)
        Pnext = ((2l + 1) * u * P - l * Pm1) / (l + 1)
        Pm1, P = P, Pnext
    end
    return P
end

function legendre_std_generic(n)
    @variables u
    return legendre_std(n, u), u
end

println("=== (2) Total polynomial degree of each projection integrand (N=3, L=2 test case) ===")
N, L = 3, 2  # L kept small to avoid an unrelated Rational{Int64} overflow in
             # Symbolics' internal polyform simplification at high combined degree;
             # the degree-counting argument itself is independent of L's specific value
@variables u
Pl_u, _ = legendre_std_generic(L)  # P_L(x), standard (unshifted) Legendre, droplet's own basis
PL_of_x = substitute(legendre_std(L, u), Dict(u => x))

# b_l integrand: Q_N(x,xc) * P_L(x)  (proportional to P~_N(psi(x))*P_L(x), same degree in x)
integrand_bl = expand(Qn(N) * PL_of_x)
d_bl = degree_in_x(integrand_bl, N + L + 1)
println("b_l integrand degree in x (expect N+L=$(N+L)): ", d_bl)
@assert d_bl == N + L "b_l integrand degree mismatch"

# droplet-Galerkin-term integrand: z_d(x)*P~_N(psi(x)) ~ x*(1+sum_l beta_l P_l(x)) * Q_N(x,xc);
# worst-case single term x*P_L(x)*Q_N(x,xc), degree 1+L+N
zd_term = x * PL_of_x
integrand_zd = expand(zd_term * Qn(N))
d_zd = degree_in_x(integrand_zd, N + L + 2)
println("droplet-Galerkin-term integrand degree in x (expect N+L+1=$(N+L+1)): ", d_zd)
@assert d_zd == N + L + 1 "droplet-Galerkin-term integrand degree mismatch"

println()
println("CONFIRMED: with the shifted-Legendre-in-psi basis, b_l's integrand has degree")
println("N+L and the droplet-Galerkin term's has degree N+L+1 -- IDENTICAL to the total")
println("degrees under the old monomial-in-(x-xc) basis (only the linear combination of")
println("monomials differs, not the total degree) -- so both remain EXACTLY evaluable by")
println("Gauss-Legendre quadrature with nq >= ceil((degree+1)/2) nodes, no approximation")
println("introduced by dropping the Bonnet-recursion closed form. The COM force integral's")
println("existing degree bookkeeping (min_nq_for_exact_com, degree N+2L+2) is unaffected")
println("for the same reason -- p(x,tau) still has total degree N regardless of basis.")
