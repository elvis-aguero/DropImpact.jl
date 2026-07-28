# CAS verification for the shifted-Legendre-in-psi pressure basis (design doc
# §subsubsec:pressure-representation).  The basis is piecewise, exactly zero outside
# [x_c,1] by construction of its support -- which, per the design doc's revised
# §subsubsec:pressure-representation, is adopted because the outer nesting makes it
# free, NOT because a weak alternative was disqualified.
#
# Two things must hold for the shifted-Legendre basis to preserve every exactness claim
# previously made for the monomial-in-(x-x_c) basis it replaced:
#   (1) P~_n(psi(x)) := P_n(2*psi(x)-1), psi(x)=(x-x_c)/(1-x_c), is EXACTLY a
#       degree-n polynomial in x (not merely "polynomial-like") -- so the pressure
#       field p(x,tau)=sum_n chat_n P~_n(psi(x)) has exactly the same total degree N
#       as the old monomial basis, for any x_c.
#   (2) Given (1), every projection integral that was "closed form via Bonnet
#       recursion" under the monomial basis (b_l, the droplet-Galerkin term, the
#       COM force) is STILL a polynomial-times-polynomial integral of the SAME total
#       degree as before -- so it remains exactly (not approximately) evaluable by
#       Gauss-Legendre quadrature with enough nodes, via the same min_nq-style degree
#       bookkeeping already used for the COM force (min_nq_for_exact_com), extended to
#       b_l and the droplet-Galerkin term, which the monomial basis got "for free"
#       from the Bonnet recursion instead.
#
# Both must hold at the PRODUCTION truncation N >= 50, which the design doc's rank law
# (eq:rank-law) pins to the bath truncation and contact radius rather than leaving free.
# That requirement is what dictates the implementation below.
#
# WHY NOT Symbolics.jl: earlier revisions of this script used Symbolics, and its
# `expand`/polyform path converts coefficients to Rational{Int64} internally, which
# OVERFLOWS on this recurrence at n >= 16 -- verified, not assumed (OverflowError in
# DynamicPolynomials' removedups_to! via SymbolicUtils.to_poly!).  Clearing the
# (1-xc)^n denominator inside the recurrence, as an earlier revision did, postpones the
# overflow but does not remove it, because the coefficients themselves grow.  Since the
# only thing being verified is a DEGREE identity for a bivariate polynomial with
# rational coefficients, this script instead carries exact bivariate polynomial
# arithmetic over Rational{BigInt} directly: no truncation, no floating-point
# comparison, no overflow at any n, and considerably faster than the CAS path.

# ---------------------------------------------------------------------------
# Exact bivariate polynomial arithmetic over Q[x, xc].
# A polynomial is a Matrix{Rational{BigInt}} with P[i+1, j+1] = coefficient of x^i xc^j.
# ---------------------------------------------------------------------------
const BiPoly = Matrix{Rational{BigInt}}

function bp(dx::Int, dxc::Int)
    return zeros(Rational{BigInt}, dx + 1, dxc + 1)
end

"""Constant polynomial."""
function bp_const(c)
    P = bp(0, 0); P[1, 1] = Rational{BigInt}(c); return P
end

"""Pad `P` so it has at least `dx+1` rows and `dxc+1` columns."""
function bp_pad(P::BiPoly, dx::Int, dxc::Int)
    Q = bp(max(size(P, 1) - 1, dx), max(size(P, 2) - 1, dxc))
    Q[1:size(P, 1), 1:size(P, 2)] .= P
    return Q
end

function bp_add(A::BiPoly, B::BiPoly)
    dx = max(size(A, 1), size(B, 1)) - 1
    dxc = max(size(A, 2), size(B, 2)) - 1
    return bp_pad(A, dx, dxc) .+ bp_pad(B, dx, dxc)
end

bp_scale(A::BiPoly, c) = A .* Rational{BigInt}(c)

function bp_mul(A::BiPoly, B::BiPoly)
    C = bp(size(A, 1) + size(B, 1) - 2, size(A, 2) + size(B, 2) - 2)
    for i in axes(A, 1), j in axes(A, 2)
        iszero(A[i, j]) && continue
        for k in axes(B, 1), l in axes(B, 2)
            iszero(B[k, l]) && continue
            C[i+k-1, j+l-1] += A[i, j] * B[k, l]
        end
    end
    return C
end

"""Exact degree in x: the largest i such that some coefficient of x^i is nonzero."""
function degree_in_x(P::BiPoly)
    for i in reverse(axes(P, 1))
        any(!iszero, @view P[i, :]) && return i - 1
    end
    return -1
end

"""Exact degree in x of a UNIVARIATE polynomial given as a coefficient vector."""
function degree_in_x(v::Vector{Rational{BigInt}})
    for i in reverse(eachindex(v))
        !iszero(v[i]) && return i - 1
    end
    return -1
end

# The two generators appearing in the recurrence, as exact bivariate polynomials.
const TWO_X_MINUS_XC_MINUS_1 = let P = bp(1, 1)
    P[2, 1] = 2       # 2x
    P[1, 2] = -1      # -xc
    P[1, 1] = -1      # -1
    P
end
const ONE_MINUS_XC_SQ = let P = bp(0, 2)
    P[1, 1] = 1; P[1, 2] = -2; P[1, 3] = 1   # (1 - xc)^2
    P
end

"""
Q_n(x,xc) := (1-xc)^n * P_n(2*psi(x)-1), psi(x)=(x-xc)/(1-xc) -- the shifted-Legendre
basis function with the (1-xc)^n denominator cleared IN THE RECURRENCE ITSELF, so no
fraction ever needs cancelling afterward.  Derived by substituting u=2*psi-1 =
(2x-xc-1)/(1-xc) into the standard three-term Legendre recurrence
(l+1)P_{l+1}(u)=(2l+1)u P_l(u)-l P_{l-1}(u) and multiplying through by (1-xc)^{l+1}:

    (l+1) Q_{l+1} = (2l+1)(2x-xc-1) Q_l - l (1-xc)^2 Q_{l-1}

Note (1-xc)*u = 2x-xc-1 is itself already a polynomial, so the recursion divides only
by (l+1).  Clearing the denominator does not change the degree in x, since (1-xc)^n is
independent of x: deg_x Q_n = deg_x P~_n.
"""
function Qn(n::Int)
    n == 0 && return bp_const(1)
    Qm1 = bp_const(1)
    Q = copy(TWO_X_MINUS_XC_MINUS_1)
    for l in 1:(n-1)
        Qnext = bp_scale(
            bp_add(bp_scale(bp_mul(TWO_X_MINUS_XC_MINUS_1, Q), 2l + 1),
                   bp_scale(bp_mul(ONE_MINUS_XC_SQ, Qm1), -l)),
            1 // (l + 1))
        Qm1, Q = Q, Qnext
    end
    return Q
end

"""Standard (unshifted) Legendre polynomial P_n(x) as an exact coefficient vector."""
function legendre_coeffs(n::Int)
    n == 0 && return Rational{BigInt}[1]
    Pm1 = Rational{BigInt}[1]
    P = Rational{BigInt}[0, 1]
    for l in 1:(n-1)
        shifted = vcat(Rational{BigInt}[0], P)                      # x * P_l
        nxt = zeros(Rational{BigInt}, max(length(shifted), length(Pm1)))
        for i in eachindex(shifted); nxt[i] += (2l + 1) * shifted[i]; end
        for i in eachindex(Pm1);     nxt[i] -= l * Pm1[i];           end
        Pm1, P = P, nxt .// (l + 1)
    end
    return P
end

"""Embed a univariate polynomial in x as a bivariate polynomial in (x, xc)."""
function bp_from_x(v::Vector{Rational{BigInt}})
    P = bp(length(v) - 1, 0)
    for i in eachindex(v); P[i, 1] = v[i]; end
    return P
end

# ---------------------------------------------------------------------------
# (1) P~_n(psi(x)) is exactly degree n in x, for every n and every xc.
# ---------------------------------------------------------------------------
println("=== (1) P~_n(psi(x)) is exactly degree n in x, for every n and every xc ===")
ok1 = true
for n in vcat(0:8, [16, 25, 40, 50, 60, 80])
    d = degree_in_x(Qn(n))
    println("n=$n: degree in x = $d", d == n ? "" : "   <-- MISMATCH, expected $n")
    d == n || global ok1 = false
end
@assert ok1 "P~_n(psi(x)) failed to be exactly degree n in x for some n -- the basis swap would invalidate the degree bookkeeping below"
println("PASS: P~_n(psi(x)) is exactly degree n in x through n=80, spanning and")
println("      exceeding the production truncation N >= 50 (design doc eq:rank-law).")
println("      Exact rational arithmetic throughout -- no floating-point degree test,")
println("      no coefficient overflow at any n.")
println()

# ---------------------------------------------------------------------------
# (2) Total-degree bookkeeping for the polynomial projections, exactly, at
# production N and L.  Unlike the Symbolics version this replaces, L need not be
# kept artificially small: there is no overflow to avoid.
# ---------------------------------------------------------------------------
println("=== (2) Total polynomial degree of each projection integrand ===")
ok2 = true
for (N, L) in ((3, 2), (50, 50), (60, 80))
    PN = Qn(N)                                  # deg_x = N, same as P~_N
    PL = bp_from_x(legendre_coeffs(L))          # deg_x = L

    # b_l integrand: P~_N(psi(x)) * P_L(x), expected degree N + L
    d_bl = degree_in_x(bp_mul(PN, PL))

    # droplet-Galerkin-term integrand: x * (1 + beta_L P_L(x)) * P~_N(psi(x)),
    # whose top-degree part is x * P_L(x) * P~_N, expected degree N + L + 1
    xpoly = bp(1, 0); xpoly[2, 1] = 1
    d_drop = degree_in_x(bp_mul(xpoly, bp_mul(PN, bp_add(bp_const(1), PL))))

    # COM force integrand: p(x) * d[r^2]/dx with r^2 = xi^2 (1-x^2), xi of degree L,
    # so r^2 has degree 2L+2 and its derivative degree 2L+1: expected N + 2L + 1
    r2 = bp_mul(bp_add(bp_const(1), PL), bp_add(bp_const(1), PL))          # xi^2, deg 2L
    onemx2 = bp(2, 0); onemx2[1, 1] = 1; onemx2[3, 1] = -1                 # 1 - x^2
    r2 = bp_mul(r2, onemx2)                                                 # deg 2L+2
    dr2 = bp(max(size(r2, 1) - 2, 0), size(r2, 2) - 1)                      # d/dx
    for i in 2:size(r2, 1), j in axes(r2, 2)
        dr2[i-1, j] += (i - 1) * r2[i, j]
    end
    d_com = degree_in_x(bp_mul(PN, dr2))

    println("  N=$N, L=$L:")
    println("    b_l integrand              degree in x = $d_bl   (expect N+L     = $(N+L))")
    println("    droplet-Galerkin integrand degree in x = $d_drop   (expect N+L+1   = $(N+L+1))")
    println("    COM-force integrand        degree in x = $d_com   (expect N+2L+1  = $(N+2L+1))")
    (d_bl == N + L && d_drop == N + L + 1 && d_com == N + 2L + 1) || global ok2 = false
end
@assert ok2 "projection integrand degrees do not match the design doc's bookkeeping"
println()
println("CONFIRMED, exactly and at production truncation: with the shifted-Legendre-in-psi")
println("basis, b_l's integrand has degree N+L, the droplet-Galerkin term's N+L+1, and the")
println("COM force's N+2L+1 -- IDENTICAL to the total degrees under the old monomial-in-")
println("(x-xc) basis, since only the linear combination of monomials differs and not the")
println("total degree.  All three therefore remain EXACTLY evaluable by Gauss-Legendre")
println("quadrature with nq >= ceil((degree+1)/2) nodes, so dropping the Bonnet-recursion")
println("closed form in favour of uniform quadrature introduces no approximation.")
