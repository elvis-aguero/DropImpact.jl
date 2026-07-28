# The pressure representation and the projections built from it: b_l (eq:b_l-def),
# the centre-of-mass force (eq:com). All generic in `eltype(chat)`/`eltype(xc)` so
# ForwardDiff.Dual flows through.
#
# BASIS (design doc §subsubsec:pressure-representation, eq:pressure-poly): shifted
# Legendre polynomials in `psi := (x-xc)/(1-xc) ∈ [0,1]`, `P̃_n(psi) := P_n(2*psi-1)`,
# NOT monomials. Monomials-in-psi have a Hilbert-matrix self-Gram (`cond ~ 1.6e4` at
# N=3, independent of xc); the shifted-Legendre basis is exactly self-orthogonal under
# the same plain-L² inner product used everywhere else in this codebase (`cond=2N+1`
# at every N and every xc) — confirmed both symbolically
# (`derivations/verify_legendre_pressure_basis.jl`) and against the full nonlinear
# Newton Jacobian (`derivations/verify_legendre_basis_full_jacobian.jl`, 2-4 orders of
# magnitude improvement at moderate-to-large contact angles; NOT a fix at the smallest
# contact angles, where the ill-conditioning is closer to a genuine identifiability
# limit than basis collinearity — tracked separately, not resolved by this basis).
#
# `P̃_n(psi(x))` is exactly a degree-n polynomial in x for every n (an affine change of
# variable composed with a degree-n polynomial), so every projection below has EXACTLY
# the same total polynomial degree it had under the old monomial basis — only the
# linear combination of monomials differs, not the total degree. This means `b_l` and
# the COM force integral remain exactly (not approximately) evaluable by Gauss-Legendre
# quadrature given enough nodes, same as the transcendental (J0) bath projections in
# bessel_moments.jl — one uniform quadrature mechanism for every projection in this
# document now, replacing the old closed-form Bonnet-recursion shortcut that was only
# available for the monomial basis.

"""
    pressure_poly_raw(chat, xc, x)

`p(x) = Σ_n chat[n+1] P̃_n(psi(x))`, `psi(x) := (x-xc)/(1-xc)`, `P̃_n(psi) := P_n(2psi-1)`
(shifted Legendre), assuming `x ≥ xc` already (no branch) — used inside
quadrature/moment integrands where `x` is known by construction to lie in `[xc,1]`.
"""
function pressure_poly_raw(chat::AbstractVector, xc, x)
    psi = (x - xc) / (1 - xc)
    N = length(chat) - 1
    Ptab = legendre_P_table(N, 2 * psi - 1)
    s = zero(promote_type(eltype(chat), typeof(psi)))
    for n in 0:N
        s += chat[n+1] * Ptab[n+1]
    end
    return s
end

"""
    pressure_poly(chat, xc, x)

Full pressure field: `pressure_poly_raw(chat,xc,x)` for `x ≥ xc`, zero otherwise. For
general evaluation (plotting, the non-intersection check eq:check-nonintersect)
rather than the quadrature/moment integrands above.
"""
function pressure_poly(chat::AbstractVector, xc, x)
    return x >= xc ? pressure_poly_raw(chat, xc, x) : zero(pressure_poly_raw(chat, xc, x))
end

"""
    b_l_all(chat, xc, x, wq, P, w, L) -> Vector

`b_l(τ) = ∫_{xc}^1 p(x,τ) x P_l(x) w(x,τ) dx` (design doc eq:b_l-selfadjoint) for
`l=0..L`, from PRECOMPUTED node data (`geometry.jl`): `x`,`wq` the mapped quadrature
nodes and weights, `P` the Legendre tables there, `w` the area weight.

The `x` and `w` factors are not decoration: without both, the compliance operator mapping
pressure to gap displacement is not self-adjoint in any inner product and the per-step
problem has no energy functional (design doc §subsubsec:compliance). With both, the whole
operator is self-adjoint in `⟨u,v⟩_w = ∫ u v w dx` to machine precision. They constitute
an O(θ_c²) change of linearisation convention relative to AlventosaEtAl2023's
`∫ p P_l dx`, and reduce to it exactly at pole contact, where `x → 1` and `w → 1`.
"""
function b_l_all(chat::AbstractVector, xc, x::AbstractVector, wq::AbstractVector,
    P, w::AbstractVector, L::Integer)
    T = promote_type(eltype(chat), typeof(xc), eltype(w))
    out = zeros(T, L + 1)
    @inbounds for i in eachindex(x)
        pv = pressure_poly_raw(chat, xc, x[i]) * x[i] * w[i] * wq[i]
        for l in 0:L
            out[l+1] += pv * P[i][l+1]
        end
    end
    return out
end

"""
    com_force_closed(chat, xc, x, wq, w) -> value

`f(τ) = 2∫_0^{r_c} p r dr = 2∫_{xc}^1 p(x,τ) w(x,τ) dx` (design doc eq:com), using
`r dr = w dx`.

SIGN: `w = -½d[r²]/dx` is positive over the patch precisely because `r` increases as
`x=cosθ` decreases (θ=0 ↔ x=1 ↔ r=0; θ=θ_c ↔ x=xc ↔ r=r_c), so `f > 0` for positive
pressure -- a decelerating force. Getting this backwards makes the pressure accelerate
the droplet further INTO the bath, the root cause of an observed runaway-penetration
pathology. Note `w > 0` requires design doc eq:check-monotone-r, enforced by the time
stepper's feasibility filter.
"""
function com_force_closed(chat::AbstractVector, xc, x::AbstractVector,
    wq::AbstractVector, w::AbstractVector)
    T = promote_type(eltype(chat), typeof(xc), eltype(w))
    s = zero(T)
    @inbounds for i in eachindex(x)
        s += pressure_poly_raw(chat, xc, x[i]) * w[i] * wq[i]
    end
    return 2 * s
end
