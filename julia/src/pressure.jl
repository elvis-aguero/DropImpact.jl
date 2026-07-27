# The pressure representation and every closed-form projection built from it: b_l
# (eq:b_l-closed), the droplet contribution to the Galerkin residual
# (eq:drop-galerkin-term), and the centre-of-mass force (eq:com). All generic in
# `eltype(chat)`/`eltype(xc)` so ForwardDiff.Dual flows through.
#
# NUMERICAL DEVIATION FROM THE LITERAL DESIGN-DOC BASIS (eq:pressure-poly), recorded
# here in detail because it materially changes what `chat`'s entries mean everywhere
# in this codebase: the doc represents pressure as `Σ ĉ_n (x-xc)^n`. Implemented
# literally, this is catastrophically ill-conditioned whenever the contact patch is
# small (`1-xc` small) — empirically confirmed here at `cond(Jacobian) ~ 1e22` for a
# perfectly ordinary N=3 case with a modest contact angle, since successive powers of
# `(x-xc)` (which never exceeds `1-xc` in magnitude on `[xc,1]`) shrink geometrically
# with `n`, making the Galerkin system's sensitivity to higher-order coefficients
# vanish relative to lower-order ones. This is the exact "Vandermonde-like
# ill-conditioning for larger N" risk the design doc's own audit trail flagged as
# unresolved (§"the closed-form projections" residual items) — it turns out to bind
# at ANY N once the contact patch is small, not only at large N.
#
# Fix: represent pressure in the rescaled variable `ψ := (x-xc)/(1-xc) ∈ [0,1]`,
# `p(x,τ) = Σ_n ĉ_n ψ(x)^n`, rather than directly in `(x-xc)^n`. This is the SAME
# polynomial family and the SAME physical pressure field — just a diagonal
# reparametrization `ĉ_n^{new} = ĉ_n^{old} (1-xc)^n` of the unknowns — but it removes
# the geometric decay of successive basis functions' magnitude on `[xc,1]`, since `ψ`
# always ranges over the fixed interval `[0,1]` regardless of the physical contact
# patch width. All closed-form projections below are stated directly in terms of
# `ψ`; the underlying Bonnet-recursion moment matrix for `(x-xc)^n` (`moment_matrix`)
# is unchanged and simply rescaled by `1/(1-xc)^n` per column.

"""
    pressure_poly_raw(chat, xc, x)

`p(x) = Σ_n chat[n+1] ψ(x)^n`, `ψ(x):=(x-xc)/(1-xc)`, assuming `x ≥ xc` already (no
branch) — used inside quadrature/moment integrands where `x` is known by construction
to lie in `[xc,1]`. See module note for why `ψ` is used rather than `(x-xc)` directly.
"""
function pressure_poly_raw(chat::AbstractVector, xc, x)
    psi = (x - xc) / (1 - xc)
    s = zero(promote_type(eltype(chat), typeof(psi)))
    p = one(psi)
    for n in eachindex(chat)
        s += chat[n] * p
        p *= psi
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
    moment_matrix(xc, lmax, nmax) -> Matrix

`M[l+1,n+1] = ∫_{xc}^{1} (x-xc)^n P_l(x) dx`, for `l=0..lmax`, `n=0..nmax`
(design doc eq:b_l-closed), via the Bonnet-recursion antiderivatives `bonnet_H` and
the binomial expansion `(x-xc)^n = Σ_j C(n,j) x^j (-xc)^{n-j}`. Rescaled to the `ψ`
basis by callers (`b_l_all`, `drop_galerkin_term`) via `moment_matrix(...) ./ (1-xc).^n`
— kept separate here since this raw form is the one directly tied to `bonnet_H`.
"""
function moment_matrix(xc::T, lmax::Integer, nmax::Integer) where {T}
    H1 = bonnet_H(lmax, nmax, one(T))
    Hxc = bonnet_H(lmax, nmax, xc)
    M = Matrix{T}(undef, lmax + 1, nmax + 1)
    for l in 0:lmax
        for n in 0:nmax
            s = zero(T)
            for j in 0:n
                s += binomial(n, j) * (-xc)^(n - j) * (H1[l+1, j+1] - Hxc[l+1, j+1])
            end
            M[l+1, n+1] = s
        end
    end
    return M
end

"""`∫_{xc}^1 ψ(x)^n P_l(x) dx = M[l,n] / (1-xc)^n`, `l=0..lmax`, `n=0..nmax`."""
function moment_matrix_psi(xc::T, lmax::Integer, nmax::Integer) where {T}
    M = moment_matrix(xc, lmax, nmax)
    w = 1 - xc
    wpow = one(T)
    for n in 0:nmax
        if n > 0
            wpow *= w
        end
        @views M[:, n+1] ./= wpow
    end
    return M
end

"""
    b_l_all(chat, xc, L) -> Vector

`b_l(τ)` (design doc eq:b_l-def) for `l=0..L` (entries `l=0,1` are formally zero
since `chat`'s moments against `P_0,P_1` are never used physically, but are computed
here anyway for a uniform 1-indexed vector matching `DropModeState`'s convention).
"""
function b_l_all(chat::AbstractVector{T}, xc, L::Integer) where {T}
    N = length(chat) - 1
    M = moment_matrix_psi(xc, L, N)
    return M * chat
end

"""
    drop_galerkin_term(chat, xc, beta, L) -> Vector

`∫_{xc}^1 x[1+Σ_{l=2}^L β_l P_l(x)]ψ(x)^n dx` for `n=0..N` (design doc
eq:drop-galerkin-term, in the `ψ` basis), the droplet contribution to the Galerkin
residual (eq:galerkin/eq:newton-residual-galerkin). Uses
`xψ^n=(1-xc)ψ^{n+1}+xcψ^n` (since `x=(1-xc)ψ+xc`) to reduce to the same moment
matrix, one degree higher (`nmax=N+1`).
"""
function drop_galerkin_term(chat::AbstractVector, xc, beta::AbstractVector, L::Integer)
    N = length(chat) - 1
    M = moment_matrix_psi(xc, L, N + 1)  # M[l+1,n+1] for n=0..N+1, in the ψ basis
    w = 1 - xc
    out = Vector{promote_type(eltype(chat), eltype(beta), typeof(xc))}(undef, N + 1)
    for n in 0:N
        # bare "x*1" contribution: ∫[(1-xc)ψ^{n+1}+xc ψ^n] dx, dx=(1-xc)dψ, ψ:0->1
        bare = w * (w / (n + 2)) + xc * (w / (n + 1))
        s = bare
        for l in 2:L
            s += beta[l+1] * (w * M[l+1, n+2] + xc * M[l+1, n+1])
        end
        out[n+1] = s
    end
    return out
end

"""
    com_force_closed(chat, xc, beta, params) -> value

`f(τ) = 2∫_0^{r_c} p(r,τ) r\\,dr = -∫_{xc}^1 p(x,τ)\\,d[r(x,τ)²]` (design doc eq:com),
with `r(x,τ)² = ξ(x,τ)²(1-x²)` a polynomial in `x` (the `√(1-x²)` in `sinθ` cancels once
squared). Evaluated by Gauss-Legendre quadrature at `params.com_nodes/com_weights`,
sized (via `min_nq_for_exact_com`) to be EXACT for this integrand's polynomial degree,
not merely approximate — unlike the genuinely transcendental `c_m`/`W_n^{(m)}`
projections in bessel_moments.jl.

SIGN, fixed here after being missed in both an earlier version of this function AND the
design doc's own eq:com derivation (its literal statement `2∫_0^{rc}pr\\,dr=∫_{xc}^1 p\\,
d[r²]` omits a minus sign): near the pole `r` INCREASES as `x=cosθ` DECREASES (θ=0 ↔
x=1 ↔ r=0; θ=θ_c ↔ x=xc ↔ r=r_c), so substituting `r→x` in `∫_0^{r_c}(·)\\,d[r²]` and
reversing the integration limits from `(0,r_c)`/`(1,xc)` to the natural `(xc,1)`
introduces an overall minus sign. Confirmed numerically for a constant test pressure
`p≡1` against the direct `r`-integral `2∫_0^{r_c}r\\,dr=r_c^2`: the unsigned form
returned exactly `-r_c^2` instead of `+r_c^2`. Getting this wrong makes the pressure
force accelerate the droplet further INTO the bath instead of decelerating it — the
root cause of an observed runaway-penetration trajectory pathology.
"""
function com_force_closed(chat::AbstractVector, xc, beta::AbstractVector, L::Integer,
    com_nodes::Vector{Float64}, com_weights::Vector{Float64})
    dr2_dx(x) = begin
        Ptab = legendre_P_table(L, x)
        dPtab = legendre_dP_table(L, x)
        xi = one(x)
        dxi = zero(x)
        for l in 2:L
            xi += beta[l+1] * Ptab[l+1]
            dxi += beta[l+1] * dPtab[l+1]
        end
        return 2 * xi * dxi * (1 - x^2) - xi^2 * 2x
    end
    integrand(x) = pressure_poly_raw(chat, xc, x) * dr2_dx(x)
    return -gauss_quad(integrand, xc, com_nodes, com_weights)
end
