# The acceleration-level (index-reduced) contact closure (design doc §subsec:accel-galerkin,
# eq:accel-constraint), replacing the position-level Galerkin rows of residual.jl to fix
# the confirmed O(δ²) Jacobian ill-conditioning of the position-level closure (newton.jl's
# module note; design doc §subsec:accel-motivation).
#
# eq:accel-constraint splits C̈(θ,τ) = Π(θ,τ;X) + K(θ,τ) into a pressure-dependent part Π
# (linear in c_m(τ), b_l(τ), f(τ), i.e. in X via the SAME Section-3 quantities computed by
# bessel_moments.jl/pressure.jl) and a state-only part K (built ENTIRELY from the previous
# converged step's τ^k state — a_m^k, ȧ_m^k, β_l^k, β̇_l^k — never from X). Both Π and K were
# mechanically re-derived and self-checked against this exact transcription by
# `derivations/verify_accel_closure.jl` (Symbolics.jl) before being written here — per the
# design doc's own extrapolation rule (§subsec:accel-galerkin), EVERY "outer" occurrence of
# a_m/β_l here (inside K entirely, and inside Π's J0/outer-bracket terms) uses the FROZEN
# τ^k state, never X-dependent a_m(X)/β_l(X); only c_m(τ), b_l(τ), f(τ) themselves carry the
# current-step X-dependence, via their existing Section-3 definitions.
#
# Two coefficient-doubling conventions to note when reading this against affine.jl: the
# drop-mode ODE β_l''=-2·damp·β_l'-ω²β_l+F·b_l (affine.jl's local `damp`,`omega2`) means the
# COEFFICIENT of β̇_l appearing here is `2*damp = 2*Oh*(2l+1)*(l-1)`, not affine.jl's bare
# `damp` — the bath side has no such doubling ambiguity since `gamma=2*Oh*km^2` there already
# IS half the β̇-coefficient by the same convention, so `2*gamma=4*Oh*km^2` is used directly.

"""`ξ_τ(x,τ) = Σ_{l=2}^L β̇_l P_l(x)` — same sum as `xi_of_x` but for the mode VELOCITIES,
with no `+1` (the constant term in ξ vanishes under the τ-derivative)."""
function xi_tau_of_x(betadot::AbstractVector, x, L::Integer)
    P = legendre_P_table(L, x)
    s = zero(promote_type(eltype(betadot), typeof(x)))
    for l in 2:L
        s += betadot[l+1] * P[l+1]
    end
    return s
end

"""`-cosθ + sinθ Σ_m a_m k_m J1(k_m r)` (design doc eq:accel-constraint's repeated bracket),
evaluated at the FROZEN state `(a, beta)` — shared, per the design doc's extrapolation rule,
between K's drop-mode term and Π's b_l(τ) term (both arise from the same ξ_ττ-substitution
step, one from its state-only remainder, one from its b_l(τ) piece).

SIGN CORRECTED 2026-07-27: `C(θ,τ):=η-z_cm+ξcosθ` (`+ξcosθ`, not `-ξcosθ` as an earlier
version of both the design doc and this function had it) — the doc's own coordinate
convention (θ=0 along -ẑ) makes a droplet-surface point's absolute height `z_cm-ξcosθ`,
confirmed independently against `postprocessing.jl`'s `south_pole_height = z_cm - ξ(0)`,
which always used the correct sign; only the contact-condition/accel-closure derivation
had the bug. Re-derived and self-checked via `derivations/verify_accel_closure.jl`; only
the direct `cosθ` term here flips (it comes straight from `ξcosθ`'s own sign in `C`) —
the `sinθ*a*k*J1(...)` term is unaffected (it comes from η's own chain rule through
`r=ξsinθ`, independent of how `ξcosθ` enters `C`)."""
function outer_bracket_of_x(x, a::AbstractVector, beta::AbstractVector, p::Params)
    r = r_of_x(beta, x, p.L)
    s = zero(promote_type(eltype(a), eltype(beta), typeof(x)))
    for m in eachindex(a)
        km = p.k[m]
        s += a[m] * km * besselj1(km * r)
    end
    return -x + sqrt(1 - x^2) * s
end

"""
    K_of_x(x, a, adot, beta, betadot, p) -> value

The state-only part of `C̈` (design doc eq:accel-constraint's `K`), built entirely from the
FROZEN τ^k state `(a,adot,beta,betadot)` — no pressure/`X` dependence anywhere.
"""
function K_of_x(x, a::AbstractVector, adot::AbstractVector, beta::AbstractVector,
    betadot::AbstractVector, p::Params)
    L = p.L
    r = r_of_x(beta, x, L)
    rtau = xi_tau_of_x(betadot, x, L) * sqrt(1 - x^2)

    bath_sum = zero(promote_type(eltype(a), eltype(adot), eltype(beta), eltype(betadot), typeof(x)))
    for m in eachindex(a)
        km = p.k[m]
        th = tanh(km * p.h0)
        J0v = besselj0(km * r)
        J1v = besselj1(km * r)
        term = (-4 * p.Oh * km^2 * adot[m] - (km^2 + p.Bo) * km * th * a[m]) * J0v
        term -= 2 * rtau * adot[m] * km * J1v
        term -= (1 // 2) * rtau^2 * a[m] * km^2 * (J0v - besselj(2, km * r))
        bath_sum += term
    end

    ob = outer_bracket_of_x(x, a, beta, p)
    P = legendre_P_table(L, x)
    drop_sum = zero(promote_type(eltype(beta), eltype(betadot), typeof(x)))
    for l in 2:L
        damp_l = 2 * p.Oh * (2l + 1) * (l - 1)
        omega2_l = l * (l - 1) * (l + 2)
        drop_sum += (damp_l * betadot[l+1] + omega2_l * beta[l+1]) * P[l+1]
    end

    return bath_sum + ob * drop_sum + p.Bo
end

"""
    Pi_of_x(x, cm, bl, f, a_frozen, beta_frozen, p) -> value

The pressure-dependent part of `C̈` (design doc eq:accel-constraint's `Π`): linear in
`cm=c_m(τ)`, `bl=b_l(τ)`, `f=f(τ)` (X-dependent, computed exactly as in Section 3's
`c_m_all`/`b_l_all`/`com_force_closed`), but with every OUTER geometric occurrence
(the `J0(k_m r)` argument and the bracket's own `a_m`,`r`) evaluated at the FROZEN
`(a_frozen, beta_frozen)` state, per the design doc's extrapolation rule.
"""
function Pi_of_x(x, cm::AbstractVector, bl::AbstractVector, f,
    a_frozen::AbstractVector, beta_frozen::AbstractVector, p::Params)
    L = p.L
    r = r_of_x(beta_frozen, x, L)

    T = promote_type(eltype(cm), eltype(bl), typeof(f), typeof(x))
    bath_pi = zero(T)
    for m in eachindex(cm)
        km = p.k[m]
        bath_pi += -2 * km * tanh(km * p.h0) * besselj0(km * r) * cm[m]
    end

    ob = outer_bracket_of_x(x, a_frozen, beta_frozen, p)
    P = legendre_P_table(L, x)
    drop_pi = zero(T)
    for l in 2:L
        drop_pi += (2l + 1) * l * P[l+1] * bl[l+1]
    end

    return bath_pi - (3 // 2) * f + ob * drop_pi
end

"""
    accel_galerkin_term(chat, xc, a_frozen, adot_frozen, beta_frozen, betadot_frozen,
                         cm, bl, f, p) -> Vector

`∫_{xc}^1 (Π(x)+K(x)) ψ(x)^n dx` for `n=0..N` (the accel-level replacement for
Section 3's `Σ_m a_m(X) W_n^{(m)}(X) - z_cm(X) elementary_moment(xc,n) - dg_n(X)`
Galerkin rows), via the same Gauss-Legendre quadrature machinery as `bessel_moments.jl`.
"""
function accel_galerkin_term(chat::AbstractVector, xc, a_frozen::AbstractVector,
    adot_frozen::AbstractVector, beta_frozen::AbstractVector, betadot_frozen::AbstractVector,
    cm::AbstractVector, bl::AbstractVector, f, p::Params)
    N = p.N
    T = promote_type(eltype(chat), typeof(xc), eltype(cm), eltype(bl), typeof(f))
    out = Vector{T}(undef, N + 1)
    w = 1 - xc
    for n in 0:N
        integrand(x) = (Pi_of_x(x, cm, bl, f, a_frozen, beta_frozen, p) +
                         K_of_x(x, a_frozen, adot_frozen, beta_frozen, betadot_frozen, p)) *
                        ((x - xc) / w)^n
        out[n+1] = gauss_quad(integrand, xc, p.gauss_nodes, p.gauss_weights)
    end
    return out
end
