# Core data types: physical/numerical parameters, per-subsystem state, the two-level
# BDF2 history, and the free-flight/in-contact phase.
#
# Bath and drop mode vectors are stored 1-indexed with a `l+1`/`m+1` offset so that
# `a[m+1] == a_m` and `beta[l+1] == beta_l` directly, matching legendre_P_table's own
# 1-indexed convention (`beta[1]`, `beta[2]` for l=0,1 are always zero/unused, per
# design doc §subsec:drop: l=0 excluded by incompressibility, l=1 governed by the
# centre of mass instead).

@enum Phase FreeFlight InContact

"""
    Params

Physical and numerical parameters for one simulation, fixed for its whole duration.

- `We, Bo, Oh`: Weber, Bond, Ohnesorge numbers (design doc §subsec:formulation).
- `M`: bath modes, m = 0..M.  `L`: drop modes, l = 2..L.  `N`: pressure polynomial degree.
- `b`: bath radius (nondim by droplet radius R).  `h0`: bath depth (nondim by R).
- `nq`: number of Gauss-Legendre quadrature points for the c_m/W_n^{(m)} projections.
- `k`: bath eigenvalues, length M+1, `k[m+1] = k_m` (roots of J1(k_m b)=0, plus k_0=0).
- `j0kb`: `J_0(k_m b)`, length M+1, precomputed once since `k,b` are fixed for the run.
  Used only by `wall=:clamped` (design doc eq:route-b-multiplier); unused otherwise.
- `gauss_nodes, gauss_weights`: precomputed nq-point Gauss-Legendre rule on [-1,1], used
  for the genuinely transcendental `c_m`/`W_n^{(m)}` projections (eq:c_m-Wn), where `nq`
  is a real numerical-accuracy trade-off chosen by the user.
- `com_nodes, com_weights`: a SEPARATE, larger Gauss-Legendre rule sized to be exactly
  (not approximately) accurate for the polynomial COM-force integrand (design doc
  eq:com), whose degree is fully determined by `N,L` rather than a free choice — see
  `min_nq_for_exact_com`.
"""
struct Params
    We::Float64
    Bo::Float64
    Oh::Float64
    M::Int
    L::Int
    N::Int
    b::Float64
    h0::Float64
    nq::Int
    wall::Symbol
    k::Vector{Float64}
    bath_norm::Vector{Float64}
    j0kb::Vector{Float64}
    gauss_nodes::Vector{Float64}
    gauss_weights::Vector{Float64}
    com_nodes::Vector{Float64}
    com_weights::Vector{Float64}
end

"""
    min_nq_for_exact_com(N, L) -> Int

Minimum number of Gauss-Legendre points making the COM-force integral (design doc
eq:com) exact: the integrand `p(x) * d[r(x)²]/dx` has degree `N + 2L + 1` in `x`
(pressure degree `N`, `r² = ξ²(1-x²)` degree `2L+2`), and an `nq`-point rule is exact
for polynomials up to degree `2nq-1`.
"""
min_nq_for_exact_com(N::Integer, L::Integer) = cld(N + 2L + 2, 2)

"""
    resolvable_rank_estimate(; M, L, b, theta_c) -> Float64

Estimate of `n_*`, the number of pressure directions the truncated bath and droplet can
respond to at a contact angle `theta_c` -- the budget that `N + 1` must stay inside.

This is a Slepian-type time--bandwidth count: restricting `span{P_l : l <= L}` to
`theta in [0, theta_c]` leaves a plateau of dimension `O(L*theta_c/pi)` before the spectrum
falls off super-exponentially, with a smaller additive bath contribution `k_M*r_c/pi`. The
mechanism (droplet-dominated, delta-independent, `n_q`-independent) is pinned in CI by
`test/test_rank_law.jl`.

!!! warning "The constants are an empirical fit, not a derivation"
    `4.5` and `1.15` were fitted at `b = 6`, `theta_c <= 0.8`, and the intercept probably
    carries a weak `log L` dependence (Slepian transition width). Use this for provisioning
    and for the warning in [`Params`](@ref); do not treat it as a sharp threshold. The
    measured quantity itself is the singular-value plateau of the assembled operator, which
    `derivations/audit_compliance_operator.jl` (AUDIT 4) computes directly.
"""
function resolvable_rank_estimate(; M::Integer, L::Integer, b::Real, theta_c::Real)
    k_M = last(bessel_zeros_J1(M)) / b
    r_c = sin(theta_c)
    return 4.5 + 1.15 * L * theta_c / pi + k_M * r_c / pi
end

# Defaults for the NUMERICAL parameters, so that callers need supply only physics.
#
# These are not round numbers picked for looks; each follows from a measured cost/accuracy
# asymmetry (walltime figures at nq=200, N=3, per contact step, this machine):
#
#   L = 120 -- the droplet truncation is very nearly FREE (measured 1.03x and 1.14x per
#       doubling, 575 -> 592 -> 673 ms/step for L = 30 -> 60 -> 120) and it is the ONLY
#       knob that buys pressure resolution, since n_* is droplet-dominated. So it is set
#       generously: at L = 120 the budget admits N + 1 <= ~9 even at a tight theta_c = 0.1,
#       which keeps the default N safe through contact onset, where n_* is smallest.
#
#   M = 60 -- the bath truncation costs LINEARLY (1.76x, 1.98x per doubling; 340 -> 597 ->
#       1183 ms/step) and buys little rank, so it is set by accuracy alone. The M sweep
#       converges by ~40 when L is held generous; 60 leaves margin.
#
#   N = 3 -- inside the budget at every contact angle reached with L = 120, and in the cheap
#       regime: N is ~free up to the budget and explodes past it (measured 6.1x walltime at
#       N = 12 versus N = 3, together with 90% of contact steps carrying negative pressure).
#
#   nq = 200 -- also linear in cost (1.95x, 1.99x per doubling). 200 comfortably exceeds
#       `min_nq_for_exact_com(3, 120) = 123`, so the COM-force integrand is integrated
#       exactly, with margin for the transcendental (J_0) bath projections.
const DEFAULT_M = 60
const DEFAULT_L = 120
const DEFAULT_N = 3
const DEFAULT_NQ = 200

"""
    Params(; We, Bo, Oh, b, h0, wall=:free, M, L, N, nq, check_budget=true)

Only the PHYSICAL inputs are required: the Weber, Bond and Ohnesorge numbers and the bath
geometry `b, h0`. The numerical truncations `M, L, N, nq` default to
`$(DEFAULT_M), $(DEFAULT_L), $(DEFAULT_N), $(DEFAULT_NQ)`, chosen from the measured cost and
resolvable-rank asymmetries documented beside `DEFAULT_M` in this file: the droplet
truncation `L` is nearly free and is the only knob that buys pressure resolution, so it is
set generously; `M` and `nq` cost linearly and are set by accuracy; `N` is kept inside the
budget, where it is also cheap.

Passing `N` outside the budget emits a warning (see [`resolvable_rank_estimate`](@ref));
`check_budget=false` suppresses it. Refining `N` at fixed `M, L` cannot converge -- see
`derivations/DIAGNOSTICS-NOTATION.md`.

`wall` selects the free-surface condition at the container wall `r = b`, which fixes both
the bath eigenvalues and the Fourier-Bessel weight (design doc eq:bessel-norm):

  `:free`   `∂η/∂r = 0` at `r = b`: a 90-degree contact line free to slide. Eigenvalues are
            the zeros of `J_1` (plus the `k_0 = 0` piston mode) and the weight is
            `2/(b J_0(k_m b))²`. This is the configuration of both parent papers and the
            default.

  `:pinned` `η = 0` at `r = b`: the free surface is pinned at the triple point, with its
            wall slope free. Eigenvalues are the zeros of `J_0` (there is no `k_0 = 0`
            mode — a surface pinned on its whole boundary cannot translate uniformly) and
            the weight is `2/(b J_1(k_m b))²`. That is the Dirichlet normalizer whose form
            AlventosaEtAl2023 print for their (free) bath, though not their exact
            expression: they evaluate `J_1` at `k_m` rather than `k_m b`, a second and
            independent error.

!!! warning "`:pinned` does not conserve bath volume. Use it as a diagnostic, not physics."
    A mode displaces volume `∫_0^b J_0(k_m r) r dr = (b/k_m) J_1(k_m b)`. For the `:free`
    eigenvalues `J_1(k_m b) = 0` by definition, so every non-piston mode is volume-neutral
    and the one volume-carrying mode (the piston) has `κ_0 = 0` and can never be driven:
    volume conservation is structural. For `:pinned` every mode carries volume and nothing
    constrains the sum. Measured over the reference impact, `|∫_0^b η r dr|` stays at
    5e-15 for `:free` but reaches 2.79 for `:pinned` — 17.5 R³ of bath volume created from
    nothing, four times the droplet's own 4π/3. For a model built to transfer displaced
    volume into capillary waves that is disqualifying.

    `:pinned` also violates no-flux at leading order (`∂φ/∂r ∝ J_1(k_m b)`, O(1) here),
    and the closure diagnostics of the design doc §subsec:contact — self-adjointness,
    semidefiniteness, conditioning, the tangency root — are all `:free` measurements that
    have NOT been repeated for `:pinned`.

    The consistent formulation instead keeps the `:free` basis and imposes pinning with a
    multiplier (the rim line force), which conserves volume exactly since `κ_0 = 0` — see
    `wall=:clamped` below. Note also that no-flux walls freeze the wall slope to O(Oh) in a
    rigid container, so a time-varying wall slope is not something a consistent model needs
    to represent in the first place. See `derivations/feasibility_pinned_contact_line.jl`
    for the derivations and measurements.

  `:clamped` The `:free` (Neumann) basis and eigenvalues, unchanged, with pinning
             `η(b,τ) = 0` imposed as a scalar constraint carried by a Lagrange multiplier Λ
             — physically the line force the rim exerts on the contact line (design doc
             eq:route-b-multiplier). Route B: conserves volume exactly (same `κ_0 = 0`
             argument as `:free`, since the multiplier only ever multiplies `κ_m`) while
             pinning exactly, at the cost of one `O(M)` dot product per step. Verified at
             operator level in `derivations/feasibility_pinned_contact_line.jl` §5: the
             constraint holds for the assembled response operator to `5e-22`, the rank-one
             correction is self-adjoint in the same weighted pairing the free operator uses
             (`1.1e-17` relative asymmetry), and positive semidefiniteness of `-𝒜` is
             preserved for any admissible step, not just the one measured. Applied at every
             step, including free flight — pinning is a constraint on `a_m` itself, not a
             byproduct of contact pressure. Measured over the reference impact, `:clamped`
             holds `|η(b,τ)|` and `|∫_0^b η r dr|` at roundoff simultaneously (7e-17 and
             2e-14), where `:pinned` traded one for the other. The step-level closure
             diagnostics of §subsec:contact/§subsubsec:compliance — conditioning of the
             assembled nonlinear system, the resolvable pressure dimension, the tangency
             root — remain `:free` measurements not yet repeated for `:clamped` at a real
             contact step, and whether `:free` and `:clamped` differ in contact time or CoR
             beyond step-controller noise at `b=6` has not been measured.
"""
function Params(; We, Bo, Oh, b, h0, wall::Symbol=:free,
                M::Integer=DEFAULT_M, L::Integer=DEFAULT_L, N::Integer=DEFAULT_N,
                nq::Integer=DEFAULT_NQ, check_budget::Bool=true)
    wall in (:free, :pinned, :clamped) ||
        throw(ArgumentError("wall must be :free, :pinned or :clamped, got $wall"))

    # Guard the one failure mode that is silent: N provisioned past the resolvable-rank
    # budget. Past it, the inner solve is ill-conditioned, the reported pointwise pressure
    # goes negative on most contact steps, and walltime rises severalfold -- none of which
    # announces itself in the output. Checked at theta_c = 0.15, a representative tight
    # angle shortly after onset rather than the onset limit itself (where n_* -> its floor
    # and no N would pass).
    if check_budget
        budget = 0.6 * resolvable_rank_estimate(M=M, L=L, b=b, theta_c=0.15)
        if N + 1 > budget
            @warn """
                Pressure order N is outside the resolvable-rank budget.
                N+1 = $(N+1) exceeds 0.6*n_* = $(round(budget, digits=1)) at theta_c = 0.15 \
                (M = $M, L = $L).
                Expect an ill-conditioned inner solve, negative pointwise pressure on most \
                contact steps, and several-fold slower steps. The cheap fix is to RAISE L \
                (nearly free: ~14% per doubling) rather than to lower N, since n_* is \
                droplet-dominated. Suppress with check_budget=false.
                """ N L budget
        end
    end
    kvals = (wall === :pinned ? bessel_zeros_J0(M) : bessel_zeros_J1(M)) ./ b
    # Squared-norm weight 2/‖J_0(k_m ·)‖² with ‖·‖² = ∫_0^b J_0(k_m r)² r dr, which equals
    # (b²/2)J_0(k_m b)² when J_1(k_m b) = 0 and (b²/2)J_1(k_m b)² when J_0(k_m b) = 0.
    # `:clamped` reuses the `:free` (Neumann) basis and weight — pinning is imposed on top
    # by a multiplier, not by changing the basis (design doc §subsubsec:wall, route B).
    bath_norm = [2 / (b * (wall === :pinned ? besselj1(k * b) : besselj0(k * b)))^2 for k in kvals]
    j0kb = [besselj0(k * b) for k in kvals]
    nodes, weights = gauss_legendre_nodes(nq)
    com_nq = min_nq_for_exact_com(N, L)
    com_nodes, com_weights = gauss_legendre_nodes(com_nq)
    return Params(We, Bo, Oh, M, L, N, b, h0, nq, wall, kvals, bath_norm, j0kb,
                  nodes, weights, com_nodes, com_weights)
end

"""BathModeState: `a[m+1] = a_m(τ)`, `adot[m+1] = ȧ_m(τ)`, m = 0..M."""
struct BathModeState
    a::Vector{Float64}
    adot::Vector{Float64}
end
BathModeState(M::Integer) = BathModeState(zeros(M + 1), zeros(M + 1))

"""DropModeState: `beta[l+1] = β_l(τ)`, `betadot[l+1] = β̇_l(τ)`, l = 0..L
(entries l=0,1 always zero — see module docstring)."""
struct DropModeState
    beta::Vector{Float64}
    betadot::Vector{Float64}
end
DropModeState(L::Integer) = DropModeState(zeros(L + 1), zeros(L + 1))

"""COMState: centre-of-mass height `z` and velocity `v`."""
struct COMState
    z::Float64
    v::Float64
end

"""
    Level

One fully-specified, immutable time level. `X` is `nothing` during free flight, or the
converged `(N+2)`-vector `(hat_c_0,...,hat_c_N,theta_c)` while in contact (design doc
eq:newton-unknowns). `dt` is the step size that PRODUCED this level (needed to form
`s^k = dt_this_step / dt` — i.e. this level's `dt` plays the role of "δ_t^{k-1}" for
the next step's BDF2 coefficients, design doc eq:bdf2).
"""
struct Level
    bath::BathModeState
    drop::DropModeState
    com::COMState
    t::Float64
    dt::Float64
    X::Union{Nothing,Vector{Float64}}
end

"""SimHistory: the two time levels BDF2 needs, `prev` (k-1) and `curr` (k)."""
mutable struct SimHistory
    prev::Level
    curr::Level
end
