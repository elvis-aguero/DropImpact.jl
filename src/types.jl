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
    selector::Symbol
    viscous::Symbol
    drop_lambda::Vector{Float64}
    drop_omega2::Vector{Float64}
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

# Contact-edge selection rule.
#
#   :crossing -- theta_c is the LARGEST crossing of the droplet and bath surfaces, HOLDING the
#                previous value when they do not cross, clamped to [asin(0.01), asin(0.9998)].
#                This is AlventosaEtAl2023's rule, read from their reference implementation
#                (harrislab-brown/BouncingDroplets, bounce_alventosa_bessel_implicit.m:245).
#                Their defaults: b = 25R, 151 bath modes, 55 drop modes, and bath eigenvalues
#                k = besselzero(-1,...)/b. Since J_{-1} = -J_1 those are the J_1 zeros, i.e. a
#                NO-FLUX wall -- the same condition we use. (An earlier note here claimed they
#                use a Dirichlet/pinned wall, from misreading the separate u = besselzero(0,...)
#                array, which is used for the pressure-projection interpolation and not for the
#                bath basis. Corrected.)
#                No feasibility predicates are consulted.
#
#   :feasible -- the former default, theta_c = inf{theta : non-intersection and monotone-r}.
#                DEGENERATE: once non-intersection stops binding, the infimum is 0 regardless of
#                the gap's magnitude, so the patch collapsed 200x in a single 1e-3 step and never
#                recovered; f then decayed to 1e-8 from above without reversing and contact
#                dragged on for 63% of its duration after the drop began rising. Retained only to
#                reproduce results generated before the default changed.
#
# DEFAULT IS :crossing, CHANGED FROM :feasible. An earlier revision of this comment said the
# default was "DELIBERATELY UNCHANGED" because :crossing fixed the patch collapse without moving
# any observable -- measured 6.3312 vs 6.3218 at We=0.0231 and 4.9553 vs 4.9531 at We=0.9985.
# THAT VERDICT WAS DRAWN FROM TOO NARROW A SAMPLE AND IS RETRACTED. Both of those points are
# low-We water. At high We the rules do not agree to three digits; one returns a physical
# trajectory and the other returns a droplet that sinks.
#
# At We = 7.307 in 5 cSt oil (Bo = 0.0563, Oh = 0.0578), :feasible holds contact for 21 steps --
# duration 0.194 against a MEASURED 5.26 -- then detaches, after which the drop free-falls to
# z_cm = -84.2 by t = 30 and never re-contacts. threshold_contact_time returns `nothing`: there is
# no contact time to report. :crossing gives contact over (0, 4.229), a rebound to z_cm = 2.239,
# and further contacts at (18.33, 22.15) and beyond 29.38.
#
# GATED BEFORE SWITCHING, on both observables the experiments report, at THIS truncation, five
# anchor points across both fluids, each rule on identical hardware (test_experiment_regression.jl,
# the `selector-regression` CI job):
#
#   fluid  We      tc :feasible  tc :crossing   delta :feasible  delta :crossing
#   water  0.7251  4.49560       4.49657        0.66951          0.66900
#   water  1.9387  4.42119       4.42242        0.95812          0.95736
#   oil    1.2158  5.07868       5.08034        0.72344          0.72301
#   oil    3.9463  5.01559       5.01686        1.07481          1.07417
#   oil    7.3070  none          5.03505        0.30009          1.33536
#
# Wherever :feasible produces an answer the two agree to four significant figures -- at most 0.03%
# in tc, 0.08% in delta -- so this cannot regress a case that previously worked; where :feasible
# fails it fails completely. A strict improvement, which is the only basis on which a default
# affecting every stored result should move.
#
# CONSEQUENCE FOR STORED RESULTS: low-We numbers regenerated after this commit differ in the fourth
# digit; high-We oil numbers change qualitatively, because they were not physical before.
#
# NOT FIXED BY THIS CHANGE: the maximum penetration depth is under-predicted, identically under both
# rules (1.0742 vs 1.0748 at oil We = 3.946), so it is not selector-related. It is about twice
# 1PKM's own deficit at moderate We while the contact time here is BETTER than 1PKM's
# (-0.85 sd against -3.61 sd at We = 7.307). Open; see derivations/tangency-selector.tex.
const DEFAULT_SELECTOR = :crossing

# Viscous model for the DROP modes.
#
#   :lamb -- lambda_l = Oh(l-1)(2l+1), omega_l^2 = l(l-1)(l+2). Lamb's (1881) small-viscosity
#            asymptotics. The published model, and what DropRebound.jl/DropSolver uses too.
#   :reid -- exact roots of Reid's (1960) characteristic equation, valid at arbitrary Oh.
#
# DEFAULT IS :reid. The reasoning, since this was reversed after review: at the small Oh where
# experiments exist, the two are experimentally INDISTINGUISHABLE (CoR identical to 4 digits),
# so :reid cannot be worse against any available data. At Oh >~ 0.05, :lamb is PROVABLY wrong by
# 23-97% while :reid is exact in theory and merely untested. Defaulting to a known-invalid model
# in the regime where neither is validated is the wrong trade -- :reid weakly dominates. Pass
# viscous=:lamb to reproduce the published small-Oh model bit-for-bit.
#
# Measured against Reid, Lamb
# OVERPREDICTS the damping by 4.1% at l=2 and 22.9% at l=120 even at production Oh = 0.006,
# and by 23-97% across l = 2..16 at Oh = 0.1. See src/reid.jl for the table and for what the
# substitution does and does not capture.
#
# READINESS OF :reid AS A FUTURE DEFAULT -- measured, all clear:
#
#   live impact    completes at Oh = 0.006, 0.05, 0.2. At production Oh the observables are
#                  essentially unchanged (CoR identical to 4 digits, t_c 3.8145 -> 3.7942, i.e.
#                  0.5%), so switching would not materially move published numbers. The
#                  difference grows with Oh and points the physically right way -- Lamb
#                  over-damps, so it loses too much energy and UNDER-predicts CoR; :reid
#                  corrects upward by +0.4% at Oh = 0.05 and +1.7% at Oh = 0.2.
#   stiffness      omega2/omega_{l,0}^2 runs 0.9992 -> 0.9663 monotonically over l = 2..120 at
#                  production Oh: 3.4% worst case, all damping positive, all finite.
#   cost           +1.26 s once per Params at L = 120, against a ~130 s run. The :reid run was
#                  in fact marginally faster overall.
#   overdamped     at Oh = 0.2, 80 modes (l = 41..120) take the real-pair Vieta branch with
#                  lambda up to ~1e3 and the run completes cleanly -- the stiff path works in a
#                  real simulation, not only in unit tests.
#
# WHAT IS STILL NOT ESTABLISHED (a caveat on the claim, no longer a reason to withhold the
# default): "more faithful to linear viscous theory" is NOT "matches experiment better", and
# only the former is verified.
# The +1.7% CoR shift at Oh = 0.2 is the one place the change is visible, and it is exactly
# there that nothing has been compared against DNS or experiment. If that shift moves AWAY from
# ground truth it would indicate the two-pole forcing gap (or something else) dominating at
# moderate Oh. A comparison against Alventosa et al. or DNS at Oh >~ 0.05 remains worth doing --
# but note it can only ever CONFIRM or REFINE :reid, since :lamb is already known wrong there.
#
# NOTE FOR THE PAPER: this switch moves t_c by 0.5% at production Oh (3.8145 -> 3.7942) with CoR
# unchanged to 4 digits. Small, but not nothing -- any figure regenerated after this commit
# differs slightly from one generated before it, and the closure should be named in the text.
const DEFAULT_VISCOUS = :reid

"""
    Params(; We, Bo, Oh, b, h0, wall=:free, M, L, N, nq,
             selector=DEFAULT_SELECTOR, viscous=DEFAULT_VISCOUS, check_budget=true)

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
                nq::Integer=DEFAULT_NQ, selector::Symbol=DEFAULT_SELECTOR,
                viscous::Symbol=DEFAULT_VISCOUS, check_budget::Bool=true)
    selector in (:crossing, :feasible) ||
        throw(ArgumentError("selector must be :crossing or :feasible, got $selector"))
    wall in (:free, :pinned, :clamped) ||
        throw(ArgumentError("wall must be :free, :pinned or :clamped, got $wall"))
    viscous in (:lamb, :reid) ||
        throw(ArgumentError("viscous must be :lamb or :reid, got $viscous"))

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
    drop_lambda, drop_omega2 = drop_viscous_coeffs(L, Oh, viscous)
    return Params(We, Bo, Oh, M, L, N, b, h0, nq, selector, viscous, drop_lambda, drop_omega2,
                  wall, kvals, bath_norm, j0kb,
                  nodes, weights, com_nodes, com_weights)
end

"""
    Params(c::ImpactConditions; b=nothing, h0=nothing, kwargs...)

Build `Params` from a measured impact, so the dimensional properties reach the solver without a
hand-computed `(We, Bo, Oh)` in between. See [`conditions`](@ref).

The bath geometry comes from `c` when [`conditions`](@ref) was given a dimensional `bath_radius`
and `bath_depth`; otherwise pass `b` and `h0` here in units of `R`, as usual. Every remaining
keyword (`M`, `L`, `N`, `nq`, `wall`, `selector`, `viscous`, `check_budget`) forwards unchanged.

```julia
c = conditions(drop=:oil, R=3.5e-4, V0=0.6)      # 5 cSt oil, 0.35 mm, 60 cm/s
p = Params(c; b=6.0, h0=3.0)
```
"""
function Params(c::ImpactConditions; b=nothing, h0=nothing, kwargs...)
    bb = b === nothing ? c.b : float(b)
    hh = h0 === nothing ? c.h0 : float(h0)
    bb === nothing && throw(ArgumentError(
        "no bath radius: give `b` here in units of R, or `bath_radius` in metres to conditions()"))
    hh === nothing && throw(ArgumentError(
        "no bath depth: give `h0` here in units of R, or `bath_depth` in metres to conditions()"))
    return Params(; We=c.We, Bo=c.Bo, Oh=c.Oh, b=bb, h0=hh, kwargs...)
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
