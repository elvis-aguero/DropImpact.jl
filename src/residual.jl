# The INNER residual of the nested contact closure (design doc eq:summary-galerkin).
#
# theta_c is NOT an unknown here. It is a parameter of the outer scalar selection
# (eq:theta-c-argmin, in timestepper.jl), fixed for the whole of each inner solve. That
# nesting is what keeps this system's conditioning independent of the time step and of
# order unity: every column carries the same O(delta^2) affine slope, and a uniform
# scalar factor cannot change a condition number. Carrying theta_c as an extra unknown
# instead mixes columns differing by delta^2 in scale and drives cond(J) to ~1e18 --
# the defect an earlier revision of the design doc misdiagnosed as intrinsic to the
# position-level closure and "fixed" with an acceleration-level reformulation, since
# deleted (design doc §subsec:corrections).
#
# The Galerkin conditions are tested in the CYLINDRICAL measure w dx = r dr against the
# SAME shifted-Legendre basis that represents the pressure. Both are required for the
# resulting matrix to be symmetric (design doc §subsubsec:compliance): pairing a
# different test basis against the trial basis destroys symmetry even when the operator
# itself is self-adjoint.

"""
    contact_quad(xc, p) -> (x, wq, P, dP, Ptil)

Node data for one contact patch, computed once per `xc` and reused by every moment
integral, the Galerkin residual, and the AD sweep over them. `Ptil[i][n+1]` is the
shifted-Legendre pressure/test basis `P̃_n(ψ(x_i))`. All plain `Float64`: these depend on
`xc` alone, never on the pressure coefficients, so they are outside the AD tape.
"""
function contact_quad(xc::Float64, p::Params)
    x, wq = mapped_nodes(xc, p.gauss_nodes, p.gauss_weights; sqrt_map=(p.alpha != 0))
    P, dP = legendre_tables(x, p.L)
    Ptil = [legendre_P_table(p.N, 2 * (xi - xc) / (1 - xc) - 1)[1:p.N+1] for xi in x]
    return x, wq, P, dP, Ptil
end

"""
    apply_clamp(am_free, kappa, p) -> am

For `p.wall == :clamped` (design doc eq:route-b-multiplier), applies the rank-one
correction carrying the rim multiplier Λ to a diagonal bath response `am_free = alpha +
kappa.*cm` (or, in free flight, `alpha` alone with `cm≡0` -- pinning must hold there too,
not only while a Newton solve is running). Λ is eliminated in closed form: solving the
pinning constraint `sum_m am[m]*j0kb[m] = 0` for Λ and substituting back gives
`Λ = -sum_m am_free[m]*j0kb[m] / D` with `D = (2/b)*sum(kappa)`. `D` is recomputed every
call since `kappa` varies with the step size; `p.j0kb[m] = J_0(k_m b)` never vanishes on
the Neumann set (never zero for m>=1, and j0kb[1]=J_0(0)=1 for the piston, whose kappa=0
kills its own term regardless).
"""
function apply_clamp(am_free::AbstractVector, kappa::Vector{Float64}, p::Params)
    D = (2 / p.b) * sum(kappa)
    Lambda = -sum(am_free .* p.j0kb) / D
    return am_free .+ (2 / p.b) .* kappa .* Lambda ./ p.j0kb
end

"""
    unpack_state(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p) -> (am, beta, zcm, f, cm, bl)

The evaluation chain of design doc §subsec:newton: pressure moments from `chat` at fixed
`xc`, then the affine state advance eq:summary-states. `q` is the `contact_quad` tuple.

One coupling is iterated rather than evaluated once: the self-adjoint `b_l`
(eq:b_l-selfadjoint) carries the area weight `w(x,τ)`, which depends on `beta`, which
depends on `b_l`. The fixed point converges because `lambda_l = O(δ²)` makes the map a
strong contraction -- typically two or three passes. This is a genuine change from the
earlier `w`-free `b_l`, for which the chain was acyclic.

`cm` and `bl` are returned alongside the advanced states so callers can inspect the pressure
moments directly -- design doc §subsubsec:compliance is explicit that only these, not the
pointwise pressure, need converge, so a convergence study has to look at them, not just at
`am`/`beta` (which also depend on the step history, confounding a pure truncation check).
"""
function unpack_state(chat::AbstractVector, xc, q, kappa::Vector{Float64},
    alpha::Vector{Float64}, lambda::Vector{Float64}, gam::Vector{Float64},
    kappa_cm::Float64, mu::Float64, p::Params)
    x, wq, P, dP, _ = q
    T = promote_type(eltype(chat), typeof(xc))
    beta = Vector{T}(undef, p.L + 1)
    @inbounds for l in 0:p.L
        beta[l+1] = gam[l+1]
    end
    local xi, r, w, bl
    for _ in 1:12
        xi, r, w = geom_at_nodes(beta, x, P, dP, p.L)
        bl = b_l_all(chat, xc, x, wq, P, w, p.L, p.alpha)
        beta_new = gam .+ lambda .* bl
        done = maximum(abs.(beta_new .- beta)) < 1e-13
        beta = beta_new
        done && break
    end
    xi, r, w = geom_at_nodes(beta, x, P, dP, p.L)
    cm = c_m_all(chat, xc, x, wq, r, w, p.k, p.bath_norm, p.alpha)
    am = alpha .+ kappa .* cm
    p.wall === :clamped && (am = apply_clamp(am, kappa, p))
    f = com_force_closed(chat, xc, x, wq, w, p.alpha)
    zcm = mu + kappa_cm * f
    return am, beta, zcm, f, cm, bl
end

"""
    residual(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p) -> Vector

The `N+1` inner Galerkin conditions of design doc eq:summary-galerkin,

    R_n = ∫_{xc}^1 [η(r(x)) - z_cm + z_d(x)] P̃_n(ψ(x)) w(x) dx = 0,   n = 0..N,

at FIXED `xc = cos(θ_c)`. `chat` is the only unknown. This is the sole function meant to
be wrapped by `ForwardDiff.jacobian`; every affine coefficient is plain `Float64` data
computed once per step by affine.jl and never differentiated.
"""
function residual(chat::AbstractVector, xc, q, kappa::Vector{Float64},
    alpha::Vector{Float64}, lambda::Vector{Float64}, gam::Vector{Float64},
    kappa_cm::Float64, mu::Float64, p::Params)
    x, wq, P, dP, Ptil = q
    am, beta, zcm, _, _, _ = unpack_state(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    xi, r, w = geom_at_nodes(beta, x, P, dP, p.L)
    T = promote_type(eltype(chat), typeof(xc))
    R = zeros(T, p.N + 1)
    @inbounds for i in eachindex(x)
        eta = zero(T)
        for m in 0:p.M
            eta += am[m+1] * besselj0(p.k[m+1] * r[i])
        end
        gapterm = (eta - zcm + xi[i] * x[i]) * w[i] * wq[i]
        for n in 0:p.N
            R[n+1] += gapterm * Ptil[i][n+1]
        end
    end
    return R
end

"""`C(θ,τ) = η(r(θ,τ),τ) - z_cm(τ) + z_d(θ,τ)` (design doc eq:pointwise-residual-sec3).
Positive means the surfaces interpenetrate."""
function C_at_theta(am::AbstractVector, beta::AbstractVector, zcm, theta, p::Params)
    r = forward_map_r(beta, theta, p.L)
    eta = zero(promote_type(eltype(am), typeof(theta)))
    for m in 0:p.M
        eta += am[m+1] * besselj0(p.k[m+1] * r)
    end
    return eta - zcm + forward_map_zd(beta, theta, p.L)
end

"""
    tangency_residual(am, beta, zcm, theta_c, p; h=1e-6) -> value

`T(θ_c,τ) = ∂C/∂θ` at `θ = θ_c` (design doc eq:tangency-selector), the objective the
outer selection eq:theta-c-argmin minimises in absolute value. Evaluated by a centred
difference: `C` is cheap, the outer iteration needs only a handful of evaluations per
step, and the analytic form would need the `J_1` chain rule through `r(θ)` for no
accuracy gain at this tolerance.
"""
function tangency_residual(am::AbstractVector, beta::AbstractVector, zcm,
    theta_c, p::Params; h::Float64=1e-6)
    return (C_at_theta(am, beta, zcm, theta_c + h, p) -
            C_at_theta(am, beta, zcm, theta_c - h, p)) / (2h)
end

"""
    check_nonintersect(am, beta, zcm, theta_c, p; nsample=400) -> Bool

Design doc eq:check-nonintersect as a FEASIBILITY FILTER on candidate `θ_c`, applied
before the selector is consulted -- exactly as AgueroEtAl2026 set `e(q) = ∞` on
candidates whose surfaces intersect. `true` iff `C(θ) < 0` throughout `(θ_c, π]`.
"""
function check_nonintersect(am::AbstractVector, beta::AbstractVector, zcm,
    theta_c, p::Params; nsample::Int=400)
    for theta in range(theta_c + 1e-4, π; length=nsample)
        C_at_theta(am, beta, zcm, theta, p) >= 0 && return false
    end
    return true
end

"""
    check_monotone_r(beta, xc, L; nsample=200) -> Bool

Design doc eq:check-monotone-r: `∂r/∂x < 0` throughout `[xc,1]`, equivalently
`r_c < r_M` in AgueroEtAl2026's notation -- the contact patch must not reach the
droplet's widest point. Without it the substitution `r dr = w dx` used throughout the
projections is invalid and `w` changes sign.
"""
function check_monotone_r(beta::AbstractVector, xc, L::Integer; nsample::Int=200)
    for x in range(xc, 1 - 1e-8; length=nsample)
        w_of_x(beta, x, L) <= 0 && return false
    end
    return true
end

"""
    check_positivity(chat, xc; nsample=200) -> Bool

Design doc eq:check-positivity. NOT imposed, but a genuine diagnostic -- of the
truncation, not of the model. Measured behaviour, over a full impact:

  * Past the resolvable-rank budget it fails almost everywhere: at `M=L=60`, raising `N`
    from 3 to 12 takes the fraction of contact steps with `min p < 0` from 2% to 90%.
    A failure here is the loudest available signal that `N+1` exceeds `0.6*n_*`
    (see `resolvable_rank_estimate`).
  * Inside the budget a small residual remains, ~1% of steps, and its magnitude is
    controlled by the BATH truncation, not by `N`: the worst excursion shrinks
    monotonically from `-21` to `-2.4` as `M` goes 40 -> 120 at fixed `L=120, N=3`,
    while being flat at `~-3.8` across `N=1..7` at fixed `M=80, L=240`.

The earlier claim here -- that this test is meaningless at large `N` because the
compliance operator is compact -- was never verified and is false; see
`test/test_rank_law.jl` and `derivations/DIAGNOSTICS-NOTATION.md`.
"""
function check_positivity(chat::AbstractVector, xc; nsample::Int=200, alpha=0.0)
    for x in range(xc, 1.0; length=nsample)
        pressure_poly_raw(chat, xc, x, alpha) < 0 && return false
    end
    return true
end
