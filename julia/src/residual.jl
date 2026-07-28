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
    x, wq = mapped_nodes(xc, p.gauss_nodes, p.gauss_weights)
    P, dP = legendre_tables(x, p.L)
    Ptil = [legendre_P_table(p.N, 2 * (xi - xc) / (1 - xc) - 1)[1:p.N+1] for xi in x]
    return x, wq, P, dP, Ptil
end

"""
    unpack_state(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p) -> (am, beta, zcm, f)

The evaluation chain of design doc §subsec:newton: pressure moments from `chat` at fixed
`xc`, then the affine state advance eq:summary-states. `q` is the `contact_quad` tuple.

One coupling is iterated rather than evaluated once: the self-adjoint `b_l`
(eq:b_l-selfadjoint) carries the area weight `w(x,τ)`, which depends on `beta`, which
depends on `b_l`. The fixed point converges because `lambda_l = O(δ²)` makes the map a
strong contraction -- typically two or three passes. This is a genuine change from the
earlier `w`-free `b_l`, for which the chain was acyclic.
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
    local xi, r, w
    for _ in 1:12
        xi, r, w = geom_at_nodes(beta, x, P, dP, p.L)
        bl = b_l_all(chat, xc, x, wq, P, w, p.L)
        beta_new = gam .+ lambda .* bl
        done = maximum(abs.(beta_new .- beta)) < 1e-13
        beta = beta_new
        done && break
    end
    xi, r, w = geom_at_nodes(beta, x, P, dP, p.L)
    cm = c_m_all(chat, xc, x, wq, r, w, p.k, p.bath_norm)
    am = alpha .+ kappa .* cm
    f = com_force_closed(chat, xc, x, wq, w)
    zcm = mu + kappa_cm * f
    return am, beta, zcm, f
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
    am, beta, zcm, _ = unpack_state(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
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

Design doc eq:check-positivity. NOT imposed and, at large `N`, not a meaningful test:
the compliance operator is compact, so the pointwise pressure is not a converged output
of the model and `min p` is measured to diverge with `N`. Retained as a diagnostic for
small `N`, where the pressure is close to fully resolved.
"""
function check_positivity(chat::AbstractVector, xc; nsample::Int=200)
    for x in range(xc, 1.0; length=nsample)
        pressure_poly_raw(chat, xc, x) < 0 && return false
    end
    return true
end
