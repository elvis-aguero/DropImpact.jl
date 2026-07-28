# SUPERSEDED -- RETAINED FOR THE RECORD ONLY.  This script verifies claims about
# a forward march under the joint crossing closure, REMOVED from the design doc.
# See paper-formulation.tex ('The contact angle: an outer selector') for the closure that replaces it.
#
# It is no longer cited by paper-formulation.tex and its conclusions no longer
# describe the model.  Do not treat a passing run here as support for current theory.

# Does the current theory's contact-angle closure (eq:theta-c-crossing, C(theta_c,tau)=0,
# joint with the pressure Newton unknowns) actually produce a continuous, monotonically
# growing theta_c(tau) through a REAL forward-marching onset transient -- warm-started
# from the previous converged step exactly as an actual simulation would -- or does it
# lose track of the physically correct branch once pressure/deformation build up?
#
# Analytically confirmed (see accompanying discussion): at the EXACT onset (all bath/
# drop coefficients zero), C(theta,tau) collapses to cos(theta)-z_cm(tau), so
# theta_c(tau)=arccos(z_cm(tau)) with dtheta_c/dtau = -zcmdot/sqrt(1-zcm^2) > 0
# automatically (Wagner-type sqrt(tau) growth), given zcmdot<0. This is a genuine,
# leading-order guarantee -- but ONLY in that trivial limit. Once a_m,beta_l are
# nonzero, dtheta_c/dtau = -(dC/dtau)/(dC/dtheta_c) via implicit differentiation, and
# NOTHING in the theory fixes the sign of dC/dtheta_c at a general solution -- so
# monotonic, physically-continuous growth is not structurally guaranteed post-onset.
# This script tests whether it holds ANYWAY, empirically, for a real forward march
# (not the earlier, misleading test that ran full Newton from isolated synthetic
# snapshots and let it wander to an unrelated root).
#
# Uses the shifted-Legendre-in-psi pressure basis (the design now being moved to
# main), crossing-row closure C(theta_c,tau)=0, current-step a_m(X),beta_l(X)
# (never frozen) in the crossing row, exactly as the design doc specifies.
#
# Run with: julia --project=../julia verify_theta_c_monotonicity.jl  (from derivations/)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralKM
using SpectralKM: BathModeState, DropModeState, COMState, Level, SimHistory,
    bath_affine, drop_affine, com_affine, bdf_derivative, gauss_quad,
    legendre_P_table, legendre_dP_table, forward_map_r, forward_map_zd, newton_solve,
    Converged, Stalled, MaxIterExceeded
using SpecialFunctions: besselj0, besselj1
using ForwardDiff
using LinearAlgebra
using Printf

# ---- shifted-Legendre-in-psi pressure basis (same as verify_legendre_basis_full_jacobian.jl) ----
function pressure_poly_raw_legendre(chat::AbstractVector, xc, x)
    psi = (x - xc) / (1 - xc)
    N = length(chat) - 1
    Ptab = legendre_P_table(N, 2 * psi - 1)
    s = zero(promote_type(eltype(chat), typeof(psi)))
    for n in 0:N
        s += chat[n+1] * Ptab[n+1]
    end
    return s
end
r_of_x_L(beta, x, L) = SpectralKM.xi_of_x(beta, x, L) * sqrt(1 - x^2)

function b_l_legendre(chat::AbstractVector, xc, L::Integer, nodes, weights)
    N = length(chat) - 1
    T = promote_type(eltype(chat), typeof(xc))
    out = Vector{T}(undef, L + 1)
    for l in 0:L
        integrand(x) = pressure_poly_raw_legendre(chat, xc, x) * legendre_P_table(L, x)[l+1]
        out[l+1] = gauss_quad(integrand, xc, nodes, weights)
    end
    return out
end

function c_m_legendre(chat::AbstractVector, xc, beta::AbstractVector, L::Integer,
    k::Vector{Float64}, b::Float64, nodes, weights)
    M = length(k) - 1
    T = promote_type(eltype(chat), typeof(xc), eltype(beta))
    out = Vector{T}(undef, M + 1)
    out[1] = zero(T)
    for m in 1:M
        km = k[m+1]
        integrand(x) = pressure_poly_raw_legendre(chat, xc, x) * besselj0(km * r_of_x_L(beta, x, L)) *
                       abs(SpectralKM.r_drdx_of_x(beta, x, L))
        norm = 2 / (b * besselj1(km))^2
        out[m+1] = norm * gauss_quad(integrand, xc, nodes, weights)
    end
    return out
end

function com_force_legendre(chat::AbstractVector, xc, beta::AbstractVector, L::Integer, com_nodes, com_weights)
    dr2_dx(x) = begin
        Ptab = legendre_P_table(L, x)
        dPtab = legendre_dP_table(L, x)
        xi = one(x); dxi = zero(x)
        for l in 2:L
            xi += beta[l+1] * Ptab[l+1]
            dxi += beta[l+1] * dPtab[l+1]
        end
        return 2 * xi * dxi * (1 - x^2) - xi^2 * 2x
    end
    integrand(x) = pressure_poly_raw_legendre(chat, xc, x) * dr2_dx(x)
    return -gauss_quad(integrand, xc, com_nodes, com_weights)
end

function accel_galerkin_term_legendre(chat, xc, a_frozen, adot_frozen, beta_frozen,
    betadot_frozen, cm, bl, f, p::Params)
    N = length(chat) - 1
    T = promote_type(eltype(chat), typeof(xc), eltype(cm), eltype(bl), typeof(f))
    out = Vector{T}(undef, N + 1)
    for n in 0:N
        integrand(x) = begin
            psi = (x - xc) / (1 - xc)
            Ptab = legendre_P_table(N, 2 * psi - 1)
            (SpectralKM.Pi_of_x(x, cm, bl, f, a_frozen, beta_frozen, p) +
             SpectralKM.K_of_x(x, a_frozen, adot_frozen, beta_frozen, betadot_frozen, p)) * Ptab[n+1]
        end
        out[n+1] = gauss_quad(integrand, xc, p.gauss_nodes, p.gauss_weights)
    end
    return out
end

function crossing_row_L(am, beta, zcm, theta_c, p::Params)
    rc = forward_map_r(beta, theta_c, p.L)
    zdc = forward_map_zd(beta, theta_c, p.L)
    T = promote_type(eltype(am), eltype(beta), typeof(theta_c), typeof(zcm))
    s = zero(T)
    for m in eachindex(am)
        s += am[m] * besselj0(p.k[m] * rc)
    end
    return s - zcm + zdc
end

function residual_accel_legendre(X::AbstractVector, kappa, alpha, lambda, gam, kappa_cm, mu,
    bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N; L = p.L
    chat = @view X[1:N+1]
    theta_c = X[N+2]
    xc = cos(theta_c)
    bl = b_l_legendre(chat, xc, L, p.gauss_nodes, p.gauss_weights)
    beta = gam .+ lambda .* bl
    f = com_force_legendre(chat, xc, beta, L, p.com_nodes, p.com_weights)
    zcm = mu + kappa_cm * f
    cm = c_m_legendre(chat, xc, beta, L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
    am = alpha .+ kappa .* cm
    galerkin = accel_galerkin_term_legendre(chat, xc, bath_frozen.a, bath_frozen.adot,
        drop_frozen.beta, drop_frozen.betadot, cm, bl, f, p)
    T = promote_type(eltype(X), Float64)
    R = Vector{T}(undef, N + 2)
    for n in 0:N
        R[n+1] = galerkin[n+1]
    end
    R[N+2] = crossing_row_L(am, beta, zcm, theta_c, p)
    return R
end

function unpack_state_L(X, kappa, alpha, lambda, gam, kappa_cm, mu, p::Params)
    N = p.N; L = p.L
    chat = @view X[1:N+1]
    theta_c = X[N+2]
    xc = cos(theta_c)
    bl = b_l_legendre(chat, xc, L, p.gauss_nodes, p.gauss_weights)
    beta = gam .+ lambda .* bl
    f = com_force_legendre(chat, xc, beta, L, p.com_nodes, p.com_weights)
    zcm = mu + kappa_cm * f
    cm = c_m_legendre(chat, xc, beta, L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
    am = alpha .+ kappa .* cm
    return am, beta, zcm
end

# ---------------------------------------------------------------------------
# Forward march: REAL BDF2 history evolving step to step (not a frozen synthetic
# snapshot), warm-started from the PREVIOUS converged X, exactly as run_simulation
# would do it -- starting from t=0 (all coefficients zero, com.z=1, v=-sqrt(We)).
# ---------------------------------------------------------------------------
p = Params(We=1.0, Bo=0.2, Oh=0.05, M=8, L=8, N=2, b=6.0, h0=3.0, nq=20)
lvl0 = Level(BathModeState(p.M), DropModeState(p.L), COMState(1.0, -sqrt(p.We)), 0.0, 0.0, nothing)
hist = SimHistory(lvl0, lvl0)

dt = 2e-4
nsteps = 40
theta_c_hist = Float64[]
theta_c_wagner = Float64[]
t_hist = Float64[]
status_hist = Symbol[]

X = nothing
for step in 1:nsteps
    global X
    dtprev = hist.curr.dt
    kappa, alpha = bath_affine(hist.curr.bath, hist.prev.bath, p, dt, dtprev)
    lambda, gam = drop_affine(hist.curr.drop, hist.prev.drop, p, dt, dtprev)
    kappa_cm, mu = com_affine(hist.curr.com, hist.prev.com, p, dt, dtprev)

    R(Xv) = residual_accel_legendre(Xv, kappa, alpha, lambda, gam, kappa_cm, mu, hist.curr.bath, hist.curr.drop, p)

    # warm start: previous converged X if available, else the leading-order Wagner
    # estimate theta_c=sqrt(2|v|*dt) with zero pressure -- exactly what a real solver
    # would use at the very first contact step.
    t_new = hist.curr.t + dt
    if X === nothing
        theta_c_guess = sqrt(max(2 * sqrt(p.We) * t_new, 1e-8))
        Xg = [zeros(p.N + 1); theta_c_guess]
    else
        Xg = copy(X)
    end

    result = newton_solve(R, Xg)
    X = result.X

    am, beta, zcm = unpack_state_L(X, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    adot = [bdf_derivative(am[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, dtprev) for m in 0:p.M]
    betadot = zeros(p.L + 1)
    for l in 2:p.L
        betadot[l+1] = bdf_derivative(beta[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, dtprev)
    end
    v = bdf_derivative(zcm, hist.curr.com.z, hist.prev.com.z, dt, dtprev)

    hist.prev = hist.curr
    hist.curr = Level(BathModeState(am, adot), DropModeState(beta, betadot), COMState(zcm, v), t_new, dt, X)

    push!(theta_c_hist, X[end])
    push!(theta_c_wagner, sqrt(max(2 * sqrt(p.We) * t_new, 0.0)))
    push!(t_hist, t_new)
    push!(status_hist, Symbol(result.status))
end

println("=" ^ 100)
println("Forward march through onset: theta_c(tau) vs leading-order Wagner estimate")
println("(N=2, dt=2e-4, warm-started from previous converged X each step)")
println("=" ^ 100)
@printf("%-4s  %-10s  %-14s  %-14s  %-10s  %-10s\n", "step", "t", "theta_c", "Wagner sqrt(2vt)", "status", "monotone?")
let prev_tc = 0.0, all_monotone = true
    for i in 1:nsteps
        is_mono = theta_c_hist[i] >= prev_tc - 1e-10
        all_monotone = all_monotone && is_mono
        @printf("%-4d  %-10.2e  %-14.6e  %-14.6e  %-10s  %-10s\n",
            i, t_hist[i], theta_c_hist[i], theta_c_wagner[i], String(status_hist[i]), is_mono ? "yes" : "NO <---")
        prev_tc = theta_c_hist[i]
    end
    println()
    println(all_monotone ? "theta_c(tau) was monotonically non-decreasing across every step." :
                            "theta_c(tau) DECREASED at some step -- non-monotonic, flagged above.")
end
