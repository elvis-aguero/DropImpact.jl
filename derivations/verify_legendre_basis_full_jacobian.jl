# SUPERSEDED -- RETAINED FOR THE RECORD ONLY.  This script verifies claims about
# a basis comparison on the joint Newton Jacobian, which no longer exists.  The
# shifted-Legendre basis is retained on the separate ground of exact self-orthogonality
# (paper-formulation.tex ('Pressure representation')); its conditioning advantage over monomials
# is no longer the reason.  Superseded by audit_nested_closure.jl.
#
# It is no longer cited by paper-formulation.tex and its conclusions no longer
# describe the model.  Do not treat a passing run here as support for current theory.

# Closes the reviewer's required gap: does substituting shifted-Legendre-in-psi for
# monomial-in-psi actually move the FULL nonlinear Newton-Jacobian conditioning (not
# just the basis-only self-Gram number, which the reviewer correctly flagged as 5-7
# orders of magnitude smaller than what actually matters)?
#
# Pressure basis swapped to p(x,tau) = sum_n chat[n] * P~_n(psi(x)), psi=(x-xc)/(1-xc),
# P~_n(psi) := P_n(2*psi-1) (shifted Legendre, orthogonal on psi in [0,1]). Still
# STRUCTURALLY piecewise (zero for x<xc), same support convention as now -- only the
# polynomial family inside the support changes.
#
# b_l/c_m/f are recomputed via direct Gauss quadrature against the new pointwise
# pressure evaluation (no closed-form Bonnet-recursion shortcut attempted here -- this
# is a numerical validation of whether the swap is WORTH doing before investing in
# that closed-form re-derivation for the .tex/production code).
#
# Run with: julia --project=../julia verify_legendre_basis_full_jacobian.jl  (from derivations/)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralKM
using SpectralKM: BathModeState, DropModeState, COMState, bath_affine, drop_affine,
    com_affine, gauss_quad, legendre_P_table, forward_map_r, forward_map_zd, newton_solve
using SpecialFunctions: besselj0
using ForwardDiff
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Shifted-Legendre-in-psi pressure basis
# ---------------------------------------------------------------------------
"""p(x,tau) = sum_n chat[n] * P_n(2*psi-1), psi=(x-xc)/(1-xc), for x>=xc (no branch,
matching pressure_poly_raw's own convention: callers only ever evaluate this on
[xc,1])."""
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
zd_of_x_L(beta, x, L) = SpectralKM.xi_of_x(beta, x, L) * x

"""b_l(chat,xc,L) via direct Gauss quadrature against the new pressure basis (no
closed-form Bonnet shortcut -- a numerical validation, not the production
implementation)."""
function b_l_legendre(chat::AbstractVector, xc, L::Integer, nodes, weights)
    N = length(chat) - 1
    T = promote_type(eltype(chat), typeof(xc))
    out = Vector{T}(undef, L + 1)
    Ptab_x(x) = legendre_P_table(L, x)
    for l in 0:L
        integrand(x) = pressure_poly_raw_legendre(chat, xc, x) * Ptab_x(x)[l+1]
        out[l+1] = gauss_quad(integrand, xc, nodes, weights)
    end
    return out
end

"""c_m(chat,xc,beta,L) via quadrature against the new pressure basis."""
function c_m_legendre(chat::AbstractVector, xc, beta::AbstractVector, L::Integer,
    k::Vector{Float64}, b::Float64, nodes, weights)
    M = length(k) - 1
    T = promote_type(eltype(chat), typeof(xc), eltype(beta))
    out = Vector{T}(undef, M + 1)
    out[1] = zero(T)
    for m in 1:M
        km = k[m+1]
        integrand(x) = begin
            r = r_of_x_L(beta, x, L)
            pressure_poly_raw_legendre(chat, xc, x) * besselj0(km * r) *
                abs(SpectralKM.r_drdx_of_x(beta, x, L) / r_of_x_L(beta, x, L) * r) # |r dr/dx|/r * r -- reuse r_drdx directly instead
        end
        # simpler: r*|dr/dx| = |r_drdx_of_x| directly (r_drdx_of_x already returns r*dr/dx)
        integrand2(x) = pressure_poly_raw_legendre(chat, xc, x) * besselj0(km * r_of_x_L(beta, x, L)) *
                         abs(SpectralKM.r_drdx_of_x(beta, x, L))
        norm = 2 / (b * besselj1_safe(km))^2
        out[m+1] = norm * gauss_quad(integrand2, xc, nodes, weights)
    end
    return out
end
besselj1_safe(x) = SpecialFunctions.besselj1(x)
using SpecialFunctions

"""f(chat,xc,beta,L) via quadrature against the new pressure basis (same d[r^2]
sign convention as com_force_closed)."""
function com_force_legendre(chat::AbstractVector, xc, beta::AbstractVector, L::Integer,
    com_nodes, com_weights)
    dr2_dx(x) = begin
        Ptab = legendre_P_table(L, x)
        dPtab = SpectralKM.legendre_dP_table(L, x)
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

"""accel-level Galerkin rows against the new basis: same Pi+K integrand
(accel_closure.jl's Pi_of_x/K_of_x, unchanged), just projected onto the new
pressure basis's OWN functions P~_n(psi) as test functions instead of psi^n."""
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

function accumulated_contact_history(p::Params)
    a = [0.15 / (m + 1) for m in 0:p.M]
    adot = [-0.4 / (m + 1) for m in 0:p.M]
    bath = BathModeState(a, adot)
    beta = zeros(p.L + 1); betadot = zeros(p.L + 1)
    for l in 2:p.L
        beta[l+1] = 0.1 / (l - 1)
        betadot[l+1] = -0.3 / (l - 1)
    end
    drop = DropModeState(beta, betadot)
    com = COMState(0.85, -0.6)
    return bath, drop, com
end

make_params(N) = Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=N, b=6.0, h0=3.0, nq=20)
representative_X0(N::Integer, theta_c0::Float64) = begin
    base = [1.0, 0.35, 0.12, 0.04]
    chat0 = [base[min(n + 1, 4)] for n in 0:N]
    [chat0; theta_c0]
end

function affine_at(p::Params, dt::Float64)
    bath, drop, com = accumulated_contact_history(p)
    kappa, alpha = bath_affine(bath, bath, p, dt, dt)
    lambda, gam = drop_affine(drop, drop, p, dt, dt)
    kappa_cm, mu = com_affine(com, com, p, dt, dt)
    return kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop
end

println("=" ^ 90)
println("(1) N-scan, theta_c=0.3, dt=1e-3: FULL Jacobian cond(J), monomial-psi vs Legendre-psi")
println("=" ^ 90)
@printf("%-3s  %-16s  %-16s\n", "N", "cond(J) monomial*", "cond(J) Legendre")
println("(*from verify_N_scaling_diagnosis.jl's earlier run, same params/history/seed)")
monomial_ref = Dict(0 => 2.3556e2, 1 => 1.6010e4, 2 => 7.0921e7, 3 => 5.1671e12)
for N in 0:3
    p = make_params(N)
    kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p, 1e-3)
    X0 = representative_X0(N, 0.3)
    R(X) = residual_accel_legendre(X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J = ForwardDiff.jacobian(R, X0)
    @printf("%-3d  %-16.4e  %-16.4e\n", N, monomial_ref[N], cond(J))
end

println()
println("=" ^ 90)
println("(2) theta_c-scan at N=3, dt=1e-3: FULL Jacobian cond(J), monomial-psi vs Legendre-psi")
println("=" ^ 90)
@printf("%-8s  %-16s  %-16s\n", "theta_c", "cond(J) monomial*", "cond(J) Legendre")
monomial_ref2 = Dict(0.05 => 3.3368e19, 0.1 => 1.5280e18, 0.3 => 5.1671e12, 0.6 => 1.9789e8, 1.0 => 3.0918e5, 1.4 => 1.4173e5)
for theta_c0 in (0.05, 0.1, 0.3, 0.6, 1.0, 1.4)
    p = make_params(3)
    kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p, 1e-3)
    X0 = representative_X0(3, theta_c0)
    R(X) = residual_accel_legendre(X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J = ForwardDiff.jacobian(R, X0)
    @printf("%-8.2f  %-16.4e  %-16.4e\n", theta_c0, monomial_ref2[theta_c0], cond(J))
end

println()
println("=" ^ 90)
println("(3) SVD spectrum at N=3, theta_c=0.3, Legendre-psi basis")
println("=" ^ 90)
p3 = make_params(3)
kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p3, 1e-3)
X0 = representative_X0(3, 0.3)
R3(X) = residual_accel_legendre(X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p3)
J3 = ForwardDiff.jacobian(R3, X0)
for (i, s) in enumerate(svdvals(J3))
    @printf("  sigma_%d = %.6e\n", i, s)
end

println()
println("=" ^ 90)
println("(4) Solve-stability (1 LM step, realistic 1e-10 relative history perturbation)")
println("    monomial-psi reference values from verify_solve_stability_smallthetac.jl")
println("=" ^ 90)
function lm_step(R, X0, mu_damp)
    Rval = R(X0)
    J = ForwardDiff.jacobian(R, X0)
    return (J' * J + mu_damp * I(length(X0))) \ (J' * Rval)
end
const REL_PERT = 1e-10
monomial_relstep = Dict(0.05 => 2.14e-11, 0.1 => 1.10e-10, 0.3 => 1.93e-10, 0.6 => 1.34e-8, 1.0 => 2.33e-8)
@printf("%-8s  %-16s  %-16s  %-16s\n", "theta_c", "cond(J) Legendre", "relstep monomial*", "relstep Legendre")
for theta_c0 in (0.05, 0.1, 0.3, 0.6, 1.0)
    p = make_params(3)
    kappa, alpha, lambda, gam, kappa_cm, mu = affine_at(p, 1e-3)[1:6]
    bath, drop = affine_at(p, 1e-3)[7:8]
    X0 = representative_X0(3, theta_c0)
    R0(X) = residual_accel_legendre(X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J0 = ForwardDiff.jacobian(R0, X0)
    dX = lm_step(R0, X0, 1e-6)
    alpha_p = alpha .* (1 + REL_PERT); gam_p = gam .* (1 + REL_PERT); mu_p = mu * (1 + REL_PERT)
    Rp(X) = residual_accel_legendre(X, kappa, alpha_p, lambda, gam_p, kappa_cm, mu_p, bath, drop, p)
    dXp = lm_step(Rp, X0, 1e-6)
    relstep = norm(dXp - dX) / norm(dX)
    @printf("%-8.2f  %-16.4e  %-16.4e  %-16.4e\n", theta_c0, cond(J0), monomial_relstep[theta_c0], relstep)
end
