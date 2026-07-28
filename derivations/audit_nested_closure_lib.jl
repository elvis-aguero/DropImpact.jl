# Shared assembly for the nested free-boundary closure audit: the inner Galerkin
# pressure solve at FIXED theta_c, and evaluation of C(theta) on a theta grid.
# Driver and findings: audit_nested_closure.jl
using SpectralKM
using SpectralKM: gauss_legendre_nodes, legendre_P_table
using SpecialFunctions: besselj0, besselj1
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "audit_compliance_operator_core.jl"))

"""
Shifted-Legendre pressure basis of eq:pressure-poly evaluated at `x` for support
[xc, 1]: returns [P~_0, ..., P~_N] with P~_n(psi) = P_n(2 psi - 1), psi=(x-xc)/(1-xc).
"""
function pbasis(N::Integer, xc, x)
    psi = (x - xc) / (1 - xc)
    return legendre_P_table(N, 2 * psi - 1)[1:N+1]
end

"""
Inner solve: for a FIXED theta_c, solve the (N+1) Galerkin kinematic conditions
eq:galerkin for chat, given the pressure-free gap residual C_free(x) = x - mu
(the undeformed onset state: all bath/drop modes zero, centre of mass at mu).

Returns (chat, cond_inner, am, beta, f, A, x, omega, w).
"""
function inner_solve(p::Params, theta_c::Float64, mu::Float64, delta::Float64)
    A, _, _, _, w, x, omega = compliance(p, theta_c, zeros(p.L + 1), delta)
    xc = cos(theta_c)
    N = p.N
    B = [pbasis(N, xc, xi) for xi in x]                       # B[i][n+1]
    n = length(x)
    G = zeros(N + 1, N + 1); rhs = zeros(N + 1)
    for nn in 0:N
        for np in 0:N
            s = 0.0
            for i in 1:n
                inner = 0.0
                for j in 1:n
                    inner += A[i, j] * B[j][np+1]
                end
                s += omega[i] * B[i][nn+1] * inner
            end
            G[nn+1, np+1] = s
        end
        rhs[nn+1] = -sum(omega[i] * B[i][nn+1] * (x[i] - mu) for i in 1:n)
    end
    chat = G \ rhs
    cond_inner = cond(G)

    # recover the physical state this pressure induces
    pv = [sum(chat[nn+1] * B[i][nn+1] for nn in 0:N) for i in 1:n]
    a = 1.5
    kappa_m = [(-2 * delta^2 * p.k[m+1] * tanh(p.k[m+1] * p.h0)) /
               (a * (a + 4 * delta * p.Oh * p.k[m+1]^2) +
                delta^2 * (p.k[m+1]^2 + p.Bo) * p.k[m+1] * tanh(p.k[m+1] * p.h0))
               for m in 0:p.M]
    lambda_l = [l < 2 ? 0.0 :
                (-delta^2 * (2l + 1) * l) /
                (a * (a + 2 * delta * p.Oh * (2l + 1) * (l - 1)) +
                 delta^2 * l * (l - 1) * (l + 2)) for l in 0:p.L]
    kappa_cm = 3 * delta^2 / (2 * a^2)
    Ptab = [legendre_P_table(p.L, xi) for xi in x]
    r = [sqrt(1 - xi^2) for xi in x]
    am = zeros(p.M + 1)
    for m in 1:p.M
        km = p.k[m+1]
        cm = 2 / (p.b * besselj1(km))^2 *
             sum(omega[j] * pv[j] * besselj0(km * r[j]) * w[j] for j in 1:n)
        am[m+1] = kappa_m[m+1] * cm
    end
    beta = zeros(p.L + 1)
    for l in 2:p.L
        bl = sum(omega[j] * pv[j] * Ptab[j][l+1] for j in 1:n)
        beta[l+1] = lambda_l[l+1] * bl
    end
    f = 2 * sum(omega[j] * pv[j] * w[j] for j in 1:n)
    return chat, cond_inner, am, beta, f, pv, x, omega, w, kappa_cm
end

"""Evaluate C(theta) = eta(r(theta)) - z_cm + xi(theta) cos(theta) on a theta grid."""
function C_of_theta(p::Params, am, beta, zcm, thetas)
    out = similar(thetas)
    for (i, th) in enumerate(thetas)
        xx = cos(th)
        Pt = legendre_P_table(p.L, xx)
        xi = 1 + sum(beta[l+1] * Pt[l+1] for l in 2:p.L)
        rr = xi * sin(th)
        eta = sum(am[m+1] * besselj0(p.k[m+1] * rr) for m in 1:p.M)
        out[i] = eta - zcm + xi * xx
    end
    return out
end

