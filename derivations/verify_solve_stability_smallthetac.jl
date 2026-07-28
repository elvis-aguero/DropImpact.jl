# SUPERSEDED -- RETAINED FOR THE RECORD ONLY.  This script verifies claims about
# the practical solve stability of the joint Newton Jacobian, which no longer exists:
# the inner system's condition number is measured at order ten, uniformly in delta.
# Superseded by audit_nested_closure.jl.
#
# It is no longer cited by docs/next-gen-KM-model.tex and its conclusions no longer
# describe the model.  Do not treat a passing run here as support for current theory.

# Does cond(J) ~ 1e12-1e19 (found at small theta_c, N=3) actually translate into a
# practically unstable/inaccurate SOLVE STEP, using the real solver this codebase uses
# (Levenberg-Marquardt-damped least squares, newton.jl -- not a naive J\R, precisely
# because newton.jl's own module note already found severe ill-conditioning and chose
# LM damping to be robust to it)? cond(J) is a worst-case linear-algebra bound; it says
# nothing on its own about whether a REALISTIC perturbation (upstream floating-point
# noise in the frozen history alpha/gam/mu, not an adversarial worst-case direction)
# actually gets amplified anywhere near that bound once the actual damped solver is used.
#
# IMPORTANT METHODOLOGICAL CORRECTION vs. an earlier attempt at this same test: running
# full `newton_solve` from a small-theta_c SEED is not a valid test of the small-theta_c
# regime -- Newton is free to (and, empirically, does) walk the iterate all the way to
# theta_c~0.94 regardless of the seed, silently escaping the catastrophic-cond(J) region
# entirely and testing nothing about it. This version instead evaluates ONE damped LM
# step AT a FIXED point with small theta_c (not iterated to convergence), which is the
# only way to actually probe the solver's local behavior in the bad region rather than
# wherever a full solve happens to wander.
#
# Run with: julia --project=../julia verify_solve_stability_smallthetac.jl  (from derivations/)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralKM
using SpectralKM: BathModeState, DropModeState, COMState, bath_affine, drop_affine,
    com_affine, gauss_quad, r_of_x, zd_of_x, b_l_all, com_force_closed, c_m_all,
    accel_galerkin_term, forward_map_r, forward_map_zd
using SpecialFunctions: besselj0
using ForwardDiff
using LinearAlgebra
using Printf

# ---- shared harness ----------------------------------------------------
function unpack_dag(X::AbstractVector, kappa, alpha, lambda, gam, kappa_cm, mu, p::Params)
    N = p.N; L = p.L
    chat = @view X[1:N+1]
    theta_c = X[N+2]
    xc = cos(theta_c)
    bl = b_l_all(chat, xc, L)
    beta = gam .+ lambda .* bl
    f = com_force_closed(chat, xc, beta, L, p.com_nodes, p.com_weights)
    zcm = mu + kappa_cm * f
    cm = c_m_all(chat, xc, beta, L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
    am = alpha .+ kappa .* cm
    return chat, theta_c, xc, beta, zcm, am
end

function crossing_row(am, beta, zcm, theta_c, p::Params)
    rc = forward_map_r(beta, theta_c, p.L)
    zdc = forward_map_zd(beta, theta_c, p.L)
    T = promote_type(eltype(am), eltype(beta), typeof(theta_c), typeof(zcm))
    s = zero(T)
    for m in eachindex(am)
        s += am[m] * besselj0(p.k[m] * rc)
    end
    return s - zcm + zdc
end

function residual_accel(X::AbstractVector, kappa, alpha, lambda, gam, kappa_cm, mu,
    bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N
    chat, theta_c, xc, beta, zcm, am = unpack_dag(X, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    cm = c_m_all(chat, xc, beta, p.L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
    bl = b_l_all(chat, xc, p.L)
    f = com_force_closed(chat, xc, beta, p.L, p.com_nodes, p.com_weights)
    galerkin = accel_galerkin_term(chat, xc, bath_frozen.a, bath_frozen.adot,
        drop_frozen.beta, drop_frozen.betadot, cm, bl, f, p)
    T = promote_type(eltype(X), Float64)
    R = Vector{T}(undef, N + 2)
    for n in 0:N
        R[n+1] = galerkin[n+1]
    end
    R[N+2] = crossing_row(am, beta, zcm, theta_c, p)
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

make_params() = Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=3, b=6.0, h0=3.0, nq=20)

representative_X0(theta_c0::Float64) = [1.0, 0.35, 0.12, 0.04, theta_c0]

"""One damped LM step, (JᵀJ+μI)ΔX = JᵀR(X0), AT a fixed X0 -- not iterated -- exactly
the linear-solve machinery newton_solve.jl uses internally at one iterate."""
function lm_step(R, X0, mu)
    Rval = R(X0)
    J = ForwardDiff.jacobian(R, X0)
    return (J' * J + mu * I(length(X0))) \ (J' * Rval)
end

const REL_PERT = 1e-10

function run_case(theta_c0::Float64, mu::Float64)
    p = make_params()
    bath, drop, com = accumulated_contact_history(p)
    dt = 1e-3
    kappa, alpha = bath_affine(bath, bath, p, dt, dt)
    lambda, gam = drop_affine(drop, drop, p, dt, dt)
    kappa_cm, mu_com = com_affine(com, com, p, dt, dt)

    X0 = representative_X0(theta_c0)
    R0(X) = residual_accel(X, kappa, alpha, lambda, gam, kappa_cm, mu_com, bath, drop, p)
    J0 = ForwardDiff.jacobian(R0, X0)
    condJ = cond(J0)
    dX = lm_step(R0, X0, mu)

    # Realistic upstream perturbation: the frozen BDF2 history intercepts, NOT the
    # pressure unknowns themselves -- exactly the kind of roundoff noise that exists
    # between one converged step and the next.
    alpha_p = alpha .* (1 + REL_PERT)
    gam_p = gam .* (1 + REL_PERT)
    mu_com_p = mu_com * (1 + REL_PERT)
    Rp(X) = residual_accel(X, kappa, alpha_p, lambda, gam_p, kappa_cm, mu_com_p, bath, drop, p)
    dXp = lm_step(Rp, X0, mu)

    d = norm(dXp - dX)
    relstep = d / norm(dX)
    naive_bound = condJ * REL_PERT

    return (; theta_c0, mu, condJ, step_norm=norm(dX), relstep, naive_bound)
end

println("=" ^ 100)
println("ONE fixed-point LM step at small theta_c: does a realistic 1e-10 relative")
println("upstream perturbation get amplified anywhere near cond(J)'s naive bound?")
println("=" ^ 100)
for mu in (1e-6, 1e-2, 1.0)
    println("\n-- LM damping mu = ", mu, " --")
    @printf("%-8s  %-10s  %-12s  %-12s  %-12s\n",
        "theta_c0", "cond(J)", "step norm", "actual relstep", "naive bound")
    for theta_c0 in (0.05, 0.1, 0.3, 0.6, 1.0)
        r = run_case(theta_c0, mu)
        @printf("%-8.2f  %-10.2e  %-12.2e  %-12.2e  %-12.2e   (ratio act/naive: %.2e)\n",
            theta_c0, r.condJ, r.step_norm, r.relstep, r.naive_bound, r.relstep / r.naive_bound)
    end
end
