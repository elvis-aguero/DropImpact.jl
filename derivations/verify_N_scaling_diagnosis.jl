# Diagnosing WHY cond(J) blows up with pressure truncation order N (found in
# verify_crossing_conditioning.jl to survive the acceleration-level fix largely
# unchanged in kind, only in magnitude) — separating two structurally different
# explanations that call for opposite fixes:
#
#   (1) NUMERICAL: the ψ(x)^n=(x-xc)^n/(1-xc)^n basis functions are nearly collinear
#       on [xc,1] (a Vandermonde/Hilbert-matrix effect, present in the monomial basis
#       ITSELF, no physics involved) — fixable by an orthogonal-on-[xc,1] basis.
#   (2) INFORMATIONAL: modes N>=2 are not independently observable from this
#       contact patch/test-function set given the actual physics coupling (the
#       nonlinear a_m(X) chain, Bessel oscillation) — a genuine identifiability
#       limit no basis change removes.
#
# This script does NOT assume which one dominates; it measures each ingredient
# separately, then looks at the actual singular-value spectrum (a cliff means (2),
# a smooth geometric decay means (1)), and varies theta_c to check whether this is
# the SAME small-contact-patch issue pressure.jl's own module note already names,
# or persists at moderate/large contact patches too.
#
# Run with: julia --project=../julia verify_N_scaling_diagnosis.jl   (from derivations/)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralKM
using SpectralKM: BathModeState, DropModeState, COMState, bath_affine, drop_affine,
    com_affine, gauss_quad, r_of_x, zd_of_x, b_l_all, com_force_closed, c_m_all,
    accel_galerkin_term, forward_map_r, forward_map_zd, moment_matrix_psi, W_nm_all
using SpecialFunctions: besselj0
using ForwardDiff
using LinearAlgebra
using Printf

# ---- shared harness (duplicated from verify_crossing_conditioning.jl for a
# standalone, self-contained diagnostic script) -----------------------------
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

function position_galerkin_term(am, beta, zcm, xc, p::Params)
    N = p.N; L = p.L; w = 1 - xc
    T = promote_type(eltype(am), eltype(beta), typeof(zcm), typeof(xc))
    out = Vector{T}(undef, N + 1)
    for n in 0:N
        function integrand(x)
            r = r_of_x(beta, x, L)
            eta = zero(T)
            for m in eachindex(am)
                eta += am[m] * besselj0(p.k[m] * r)
            end
            return (eta - zcm + zd_of_x(beta, x, L)) * ((x - xc) / w)^n
        end
        out[n+1] = gauss_quad(integrand, xc, p.gauss_nodes, p.gauss_weights)
    end
    return out
end

function residual_variant(variant::Symbol, X::AbstractVector, kappa, alpha, lambda, gam,
    kappa_cm, mu, bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N
    chat, theta_c, xc, beta, zcm, am = unpack_dag(X, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    galerkin = if variant == :position
        position_galerkin_term(am, beta, zcm, xc, p)
    else
        cm = c_m_all(chat, xc, beta, p.L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
        bl = b_l_all(chat, xc, p.L)
        f = com_force_closed(chat, xc, beta, p.L, p.com_nodes, p.com_weights)
        accel_galerkin_term(chat, xc, bath_frozen.a, bath_frozen.adot, drop_frozen.beta,
            drop_frozen.betadot, cm, bl, f, p)
    end
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

function affine_at(p::Params, dt::Float64)
    bath, drop, com = accumulated_contact_history(p)
    kappa, alpha = bath_affine(bath, bath, p, dt, dt)
    lambda, gam = drop_affine(drop, drop, p, dt, dt)
    kappa_cm, mu = com_affine(com, com, p, dt, dt)
    return kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop
end

representative_X0(N::Integer, theta_c0::Float64) = begin
    base = [1.0, 0.35, 0.12, 0.04]
    chat0 = [base[min(n + 1, 4)] for n in 0:N]
    [chat0; theta_c0]
end

make_params(N::Integer) = Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=N, b=6.0, h0=3.0, nq=20)

# ---------------------------------------------------------------------------
# (1) Pure basis-only conditioning: no physics, no a_m(X)/beta_l(X) coupling at all.
#
# (1a) Hilbert-matrix-like Gram matrix of the ψ^n basis alone:
#      G[n,m] = ∫_{xc}^1 ψ(x)^n ψ(x)^m dx = (1-xc)/(n+m+1)   (n,m=0..N).
#      Its conditioning is INDEPENDENT of xc (an overall positive scale factor
#      does not change cond()) and independent of physics entirely — pure
#      monomial-basis collinearity on the unit interval, the textbook Hilbert
#      matrix effect.
# (1b) moment_matrix_psi(xc,L,N): the actual linear map chat -> b_l used
#      throughout the codebase (droplet-side Legendre projection of the
#      pressure basis) — still no a_m(X)/Bessel coupling, but uses the REAL
#      Legendre test functions instead of a synthetic self-Gram matrix.
# ---------------------------------------------------------------------------
println("=" ^ 78)
println("(1) Pure basis-only conditioning (no physics)")
println("=" ^ 78)
println("\n-- (1a) Hilbert-like Gram matrix of psi^n basis alone, cond independent of xc --")
@printf("%-3s  %-14s\n", "N", "cond(Gram)")
for N in 0:3
    G = [1.0 / (n + m + 1) for n in 0:N, m in 0:N]
    @printf("%-3d  %-14.4e\n", N, cond(G))
end

println("\n-- (1b) moment_matrix_psi(xc,L,N): chat -> b_l (Legendre) projection, xc=cos(0.3) --")
xc_ref = cos(0.3)
@printf("%-3s  %-14s\n", "N", "cond(M)")
for N in 0:3
    M = moment_matrix_psi(xc_ref, 10, N)
    @printf("%-3d  %-14.4e\n", N, cond(M))
end

println("\n-- (1b') W_nm_all(xc,beta,L,k,N,...): a_m -> Galerkin bath-contribution map, xc=cos(0.3), beta=0 --")
p_ref = make_params(3)
@printf("%-3s  %-14s\n", "N", "cond(W)")
for N in 0:3
    Wm = W_nm_all(xc_ref, zeros(11), 10, p_ref.k, N, p_ref.gauss_nodes, p_ref.gauss_weights)
    @printf("%-3d  %-14.4e\n", N, cond(Wm))
end

# ---------------------------------------------------------------------------
# (2) Full nonlinear Jacobian conditioning (physics included), for direct
# comparison against (1) at the same N -- if (2) tracks (1) closely, the basis
# alone explains the blowup; if (2) is orders of magnitude worse, the nonlinear
# a_m(X)/Bessel coupling is the dominant contributor, not the basis.
# ---------------------------------------------------------------------------
println()
println("=" ^ 78)
println("(2) Full nonlinear Jacobian cond(J) (accel-level rows), same N, theta_c=0.3, dt=1e-3")
println("=" ^ 78)
@printf("%-3s  %-14s\n", "N", "cond(J) accel")
for N in 0:3
    p = make_params(N)
    kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p, 1e-3)
    X0 = representative_X0(N, 0.3)
    R(X) = residual_variant(:accel, X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J = ForwardDiff.jacobian(R, X0)
    @printf("%-3d  %-14.4e\n", N, cond(J))
end

# ---------------------------------------------------------------------------
# (3) Singular-value SPECTRUM at N=3 (not just the min/max ratio): a smooth
# geometric decay across all 5 singular values indicates ordinary (basis)
# ill-conditioning; a sharp cliff after the 2nd or 3rd indicates a genuine
# identifiability gap (those directions in X-space are simply not observed).
# ---------------------------------------------------------------------------
println()
println("=" ^ 78)
println("(3) Full singular-value spectrum at N=3 (accel-level rows), theta_c=0.3")
println("=" ^ 78)
p3 = make_params(3)
kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p3, 1e-3)
X0 = representative_X0(3, 0.3)
R3(X) = residual_variant(:accel, X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p3)
J3 = ForwardDiff.jacobian(R3, X0)
sv = svdvals(J3)
for (i, s) in enumerate(sv)
    @printf("  sigma_%d = %.6e   (ratio to sigma_1: %.3e)\n", i, s, s / sv[1])
end

# ---------------------------------------------------------------------------
# (4) theta_c sweep at fixed N=3: is this the SAME small-contact-patch
# Vandermonde issue pressure.jl's own module note already names (should get
# BETTER as theta_c grows / contact patch widens), or does it persist at
# moderate/large patches too (a different, undisclosed issue)?
# ---------------------------------------------------------------------------
println()
println("=" ^ 78)
println("(4) theta_c sweep at fixed N=3 (accel-level rows), dt=1e-3")
println("=" ^ 78)
@printf("%-10s  %-14s\n", "theta_c", "cond(J) accel")
for theta_c0 in (0.05, 0.1, 0.3, 0.6, 1.0, 1.4)
    p = make_params(3)
    kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(p, 1e-3)
    X0 = representative_X0(3, theta_c0)
    R(X) = residual_variant(:accel, X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J = ForwardDiff.jacobian(R, X0)
    @printf("%-10.2f  %-14.4e\n", theta_c0, cond(J))
end
