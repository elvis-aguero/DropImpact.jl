# Numerical conditioning comparison: does replacing the POSITION-level Galerkin rows
# (eq:galerkin) with the ACCELERATION-level rows (eq:accel-constraint/eq:accel-galerkin,
# Section 4 of the design doc) actually fix the catastrophic O(δ²) Jacobian
# ill-conditioning, once the closing row is the CURRENT, approved theory's contact-angle
# equation C(θ_c,τ)=0 (eq:theta-c-crossing) rather than the old, now-abandoned tangency
# row? This has never been checked directly: the design doc's own N=0..3 conditioning
# numbers (§subsec:accel-motivation, 5e5/5e9/1.4e14/7e17) were measured on the
# POSITION-level system before Section 4's fix existed, and the codebase's own regression
# test (test/test_conditioning.jl) only checks flatness-in-dt for the ACCEL-level system
# closed by the OLD tangency/edge-condition rows — never for the crossing-row closure
# this design doc now specifies. This script builds BOTH residual variants directly from
# SpectralKM's own internals (not a re-implementation) so the comparison is apples-to-
# apples: same Galerkin-vs-accel choice for the first N+1 rows, IDENTICAL crossing row
# for the last, for every case.
#
# Answers exactly two questions:
#   (1) N-scan at fixed dt: does cond(J) blow up with N under the position-level rows
#       the way the design doc's own historical numbers show, and does the accel-level
#       replacement actually stay flat instead?
#   (2) dt-scan at fixed N: same question, along the axis the design doc's O(δ²)
#       argument is stated in terms of.
#
# Run with: julia --project=../julia verify_crossing_conditioning.jl   (from derivations/)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralKM
using SpectralKM: BathModeState, DropModeState, COMState, bath_affine, drop_affine,
    com_affine, gauss_quad, r_of_x, zd_of_x, xi_of_x, b_l_all, com_force_closed,
    c_m_all, accel_galerkin_term, forward_map_r, forward_map_zd
using SpecialFunctions: besselj0
using ForwardDiff
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Shared DAG: chat,theta_c -> bl,beta -> f,zcm -> cm,am  (identical to residual.jl's
# own acyclic chain — reused here, not re-derived, so this script cannot silently
# diverge from the production evaluation order).
# ---------------------------------------------------------------------------
function unpack_dag(X::AbstractVector, kappa, alpha, lambda, gam, kappa_cm, mu, p::Params)
    N = p.N
    L = p.L
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

"""`C(θ_c,τ;X) = Σ_m a_m(X) J0(k_m r_c) - z_cm(X) + z_d(θ_c;X)` — eq:theta-c-crossing,
evaluated with the pressure trial's OWN a_m(X),β_l(X) (never frozen state), exactly as
§subsubsec:contact-angle specifies. Shared, unmodified, between both residual variants
below — the whole point of this comparison is to isolate the effect of the Galerkin-row
formulation, not to also vary the closing row."""
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

"""Position-level Galerkin rows (eq:galerkin): `∫_{xc}^1[η-z_cm+z_d](x-xc)^n dx`,
built directly from `am,beta,zcm` (the SAME quantities the accel-level variant uses),
in the same ψ=(x-xc)/(1-xc) rescaled basis `accel_galerkin_term` uses, for an
apples-to-apples comparison of conditioning, not of basis choice (design doc's own
argument, §subsec:accel-motivation: a basis change alone cannot fix this)."""
function position_galerkin_term(am, beta, zcm, xc, p::Params)
    N = p.N
    L = p.L
    w = 1 - xc
    T = promote_type(eltype(am), eltype(beta), typeof(zcm), typeof(xc))
    out = Vector{T}(undef, N + 1)
    for n in 0:N
        function integrand(x)
            r = r_of_x(beta, x, L)
            eta = zero(T)
            for m in eachindex(am)
                eta += am[m] * besselj0(p.k[m] * r)
            end
            zd = zd_of_x(beta, x, L)
            return (eta - zcm + zd) * ((x - xc) / w)^n
        end
        out[n+1] = gauss_quad(integrand, xc, p.gauss_nodes, p.gauss_weights)
    end
    return out
end

"""Full R(X) ∈ R^{N+2}: first N+1 rows per `variant` (:position or :accel), last row
always `crossing_row` (eq:theta-c-crossing) — the CURRENT, approved design, not the
tangency/edge-condition rows `residual.jl`/`residual_edge` still implement."""
function residual_variant(variant::Symbol, X::AbstractVector, kappa, alpha, lambda, gam,
    kappa_cm, mu, bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N
    chat, theta_c, xc, beta, zcm, am = unpack_dag(X, kappa, alpha, lambda, gam, kappa_cm, mu, p)

    galerkin = if variant == :position
        position_galerkin_term(am, beta, zcm, xc, p)
    elseif variant == :accel
        cm = c_m_all(chat, xc, beta, p.L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
        bl = b_l_all(chat, xc, p.L)
        f = com_force_closed(chat, xc, beta, p.L, p.com_nodes, p.com_weights)
        accel_galerkin_term(chat, xc, bath_frozen.a, bath_frozen.adot, drop_frozen.beta,
            drop_frozen.betadot, cm, bl, f, p)
    else
        error("unknown variant $variant")
    end

    T = promote_type(eltype(X), Float64)
    R = Vector{T}(undef, N + 2)
    for n in 0:N
        R[n+1] = galerkin[n+1]
    end
    R[N+2] = crossing_row(am, beta, zcm, theta_c, p)
    return R
end

# ---------------------------------------------------------------------------
# Representative synthetic states. Two cases, since the accel-level fix's own
# motivation is about history-dependence (κ,λ ~ O(δ²) regardless of the state they're
# applied to): a FRESH-CONTACT case (all history zero, matching test_conditioning.jl's
# existing regression convention exactly, so results are comparable to it) and an
# ACCUMULATED-CONTACT case (nonzero a_m,β_l/rates, order-of-magnitude matched to the
# real We≈0.96 pressure/deflection data gathered earlier this session — pressure O(1),
# deflection O(0.1) — so the comparison isn't an artifact of starting from all-zero
# history).
# ---------------------------------------------------------------------------
function fresh_contact_history(p::Params)
    bath = BathModeState(p.M)
    drop = DropModeState(p.L)
    com = COMState(0.98, -0.9)
    return bath, drop, com
end

function accumulated_contact_history(p::Params)
    a = [0.15 / (m + 1) for m in 0:p.M]
    adot = [-0.4 / (m + 1) for m in 0:p.M]
    bath = BathModeState(a, adot)
    beta = zeros(p.L + 1)
    betadot = zeros(p.L + 1)
    for l in 2:p.L
        beta[l+1] = 0.1 / (l - 1)
        betadot[l+1] = -0.3 / (l - 1)
    end
    drop = DropModeState(beta, betadot)
    com = COMState(0.85, -0.6)
    return bath, drop, com
end

"""Build (kappa,alpha,lambda,gam,kappa_cm,mu,bath_curr,drop_curr) at a representative
step from a synthetic (curr,curr) history pair (dtprev=dt, steady-stepping BDF2, not
the BDF1 cold-start limit) at trial step size `dt`."""
function affine_at(history_fn, p::Params, dt::Float64)
    bath, drop, com = history_fn(p)
    kappa, alpha = bath_affine(bath, bath, p, dt, dt)
    lambda, gam = drop_affine(drop, drop, p, dt, dt)
    kappa_cm, mu = com_affine(com, com, p, dt, dt)
    return kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop
end

"""A representative X0 = (ĉ_0,...,ĉ_N,θ_c), decaying-magnitude pressure coefficients
matched to the real We≈0.96 data's O(1) peak, θ_c in the moderate small-deformation
range this model targets."""
function representative_X0(N::Integer)
    base = [1.0, 0.35, 0.12, 0.04]
    chat0 = [base[min(n + 1, 4)] for n in 0:N]
    return [chat0; 0.3]
end

function cond_for(variant::Symbol, history_fn, p::Params, dt::Float64)
    kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop = affine_at(history_fn, p, dt)
    X0 = representative_X0(p.N)
    R(X) = residual_variant(variant, X, kappa, alpha, lambda, gam, kappa_cm, mu, bath, drop, p)
    J = ForwardDiff.jacobian(R, X0)
    return cond(J)
end

function make_params(N::Integer)
    return Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=N, b=6.0, h0=3.0, nq=20)
end

println("=" ^ 78)
println("(1) N-scan at fixed dt=1e-3: position-level vs acceleration-level Galerkin rows")
println("    (crossing row eq:theta-c-crossing identical in both, per-N)")
println("=" ^ 78)
for history_name in ("fresh-contact (all history zero)", "accumulated-contact (nonzero a_m,beta_l)")
    history_fn = history_name[1] == 'f' ? fresh_contact_history : accumulated_contact_history
    println("\n-- history: ", history_name, " --")
    @printf("%-3s  %-14s  %-14s\n", "N", "cond(J) pos.", "cond(J) accel")
    for N in 0:3
        p = make_params(N)
        cp = cond_for(:position, history_fn, p, 1e-3)
        ca = cond_for(:accel, history_fn, p, 1e-3)
        @printf("%-3d  %-14.4e  %-14.4e\n", N, cp, ca)
    end
end

println()
println("=" ^ 78)
println("(2) dt-scan at fixed N=1: position-level vs acceleration-level Galerkin rows")
println("=" ^ 78)
for history_name in ("fresh-contact (all history zero)", "accumulated-contact (nonzero a_m,beta_l)")
    history_fn = history_name[1] == 'f' ? fresh_contact_history : accumulated_contact_history
    println("\n-- history: ", history_name, " --")
    @printf("%-10s  %-14s  %-14s\n", "dt", "cond(J) pos.", "cond(J) accel")
    p = make_params(1)
    for dt in (1e-2, 1e-3, 1e-4, 1e-5)
        cp = cond_for(:position, history_fn, p, dt)
        ca = cond_for(:accel, history_fn, p, dt)
        @printf("%-10.1e  %-14.4e  %-14.4e\n", dt, cp, ca)
    end
end
