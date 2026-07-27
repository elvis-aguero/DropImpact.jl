# The joint residual R(X) (design doc eq:newton-residual-tangency and, since Section 4,
# eq:accel-constraint in place of eq:newton-residual-galerkin, §subsec:newton/
# §subsec:accel-galerkin): the sole function meant to be wrapped by ForwardDiff.jacobian.
#
# X = (hat_c_0,...,hat_c_N, theta_c) ∈ R^{N+2} (eq:newton-unknowns). Every other
# quantity is computed from X via the explicit, acyclic (DAG) chain established in
# the design doc: p(x;X) -> b_l(X) -> beta(X) -> {xi,r,z_d}(X) -> c_m(X)/W_n^{(m)}(X)
# -> a_m(X) -> R(X). The affine coefficients (kappa,alpha,lambda,gam,kappa_cm,mu) are
# passed in as plain Float64 data, computed ONCE per timestep by affine.jl and never
# re-differentiated — X is the only argument whose type flows through generically.
#
# Since the acceleration-level fix (Section 4, accel_closure.jl): the first `N+1` rows
# are now the Galerkin projection of C̈=Π+K (eq:accel-constraint), not the position-level
# C — this is the actual fix for the confirmed O(δ²) Jacobian ill-conditioning (newton.jl's
# module note). This requires the FROZEN τ^k state (`bath_frozen`,`drop_frozen` — the
# previous converged Level's a_m,ȧ_m,β_l,β̇_l) as extra plain-Float64 arguments, alongside
# X, never differentiated (same status as kappa/alpha/etc.). The tangency row (last entry)
# is explicitly UNCHANGED — still the position-level eq:tangency-explicit, per the design
# doc's own scoping (§subsec:accel-open: tangency's own O(δ²) sensitivity is disclosed as
# a separate, still-open issue, not fixed here).

"""`(1-xc)/(n+1) = ∫_{xc}^1 ψ(x)^n dx`, `ψ:=(x-xc)/(1-xc)` (dx=(1-xc)dψ, ψ:0→1), the
elementary antiderivative needed for the constant `z_cm` term inside the Galerkin
integral (design doc eq:galerkin, in the rescaled ψ basis — see pressure.jl)."""
elementary_moment(xc, n::Integer) = (1 - xc) / (n + 1)

"""
    residual(X, kappa, alpha, lambda, gam, kappa_cm, mu, bath_frozen, drop_frozen, p) -> Vector

`R(X) ∈ R^{N+2}`: the first `p.N+1` entries are the acceleration-level Galerkin residuals
(eq:accel-constraint, §subsec:accel-galerkin), the last is the (still position-level)
tangency residual (eq:newton-residual-tangency). Generic in `eltype(X)`; `bath_frozen`/
`drop_frozen` (the previous converged Level's `BathModeState`/`DropModeState`, i.e. τ^k)
are plain Float64 data, never differentiated — same status as `kappa`,`alpha`, etc.
"""
function residual(X::AbstractVector, kappa::Vector{Float64}, alpha::Vector{Float64},
    lambda::Vector{Float64}, gam::Vector{Float64}, kappa_cm::Float64, mu::Float64,
    bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N
    L = p.L
    chat = @view X[1:N+1]
    theta_c = X[N+2]
    xc = cos(theta_c)

    # b_l(X) -> beta(X): the acyclic step — depends only on chat, xc, never on geometry.
    bl = b_l_all(chat, xc, L)
    beta = gam .+ lambda .* bl   # length L+1, entries l=0,1 are 0+0*bl=0 automatically

    # f(X) -> z_cm(X)
    f = com_force_closed(chat, xc, beta, L, p.com_nodes, p.com_weights)
    zcm = mu + kappa_cm * f

    # c_m(X) -> a_m(X)
    cm = c_m_all(chat, xc, beta, L, p.k, p.b, p.gauss_nodes, p.gauss_weights)
    am = alpha .+ kappa .* cm

    # acceleration-level Galerkin residual (eq:accel-constraint): Π(x;X) uses cm/bl/f
    # above (current-step pressure) but the FROZEN a_m^k/β_l^k for every outer geometric
    # occurrence; K(x) is built entirely from the frozen state. zcm/am computed above are
    # still needed (for unpack_state / advancing the next Level and the tangency row
    # below), even though they no longer feed the Galerkin rows directly.
    galerkin = accel_galerkin_term(chat, xc, bath_frozen.a, bath_frozen.adot,
        drop_frozen.beta, drop_frozen.betadot, cm, bl, f, p)

    T = promote_type(eltype(X), Float64)
    R = Vector{T}(undef, N + 2)
    for n in 0:N
        R[n+1] = galerkin[n+1]
    end

    # tangency residual (eq:tangency-explicit), reusing geometry.jl's theta-based forms
    rc = forward_map_r(beta, theta_c, L)
    rth = r_theta(beta, theta_c, L)
    zdth = zd_theta(beta, theta_c, L)
    bessel_sum = zero(T)
    for m in eachindex(am)
        km = p.k[m]
        bessel_sum += am[m] * km * besselj1(km * rc)
    end
    R[N+2] = -bessel_sum * rth - zdth

    return R
end

"""
    residual_edge(Y, kappa, alpha, lambda, gam, kappa_cm, mu, bath_frozen, drop_frozen, p) -> Vector

Joint residual with the pressure-vanishing free-boundary condition `p(x_c,τ)=0` used IN
PLACE OF the position-level tangency row (`eq:tangency-explicit`). Tangency was found
(2026-07-26/27 session) to be doubly degenerate: `∂_θC` is odd in `θ` and vanishes
identically at `θ=0` for ANY state (a coordinate-singularity triviality near the axis,
not an `O(δ²)`-conditioning defect further differentiation removes), and separately it
depends on `a_m(X)`, whose history-intercept `α_m` never grows because pressure itself
never got the chance to build it (the `K`/drift issue, `§subsec:accel-open`) — so it was
ALSO near-zero at moderate `θ_c`, not just as `θ_c→0`. `p(x_c,τ)=0` depends only on `ĉ`
(this step's genuinely `O(1)`-responsive pressure, confirmed by direct testing of
`com_force_closed`), sidestepping both failure modes — and is the standard Signorini/
obstacle-problem complementarity condition for a regular pressure field at a free
boundary, not an ad hoc substitute.

In the `ψ`-monomial pressure basis, `p(x_c,τ)=ĉ_0` exactly (`ψ(x_c)=0` kills every other
term), so the condition is simply `ĉ_0≡0` — implemented by removing it from the unknown
vector rather than adding a new equation: `Y=(ĉ_1,...,ĉ_N,θ_c) ∈ R^{N+1}`, closed by the
SAME `N+1` accel-level Galerkin rows `residual` already computes (the tangency row is
dropped entirely, not replaced one-for-one).

Confirmed empirically to escape the universal collapse-to-`θ_c=0` failure that the
tangency-based joint solve exhibited from every tested seed: different seeds now
converge to different, genuinely nonzero `θ_c`, tracking the seed (multiple roots exist,
consistent with the design doc's own disclosed `J0`-multi-rootedness risk — a real but
much more tractable concern than universal collapse, addressed the same way as any
multi-rooted Newton problem, by warm-starting from the previous converged step).
"""
function residual_edge(Y::AbstractVector, kappa::Vector{Float64}, alpha::Vector{Float64},
    lambda::Vector{Float64}, gam::Vector{Float64}, kappa_cm::Float64, mu::Float64,
    bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    N = p.N
    T = eltype(Y)
    chat = Vector{T}(undef, N + 1)
    chat[1] = zero(T)
    for n in 1:N
        chat[n+1] = Y[n]
    end
    theta_c = Y[N+1]
    full = residual(vcat(chat, theta_c), kappa, alpha, lambda, gam, kappa_cm, mu, bath_frozen, drop_frozen, p)
    return full[1:N+1]
end

"""
    unpack_Y_edge(Y, N) -> X

Expands the reduced edge-condition unknown `Y=(ĉ_1,...,ĉ_N,θ_c)` back to the full
`X=(ĉ_0≡0,ĉ_1,...,ĉ_N,θ_c)` convention `unpack_state`/postprocessing/reconstruction
already use, so nothing downstream of the Newton solve needs to know this condition is
in use.
"""
function unpack_Y_edge(Y::AbstractVector{Float64}, N::Integer)
    return [0.0; Y[1:N]; Y[N+1]]
end

"""
    unpack_state(X, kappa, alpha, lambda, gam, kappa_cm, mu, p) -> (am, beta, zcm)

Recompute `a_m(X)`, `β_l(X)` (length `M+1`/`L+1`, 1-indexed), and `z_cm(X)` from a
converged `X` — the same DAG steps `residual` uses internally, exposed separately so
the timestepper can build the next `Level`'s state without re-deriving them.
"""
function unpack_state(X::AbstractVector{Float64}, kappa::Vector{Float64}, alpha::Vector{Float64},
    lambda::Vector{Float64}, gam::Vector{Float64}, kappa_cm::Float64, mu::Float64, p::Params)
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
    return am, beta, zcm
end
