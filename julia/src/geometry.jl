# The forward geometric map (design doc eq:forward-map) and its theta-derivatives for
# the tangency condition (eq:tangency-lhs/rhs). Two equivalent parametrizations are
# provided deliberately: the x=cos(theta) forms (used by the Galerkin/quadrature
# integrands in pressure.jl/bessel_moments.jl, where x is the natural integration
# variable) and the literal theta forms (used only for the tangency row of R(X), kept
# as close as possible to the audited eq:tangency-lhs/rhs/explicit rather than
# re-derived via a chain-rule substitution into x, to avoid introducing a fresh,
# unaudited sign error). Both are generic in `eltype(beta)`.
#
# The forward map r(theta,tau)=xi(theta,tau)*sin(theta) is used ONLY forward here —
# never inverted for theta given r — consistent with the whole point of this closure.

"""`ξ(x,τ) = 1 + Σ_{l=2}^L β_l P_l(x)` (design doc eq:eta-xi), as a function of `x=cosθ`."""
function xi_of_x(beta::AbstractVector, x, L::Integer)
    P = legendre_P_table(L, x)
    s = one(promote_type(eltype(beta), typeof(x)))
    for l in 2:L
        s += beta[l+1] * P[l+1]
    end
    return s
end

"""`r(x,τ) = ξ(x,τ)√(1-x²)` (design doc eq:forward-map, in the `x=cosθ` parametrization)."""
r_of_x(beta::AbstractVector, x, L::Integer) = xi_of_x(beta, x, L) * sqrt(1 - x^2)

"""`z_d(x,τ) = ξ(x,τ) x` (design doc eq:forward-map, in the `x=cosθ` parametrization)."""
zd_of_x(beta::AbstractVector, x, L::Integer) = xi_of_x(beta, x, L) * x

"""`dξ/dx` (derivative w.r.t. `x` itself, not `θ` — distinct from `xi_theta` below;
needed for `dr/dx` inside the bath-side quadrature integrands in bessel_moments.jl)."""
function xi_prime_of_x(beta::AbstractVector, x, L::Integer)
    dP = legendre_dP_table(L, x)
    s = zero(promote_type(eltype(beta), typeof(x)))
    for l in 2:L
        s += beta[l+1] * dP[l+1]
    end
    return s
end

"""`dr/dx = ξ'(x)√(1-x²) - ξ(x) x/√(1-x²)`, from `r(x,τ)=ξ(x,τ)√(1-x²)`.

Individually singular at `x=1` (the symmetry axis, θ=0): the `1/√(1-x²)` term diverges.
Gauss-Legendre quadrature nodes never land exactly on `x=1`, so this is never actually
evaluated there in practice, but for the `r * dr/dx` COMBINATION that appears in the
`c_m` integrand (design doc eq:c_m-Wn), use `r_drdx_of_x` below instead — the
`1/√(1-x²)` cancels analytically in that product, so evaluating the simplified,
non-singular closed form avoids the `0×∞` floating-point cancellation this function
would otherwise suffer arbitrarily close to `x=1`."""
function dr_dx(beta::AbstractVector, x, L::Integer)
    s = sqrt(1 - x^2)
    return xi_prime_of_x(beta, x, L) * s - xi_of_x(beta, x, L) * x / s
end

"""`r(x,τ) dr/dx`, in the analytically-simplified, non-singular form
`ξ ξ' (1-x²) - ξ² x` (the `1/√(1-x²)` from `dr/dx` cancels exactly against the
`√(1-x²)` from `r`) — equivalently `(1/2) d[r²]/dx`, matching the same simplification
already used in `com_force_closed`'s integrand."""
function r_drdx_of_x(beta::AbstractVector, x, L::Integer)
    xi = xi_of_x(beta, x, L)
    dxi = xi_prime_of_x(beta, x, L)
    return xi * dxi * (1 - x^2) - xi^2 * x
end

xi_of_theta(beta::AbstractVector, theta, L::Integer) = xi_of_x(beta, cos(theta), L)

"""`r(θ,τ) = ξ(θ,τ) sinθ` (design doc eq:forward-map)."""
forward_map_r(beta::AbstractVector, theta, L::Integer) = xi_of_theta(beta, theta, L) * sin(theta)

"""`z_d(θ,τ) = ξ(θ,τ) cosθ` (design doc eq:forward-map)."""
forward_map_zd(beta::AbstractVector, theta, L::Integer) = xi_of_theta(beta, theta, L) * cos(theta)

"""`w(x,τ) := -½ d[r²]/dx = ξ²x - ξξ'(1-x²)`, the cylindrical area-element weight, so
that `r dr = w dx` (design doc eq:compliance-kernel).

Equals `r|dr/dx|` — and hence is non-negative — only where `dr/dx ≤ 0`, i.e. only for
contact patches not reaching the droplet's widest point. That restriction is design doc
eq:check-monotone-r (equivalently `r_c < r_M` in AgueroEtAl2026's notation) and is
checked by the time stepper's admissibility filter, not assumed here."""
w_of_x(beta::AbstractVector, x, L::Integer) = -r_drdx_of_x(beta, x, L)

# ---------------------------------------------------------------------------
# Precomputed node data.
#
# PERFORMANCE, not cosmetics: the moment integrals (b_l, c_m, f) and the Galerkin
# residual all sweep the same quadrature nodes, and each needs the full Legendre table
# there. Recomputing `legendre_P_table(L, x)` inside a mode loop -- as earlier versions
# of b_l_all and c_m_all did -- costs O(L) per mode per node, i.e. O(L^2 n_q) per call,
# and that call sits inside a fixed-point iteration inside an AD sweep inside a
# candidate loop inside a time step. At production L this is the difference between a
# simulation that finishes and one that does not.
#
# The Legendre tables depend only on the nodes, hence only on `xc` -- NOT on `beta` or
# on the pressure coefficients -- so they are plain Float64 and are never differentiated.
# Only the geometry built from them (`xi`, `r`, `w`) carries the generic element type.
# ---------------------------------------------------------------------------

"""Quadrature nodes and weights mapped from `[-1,1]` onto `[xc,1]` (design doc
eq:gauss-quad)."""
function mapped_nodes(xc, nodes::Vector{Float64}, weights::Vector{Float64})
    x = similar(nodes, typeof(xc))
    wq = similar(nodes, typeof(xc))
    half = (1 - xc) / 2
    @inbounds for i in eachindex(nodes)
        x[i] = xc + (1 + nodes[i]) * half
        wq[i] = weights[i] * half
    end
    return x, wq
end

"""`P[i][l+1] = P_l(x_i)` and `dP[i][l+1] = P_l'(x_i)`, computed once per node."""
function legendre_tables(x::AbstractVector, L::Integer)
    P = [legendre_P_table(L, xi) for xi in x]
    dP = [legendre_dP_table(L, xi) for xi in x]
    return P, dP
end

"""`xi`, `r`, and `w = -½d[r²]/dx` at every node, from precomputed Legendre tables."""
function geom_at_nodes(beta::AbstractVector, x::AbstractVector, P, dP, L::Integer)
    T = promote_type(eltype(beta), eltype(x))
    n = length(x)
    xi = Vector{T}(undef, n); r = Vector{T}(undef, n); w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s = one(T); ds = zero(T)
        for l in 2:L
            s += beta[l+1] * P[i][l+1]
            ds += beta[l+1] * dP[i][l+1]
        end
        xi[i] = s
        r[i] = s * sqrt(1 - x[i]^2)
        w[i] = s^2 * x[i] - s * ds * (1 - x[i]^2)     # = -½ d[r²]/dx
    end
    return xi, r, w
end
