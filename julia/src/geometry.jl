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

"""
`ξ_θ(θ,τ) = -sinθ Σ_{l=2}^L β_l P_l'(cosθ)` (design doc eq:tangency-rhs), the θ-derivative
of `ξ` via the chain rule `d/dθ[P_l(cosθ)] = -sinθ P_l'(cosθ)`.
"""
function xi_theta(beta::AbstractVector, theta, L::Integer)
    x = cos(theta)
    dP = legendre_dP_table(L, x)
    s = zero(promote_type(eltype(beta), typeof(theta)))
    for l in 2:L
        s += beta[l+1] * dP[l+1]
    end
    return -sin(theta) * s
end

"""`r_θ(θ,τ) = ξ_θ sinθ + ξ cosθ` (design doc eq:tangency-lhs)."""
function r_theta(beta::AbstractVector, theta, L::Integer)
    return xi_theta(beta, theta, L) * sin(theta) + xi_of_theta(beta, theta, L) * cos(theta)
end

"""`d[z_d]/dθ = ξ_θ cosθ - ξ sinθ` (design doc eq:tangency-rhs)."""
function zd_theta(beta::AbstractVector, theta, L::Integer)
    return xi_theta(beta, theta, L) * cos(theta) - xi_of_theta(beta, theta, L) * sin(theta)
end
