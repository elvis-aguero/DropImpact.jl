# The bath-side projections c_m(X), W_n^{(m)}(X) (design doc eq:c_m-Wn) — the AD-
# criticality hotspot, since J0 composed with the geometric map r(x,τ) has no
# elementary antiderivative and must be evaluated by Gauss-Legendre quadrature
# (quadrature.jl), not the closed-form Bonnet machinery used in pressure.jl.
#
# Special case: the m=0 "piston" bath mode has k_0=0, and eq:c_m-def's normalization
# 2/(b J1(k_m))² is singular at k_0=0 (J1(0)=0). This is harmless rather than a bug:
# k_0=0 also multiplies the forcing term in the bath mode ODE (eq:bath-mode) and the
# affine slope κ_0 (eq:kappa-m) is exactly zero, so c_0's actual value never affects
# anything physical — a_0 evolves purely from BDF2 history, decoupled from pressure
# entirely (this degeneracy was flagged during the design's own derivation audit).
# c_0 is therefore fixed to 0 by convention rather than evaluating the singular
# normalization. W_n^{(0)}, by contrast, carries no such normalization and is well
# defined (J0(0)=1), so it is computed normally.

"""
    c_m_all(chat, xc, beta, L, k, b, nodes, weights) -> Vector

`c_m(τ)` (design doc eq:c_m-def, via eq:c_m-Wn) for `m = 0..M` (`length(k)-1`).
`c_m_all[1]` (the `m=0` piston mode) is fixed to `0` by convention — see module note.
"""
function c_m_all(chat::AbstractVector, xc, beta::AbstractVector, L::Integer,
    k::Vector{Float64}, b::Float64, nodes::Vector{Float64}, weights::Vector{Float64})
    M = length(k) - 1
    T = promote_type(eltype(chat), typeof(xc), eltype(beta))
    out = Vector{T}(undef, M + 1)
    out[1] = zero(T)
    for m in 1:M
        km = k[m+1]
        integrand(x) = begin
            r = r_of_x(beta, x, L)
            pressure_poly_raw(chat, xc, x) * besselj0(km * r) * abs(r_drdx_of_x(beta, x, L))
        end
        norm = 2 / (b * besselj1(km))^2
        out[m+1] = norm * gauss_quad(integrand, xc, nodes, weights)
    end
    return out
end

"""
    W_nm_all(chat_unused_for_shape, xc, beta, L, k, N, nodes, weights) -> Matrix

`W_n^{(m)}(τ)` (design doc eq:c_m-Wn, in the rescaled `ψ=(x-xc)/(1-xc)` basis — see
pressure.jl's module note) for `n=0..N`, `m=0..M`, returned as an `(N+1)×(M+1)`
matrix, used for the bath contribution to the Galerkin residual
(`Σ_m a_m(τ) W_n^{(m)}(τ)`). Does not depend on `chat` (unlike `c_m`) — only on the
pressure basis's support `[xc,1]` via the `ψ(x)^n` test-function weights — but takes
`N` explicitly rather than inferring it from `chat` since it is used before `chat` is
otherwise needed in some call sites.
"""
function W_nm_all(xc, beta::AbstractVector, L::Integer, k::Vector{Float64}, N::Integer,
    nodes::Vector{Float64}, weights::Vector{Float64})
    M = length(k) - 1
    T = promote_type(typeof(xc), eltype(beta))
    out = Matrix{T}(undef, N + 1, M + 1)
    w = 1 - xc
    for m in 0:M
        km = k[m+1]
        for n in 0:N
            integrand(x) = besselj0(km * r_of_x(beta, x, L)) * ((x - xc) / w)^n
            out[n+1, m+1] = gauss_quad(integrand, xc, nodes, weights)
        end
    end
    return out
end
