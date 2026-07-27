# Core data types: physical/numerical parameters, per-subsystem state, the two-level
# BDF2 history, and the free-flight/in-contact phase.
#
# Bath and drop mode vectors are stored 1-indexed with a `l+1`/`m+1` offset so that
# `a[m+1] == a_m` and `beta[l+1] == beta_l` directly, matching legendre_P_table's own
# 1-indexed convention (`beta[1]`, `beta[2]` for l=0,1 are always zero/unused, per
# design doc §subsec:drop: l=0 excluded by incompressibility, l=1 governed by the
# centre of mass instead).

@enum Phase FreeFlight InContact

"""
    Params

Physical and numerical parameters for one simulation, fixed for its whole duration.

- `We, Bo, Oh`: Weber, Bond, Ohnesorge numbers (design doc §subsec:formulation).
- `M`: bath modes, m = 0..M.  `L`: drop modes, l = 2..L.  `N`: pressure polynomial degree.
- `b`: bath radius (nondim by droplet radius R).  `h0`: bath depth (nondim by R).
- `nq`: number of Gauss-Legendre quadrature points for the c_m/W_n^{(m)} projections.
- `k`: bath eigenvalues, length M+1, `k[m+1] = k_m` (roots of J1(k_m b)=0, plus k_0=0).
- `gauss_nodes, gauss_weights`: precomputed nq-point Gauss-Legendre rule on [-1,1], used
  for the genuinely transcendental `c_m`/`W_n^{(m)}` projections (eq:c_m-Wn), where `nq`
  is a real numerical-accuracy trade-off chosen by the user.
- `com_nodes, com_weights`: a SEPARATE, larger Gauss-Legendre rule sized to be exactly
  (not approximately) accurate for the polynomial COM-force integrand (design doc
  eq:com), whose degree is fully determined by `N,L` rather than a free choice — see
  `min_nq_for_exact_com`.
"""
struct Params
    We::Float64
    Bo::Float64
    Oh::Float64
    M::Int
    L::Int
    N::Int
    b::Float64
    h0::Float64
    nq::Int
    k::Vector{Float64}
    gauss_nodes::Vector{Float64}
    gauss_weights::Vector{Float64}
    com_nodes::Vector{Float64}
    com_weights::Vector{Float64}
end

"""
    min_nq_for_exact_com(N, L) -> Int

Minimum number of Gauss-Legendre points making the COM-force integral (design doc
eq:com) exact: the integrand `p(x) * d[r(x)²]/dx` has degree `N + 2L + 1` in `x`
(pressure degree `N`, `r² = ξ²(1-x²)` degree `2L+2`), and an `nq`-point rule is exact
for polynomials up to degree `2nq-1`.
"""
min_nq_for_exact_com(N::Integer, L::Integer) = cld(N + 2L + 2, 2)

function Params(; We, Bo, Oh, M, L, N, b, h0, nq)
    kvals = bessel_zeros_J1(M) ./ b
    nodes, weights = gauss_legendre_nodes(nq)
    com_nq = min_nq_for_exact_com(N, L)
    com_nodes, com_weights = gauss_legendre_nodes(com_nq)
    return Params(We, Bo, Oh, M, L, N, b, h0, nq, kvals, nodes, weights, com_nodes, com_weights)
end

"""BathModeState: `a[m+1] = a_m(τ)`, `adot[m+1] = ȧ_m(τ)`, m = 0..M."""
struct BathModeState
    a::Vector{Float64}
    adot::Vector{Float64}
end
BathModeState(M::Integer) = BathModeState(zeros(M + 1), zeros(M + 1))

"""DropModeState: `beta[l+1] = β_l(τ)`, `betadot[l+1] = β̇_l(τ)`, l = 0..L
(entries l=0,1 always zero — see module docstring)."""
struct DropModeState
    beta::Vector{Float64}
    betadot::Vector{Float64}
end
DropModeState(L::Integer) = DropModeState(zeros(L + 1), zeros(L + 1))

"""COMState: centre-of-mass height `z` and velocity `v`."""
struct COMState
    z::Float64
    v::Float64
end

"""
    Level

One fully-specified, immutable time level. `X` is `nothing` during free flight, or the
converged `(N+2)`-vector `(hat_c_0,...,hat_c_N,theta_c)` while in contact (design doc
eq:newton-unknowns). `dt` is the step size that PRODUCED this level (needed to form
`s^k = dt_this_step / dt` — i.e. this level's `dt` plays the role of "δ_t^{k-1}" for
the next step's BDF2 coefficients, design doc eq:bdf2).
"""
struct Level
    bath::BathModeState
    drop::DropModeState
    com::COMState
    t::Float64
    dt::Float64
    X::Union{Nothing,Vector{Float64}}
end

"""SimHistory: the two time levels BDF2 needs, `prev` (k-1) and `curr` (k)."""
mutable struct SimHistory
    prev::Level
    curr::Level
end
