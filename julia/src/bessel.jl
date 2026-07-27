# Bath eigenvalues k_m: roots of J_0'(k_m b) = 0, i.e. (since J_0' = -J_1) positive
# zeros of J_1, plus the trivial k_0 = 0 "piston" mode (design doc §subsec:bath).
#
# SpecialFunctions.jl is used for the Bessel function EVALUATIONS themselves (J0, J1) —
# hand-rolling a robust Bessel function is materially riskier than hand-rolling Legendre
# (large-argument asymptotic behavior is easy to get subtly wrong), so this is a
# correctness-critical dependency rather than a style one. The zero-FINDER, however, is
# hand-rolled (Newton's method seeded by McMahon's asymptotic formula), matching the
# house style of the sister repos and giving exact control over convergence tolerance.

"""
    mcmahon_seed_J1(m) -> Float64

McMahon's asymptotic approximation to the `m`-th positive zero of `J_1`, `m ≥ 1`,
used only as a Newton-iteration seed (not returned as a final answer).
"""
function mcmahon_seed_J1(m::Integer)
    m < 1 && throw(ArgumentError("m must be ≥ 1 (m=0 is the trivial k_0=0 mode)"))
    β = (m + 1//4) * π
    μ = 4.0  # 4ν² at ν=1
    return β - (μ - 1) / (8β)
end

besselj1_prime(x) = x == 0 ? 0.5 : besselj0(x) - besselj1(x) / x

"""
    refine_zero_J1(seed; tol=1e-13, maxiter=50) -> Float64

Newton's method on `J_1(x) = 0` from the given seed, using
`J_1'(x) = J_0(x) - J_1(x)/x`.
"""
function refine_zero_J1(seed::Float64; tol::Float64=1e-13, maxiter::Int=50)
    x = seed
    for _ in 1:maxiter
        fx = besselj1(x)
        abs(fx) < tol && return x
        x -= fx / besselj1_prime(x)
    end
    return x
end

"""
    bessel_zeros_J1(M) -> Vector{Float64}

Length `M+1` vector `[0.0, j_{1,1}, ..., j_{1,M}]`: the trivial `k_0=0` bath mode
followed by the first `M` positive zeros of `J_1`, refined to `~1e-13`.
"""
function bessel_zeros_J1(M::Integer)
    zeros_vec = Vector{Float64}(undef, M + 1)
    zeros_vec[1] = 0.0
    for m in 1:M
        zeros_vec[m+1] = refine_zero_J1(mcmahon_seed_J1(m))
    end
    return zeros_vec
end
