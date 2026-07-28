# The bath-side projection c_m(X) (design doc eq:c_m-def) — the AD-criticality hotspot,
# since J0 composed with the geometric map r(x,τ) has no elementary antiderivative and
# must be evaluated by Gauss-Legendre quadrature (quadrature.jl) regardless of pressure
# basis, unlike the polynomial projections in pressure.jl.
#
# NORMALIZATION (design doc eq:bessel-norm): the no-flux wall condition gives
# J_0'(k_m b) = 0, i.e. J_1(k_m b) = 0, so ∫_0^b J_0(k_m r)² r dr = (b²/2) J_0(k_m b)²
# and the Fourier-Bessel weight is 2/(b J_0(k_m b))². An earlier version of this file
# used 2/(b J_1(k_m))², reproduced verbatim from AlventosaEtAl2023 eq. (300); that is
# the DIRICHLET normalizer, evaluated at k_m rather than k_m b, and is inconsistent
# with the no-flux basis both that paper and this one impose. The error is
# mode-dependent and locally severe: at b=6 the ratio of their weight to the correct
# one ranges over 0.16 to 1.8 for m=1..8 but reaches 116 at m=7, where k_7=3.7933 falls
# close to the first zero of J_1 at 3.8317, so J_1(k_m) → 0 and their weight diverges.
# (An earlier version of this comment printed that list inverted.)
#
# The m=0 "piston" mode (k_0=0) needs no special case under the correct normalization:
# J_0(0)=1, so its weight is the finite 2/b², matching ∫_0^b 1·r dr = b²/2. Its affine
# slope κ_0 (eq:kappa-m) is exactly zero regardless, so a_0 evolves purely from BDF2
# history, decoupled from the pressure.

"""
    c_m_all(chat, xc, x, wq, r, w, k, bath_norm) -> Vector

`c_m(τ)` (design doc eq:c_m-def) for `m = 0..M`, from PRECOMPUTED node data: the
Fourier-Bessel projection of the contact pressure against `J_0(k_m r)` in the cylindrical
area measure `r dr = w dx`.

This is the one projection with no closed form regardless of pressure basis -- `J_0`
composed with the geometric map has no elementary antiderivative -- so quadrature here is
not a convenience but a necessity (design doc §subsubsec:quadrature).
"""
function c_m_all(chat::AbstractVector, xc, x::AbstractVector, wq::AbstractVector,
    r::AbstractVector, w::AbstractVector, k::Vector{Float64}, bath_norm::Vector{Float64})
    M = length(k) - 1
    T = promote_type(eltype(chat), typeof(xc), eltype(w))
    out = Vector{T}(undef, M + 1)
    pw = [pressure_poly_raw(chat, xc, x[i]) * w[i] * wq[i] for i in eachindex(x)]
    @inbounds for m in 0:M
        km = k[m+1]
        s = zero(T)
        for i in eachindex(x)
            s += pw[i] * besselj0(km * r[i])
        end
        out[m+1] = bath_norm[m+1] * s
    end
    return out
end
