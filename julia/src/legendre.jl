# Legendre polynomials, their derivatives, and the arbitrary-order Bonnet-recursion
# antiderivatives H_l^{(k)}(x) = ∫ x^k P_l(x) dx (design doc eq:bonnet).
#
# Hand-rolled via the standard three-term recurrence rather than a library call:
# the recurrence is the numerically stable way to evaluate Legendre polynomials at
# high degree (MATLAB's builtin `legendre()` is known to become unstable around
# l ~ 100 for exactly this kind of high-order evaluation; the recurrence itself does
# not share that failure mode). Generic in `eltype(x)` throughout so ForwardDiff.Dual
# flows through untouched.

"""
    legendre_P_table(lmax, x) -> Vector

`P_0(x), ..., P_{lmax}(x)` via the three-term recurrence
`(l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)`.
"""
function legendre_P_table(lmax::Integer, x::T) where {T}
    P = Vector{T}(undef, lmax + 1)
    P[1] = one(T)
    if lmax >= 1
        P[2] = x
    end
    for l in 1:(lmax-1)
        P[l+2] = ((2l + 1) * x * P[l+1] - l * P[l]) / (l + 1)
    end
    return P
end

legendre_P(l::Integer, x) = legendre_P_table(l, x)[l+1]

"""
    legendre_dP_table(lmax, x) -> Vector

`P_0'(x), ..., P_{lmax}'(x)` via the stable forward recurrence
`P_{l+1}'(x) = x P_l'(x) + (l+1) P_l(x)`, `P_0'=0, P_1'=1`
(avoids the `1/(1-x^2)` singular form at `x = ±1`).
"""
function legendre_dP_table(lmax::Integer, x::T) where {T}
    P = legendre_P_table(lmax, x)
    dP = Vector{T}(undef, lmax + 1)
    dP[1] = zero(T)
    if lmax >= 1
        dP[2] = one(T)
    end
    for l in 1:(lmax-1)
        dP[l+2] = x * dP[l+1] + (l + 1) * P[l+1]
    end
    return dP
end

legendre_dP(l::Integer, x) = legendre_dP_table(l, x)[l+1]

"""
    bonnet_H(lmax, kmax, x) -> Matrix

`H[l+1, k+1] = H_l^{(k)}(x) := ∫ x^k P_l(x) dx`, for `l = 0..lmax`, `k = 0..kmax`
(design doc eq:bonnet), via
`H_l^{(0)} = F_l = (P_{l+1}(x) - P_{l-1}(x)) / (2l+1)` (with the convention `P_{-1} ≡ 0`,
which reproduces `F_0(x) = x = ∫ P_0 dx` exactly), and
`H_l^{(k)} = [(l+1) H_{l+1}^{(k-1)} + l H_{l-1}^{(k-1)}] / (2l+1)` for `k ≥ 1`
(with the convention `H_{-1}^{(k-1)} ≡ 0`, never actually needed since its coefficient
is `l = 0`).

Computing `H_l^{(k)}` for `l` up to `lmax` at order `k` requires `H_{l+1}^{(·)}` at
order `k-1`, so the underlying `P_l` / `H_l^{(0)}` table is built out to `l = lmax + kmax`
and then narrowed one row per increasing `k`.
"""
function bonnet_H(lmax::Integer, kmax::Integer, x::T) where {T}
    lext = lmax + kmax
    P = legendre_P_table(lext + 1, x)  # P[l+1] = P_l(x), l = 0..lext+1

    # H0[l+1] = H_l^{(0)}(x) = F_l(x), l = 0..lext  (P_{-1} := 0)
    H0 = Vector{T}(undef, lext + 1)
    for l in 0:lext
        Pl_minus1 = l == 0 ? zero(T) : P[l]      # P[l] = P_{l-1}
        H0[l+1] = (P[l+2] - Pl_minus1) / (2l + 1)
    end

    H = Matrix{T}(undef, lmax + 1, kmax + 1)
    Hprev = H0
    for l in 0:lmax
        H[l+1, 1] = Hprev[l+1]
    end
    for k in 1:kmax
        lrange = lext - k  # valid upper l-index of Hcur, shrinks by 1 each k
        Hcur = Vector{T}(undef, lrange + 1)
        for l in 0:lrange
            Hl_minus1 = l == 0 ? zero(T) : Hprev[l]      # Hprev[l] = H_{l-1}^{(k-1)}
            Hl_plus1 = Hprev[l+2]                        # Hprev[l+2] = H_{l+1}^{(k-1)}
            Hcur[l+1] = ((l + 1) * Hl_plus1 + l * Hl_minus1) / (2l + 1)
        end
        for l in 0:min(lmax, lrange)
            H[l+1, k+1] = Hcur[l+1]
        end
        Hprev = Hcur
    end
    return H
end
