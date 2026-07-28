# Shared core for the audit scripts: assembly of the model's pressure -> gap
# compliance operator and the weighted-asymmetry measure.  Extracted verbatim from
# audit_compliance_operator.jl so audit_nested_closure.jl can reuse it.
using SpectralKM
using SpectralKM: gauss_legendre_nodes, legendre_P_table
using SpecialFunctions: besselj0, besselj1
using LinearAlgebra

"""
Assemble the discrete pressure -> C-displacement operator at frozen geometry.

Pressure is represented POINTWISE at the `nq` Gauss nodes of [x_c, 1] -- no
polynomial truncation -- so what is measured is a property of the continuous
model, not of a basis choice.  Returns (A, w, x, omega) where

    C_pressure(x_i) = sum_j A[i,j] p_j ,

`w` is the cylindrical area element weight (r |dr/dx|, so that r dr = w dx),
`x` the nodes, `omega` the quadrature weights on [x_c, 1].

`wall = :clamped` adds the pinning correction of `apply_clamp`
(src/residual.jl): with `am_free = alpha + kappa.*cm`, the clamped response is
`am = am_free + (2/b)*kappa.*Lambda./j0kb`, `Lambda = -sum(am_free.*j0kb)/D`,
`D = (2/b)*sum(kappa)`. Only the `cm`-linear (pressure-dependent) part of
`am_free` feeds the compliance operator (`alpha` is background forcing, not a
function of the current pressure), so substituting `cm[n] = bath_norm[n+1] *
sum_j p_j J_0(k_n r_j) w_j omega_j` (`c_m_all`, src/bessel_moments.jl) turns the
correction into a RANK-ONE update of `A_bath`:

    A_bath_clamped[i,j] = A_bath_free[i,j] - (2/(b*D)) * v[i] * u[j] * omega[j] * w[j]
    v[i] = sum_m (kappa_m/j0kb_m) J_0(k_m r_i)
    u[j] = sum_m kappa_m * j0kb_m * bath_norm_m * J_0(k_m r_j)

(sum over m=1:M; the m=0 piston has kappa_0=0 and drops out, matching
`apply_clamp` itself, which sums over the full range but gets zero from that
term). This is exactly the operator identity the re-audit needs to check: does
volume-conserving pinning preserve the free-wall operator's self-adjoint,
positive-definite structure, or does the rank-one correction break it?
"""
function compliance(p::Params, theta_c::Float64, beta::Vector{Float64}, delta::Float64;
    wall::Symbol=:free)
    xc = cos(theta_c)
    s, wq = gauss_legendre_nodes(p.nq)
    x = @. xc + (1 + s) * (1 - xc) / 2
    omega = @. wq * (1 - xc) / 2

    # frozen droplet geometry: xi(x), r(x), and w = r |dr/dx|
    Ptab = [legendre_P_table(p.L, xi) for xi in x]
    xi = [1 + sum(beta[l+1] * Ptab[i][l+1] for l in 2:p.L) for i in eachindex(x)]
    r = @. xi * sqrt(1 - x^2)
    # d(r^2)/dx by finite difference of the analytic polynomial r^2 = xi^2 (1-x^2)
    r2(xx) = begin
        Pt = legendre_P_table(p.L, xx)
        xiv = 1 + sum(beta[l+1] * Pt[l+1] for l in 2:p.L)
        xiv^2 * (1 - xx^2)
    end
    h = 1e-7
    dr2 = [(r2(xx + h) - r2(xx - h)) / (2h) for xx in x]
    w = @. -dr2 / 2                       # >= 0 over the patch (r grows as x falls)

    # BDF2 affine slopes at a single step of size delta (s = 1, so a = 3/2)
    a = 1.5
    kappa_m = [(-2 * delta^2 * p.k[m+1] * tanh(p.k[m+1] * p.h0)) /
               (a * (a + 4 * delta * p.Oh * p.k[m+1]^2) +
                delta^2 * (p.k[m+1]^2 + p.Bo) * p.k[m+1] * tanh(p.k[m+1] * p.h0))
               for m in 0:p.M]
    lambda_l = [l < 2 ? 0.0 :
                (-delta^2 * (2l + 1) * l) /
                (a * (a + 2 * delta * p.Oh * (2l + 1) * (l - 1)) +
                 delta^2 * l * (l - 1) * (l + 2)) for l in 0:p.L]
    kappa_cm = 3 * delta^2 / (2 * a^2)

    n = length(x)
    A_bath = zeros(n, n); A_drop = zeros(n, n); A_com = zeros(n, n)
    # m=0 (k_0=0) is the piston mode: kappa_0 = 0 exactly and its normalization is
    # singular, so it is excluded by the same convention bessel_moments.jl uses.
    for m in 1:p.M
        km = p.k[m+1]
        # Fourier-Bessel normalization for the NO-FLUX basis J_0'(k_m b)=0: the squared
        # norm is int_0^b J_0(k_m r)^2 r dr = (b^2/2) J_0(k_m b)^2, so the coefficient
        # weight is 2/(b J_0(k_m b))^2.  NOT 2/(b J_1(k_m))^2, which is the Dirichlet
        # normalizer with the wrong argument -- an error inherited from
        # AlventosaEtAl2023 eq. (300) and corrected in docs eq:c_m-def.
        pref = kappa_m[m+1] * 2 / (p.b * besselj0(km * p.b))^2
        J = [besselj0(km * ri) for ri in r]
        for i in 1:n, j in 1:n
            A_bath[i, j] += pref * J[i] * J[j] * omega[j] * w[j]
        end
    end
    for l in 2:p.L
        for i in 1:n, j in 1:n
            A_drop[i, j] += lambda_l[l+1] * x[i] * Ptab[i][l+1] * Ptab[j][l+1] * omega[j]
        end
    end
    for i in 1:n, j in 1:n
        A_com[i, j] = -2 * kappa_cm * omega[j] * w[j]
    end

    if wall === :clamped
        v = zeros(n); u = zeros(n)
        for m in 1:p.M
            km = p.k[m+1]
            J = [besselj0(km * ri) for ri in r]
            v .+= (kappa_m[m+1] / p.j0kb[m+1]) .* J
            u .+= (kappa_m[m+1] * p.j0kb[m+1] * p.bath_norm[m+1]) .* J
        end
        D = (2 / p.b) * sum(kappa_m)
        correction = [-(2 / (p.b * D)) * v[i] * u[j] * omega[j] * w[j] for i in 1:n, j in 1:n]
        A_bath = A_bath + correction
    end

    return A_bath + A_drop + A_com, A_bath, A_drop, A_com, w, x, omega
end

"""
Relative asymmetry of `A` in the weighted inner product <u,v> = sum_i mu_i omega_i u_i v_i,
i.e. how far mu_i A_ij / omega_j is from symmetric.  Returns the relative Frobenius
deviation; 0 means exactly self-adjoint (Betti-reciprocal).
"""
function asymmetry(A, mu, omega)
    n = size(A, 1)
    B = [mu[i] * A[i, j] / omega[j] for i in 1:n, j in 1:n]
    return norm(B - B') / norm(B + B')
end

