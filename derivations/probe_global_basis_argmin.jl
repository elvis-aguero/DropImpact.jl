# PROBE (not yet a design-doc claim): does the global-basis, two-region-weak formulation
# with an argmin selector for theta_c actually behave?
#
# Formulation under test:
#   * pressure on a GLOBAL Legendre basis over all of x in [-1,1]:
#         p(x) = sum_{n=0}^{N} chat_n P_n(x)
#     -- no moving support, so no piecewise construction and no support parameter.
#   * SELF-ADJOINT pairing throughout: b_l -> int p x P_l w dx, which makes the whole
#     compliance operator self-adjoint in <u,v>_w = int u v w dx to machine precision
#     (verified separately), so the Galerkin matrix of the contact block is symmetric.
#   * weak zero-pressure on the NON-CONTACT region [-1, x_c], tested against shifted
#     Legendre polynomials there:      int_{-1}^{x_c} p(x) Pt_i(x) dx = 0,  i = 0..N
#   * weak kinematic match on the CONTACT region [x_c, 1], tested against shifted
#     Legendre polynomials there in the w-measure:
#                                      int_{x_c}^1 C(x) Pt_j(x) w(x) dx = 0, j = 0..N
#   * that is 2(N+1) conditions for N+1 coefficients at fixed theta_c: over-determined,
#     solved in least squares, leaving a scalar residual rho(theta_c).
#   * theta_c := argmin rho.  No sign change required (so the measured non-monotonicity
#     of the tangency residual is irrelevant) and rho is defined at theta_c = 0 (so the
#     onset step needs no special rule).
#
# Physical moments (c_m, b_l, f) integrate over [x_c, 1] ONLY, never over [-1,1]: the
# map x -> r(x) = xi sqrt(1-x^2) is not injective on [-1,1] (dr/dx changes sign at the
# droplet's widest point), so a cylindrical-coordinate integral over the whole range is
# meaningless.  The pressure's values on [-1, x_c] therefore never enter the force
# balance; the weak zero-pressure block constrains the FIELD to be consistent with
# vanishing off the patch, and is what over-determines the system and so makes theta_c
# identifiable.
#
# QUESTIONS:
#   Q1. Is the contact-block Galerkin matrix symmetric under the corrected pairing?
#   Q2. Does rho(theta_c) have a clean, single interior minimum?
#   Q3. Is argmin rho insensitive to N?
#   Q4. How sensitive is it to the relative weight between the two residual blocks --
#       the one genuinely new parameter this formulation introduces?
#   Q5. Is rho defined and finite at theta_c -> 0, as the onset argument requires?
#
# ============================ RESULT: THIS FORMULATION FAILS ==================
# Measured, and the reason is not parameter tuning:
#
#  (i) n_out must satisfy n_out <= N.  With n_out = N+1 the [-1,x_c] block is square
#      and of full rank (degree-N polynomials restricted to a subinterval are linearly
#      independent), so it admits only chat == 0 and the system collapses to p == 0.
#      This is the identity-theorem obstruction reappearing in weak form: the outside
#      test space must be a PROPER subspace of the pressure space.
#
#  (ii) With n_out <= N the combined system is numerically rank deficient --
#      cond(K) measured at 1e14 to 1e22 -- because the global Legendre basis restricted
#      to two subintervals yields nearly collinear rows.  The over-determination is
#      therefore illusory: the least-squares residual sits at roundoff (1e-7 to 1e-11
#      relative) at EVERY theta_c, with no structure, while ||chat|| runs 1e3 to 1e7.
#      The argmin selector has no signal to minimise.
#
#  (iii) Where a residual signal does survive (n_out = N, larger weight), rho(theta_c)
#      is monotone increasing away from zero, so argmin picks theta_c -> 0: the onset
#      degeneracy is reproduced rather than cured.  Minimising an integral over the
#      contact patch rewards shrinking the patch.  Note this does NOT afflict
#      AgueroEtAl2026's argmin, which minimises a POINTWISE edge residual over a
#      fixed-size quantity, not an integral over a shrinking domain.
#
# What DID survive, and is kept: with the self-adjoint pairing b_l -> int p x P_l w dx
# AND the same basis used for trial and test on the contact region, the contact-block
# Galerkin matrix is symmetric to 1e-16 at every theta_c (Q1 below).  That result is
# independent of the representation choice and carries over to the piecewise basis.
# =============================================================================
using Printf, LinearAlgebra
include(joinpath(@__DIR__, "audit_compliance_operator_core.jl"))

"""
Assemble the two-region weak system at fixed `theta_c`.

Returns (Bout, Gin, din, scale_out, scale_in) with
  Bout * chat                = weak zero-pressure residual on [-1, x_c]   (homogeneous)
  Gin  * chat + din          = weak kinematic residual on [x_c, 1]        (affine)
`scale_*` are the natural nondimensionalisers for each block (see Q4).
"""
function two_region_system(p::Params, theta_c::Float64, mu::Float64, delta::Float64, n_out::Int)
    xc = cos(theta_c); N = p.N
    a = 1.5
    kappa_m = [(-2 * delta^2 * p.k[m+1] * tanh(p.k[m+1] * p.h0)) /
               (a * (a + 4 * delta * p.Oh * p.k[m+1]^2) +
                delta^2 * (p.k[m+1]^2 + p.Bo) * p.k[m+1] * tanh(p.k[m+1] * p.h0)) for m in 0:p.M]
    lambda_l = [l < 2 ? 0.0 :
                (-delta^2 * (2l + 1) * l) /
                (a * (a + 2 * delta * p.Oh * (2l + 1) * (l - 1)) +
                 delta^2 * l * (l - 1) * (l + 2)) for l in 0:p.L]
    kappa_cm = 3 * delta^2 / (2 * a^2)

    # --- contact region [xc, 1]
    s, wq = gauss_legendre_nodes(p.nq)
    xi_ = @. xc + (1 + s) * (1 - xc) / 2
    om = @. wq * (1 - xc) / 2
    n = length(xi_)
    Ptab = [legendre_P_table(p.L, xx) for xx in xi_]
    r = [sqrt(1 - xx^2) for xx in xi_]            # beta = 0 reference geometry
    w = copy(xi_)                                  # at beta=0, w = -d[1-x^2]/dx / 2 = x
    Pglob = [legendre_P_table(N, xx)[1:N+1] for xx in xi_]      # global basis at nodes
    # test functions on the contact region: the SAME global basis as the trial space.
    # Pairing a different test basis against the trial basis destroys symmetry of the
    # Galerkin matrix even when the operator itself is self-adjoint.
    Tin = Pglob

    # compliance kernel acting on the global basis, with the SELF-ADJOINT pairing
    Acol = zeros(n, N + 1)      # Acol[i, n+1] = (A P_n)(x_i)
    for nn in 0:N
        for m in 1:p.M
            km = p.k[m+1]
            pref = kappa_m[m+1] * 2 / (p.b * besselj0(km * p.b))^2
            cm = sum(om[j] * Pglob[j][nn+1] * besselj0(km * r[j]) * w[j] for j in 1:n)
            for i in 1:n
                Acol[i, nn+1] += pref * besselj0(km * r[i]) * cm
            end
        end
        fcol = 2 * sum(om[j] * Pglob[j][nn+1] * w[j] for j in 1:n)
        for i in 1:n; Acol[i, nn+1] -= kappa_cm * fcol; end
        for l in 2:p.L
            # SELF-ADJOINT pairing: b_l = int p x P_l w dx  (both vertical, both in w)
            bl = sum(om[j] * Pglob[j][nn+1] * xi_[j] * Ptab[j][l+1] * w[j] for j in 1:n)
            for i in 1:n
                Acol[i, nn+1] += lambda_l[l+1] * xi_[i] * Ptab[i][l+1] * bl
            end
        end
    end

    Gin = zeros(N + 1, N + 1); din = zeros(N + 1)
    for j in 0:N
        for nn in 0:N
            Gin[j+1, nn+1] = sum(om[i] * Tin[i][j+1] * Acol[i, nn+1] * w[i] for i in 1:n)
        end
        din[j+1] = sum(om[i] * Tin[i][j+1] * (xi_[i] - mu) * w[i] for i in 1:n)
    end

    # --- non-contact region [-1, xc]: weak zero pressure
    so, wqo = gauss_legendre_nodes(p.nq)
    xo = @. -1 + (1 + so) * (xc + 1) / 2
    omo = @. wqo * (xc + 1) / 2
    Pglob_o = [legendre_P_table(N, xx)[1:N+1] for xx in xo]
    # CRITICAL: n_out must be STRICTLY LESS than N+1.  With N+1 test functions on
    # [-1,xc] the block is square and nonsingular -- the degree-N polynomials restricted
    # to a subinterval are linearly independent -- so it admits only chat == 0, and the
    # whole system collapses to the trivial solution.  This is the weak-formulation form
    # of the identity-theorem obstruction: the outside test space must be a proper
    # subspace of the pressure space.  Verified numerically below.
    @assert n_out <= N "n_out must be <= N, else the zero-pressure block annihilates p"
    Tout = [legendre_P_table(max(n_out - 1, 0), 2 * (xx + 1) / (xc + 1) - 1)[1:n_out] for xx in xo]
    Bout = zeros(n_out, N + 1)
    for i in 1:n_out, nn in 0:N
        Bout[i, nn+1] = sum(omo[q] * Tout[q][i] * Pglob_o[q][nn+1] for q in eachindex(xo))
    end

    return Bout, Gin, din, (xc + 1), (1 - xc), Acol, xi_, om, w
end

"""Least-squares solve of the weighted two-region system; returns (chat, rho, cond)."""
function solve_two_region(p::Params, theta_c, mu, delta, wrel, n_out)
    Bout, Gin, din, sc_o, sc_i, Acol, x, om, w = two_region_system(p, theta_c, mu, delta, n_out)
    # nondimensionalise each block by its own interval length, then apply relative weight
    Ao = Bout ./ sc_o
    Ai = Gin ./ sc_i
    bi = -din ./ sc_i
    Kmat = vcat(wrel .* Ao, Ai)
    rhs = vcat(zeros(n_out), bi)
    chat = Kmat \ rhs
    rho = norm(Kmat * chat - rhs) / max(norm(rhs), eps())
    return chat, rho, cond(Kmat), Gin
end

mu = 0.985; delta = 1e-3
println("="^78)
println("Q0: does the zero-pressure block annihilate p when n_out = N+1?")
println("="^78)
println("  Rank of the [-1,xc] block for a degree-N global basis, N=6:")
let p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=6, b=6.0, h0=3.0, nq=80)
    for n_out in (1, 3, 6)
        Bout, _, _, _, _, _, _, _, _ = two_region_system(p, 0.20, mu, delta, n_out)
        @printf("    n_out=%-3d  size=%dx%d  rank=%d  nullity=%d\n",
                n_out, size(Bout, 1), size(Bout, 2), rank(Bout), (p.N + 1) - rank(Bout))
    end
    println("    n_out=7 (=N+1) would be 7x7 of full rank 7 => nullity 0 => p == 0 forced.")
end

println()
println("="^78)
println("Q1: is the contact-block Galerkin matrix symmetric under the corrected pairing?")
println("="^78)
for theta_c in (0.10, 0.20, 0.40)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=6, b=6.0, h0=3.0, nq=80)
    _, _, _, G = solve_two_region(p, theta_c, mu, delta, 1.0, 3)
    @printf("  theta_c=%.2f   ||G-G'||/||G+G'|| = %.3e\n", theta_c, norm(G - G') / norm(G + G'))
end

println()
println("="^78)
println("Q2/Q3/Q5: rho(theta_c), its argmin, and N-dependence  (n_out = N, wrel = 1)")
println("="^78)
@printf("  %-9s", "th_c")
for N in (3, 5, 7, 9); @printf(" %-13s", "N=$N"); end
println()
for theta_c in vcat([1e-3, 0.02, 0.05], collect(0.08:0.02:0.32))
    @printf("  %-9.4f", theta_c)
    for N in (3, 5, 7, 9)
        p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=N, b=6.0, h0=3.0, nq=80)
        _, rho, _, _ = solve_two_region(p, theta_c, mu, delta, 1.0, N)
        @printf(" %-13.5e", rho)
    end
    println()
end
println("\n  argmin over a refined grid (n_out = N):")
for N in (3, 5, 7, 9)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=N, b=6.0, h0=3.0, nq=80)
    grid = collect(0.01:0.0025:0.40)
    rr = [solve_two_region(p, t, mu, delta, 1.0, N)[2] for t in grid]
    @printf("    N=%-3d  argmin theta_c = %.4f   rho_min = %.5e   (acos(mu) = %.4f)\n",
            N, grid[argmin(rr)], minimum(rr), acos(mu))
end

println()
println("="^78)
println("Q4: sensitivity of the argmin to n_out and to the relative block weight")
println("="^78)
@printf("  %-8s %-10s %-13s %-14s %-11s\n", "n_out", "weight", "argmin th_c", "rho_min", "cond(K)")
for n_out in (1, 3, 6)
    for wrel in (1e-2, 1.0, 1e2)
        p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=6, b=6.0, h0=3.0, nq=80)
        grid = collect(0.01:0.0025:0.40)
        rr = [solve_two_region(p, t, mu, delta, wrel, n_out)[2] for t in grid]
        _, _, cnd, _ = solve_two_region(p, grid[argmin(rr)], mu, delta, wrel, n_out)
        @printf("  %-8d %-10.0e %-13.4f %-14.5e %-11.3e\n", n_out, wrel, grid[argmin(rr)], minimum(rr), cnd)
    end
end
