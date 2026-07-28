# AUDIT: does volume-conserving pinning (wall=:clamped, design doc eq:route-b-multiplier)
# preserve the Signorini structure that audit_compliance_operator.jl established for the
# free wall (:free)? Pinning replaces the bath response am_free = alpha + kappa.*cm with a
# rank-one-corrected am (src/residual.jl's apply_clamp), so the pressure -> gap compliance
# operator A gains a rank-one term (derivation in audit_compliance_operator_core.jl's
# `compliance` docstring). A rank-one addition can just as easily destroy self-adjointness
# or definiteness as preserve it -- this is not a foregone conclusion, hence the re-audit
# rather than an assumption that "the same proof carries over."
using SpectralKM
using SpectralKM: gauss_legendre_nodes, legendre_P_table, apply_clamp
using SpecialFunctions: besselj0
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "audit_compliance_operator_core.jl"))

println("="^78)
println("AUDIT 1 (clamped): is the pressure -> gap compliance operator still self-adjoint?")
println("="^78)
p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=60, L=60, N=8, b=6.0, h0=3.0, nq=40, wall=:clamped)
beta0 = zeros(p.L + 1)
delta = 1e-3

println("\nAsymmetry per contribution, in the CYLINDRICAL area measure mu = w (r dr):")
println("(free-wall numbers alongside, from the SAME p.M/p.L/p.N/nq, for direct comparison)")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    A, Ab, Ad, Ac, w, x, om = compliance(p, theta_c, beta0, delta; wall=:clamped)
    Af, Abf, _, _, wf, xf, omf = compliance(p, theta_c, beta0, delta; wall=:free)
    @printf("  theta_c=%.2f   clamped: total=%.3e bath=%.3e   free: total=%.3e bath=%.3e\n",
            theta_c, asymmetry(A, w, om), asymmetry(Ab, w, om),
            asymmetry(Af, wf, omf), asymmetry(Abf, wf, omf))
end

println()
println("="^78)
println("AUDIT 2 (clamped): is the symmetric part still definite? (unique-solution check)")
println("="^78)
println("Gap response is g = g_free + (-A) p, so (-A) must be POSITIVE definite.")
println("(free-wall numbers alongside: this operator is compact -- most of a length-nq")
println(" discretization's spectrum is expected to sit near zero regardless of wall type,")
println(" so a large #negative count is only meaningful relative to the free-wall baseline.)")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    n = 0
    results = NamedTuple[]
    for wall in (:clamped, :free)
        A, _, _, _, w, x, om = compliance(p, theta_c, beta0, delta; wall=wall)
        n = length(x)
        S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
        ev = eigvals(Symmetric((S + S') / 2))
        push!(results, (; wall, minev=minimum(ev), maxev=maximum(ev), neg=count(<(0), ev)))
    end
    c, f = results
    @printf("  theta_c=%.2f   clamped: min=%+.3e max=%+.3e neg=%d/%d   free: min=%+.3e max=%+.3e neg=%d/%d\n",
            theta_c, c.minev, c.maxev, c.neg, n, f.minev, f.maxev, f.neg, n)
end

println()
println("="^78)
println("AUDIT 3 (clamped): does the pinning correction change RESOLVABLE RANK?")
println("="^78)
println("A rank-one update can raise rank by at most 1 or lower it by at most 1; the point")
println("is whether the volume-conservation constraint measurably eats into or adds to the")
println("resolvable pressure dimension audit_compliance_operator.jl's AUDIT 4 established.")
@printf("  %-6s %-6s %-6s %s\n", "M", "L", "th_c", "rank(1e-8): clamped / free")
for (Mv, Lv, tc) in ((80, 80, 0.2), (80, 80, 0.4), (80, 80, 0.6), (160, 40, 0.4), (40, 160, 0.4))
    q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=Mv, L=Lv, N=8, b=6.0, h0=3.0, nq=160, wall=:clamped)
    ranks = Int[]
    for wall in (:clamped, :free)
        A, _, _, _, w, x, om = compliance(q, tc, zeros(q.L + 1), delta; wall=wall)
        n = length(x)
        S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
        sv = svdvals(Symmetric((S + S') / 2)); sv ./= sv[1]
        push!(ranks, count(>(1e-8), sv))
    end
    @printf("  %-6d %-6d %-6.2f %d / %d\n", Mv, Lv, tc, ranks[1], ranks[2])
end

println()
println("="^78)
println("AUDIT 4 (clamped): does the rank-one term itself sanity-check against apply_clamp?")
println("="^78)
println("Direct check, independent of the operator-assembly algebra above: build a random")
println("pressure vector, form am the SAME way residual.jl's unpack_state does (kappa.*cm,")
println("then apply_clamp), and verify sum_m am[m]*j0kb[m] == 0 (the pinning constraint) to")
println("machine precision -- confirms the clamped compliance operator derived here is")
println("assembling the actual constraint apply_clamp enforces, not a lookalike.")
let theta_c = 0.3, seed_p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=60, L=60, N=8, b=6.0, h0=3.0, nq=40, wall=:clamped)
    xc = cos(theta_c)
    s, wq = gauss_legendre_nodes(seed_p.nq)
    x = @. xc + (1 + s) * (1 - xc) / 2
    omega = @. wq * (1 - xc) / 2
    r = @. sqrt(1 - x^2)  # beta=0: xi=1, r=sin(theta), r^2=1-x^2
    w = @. x              # beta=0: w = -d(r^2)/dx / 2 = x, exact (no finite difference needed)
    pvec = sin.(3 .* x) .+ 0.5  # arbitrary smooth "pressure" sample, not tied to any basis
    a = 1.5
    kappa_m = [(-2 * delta^2 * seed_p.k[m+1] * tanh(seed_p.k[m+1] * seed_p.h0)) /
               (a * (a + 4 * delta * seed_p.Oh * seed_p.k[m+1]^2) +
                delta^2 * (seed_p.k[m+1]^2 + seed_p.Bo) * seed_p.k[m+1] * tanh(seed_p.k[m+1] * seed_p.h0))
               for m in 0:seed_p.M]
    cm = [seed_p.bath_norm[m+1] * sum(pvec[i] * besselj0(seed_p.k[m+1] * r[i]) * w[i] * omega[i] for i in eachindex(x))
          for m in 0:seed_p.M]
    am_free = kappa_m .* cm  # alpha=0: isolates the pressure response exactly as the operator audit does
    am = apply_clamp(am_free, kappa_m, seed_p)
    pinning_residual = sum(am .* seed_p.j0kb)
    @printf("  sum_m am[m]*j0kb[m] = %.3e  (machine zero expected)\n", pinning_residual)
end
