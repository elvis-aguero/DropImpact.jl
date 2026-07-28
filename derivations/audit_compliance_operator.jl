# AUDIT: does the model of paper-formulation.tex possess a variational
# (Signorini / obstacle-problem) structure at the CONTINUOUS level?
#
# A Signorini contact problem is well posed iff the compliance operator A mapping
# contact pressure -> gap displacement is symmetric and positive definite in SOME
# inner product. Symmetry is Betti reciprocity (equivalently: the pressure-gap
# pairing derives from an energy); positive definiteness makes the resulting
# variational inequality strictly convex, hence uniquely solvable.  If A has that
# structure, the free boundary needs NO closing equation at all: it is the edge of
# {p > 0} in the unique minimizer of
#       min_{p >= 0}  1/2 <p, A p> + <p, g_free>.
#
# This script assembles A numerically from the model's OWN projections
# (eq:c_m-def, eq:b_l-def, eq:com, eq:kappa-m--eq:kappa-cm) at frozen geometry and
# measures (i) how far A is from symmetric, (ii) in which term the asymmetry lives,
# (iii) how the asymmetry scales with theta_c, (iv) whether the symmetric part is
# definite, and (v) whether the delta^2 factor is uniform across A (which would make
# it irrelevant to conditioning, since cond(cA) = cond(A) for scalar c).
using SpectralKM
using SpectralKM: gauss_legendre_nodes, legendre_P_table, bessel_zeros_J1
using SpecialFunctions: besselj0, besselj1
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "audit_compliance_operator_core.jl"))

println("="^78)
println("AUDIT 1: is the pressure -> gap compliance operator self-adjoint?")
println("="^78)
p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=60, L=60, N=8, b=6.0, h0=3.0, nq=40)
beta0 = zeros(p.L + 1)
delta = 1e-3

println("\nAsymmetry per contribution, in the CYLINDRICAL area measure mu = w (r dr):")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    A, Ab, Ad, Ac, w, x, om = compliance(p, theta_c, beta0, delta)
    @printf("  theta_c=%.2f   total=%.3e   bath=%.3e   drop=%.3e   com=%.3e\n",
            theta_c, asymmetry(A, w, om), asymmetry(Ab, w, om),
            asymmetry(Ad, w, om), asymmetry(Ac, w, om))
end

println("\nSame, in the SPHERICAL area measure mu = 1 (dx = sin(theta) d(theta)):")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    A, Ab, Ad, Ac, w, x, om = compliance(p, theta_c, beta0, delta)
    one = ones(length(x))
    @printf("  theta_c=%.2f   total=%.3e   bath=%.3e   drop=%.3e   com=%.3e\n",
            theta_c, asymmetry(A, one, om), asymmetry(Ab, one, om),
            asymmetry(Ad, one, om), asymmetry(Ac, one, om))
end

println("\n--- Is the droplet asymmetry the cos(theta) projection mismatch? ---")
println("The drop term pairs pressure against P_l via b_l = int p P_l dx (radial")
println("normal displacement) but returns it to the gap as x*xi (vertical component).")
println("Test: replace b_l -> int p x P_l dx, making both sides vertical, then")
println("re-measure asymmetry in the CONSTANT measure mu = 1.")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    xc = cos(theta_c)
    s, wq = gauss_legendre_nodes(p.nq)
    x = @. xc + (1 + s) * (1 - xc) / 2
    omega = @. wq * (1 - xc) / 2
    Ptab = [legendre_P_table(p.L, xi) for xi in x]
    a = 1.5
    lambda_l = [l < 2 ? 0.0 :
                (-delta^2 * (2l + 1) * l) /
                (a * (a + 2 * delta * p.Oh * (2l + 1) * (l - 1)) +
                 delta^2 * l * (l - 1) * (l + 2)) for l in 0:p.L]
    n = length(x)
    Ad_sym = zeros(n, n)
    for l in 2:p.L, i in 1:n, j in 1:n
        Ad_sym[i, j] += lambda_l[l+1] * x[i] * Ptab[i][l+1] * Ptab[j][l+1] * x[j] * omega[j]
    end
    @printf("  theta_c=%.2f   drop asymmetry with vertical-vertical pairing = %.3e\n",
            theta_c, asymmetry(Ad_sym, ones(n), omega))
end

println()
println("="^78)
println("AUDIT 2: is the symmetric part definite? (convexity => unique solution)")
println("="^78)
println("Gap response is g = g_free + (-A) p, so (-A) must be POSITIVE definite.")
for theta_c in (0.05, 0.15, 0.3, 0.6)
    A, Ab, Ad, Ac, w, x, om = compliance(p, theta_c, beta0, delta)
    n = length(x)
    # symmetrize in the w-measure: S = D_w A D_om^{-1}, take (S+S')/2, eigenvalues
    S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
    ev = eigvals(Symmetric((S + S') / 2))
    @printf("  theta_c=%.2f   min eig=%+.4e   max eig=%+.4e   #negative=%d/%d\n",
            theta_c, minimum(ev), maximum(ev), count(<(0), ev), n)
end

println()
println("="^78)
println("AUDIT 3: is the delta^2 factor UNIFORM across A?")
println("="^78)
println("If A = delta^2 * Ahat with Ahat delta-independent, then cond(A) = cond(Ahat)")
println("and the delta^2 sensitivity that motivated the acceleration-level closure of")
println("Sec. 4 cannot by itself cause any ill-conditioning.")
for delta_test in (1e-2, 1e-3, 1e-4, 1e-5)
    A, _, _, _, w, x, om = compliance(p, 0.3, beta0, delta_test)
    n = length(x)
    S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
    sv = svdvals(S)
    @printf("  delta=%.0e   cond(A)=%.4e   ||A||/delta^2=%.6e\n",
            delta_test, sv[1] / sv[end], sv[1] / delta_test^2)
end

println()
println("="^78)
println("AUDIT 4: what actually sets the resolvable pressure rank?")
println("="^78)
println("An earlier revision of this script fitted rank ~ 0.75*k_M*r_c + 2 and attributed")
println("it to the BATH truncation.  That study set M = L throughout, in which k_M*r_c and")
println("the droplet-side L*theta_c are collinear, so it had no power to attribute the rank")
println("to either block.  It also held n_q fixed, and since the pressure is discretized at")
println("n_q nodes, rank <= n_q caps the largest cases.  Both confounds are removed here:")
println("M and L are varied independently, and n_q is refined until the rank stops moving.")
println()
println("(a) M and L decoupled, n_q refined to independence:")
@printf("  %-6s %-6s %-6s %-9s %-9s %s\n", "M", "L", "th_c", "k_M*r_c", "L*th_c", "rank(1e-8) at nq=80/160/240")
let delta = 1e-3
    for (Mv, Lv, tc) in ((160,160,0.4), (160,40,0.4), (160,10,0.4), (40,160,0.4), (10,160,0.4),
                         (320,20,0.8), (20,320,0.8), (80,80,0.2), (80,80,0.6))
        ranks = Int[]
        for nq in (80, 160, 240)
            q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=Mv, L=Lv, N=8, b=6.0, h0=3.0, nq=nq)
            A, _, _, _, w, x, om = compliance(q, tc, zeros(q.L + 1), delta)
            n = length(x)
            S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
            sv = svdvals(Symmetric((S + S') / 2)); sv ./= sv[1]
            push!(ranks, count(>(1e-8), sv))
        end
        q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=Mv, L=Lv, N=8, b=6.0, h0=3.0, nq=80)
        @printf("  %-6d %-6d %-6.2f %-9.1f %-9.1f %d / %d / %d\n",
                Mv, Lv, tc, q.k[end]*sin(tc), Lv*tc, ranks[1], ranks[2], ranks[3])
    end
end
println()
println("(b) Why the droplet block dominates: the affine slopes' decay rates.")
println("    kappa_m ~ -2 k tanh(k h0)/(a^2 + delta^2 k^3 tanh) decays like 1/k^2 at large k,")
println("    so high bath modes stop responding to pressure; lambda_l ~ -(2l+1)l/(a^2 +")
println("    delta^2 l(l-1)(l+2)) decays only like 1/l, so droplet modes keep contributing.")
let delta = 1e-3, a = 1.5
    q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=320, L=320, N=8, b=6.0, h0=3.0, nq=80)
    @printf("  %-6s %-14s %-14s %-14s %-14s\n", "index", "k_m", "|kappa_m|", "|kappa_m|/|kappa_1|", "|lambda_l|/|lambda_2|")
    km1 = abs((-2*delta^2*q.k[2]*tanh(q.k[2]*q.h0))/(a*(a+4*delta*q.Oh*q.k[2]^2)+delta^2*(q.k[2]^2+q.Bo)*q.k[2]*tanh(q.k[2]*q.h0)))
    lam2 = abs((-delta^2*5*2)/(a*(a+2*delta*q.Oh*5*1)+delta^2*2*1*4))
    for idx in (2, 10, 40, 160, 320)
        km = q.k[idx]
        kap = abs((-2*delta^2*km*tanh(km*q.h0))/(a*(a+4*delta*q.Oh*km^2)+delta^2*(km^2+q.Bo)*km*tanh(km*q.h0)))
        l = idx
        lam = abs((-delta^2*(2l+1)*l)/(a*(a+2*delta*q.Oh*(2l+1)*(l-1))+delta^2*l*(l-1)*(l+2)))
        @printf("  %-6d %-14.3f %-14.4e %-14.4e %-14.4e\n", idx, km, kap, kap/km1, lam/lam2)
    end
end
