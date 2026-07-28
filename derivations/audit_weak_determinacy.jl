# AUDIT (part 3): the claims of paper-formulation.tex ('The compliance operator') that an independent review
# found unsupported by the earlier audit scripts.  Each block here produces numbers the
# design doc cites, so that no cited measurement lives only in a scratch file.
#
# (1) IDENTIFIABILITY of the droplet self-adjointness diagnosis.  At beta = 0 the
#     cylindrical area element w(x) = -d[xi^2(1-x^2)]/dx / 2 reduces to exactly x, so
#     the two candidate mechanisms for the droplet block's asymmetry --
#       (i) radial-vs-vertical: b_l pairs p against P_l radially but returns x*xi, and
#       (ii) spherical-vs-cylindrical: c_m and f pair p against w dx = r dr while b_l
#            pairs it against dx,
#     are the SAME correction and cannot be told apart.  They separate only at beta != 0.
#
# (2) DEFINITENESS, reported as measured: the symmetric part of -A is positive
#     SEMIdefinite with a large nullspace, not definite.
#
# (3) The delta^2 factor's actual uniformity, per mode rather than in norm.
#
# (4) WEAK DETERMINACY across more than one functional, more than one state and more
#     than one M: does the pressure's action (c_m, b_l, f) converge in N while the
#     pointwise pressure does not, and is any N-drift distinguishable from the spectral
#     filter's own cutoff sensitivity?
using Printf, LinearAlgebra
include(joinpath(@__DIR__, "audit_nested_closure_lib.jl"))

println("="^78)
println("(1) Identifiability: at beta=0, is w(x) identically x?")
println("="^78)
let q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=3, b=6.0, h0=3.0, nq=40)
    for tc in (0.16, 0.30, 0.60)
        _, _, _, _, w, x, _ = compliance(q, tc, zeros(q.L + 1), 1e-3)
        @printf("  theta_c=%.2f   max|w(x) - x| = %.3e   (pure finite-difference noise)\n",
                tc, maximum(abs.(w .- x)))
    end
end
println()
println("  Separating the two mechanisms at beta != 0.  Droplet-block asymmetry measured in")
println("  the cylindrical measure mu = w, under three pairings:")
println("    as-is        : kernel lambda_l * x P_l(x) P_l(x')            (eq:compliance-kernel)")
println("    vertical fix : kernel lambda_l * x P_l(x) P_l(x') x'         (b_l -> int p x P_l dx)")
println("    measure fix  : kernel lambda_l * x P_l(x) P_l(x') w(x')      (b_l against w dx)")
println("    BOTH         : kernel lambda_l * x P_l(x) P_l(x') x' w(x')   (eq:b_l-selfadjoint)")
@printf("  %-10s %-13s %-13s %-13s %-13s %-13s\n", "beta_2", "as-is", "vertical", "measure", "BOTH(drop)", "BOTH(full A)")
let q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=3, b=6.0, h0=3.0, nq=40),
    delta = 1e-3, tc = 0.30, a = 1.5
    lambda_l = [l < 2 ? 0.0 :
                (-delta^2 * (2l + 1) * l) /
                (a * (a + 2 * delta * q.Oh * (2l + 1) * (l - 1)) +
                 delta^2 * l * (l - 1) * (l + 2)) for l in 0:q.L]
    for b2 in (0.0, 0.02, 0.05, 0.10)
        beta = zeros(q.L + 1); beta[3] = b2
        _, Ab, _, Ac, w, x, om = compliance(q, tc, beta, delta)
        Ptab = [legendre_P_table(q.L, xi) for xi in x]
        n = length(x)
        A0 = zeros(n, n); A1 = zeros(n, n); A2 = zeros(n, n); A3 = zeros(n, n)
        for l in 2:q.L, i in 1:n, j in 1:n
            base = lambda_l[l+1] * x[i] * Ptab[i][l+1] * Ptab[j][l+1] * om[j]
            A0[i, j] += base
            A1[i, j] += base * x[j]          # vertical-vertical pairing only
            A2[i, j] += base * w[j]          # cylindrical measure only
            A3[i, j] += base * x[j] * w[j]   # BOTH -- eq:b_l-selfadjoint
        end
        # and the FULL operator with the doubly-corrected droplet block
        Afull = Ab .+ Ac .+ A3
        @printf("  %-10.2f %-13.3e %-13.3e %-13.3e %-13.3e %-13.3e\n", b2,
                asymmetry(A0, w, om), asymmetry(A1, w, om), asymmetry(A2, w, om),
                asymmetry(A3, w, om), asymmetry(Afull, w, om))
    end
end

println()
println("="^78)
println("(2) Definiteness of the symmetric part of -A, reported as measured")
println("="^78)
let q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=3, b=6.0, h0=3.0, nq=40)
    @printf("  %-8s %-13s %-13s %-11s %-13s\n",
            "th_c", "min eig", "max eig", "min/max", "#|ev|<1e-12*max")
    for tc in (0.05, 0.15, 0.30, 0.60)
        A, _, _, _, w, x, om = compliance(q, tc, zeros(q.L + 1), 1e-3)
        n = length(x)
        S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
        ev = eigvals(Symmetric((S + S') / 2))
        @printf("  %-8.2f %-+13.3e %-+13.3e %-+11.2e %d of %d\n",
                tc, minimum(ev), maximum(ev), minimum(ev) / maximum(ev),
                count(<(1e-12 * maximum(ev)), abs.(ev)), n)
    end
end
println("  => positive SEMIdefinite with a nullspace over half the discretization.")
println("     Uniqueness of the GAP RESPONSE A p follows; uniqueness of p does not.")

println()
println("="^78)
println("(3) Is the delta^2 factor uniform PER MODE, not merely in norm?")
println("="^78)
println("  Relative deviation of kappa_m/delta^2 from its delta->0 limit")
println("  -2 k_m tanh(k_m h0)/a^2, maximised over m; likewise for lambda_l.")
@printf("  %-8s %-12s %-14s %-14s\n", "M=L", "delta", "max dev kappa", "max dev lambda")
let a = 1.5
    for ML in (40, 160, 700)
        q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=ML, L=ML, N=3, b=6.0, h0=3.0, nq=20)
        for delta in (1e-2, 1e-3, 1e-4)
            dk = 0.0; dl = 0.0
            for m in 1:ML
                km = q.k[m+1]
                exact = (-2 * delta^2 * km * tanh(km * q.h0)) /
                        (a * (a + 4 * delta * q.Oh * km^2) + delta^2 * (km^2 + q.Bo) * km * tanh(km * q.h0))
                lim = -2 * delta^2 * km * tanh(km * q.h0) / a^2
                dk = max(dk, abs(exact / lim - 1))
            end
            for l in 2:ML
                exact = (-delta^2 * (2l + 1) * l) /
                        (a * (a + 2 * delta * q.Oh * (2l + 1) * (l - 1)) + delta^2 * l * (l - 1) * (l + 2))
                lim = -delta^2 * (2l + 1) * l / a^2
                dl = max(dl, abs(exact / lim - 1))
            end
            @printf("  %-8d %-12.0e %-14.3f %-14.3f\n", ML, delta, dk, dl)
        end
    end
end
println("  => uniformity holds only while delta*k_M^2 << 1 and delta^2 L^3 << 1.")

println()
println("="^78)
println("(4) Weak determinacy: which functionals converge in N, and vs the filter cutoff")
println("="^78)

"""Inner solve at fixed theta_c with a truncated-SVD (spectral-filter) linear solve."""
function inner_filtered(p::Params, theta_c, mu, delta, eps_rel)
    A, _, _, _, w, x, omega = compliance(p, theta_c, zeros(p.L + 1), delta)
    xc = cos(theta_c); N = p.N; n = length(x)
    B = [pbasis(N, xc, xi) for xi in x]
    G = zeros(N + 1, N + 1); rhs = zeros(N + 1)
    for nn in 0:N
        for np in 0:N
            s = 0.0
            for i in 1:n
                inner = 0.0
                for j in 1:n; inner += A[i, j] * B[j][np+1]; end
                s += omega[i] * B[i][nn+1] * inner
            end
            G[nn+1, np+1] = s
        end
        rhs[nn+1] = -sum(omega[i] * B[i][nn+1] * (x[i] - mu) for i in 1:n)
    end
    F = svd(G); keep = F.S .> eps_rel * F.S[1]
    chat = F.V[:, keep] * ((F.U[:, keep]' * rhs) ./ F.S[keep])
    pv = [sum(chat[nn+1] * B[i][nn+1] for nn in 0:N) for i in 1:n]
    Ptab = [legendre_P_table(p.L, xi) for xi in x]
    r = [sqrt(1 - xi^2) for xi in x]
    f = 2 * sum(omega[j] * pv[j] * w[j] for j in 1:n)
    bl = [sum(omega[j] * pv[j] * Ptab[j][l+1] for j in 1:n) for l in 2:min(p.L, 12)]
    cm = [2 / (p.b * besselj0(p.k[m+1] * p.b))^2 *
          sum(omega[j] * pv[j] * besselj0(p.k[m+1] * r[j]) * w[j] for j in 1:n) for m in 1:min(p.M, 12)]
    return count(keep), minimum(pv), f, bl, cm
end

for (Mv, tc) in ((400, 0.16), (800, 0.16), (800, 0.40))
    println("\n  M = L = $Mv, theta_c = $tc, cutoff 1e-10.  Relative change vs previous N:")
    @printf("    %-5s %-6s %-12s %-12s %-11s %-11s\n", "N", "kept", "f", "min p", "d(b_l)rel", "d(c_m)rel")
    prev_bl = nothing; prev_cm = nothing
    for N in (10, 20, 40, 80)
        q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=Mv, L=Mv, N=N, b=6.0, h0=3.0, nq=140)
        k, mp, f, bl, cm = inner_filtered(q, tc, 0.985, 1e-3, 1e-10)
        dbl = prev_bl === nothing ? NaN : norm(bl - prev_bl) / norm(bl)
        dcm = prev_cm === nothing ? NaN : norm(cm - prev_cm) / norm(cm)
        @printf("    %-5d %-6d %-12.6f %-12.3e %-11.3e %-11.3e\n", N, k, f, mp, dbl, dcm)
        prev_bl = bl; prev_cm = cm
    end
end

println("\n  Cutoff sensitivity at M=L=800, theta_c=0.16, N=40 -- the same quantities,")
println("  so that N-drift and cutoff-drift can be compared on one scale:")
@printf("    %-10s %-6s %-12s %-12s %-11s %-11s\n", "cutoff", "kept", "f", "min p", "d(b_l)rel", "d(c_m)rel")
let prev_bl = nothing, prev_cm = nothing
    for e in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
        q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=800, L=800, N=40, b=6.0, h0=3.0, nq=140)
        k, mp, f, bl, cm = inner_filtered(q, 0.16, 0.985, 1e-3, e)
        dbl = prev_bl === nothing ? NaN : norm(bl - prev_bl) / norm(bl)
        dcm = prev_cm === nothing ? NaN : norm(cm - prev_cm) / norm(cm)
        @printf("    %-10.0e %-6d %-12.6f %-12.3e %-11.3e %-11.3e\n", e, k, f, mp, dbl, dcm)
        prev_bl = bl; prev_cm = cm
    end
end


println()
println("="^78)
println("(5) Resolved singular-value spectrum: is A a genuine limit object?")
println("="^78)
println("  Compactness is argued from the decay of the RESOLVED singular values, not from")
println("  a computed condition number (at ~1e18 the latter exceeds 1/eps_mach and reports")
println("  only that sigma_min is unresolvable).  sigma_i/sigma_1 at increasing M=L:")
let delta = 1e-3, tc = 0.30, nq = 60
    specs = Dict{Int,Vector{Float64}}()
    for ML in (20, 40, 80, 160)
        q = Params(We=1.0958, Bo=0.017, Oh=0.006, M=ML, L=ML, N=8, b=6.0, h0=3.0, nq=nq)
        A, _, _, _, w, x, om = compliance(q, tc, zeros(q.L + 1), delta)
        n = length(x)
        S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
        sv = svdvals(Symmetric((S + S') / 2))
        specs[ML] = sv ./ sv[1]
    end
    @printf("  %-4s", "i")
    for ML in (20, 40, 80, 160); @printf(" %-14s", "M=L=$ML"); end
    println()
    for i in (2, 3, 4, 5, 8, 14)
        @printf("  %-4d", i)
        for ML in (20, 40, 80, 160); @printf(" %-14.4e", specs[ML][i]); end
        println()
    end
    @printf("\n  sigma_14/sigma_2 at M=L=160 = %.4f   (algebraic 2/14 = %.4f; geometric 0.78^12 = %.4f)\n",
            specs[160][14] / specs[160][2], 2 / 14, 0.78^12)
end
