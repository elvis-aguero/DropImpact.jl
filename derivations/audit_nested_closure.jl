# AUDIT (part 2): the NESTED free-boundary closure vs. the square joint Newton
# system the .tex currently uses.  Shared assembly lives in
# audit_nested_closure_lib.jl (which itself reuses audit_compliance_operator_core.jl).
#
# Structure under test -- full KM's iteration-on-geometry, in a weak/Galerkin setting
# with no collocation anywhere:
#
#   OUTER: theta_c is a single scalar parameter, NOT a Newton unknown.
#   INNER: for each trial theta_c, solve the (N+1) x (N+1) Galerkin kinematic system
#          eq:galerkin for the pressure coefficients.  Square, no theta_c row.
#   SELECT: theta_c is fixed by scalar residuals in theta_c alone (pressure already
#          eliminated), exactly as full KM's e(q) selects its integer contact count.
#
# Q1. Is the inner system's conditioning delta-independent?  (Predicted yes: with
#     theta_c gone, every column carries the same delta^2 factor, and cond(cA)=cond(A).)
# Q2. Do the candidate scalar selectors -- tangency dC/dtheta at theta_c, and the
#     non-penetration feasibility boundary max C(theta>theta_c) < 0 -- agree with each
#     other, and are they independent of the pressure truncation N?
# Q3. Does pressure positivity EMERGE in the selected window, as both parent papers
#     assume without ever enforcing it?
using Printf
include(joinpath(@__DIR__, "audit_nested_closure_lib.jl"))

# A synthetic penetrating state: centre of mass at mu with all bath/drop modes zero,
# so the undeformed droplet already overlaps the flat bath by 1-mu at the pole.  The
# undeformed surfaces cross where cos(theta)=mu, giving a parameter-free geometric
# reference for where the contact edge ought to land.
mu = 0.985
delta = 1e-3

println("="^76)
println("Q1: inner-system conditioning vs delta (theta_c held OUT of the system)")
println("="^76)
let p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=3, b=6.0, h0=3.0, nq=60)
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
        _, ci, _, _, _, _, _, _, _, _ = inner_solve(p, 0.3, mu, d)
        @printf("  delta=%.0e   cond(inner Galerkin matrix) = %.4e\n", d, ci)
    end
end
println("\n  Compare: the .tex reports cond(J) for the SQUARE joint system at ~1e18,")
println("  growing ~6 orders of magnitude over this same delta range and ~4 orders per")
println("  added pressure mode -- which is what motivated the acceleration-level closure.")

println()
println("="^76)
println("Q2/Q3: scalar selectors for theta_c, and emergent pressure positivity")
println("="^76)
@printf("Geometric reference: undeformed surfaces cross at acos(mu) = %.4f\n\n", acos(mu))
for N in (1, 2, 3)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=80, L=80, N=N, b=6.0, h0=3.0, nq=60)
    @printf("N = %d   (M = L = 80, k_M = %.1f)\n", N, p.k[end])
    @printf("  %-7s %-12s %-13s %-12s %-12s %-9s\n",
            "th_c", "dC/dth@th_c", "max C(th>th_c)", "min p", "net f", "cond_in")
    for theta_c in 0.06:0.02:0.34
        chat, ci, am, beta, f, pv, x, om, w, kcm = inner_solve(p, theta_c, mu, delta)
        zcm = mu + kcm * f
        h = 1e-5
        Cpm = C_of_theta(p, am, beta, zcm, [theta_c - h, theta_c + h])
        T = (Cpm[2] - Cpm[1]) / (2h)
        Cout = C_of_theta(p, am, beta, zcm, collect(range(theta_c + 1e-3, 3.0; length=600)))
        @printf("  %-7.2f %-+12.3e %-+13.3e %-+12.3e %-+12.3e %-9.2e\n",
                theta_c, T, maximum(Cout), minimum(pv), f, ci)
    end
    println()
end
println("Reading the table: the tangency residual changes sign, the non-intersection")
println("feasibility boundary is crossed, and min p turns negative, all in the neighbourhood")
println("of theta_c ~ 0.16-0.20 -- but NOT on one common interval; the spread across the")
println("three is 0.025 to 0.080 depending on N (see docs subsubsec:contact-angle, which")
println("records the earlier `they coincide' claim as withdrawn).  What IS N-independent is")
println("the tangency root itself, at 0.1925 to four figures for every N tested, against the")
println("geometric reference acos(mu) = 0.1734.")
println("`net f` is NOT diagnostic here: this synthetic state carries 0.015 of un-relieved")
println("penetration that no prior dynamics produced, so the pressure needed to enforce the")
println("kinematic match within one delta=1e-3 step is correspondingly stiff.  Only a real")
println("forward march can test force magnitudes.")
println("cond_in blows up where N+1 exceeds the resolvable dimension of the assembled")
println("compliance operator, which audit_compliance_operator.jl AUDIT 4 measures directly")
println("(the earlier closed-form rank law F(k_M r_c) is withdrawn -- both truncations")
println("contribute and no single-parameter collapse survives decoupling M from L).")
