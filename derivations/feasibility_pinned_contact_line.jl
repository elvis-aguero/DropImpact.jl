# FEASIBILITY STUDY: finite container, no-flux everywhere, free surface PINNED at the
# triple point.
#
# The current model has no-flux walls for the flow AND a free (90-degree) contact line:
# eta is expanded on J_0(k_m r) with J_1(k_m b) = 0, the NEUMANN set, every member of
# which satisfies d(eta)/dr = 0 at r = b. Pinning instead demands eta(b, tau) = 0 with the
# WALL SLOPE FREE -- a generic pinned meniscus meets the wall at whatever angle the
# dynamics dictate.
#
# That is a genuine conflict, and the two obvious routes each break something:
#
#   (A) DIRICHLET basis, k_m b = zeros of J_0. Then eta(b) = 0 identically, mode by mode,
#       and the wall slope is free -- exactly right for pinning. But the velocity potential
#       shares this horizontal basis (they are coupled by d(eta)/dt = d(phi)/dz at z = 0),
#       and d(phi)/dr at r = b goes like J_1(k_m b), which is NOT zero for this set. The
#       wall leaks: no-flux is violated.
#
#       Aside worth recording: the Fourier-Bessel weight for THIS basis is
#       2/(b J_1(k_m b))^2 -- precisely the expression AlventosaEtAl2023 print and which
#       this project corrected to 2/(b J_0(k_m b))^2 for the no-flux basis. Their formula
#       is the right one for a pinned bath, applied to a free one.
#
#   (B) NEUMANN basis (no-flux honoured exactly) with eta(b) = 0 imposed as a single
#       scalar constraint carried by a Lagrange multiplier -- physically the line force
#       the pinning exerts on the contact line. The flow boundary condition is then exact
#       and the pinning is exact, but every basis function still has zero wall slope, so
#       the representable surfaces all have d(eta)/dr(b) = 0. A pinned meniscus with a
#       nonzero wall slope can only be approached, never represented.
#
# Route (B) is the physically consistent one -- pinning IS a constraint force, and it
# leaves the flow boundary condition intact -- so the question that decides feasibility is
# quantitative: HOW BADLY does the Neumann set converge to a profile with a nonzero wall
# slope? That is what this script measures.
#
# It also checks the two structural questions route (B) raises: whether eliminating the
# multiplier keeps the bath response AFFINE in the pressure (so the nested closure of
# docs/next-gen-KM-model.tex survives untouched), and whether that elimination is
# well conditioned.
using SpectralKM
using SpectralKM: bessel_zeros_J1
using SpecialFunctions
using LinearAlgebra
using Printf

const B = 6.0

"""Zeros of J_0, for the Dirichlet (pinned-by-construction) basis of route (A)."""
function bessel_zeros_J0(M::Integer)
    z = Vector{Float64}(undef, M)
    for m in 1:M
        x = (m - 0.25) * π                     # McMahon asymptotic seed
        for _ in 1:60
            x -= besselj0(x) / (-besselj1(x))
        end
        z[m] = x
    end
    return z
end

println("="^78)
println("(1) Can the NEUMANN set represent a pinned surface? (route B's central risk)")
println("="^78)
println("Target: a profile that is pinned at the wall with a NONZERO slope there --")
println("eta(r) = J_0(j01 r/b), which has eta(b) = 0 and d(eta)/dr(b) != 0.")
println("Expanded on the no-flux set {J_0(k_m r) : J_1(k_m b) = 0}, whose every member has")
println("zero wall slope. Errors are relative L2 over the bath, and pointwise near r = b.\n")

let j01 = bessel_zeros_J0(1)[1]
    target(r) = besselj0(j01 * r / B)
    nq = 4000
    r = range(1e-9, B; length=nq)
    dr = step(r)
    w = r .* dr                                     # cylindrical measure r dr
    tg = target.(r)
    nrm = sqrt(sum(w .* tg .^ 2))
    @printf("  %-6s %-13s %-13s %-13s %-13s\n", "M", "rel L2", "|err| at 0.99b", "|err| at b", "slope at b")
    for M in (10, 20, 40, 80, 160, 320, 640)
        k = bessel_zeros_J1(M) ./ B                 # includes k_0 = 0 (the piston mode)
        # least-squares fit in the r dr measure = plain projection, the basis is orthogonal
        coef = zeros(length(k))
        for (i, km) in enumerate(k)
            ϕ = besselj0.(km .* r)
            nn = sum(w .* ϕ .^ 2)
            nn <= 0 && continue
            coef[i] = sum(w .* ϕ .* tg) / nn
        end
        fit = zeros(nq)
        for (i, km) in enumerate(k)
            fit .+= coef[i] .* besselj0.(km .* r)
        end
        relL2 = sqrt(sum(w .* (fit .- tg) .^ 2)) / nrm
        i99 = searchsortedfirst(collect(r), 0.99B)
        slope = (fit[end] - fit[end-1]) / dr
        @printf("  %-6d %-13.3e %-13.3e %-13.3e %-13.3e\n",
                M, relL2, abs(fit[i99] - tg[i99]), abs(fit[end] - tg[end]), slope)
    end
    @printf("\n  exact wall slope of the target: %.4f\n", -j01 / B * besselj1(j01))
    println("  The slope column is a finite difference taken just INSIDE the wall, not at it.")
    println("  AT r = b every basis function has zero derivative, so the truncated sum does")
    println("  too, exactly. Just inside, the fit develops a boundary layer that thins as M")
    println("  grows -- which is why the measured slope climbs (5.7e-4 -> 3.3e-2) while still")
    println("  falling six-fold short of the true -0.2081 at M = 640. That is the signature")
    println("  of the Gibbs-type mismatch, and it is the quantity a contact-line force would")
    println("  depend on.")
end

println()
println("="^78)
println("(2) For contrast: the same fit for a target the Neumann set CAN represent")
println("="^78)
let
    k = bessel_zeros_J1(60) ./ B
    target(r) = besselj0(k[4] * r) + 0.5 * besselj0(k[7] * r)     # zero wall slope
    nq = 4000
    r = range(1e-9, B; length=nq); dr = step(r); w = r .* dr
    tg = target.(r); nrm = sqrt(sum(w .* tg .^ 2))
    @printf("  %-6s %-13s\n", "M", "rel L2")
    for M in (10, 20, 40)
        kk = bessel_zeros_J1(M) ./ B
        fit = zeros(nq)
        for km in kk
            ϕ = besselj0.(km .* r); nn = sum(w .* ϕ .^ 2)
            nn <= 0 && continue
            fit .+= (sum(w .* ϕ .* tg) / nn) .* ϕ
        end
        @printf("  %-6d %-13.3e\n", M, sqrt(sum(w .* (fit .- tg) .^ 2)) / nrm)
    end
    println("  Spectral (machine-precision once the modes are included), as expected.")
end

println()
println("="^78)
println("(3) Route (A)'s cost: how much flux leaks through the wall?")
println("="^78)
println("Using the Dirichlet set, d(phi)/dr at r = b goes like J_1(k_m b), which does not")
println("vanish. Reported relative to the mode's own interior scale.")
let
    z0 = bessel_zeros_J0(8)
    @printf("  %-4s %-12s %-14s %-14s\n", "m", "k_m b", "J_0(k_m b)", "J_1(k_m b)")
    for (m, z) in enumerate(z0)
        @printf("  %-4d %-12.5f %-14.2e %-14.4f\n", m, z, besselj0(z), besselj1(z))
    end
    println("  J_0 vanishes (pinning exact) while J_1 is O(1): the wall-normal velocity is")
    println("  NOT small, so no-flux fails at leading order rather than marginally.")
end

println()
println("="^78)
println("(4) Route (B) structure: does eliminating the multiplier keep the bath AFFINE?")
println("="^78)
println("Constraint  C(a) = sum_m a_m J_0(k_m b) = 0.  By virtual work the multiplier's")
println("generalized force on mode m is Lambda * dC/da_m = Lambda * J_0(k_m b), so the BDF2")
println("affine relation gains one term,")
println("      a_m = alpha_m + kappa_m c_m + kappa_m J_0(k_m b) Lambda / F_m,")
println("and the constraint determines Lambda linearly. Substituting back leaves a_m affine")
println("in c with a RANK-ONE correction to the bath compliance -- so the inner Galerkin")
println("system stays square, linear in the same unknowns, and the nested closure is")
println("untouched. What must be checked is that the denominator does not vanish.\n")
let delta = 1e-3, a = 1.5, h0 = 3.0, Oh = 0.006, Bo = 0.017
    @printf("  %-6s %-16s %-16s %-12s\n", "M", "denominator D", "|D| / max term", "cond proxy")
    for M in (20, 60, 120, 300)
        k = bessel_zeros_J1(M) ./ B
        terms = Float64[]
        for m in 1:M                      # skip the k=0 piston mode: kappa_0 = 0
            km = k[m+1]
            kap = (-2 * delta^2 * km * tanh(km * h0)) /
                  (a * (a + 4 * delta * Oh * km^2) + delta^2 * (km^2 + Bo) * km * tanh(km * h0))
            F = -2 * km * tanh(km * h0)
            push!(terms, besselj0(km * B)^2 * kap / F)
        end
        D = sum(terms)
        @printf("  %-6d %-16.6e %-16.3e %-12.3e\n", M, D, abs(D) / maximum(abs, terms),
                maximum(abs, terms) / abs(D))
    end
    println("  D is a sum of same-sign terms (kappa/F > 0), so no cancellation: the")
    println("  elimination is unconditionally well posed and costs one O(M) dot product.")
end
