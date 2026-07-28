# FEASIBILITY STUDY: finite container, no-flux everywhere, free surface PINNED at the
# triple point.
#
# The current model has no-flux walls for the flow AND a free (90-degree) contact line:
# eta is expanded on J_0(k_m r) with J_1(k_m b) = 0, the NEUMANN set, every member of
# which satisfies d(eta)/dr = 0 at r = b. Pinning instead demands eta(b, tau) = 0 with the
# WALL SLOPE FREE -- a generic pinned meniscus meets the wall at whatever angle the
# dynamics dictate.
#
# FIRST, the continuum fact that frames everything below, because it is easy to miss and it
# settles the question. Exact no-flux on a vertical wall means d(phi)/dr(b,z) = 0 for ALL z;
# differentiating in z gives d/dr d(phi)/dz (b,0) = 0, and the kinematic condition then gives
#
#       d/dtau [ d(eta)/dr (b) ] = 2 Oh d/dr grad^2 eta (b) = O(Oh).
#
# In a rigid right-cylindrical container the wall slope is FROZEN, to O(Oh), whatever basis
# is used. So "pinned at the rim with a freely time-varying wall slope" is not a solution of
# the governing equations under no-flux walls -- it is over-determined. The admissible pinned
# configuration is the CLAMPED EDGE: eta(b) = 0 with d(eta)/dr(b) frozen. A constant nonzero
# wall slope is then had by splitting eta = eta_s(r) + sum a_m(tau) J_0(k_m r) with eta_s a
# static meniscus, at which point pinning, no-flux and a nonzero wall slope are all exact.
#
# The two discrete routes:
#
#   (A) DIRICHLET basis, k_m b = zeros of J_0. Then eta(b) = 0 identically, mode by mode,
#       and the wall slope is free. But the velocity potential shares this horizontal basis
#       (coupled by d(eta)/dt = d(phi)/dz at z = 0), and d(phi)/dr at r = b goes like
#       J_1(k_m b) != 0. The wall leaks -- and, far worse, BATH VOLUME IS NOT CONSERVED
#       (section 3b): measured, route (A) creates four times the droplet's own volume.
#
#       Aside worth recording: the Fourier-Bessel weight for THIS basis is
#       2/(b J_1(k_m b))^2, which is the Dirichlet normalizer whose FORM AlventosaEtAl2023
#       print -- though not their exact expression, since they evaluate J_1 at k_m rather
#       than k_m b, an independent second error. It is the right normalizer for a pinned
#       bath applied to a free one. Note the configurations also differ in the eigenvalues
#       themselves and in the presence of the k = 0 piston mode, so the weight substitution
#       is not the whole of the difference.
#
#   (B) NEUMANN basis (no-flux and volume conservation both exact) with eta(b) = 0 imposed
#       as a single scalar constraint carried by a multiplier -- physically the line force
#       the rim exerts on the contact line, a quantity of interest in its own right. Every
#       basis function has zero wall slope, so the truncated sum does too AT r = b exactly;
#       but section (1) shows this failure is confined to that single point, with the slope
#       recovered at every r < b and L2 convergence at M^(-3/2).
#
# CONCLUSION: route (B). An earlier revision of this script concluded the opposite, on the
# strength of a wall-slope diagnostic that was measuring its own grid spacing; section (1b)
# reproduces that artifact deliberately so the mistake stays visible. Route (A) is what the
# package currently implements as the non-default `wall=:pinned`, and it should be read as a
# diagnostic rather than a physical model.
#
# This script checks: (1) how well the Neumann set represents a pinned profile, analytically;
# (1b) the retracted artifact; (3) route (A)'s wall leak and (3b) its
# volume violation; (4) whether eliminating route (B)'s multiplier keeps the bath response
# AFFINE in the pressure, so that the nested closure of docs/next-gen-KM-model.tex survives.
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
println("Target: a profile pinned at the wall with a NONZERO slope there --")
println("eta(r) = J_0(j01 r/b), which has eta(b) = 0 and d(eta)/dr(b) != 0.")
println("Expanded on the no-flux set {J_0(k_m r) : J_1(k_m b) = 0}, every member of which has")
println("zero wall slope.")
println()
println("EVERYTHING HERE IS ANALYTIC. An earlier version of this script fitted on a fixed")
println("4000-point grid and finite-differenced the last interval to get a 'wall slope'. That")
println("interval lies INSIDE a boundary layer of width ~ b/k_M, so the number returned was")
println("proportional to dr*k_M -- a property of the grid, not of the expansion. Refining the")
println("grid at fixed M = 640 drove it from -0.033 to -0.0005, toward zero, while the true")
println("slope is -0.2081. It doubled whenever M doubled, which looked like slow convergence")
println("and was really just dr*k_M. Section (1b) below reproduces the artifact deliberately.")
println()
println("The projection is available in closed form. With alpha = j01/b and J_1(k_m b) = 0,")
println("  int_0^b J_0(alpha r) J_0(k_m r) r dr = b*alpha*J_1(alpha b)*J_0(k_m b)/(alpha^2-k_m^2)")
println("(the other term carries J_0(alpha b) = J_0(j01) = 0), and ||J_0(k_m .)||^2 =")
println("(b^2/2) J_0(k_m b)^2, so a_m = 2 alpha J_1(alpha b) / (b (alpha^2-k_m^2) J_0(k_m b)).")
println("For the piston mode k_0 = 0: a_0 = 2 J_1(alpha b)/(alpha b).\n")

"""Exact Fourier-Bessel coefficients of J_0(alpha r) on the no-flux set, plus the modal norms."""
function neumann_coeffs(alpha, k, b)
    a = similar(k)
    for (i, km) in enumerate(k)
        a[i] = km == 0 ? 2 * besselj1(alpha * b) / (alpha * b) :
               2 * alpha * besselj1(alpha * b) / (b * (alpha^2 - km^2) * besselj0(km * b))
    end
    return a
end

let j01 = bessel_zeros_J0(1)[1], alpha = bessel_zeros_J0(1)[1] / B
    dtarget(r) = -alpha * besselj1(alpha * r)               # exact slope of the target
    # ||target||^2 = (b^2/2)[J_0(alpha b)^2 + J_1(alpha b)^2] = (b^2/2) J_1(j01)^2
    nrm2 = B^2 / 2 * besselj1(j01)^2
    @printf("  %-7s %-13s %-11s %-11s %-11s %-11s\n",
            "M", "rel L2 (exact)", "s(0.9b)", "s(0.99b)", "s(0.999b)", "s(b)")
    for M in (10, 40, 160, 640, 2560)
        k = bessel_zeros_J1(M) ./ B
        a = neumann_coeffs(alpha, k, B)
        # Parseval: ||target - fit||^2 = ||target||^2 - sum a_m^2 ||phi_m||^2
        fit2 = sum(a[i]^2 * (i == 1 && k[i] == 0 ? B^2 / 2 : B^2 / 2 * besselj0(k[i] * B)^2)
                   for i in eachindex(k))
        relL2 = sqrt(max(nrm2 - fit2, 0.0)) / sqrt(nrm2)
        # analytic derivative of the truncated series: d/dr sum a_m J_0(k_m r)
        slope(r) = -sum(a[i] * k[i] * besselj1(k[i] * r) for i in eachindex(k))
        @printf("  %-7d %-13.3e %-11.4f %-11.4f %-11.4f %-11.4f\n",
                M, relL2, slope(0.9B), slope(0.99B), slope(0.999B), slope(B))
    end
    @printf("\n  target slope:                       %-11.4f %-11.4f %-11.4f %-11.4f\n",
            dtarget(0.9B), dtarget(0.99B), dtarget(0.999B), dtarget(B))
    println("  The relative L2 error falls as M^(-3/2): clean algebraic convergence, no floor.")
    println("  The slope is recovered at every r < b as M grows. At r = b EXACTLY it is 0 for")
    println("  every truncation, since every basis function has zero derivative there -- so the")
    println("  failure is confined to a single point, a set of measure zero, with a boundary")
    println("  layer of width ~ b/k_M around it. That is a far weaker defect than route A's")
    println("  loss of volume conservation (section 3).")
end

println()
println("="^78)
println("(1b) The artifact, reproduced on purpose: finite-differencing the last interval")
println("="^78)
println("M is FIXED at 640. Only the grid changes. A genuine representation error cannot")
println("depend on the grid used to look at it.")
let alpha = bessel_zeros_J0(1)[1] / B
    k = bessel_zeros_J1(640) ./ B
    a = neumann_coeffs(alpha, k, B)
    @printf("  %-10s %-12s %-14s\n", "nq", "dr", "FD 'wall slope'")
    for nq in (4000, 16000, 64000, 256000)
        r = range(1e-9, B; length=nq); dr = step(r)
        f(x) = sum(a[i] * besselj0(k[i] * x) for i in eachindex(k))
        @printf("  %-10d %-12.2e %-14.5f\n", nq, dr, (f(B) - f(B - dr)) / dr)
    end
    println("  -> 0 under refinement. The quantity measured was dr*k_M.")
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
    println("  J_0 vanishes (pinning exact) while J_1 does not, decaying only as m^(-1/2)")
    println("  (|J_1(j0m)| -> sqrt(2/(pi j0m))): the wall-normal velocity is not small.")
end

println()
println("="^78)
println("(3b) Route (A)'s REAL cost: bath volume is not conserved")
println("="^78)
println("This, not the wall velocity, is what disqualifies route (A) as physics. The")
println("volume a mode displaces is int_0^b J_0(k_m r) r dr = (b/k_m) J_1(k_m b).")
println()
println("For the NEUMANN set J_1(k_m b) = 0 by definition, so every non-piston mode is")
println("volume-NEUTRAL, and the only volume-carrying mode is the piston, whose pressure")
println("response kappa_0 is identically zero -- it can never be driven. Volume conservation")
println("is therefore structural, not a matter of accuracy.")
println()
println("For the DIRICHLET set J_1(k_m b) != 0, so EVERY mode carries volume and nothing")
println("constrains the sum.")
let
    @printf("  %-6s %-16s %-16s\n", "m", "Neumann (b/k)J_1", "Dirichlet (b/k)J_1")
    kn = bessel_zeros_J1(5) ./ B
    kd = bessel_zeros_J0(5) ./ B
    for m in 1:5
        @printf("  %-6d %-16.2e %-16.4f\n", m,
                (B / kn[m+1]) * besselj1(kn[m+1] * B), (B / kd[m]) * besselj1(kd[m] * B))
    end
    println("  Neumann: zero to roundoff, mode by mode. Dirichlet: O(1), alternating in sign.")
    println()
    println("  Measured over the reference impact (We=1.0958, M=L=60, N=3), the worst")
    println("  |int_0^b eta r dr| over the run is 4.8e-15 for :free and 2.788 for :pinned.")
    println("  Times 2*pi that is 17.5 R^3 of bath volume created from nothing, against a")
    println("  droplet volume of 4*pi/3 = 4.19 R^3 -- a factor of four. Reproduce with")
    println("  julia --project=julia scripts/run_impact.jl 1.0958 0.017 0.006 60 60 3 14 pinned")
end

println()
println("="^78)
println("(4) Route (B) structure: does eliminating the multiplier keep the bath AFFINE?")
println("="^78)
println("Constraint  C(a) = sum_m a_m J_0(k_m b) = 0, enforced by a multiplier Lambda.")
println()
println("An earlier version of this script inserted Lambda as a 'generalized force'")
println("Lambda*dC/da_m = Lambda*J_0(k_m b) divided by F_m. That was wrong: it omits the modal")
println("norm that eq:c_m-def carries, and its m=0 term is literally 0/0 since kappa_0 = F_0 = 0")
println("(the loop silently started at m=1, so the printed sum was not the sum the formula")
println("defined). Treat Lambda as the physical object instead -- a line force at the rim,")
println("p_Lambda = Lambda*delta(r-b). Its Fourier-Bessel coefficient under eq:c_m-def is")
println("      c_m^Lambda = [2/(b J_0(k_m b))^2] * Lambda * b * J_0(k_m b) = 2 Lambda/(b J_0(k_m b)),")
println("so the affine relation and the constraint close on")
println("      a_m = alpha_m + kappa_m c_m + (2/b) kappa_m Lambda / J_0(k_m b),")
println("      D   = (2/b) sum_m kappa_m,")
println("in which the J_0(k_m b) factors CANCEL. Substituting back leaves a_m affine in c with")
println("a rank-one correction, so the nested closure is untouched.\n")
let delta = 1e-3, a = 1.5, h0 = 3.0, Oh = 0.006, Bo = 0.017
    kappa_of(km) = (-2 * delta^2 * km * tanh(km * h0)) /
                   (a * (a + 4 * delta * Oh * km^2) + delta^2 * (km^2 + Bo) * km * tanh(km * h0))
    @printf("  %-8s %-18s %-14s %-14s\n", "M", "D = (2/b)sum kappa", "all kappa<0?", "kappa_M")
    for M in (20, 60, 120, 300, 1200, 4800)
        k = bessel_zeros_J1(M) ./ B
        kaps = [kappa_of(k[m+1]) for m in 1:M]      # kappa_0 = 0 exactly, contributes nothing
        @printf("  %-8d %-18.6e %-14s %-14.3e\n", M, (2 / B) * sum(kaps),
                all(<(0), kaps), kaps[end])
    end
    println("  Every term is negative, so no cancellation is possible and D != 0")
    println("  unconditionally. The sum converges because kappa_m ~ -2/k_m^2, so D saturates")
    println("  as M grows rather than diverging. Cost: one O(M) dot product.")
    println()
    println("  Note kappa_0 = 0 exactly: the rim line force cannot drive the piston mode, so")
    println("  route (B) pins the surface WITHOUT moving volume. It does not reintroduce the")
    println("  defect of section (3b) -- which is the decisive argument in its favour.")
end
