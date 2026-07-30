# VALIDATION, NOT A TEST. Deliberately outside test/ and outside CI.
#
# THE QUESTION. The drop mode is carried by a two-pole model (one second-order ODE per `l`).
# Reid's spectrum, however, contains the capillary pair PLUS an infinite diffusive ladder --
# internal Stokes relaxations at rates ~ Oh * q_j^2, with q_j the zeros of j_l. So the exact
# FORCED response
#
#     c_l(t) = integral K_l(t - t') p_l(t') dt',      K_l = sum_j r_j exp(-sigma_j t)
#
# carries memory from poles the two-pole model discards. Free decay is unaffected (that is
# asserted in test/test_reid.jl); forcing is not. Does it matter for impact?
#
# WHY THIS IS NOT IN THE TESTBED. Answering it needs the RESIDUES r_j, i.e. the numerator of
# the transfer function G_l(s), and Reid solves only the homogeneous problem -- it gives the
# poles (the denominator) and nothing else. Obtaining N(s) means redoing the interior
# boundary-value problem with an inhomogeneous normal-stress condition, which is not done. Any
# assertion about residues would therefore be a guess dressed as a check, so this file reports
# what IS computable and states plainly what is not.
#
# WHAT IS COMPUTABLE: the timescale overlap. A discarded mode can only matter if it is still
# alive while pressure is being applied. That is a necessary condition, not a sufficient one --
# a long-lived mode with a tiny residue is still irrelevant. So a "yes" here means "cannot be
# ruled out", and only a "no" is conclusive.
#
# Run: julia --project=. derivations/validate_residue_question.jl

using SpectralKM
using SpecialFunctions
using Printf

# Zeros of j_l = zeros of J_{l+1/2}, ascending, by bisection on besselj (well scaled near q~l).
function jl_zeros(l::Integer, n::Int)
    nu = l + 0.5
    out = Float64[]
    q = max(nu, 1.0) + 1e-6
    step = 0.05 * max(1.0, nu / 4)
    fa = besselj(nu, q)
    while length(out) < n && q < 40 * nu + 400
        q2 = q + step
        fb = besselj(nu, q2)
        if isfinite(fa) && isfinite(fb) && sign(fa) != sign(fb)
            a, b, ga = q, q2, fa
            for _ in 1:200
                m = (a + b) / 2
                gm = besselj(nu, m)
                sign(gm) == sign(ga) ? (a, ga = m, gm) : (b = m)
            end
            push!(out, (a + b) / 2)
        end
        q, fa = q2, fb
    end
    return out
end

println("="^92)
println("VALIDATION: can the discarded diffusive modes be ruled out for impact forcing?")
println("="^92)
println()
println("Contact in this model lasts a few tau_cap (measured ~3-4 tau_cap at We ~ 1).")
println("A discarded mode is IRRELEVANT if its decay time is short compared with that, since")
println("it then relaxes quasi-statically and is absorbed into an effective coefficient.")
println()

const T_CONTACT = 3.5     # tau_cap, representative of the runs in this repo

@printf("%-5s %-7s | %-12s %-12s | %-11s %-11s %-11s | %-s\n",
        "l", "Oh", "capillary", "tau_cap/dec", "diff j=1", "j=2", "j=3", "verdict")
@printf("%-5s %-7s | %-12s %-12s | %-11s %-11s %-11s | %-s\n",
        "", "", "lambda", "time", "decay time", "decay time", "decay time", "(on timescales alone)")
println("-"^92)

for l in (2, 4, 16)
    qz = jl_zeros(l, 3)
    for Oh in (0.006, 0.05, 0.3, 1.0)
        lam, om, _ = reid_root_tracked(l, Oh)
        isfinite(lam) || continue
        tcap = 1 / lam
        tdiff = [1 / (Oh * z^2) for z in qz]
        # "alive during contact" if the slowest discarded mode decays no faster than the
        # contact duration itself
        alive = tdiff[1] > 0.3 * T_CONTACT
        @printf("%-5d %-7.3f | %-12.4f %-12.2f | %-11.3f %-11.3f %-11.3f | %s\n",
                l, Oh, lam, tcap, tdiff[1], tdiff[2], tdiff[3],
                alive ? "CANNOT rule out" : "ruled out (quasi-static)")
    end
    println()
end

println("="^92)
println("READING")
println("="^92)
println("""
The diffusive rates scale as Oh*q_j^2, so they get FASTER as Oh grows. That inverts the naive
expectation, and the measured picture is sharper than "it might matter":

  * RULED OUT everywhere for Oh >= 0.05, at every l tested. The ladder relaxes in 0.6 tau_cap
    or less against a ~3.5 tau_cap contact, so it is quasi-static and folds into an effective
    coefficient. IMPORTANTLY, this covers the whole regime the arbitrary-Oh work targets: the
    two-pole Reid model is on solid ground exactly where we want to use it.

  * RULED OUT for high modes even at small Oh: at l = 16, Oh = 0.006 the slowest discarded mode
    decays in 0.36 tau_cap.

  * CANNOT be ruled out in one corner only: LOW l at SMALL Oh. At Oh = 0.006 the slowest
    discarded mode lives 5.02 tau_cap (l = 2) and 2.50 tau_cap (l = 4), against a contact of
    ~3.5. That corner is this repo's production setting, and l = 2, 4 are the modes carrying
    most of the deformation energy in a low-We impact -- so it is not a corner one can dismiss
    as peripheral.

The irony is worth stating: the concern does NOT apply to the large-Oh regime this whole
exercise was built for, and does apply to the small-Oh regime where Lamb was already adequate.
Switching to :reid therefore does not inherit a new forcing-memory problem -- it inherits an
old one that was always present in the Lamb model too, since Lamb is likewise a two-pole
closure.

WHAT THIS DOES NOT SETTLE. Amplitude. Excitation of these modes is governed by the residues,
and at small Oh the surface boundary layer is thin (delta_nu/R = 0.046 at Oh = 0.006), so
surface pressure may couple only weakly to bulk vorticity structures. A long-lived mode with a
negligible residue changes nothing. Timescale overlap is necessary, not sufficient.

TO ANSWER IT PROPERLY, in order of increasing effort:
  1. Derive G_l(s) by redoing the interior BVP with an inhomogeneous normal-stress condition;
     residues then follow as N(s_j)/D'(s_j).
  2. Or skip the analytics: evaluate G_l(i*omega) numerically by solving the three-boundary-
     condition system at each frequency, then rational-fit (AAA / vector fitting). This also
     yields the n-pole realisation directly and validates route 1 if both are done.
  3. Then compare a 2-pole against an n-pole run under the SAME contact forcing and measure the
     difference in the observables we actually report (CoR, contact time, peak force).

Until then the two-pole model should be described as exact for free decay -- which is now
asserted end-to-end in test/test_reid.jl -- and as an approximation under forcing whose error
is unquantified. That is a weaker claim than 'validated', and it is the claim the evidence
supports.
""")
