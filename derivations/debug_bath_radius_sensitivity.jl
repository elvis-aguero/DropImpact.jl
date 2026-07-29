using SpectralKM, Printf
# PHASE 3 HYPOTHESIS TEST (single variable): is contact time overpredicted because the bath
# domain b = 6 is too small, letting reflected capillary waves return during contact?
#
# WHY THIS HYPOTHESIS. AlventosaEtAl2023 §"Numerical methods" use b = 25R, "determined to be
# sufficiently large such that reflected waves did not influence the droplet during impact".
# We default to b = 6. And paper-formulation.tex:141 already flags this: "There is no basis, on
# dimensional grounds alone, for treating the wall condition as negligible once b is only a few
# droplet radii; whether it is negligible at any particular b is a question the model above is
# built to let one ask, not one this section prejudges." It was never asked -- there is no
# b-convergence study anywhere in the repo.
#
# TIMESCALE SUPPORT. In these units capillary group velocity is c_g ~ 1.5*sqrt(k), so a wall
# round trip takes 2b/c_g ~ 8 tau_cap at k ~ 1 and ~4 tau_cap at k ~ 4 -- INSIDE the ~5 tau_cap
# contact. At b = 25 the round trip is ~33 tau_cap, comfortably outside.
#
# CONFOUND CONTROLLED. k_max = k_M/b, so M is scaled proportionally to b to hold the resolved
# wavenumber range fixed; otherwise growing b would simultaneously coarsen resolution and the
# test would not isolate domain size.
#
# RESULT: HYPOTHESIS FALSIFIED. Measured tc(threshold) = 4.9402, 4.9613, 4.9614, 4.9614 for
# b = 6, 12, 20, 30 -- flat to 5 digits from b = 20 onward, and only 0.4% different from b = 6.
# Reflected capillary waves are NOT the cause of the contact-time overprediction. Retained as a
# negative result: b = 6 is flagged as unvalidated in provenance.tex:39 and paper-formulation.tex:141,
# and this settles that it is not responsible for THIS symptom (it may still matter elsewhere).
#
# Fixed at one of the swept experimental points, where experiment gives tc/t_sigma = 2.498
# (IQR 2.433-2.592) and we produced 4.94.
const WE, BO, OH = 0.998546, 0.047169, 0.048658
const TC_EXP = 2.4981
# wall = :free (no-flux, d(eta)/dr = 0 at r = b) throughout -- the physically correct BC, and
# the package default. Not :pinned, not :clamped.

@printf("%-6s %-6s %-9s %-11s %-13s %-15s %-9s %-s\n",
        "b", "M", "k_max", "lambda_max", "tc(thresh)", "tc(InContact)", "CoR", "tc - exp")
flush(stdout)
for (b, M) in ((6.0, 60), (12.0, 120), (20.0, 200), (30.0, 300))
    p = Params(We=WE, Bo=BO, Oh=OH, b=b, h0=3.0, M=M, L=120, N=3, nq=200)
    kmax = p.k[end]
    lammax = 2pi / p.k[2]            # longest resolvable wave; k[1] = 0 is the piston mode
    levels, diag, phases = run_simulation(p; t_end=14.0)
    ts = [l.t for l in levels]
    tct = threshold_contact_time(ts, levels)
    tci = contact_time(ts, phases)
    cor = coefficient_of_restitution(ts, levels, phases)
    @printf("%-6.1f %-6d %-9.2f %-11.2f %-13s %-15.4f %-9s %s\n",
            b, M, kmax, lammax,
            tct === nothing ? "no rebound" : @sprintf("%.4f", tct), tci,
            cor === nothing ? "n/a" : @sprintf("%.4f", cor),
            tct === nothing ? "n/a" : @sprintf("%+.3f", tct - TC_EXP))
    flush(stdout)
end
@printf("\nexperiment at this point: tc/t_sigma = %.4f (IQR 2.433-2.592)\n", TC_EXP)
println("IF THE HYPOTHESIS HOLDS: tc falls with b and plateaus near the experimental value.")
println("IF tc IS FLAT IN b:      the hypothesis is WRONG and the cause lies elsewhere.")
