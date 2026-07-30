using SpectralKM
using SpectralKM: forward_map_r
using Printf

# WHY inf{feasible} collapses 200x in one step, from theta_c = 0.0475 at t = 0.009 to 0.00024 at
# t = 0.010, after which the contact patch stays pinned near zero for the rest of the impact.
#
# feasible_at gates on THREE conditions, and the previous diagnostic only reported their AND:
#   1. the inner Newton solve converges (or gets within practical_resid_tol)
#   2. check_monotone_r(beta, xc, L)          -- r must increase monotonically along the patch
#   3. check_nonintersect(am, beta, zcm, theta_c, p) -- surfaces must not overlap beyond the edge
# Since theta_c = inf{feasible}, whichever gate stops failing at small theta is what releases the
# lower edge and lets the patch collapse. This separates them.
#
# Also reports check_positivity, which is NOT part of feasible_at but is the physical
# complementarity condition the selector's docstring appeals to -- worth seeing whether it would
# have excluded the collapsed patch.

const WE, BO, OH = 0.0231, 0.02, 0.03
p = Params(We=WE, Bo=BO, Oh=OH, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)
dt = 1e-3

lvl0 = initial_level(p)
hist = SimHistory(lvl0, lvl0)
theta_seed = nothing
for _ in 1:20_000
    th = onset_theta_c(hist, dt, p)
    th !== nothing && (global theta_seed = th; break)
    trial = free_flight_step(hist, dt, p)
    hist.prev = hist.curr; hist.curr = trial
end
theta_seed === nothing && error("no onset")

grid = vcat(range(1e-4, 0.01; length=12), range(0.015, 0.30; length=26))
theta_prev = theta_seed
chat_guess = zeros(p.N + 1); chat_guess[1] = 1e-3

for step in 1:12
    show_this = step in (9, 10, 11, 12)
    if show_this
        @printf("\n===== step %d, t = %.4f, theta_prev = %.5f =====\n", step, hist.curr.t, theta_prev)
        @printf("%-9s %-9s %-9s %-9s %-9s %-9s %-11s\n",
                "theta", "solved", "mono_r", "nonint", "FEASIBLE", "pos(p>0)", "f")
    end
    for th in grid
        chat, result, am, beta, zcm, f, cm, bl = inner_solve(hist, dt, th, chat_guess, p)
        solved = result.status == Converged || result.resid_norm_hist[end] < 1e-4
        xc = cos(th)
        mono = solved ? check_monotone_r(beta, xc, p.L) : false
        noni = (solved && mono) ? check_nonintersect(am, beta, zcm, th, p) : false
        pos = solved ? check_positivity(chat, xc) : false
        if show_this
            @printf("%-9.5f %-9s %-9s %-9s %-9s %-9s %-11.3e\n",
                    th, solved, mono, noni, solved && mono && noni, pos, f)
        end
    end
    lvl, info = contact_step(hist, dt, theta_prev, chat_guess, p)
    lvl === nothing && (println("\ncontact_step returned nothing at step $step"); break)
    hist.prev = hist.curr; hist.curr = lvl
    global theta_prev = info.theta_c
    global chat_guess = lvl.X[1:p.N+1]
end

println("""

READING
  The gate that is FALSE at small theta in the last good step and TRUE at small theta in the
  collapse step is the one that released the lower edge. If check_nonintersect is the culprit,
  the surfaces stopped overlapping beyond a tiny patch, so a tiny patch became admissible and
  inf{feasible} legitimately follows -- meaning the SELECTION RULE (take the infimum) is what
  pins the patch small, not a bug in the predicate.
  The pos(p>0) column shows whether the physical positivity condition would have excluded the
  collapsed patch. If the tiny patch has p < 0, then complementarity -- which the selector's
  docstring cites as its justification -- actually forbids it, and the rule is inconsistent with
  its own stated basis.
""")
