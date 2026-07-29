using SpectralKM
using SpectralKM: forward_map_r
using Printf

# Is the contact patch's slow growth caused by the SEARCH in select_theta_c, or by the
# FEASIBILITY CRITERION it searches over?
#
# CONTEXT. At We=0.0231, Bo=0.02, Oh=0.03 the measured contact radius creeps to a peak of only
# 0.378 R at t = 2.11, while experiment, DNS and Alventosa's own KM model all rise to
# rc ~ 1.05-1.11 R within t = 0.42-0.67. The spreading phase is essentially absent.
#
# select_theta_c returns inf{theta : non-intersection and monotone-r hold}, found by walking
# outward from the PREVIOUS step's theta_c in steps of max(0.15*theta_prev, 5e-3), at most
# nwalk=14 of them, then bisecting. Two possibilities:
#
#   (a) SEARCH-LIMITED: the feasible band's true lower edge is already large, but the local
#       walk cannot reach it in 14 steps from a small theta_prev, so theta_c is dragged along
#       behind the physics.
#   (b) CRITERION-LIMITED: the true inf{feasible} really is small -- the non-intersection and
#       monotone-r predicates genuinely permit a tiny patch -- in which case the search is
#       faithful and the criterion is what disagrees with reality.
#
# These are distinguished by BRUTE FORCE: scan feasibility over a dense theta grid at each early
# contact step and compare the true inf{feasible} against what the selector returned.

const WE, BO, OH = 0.0231, 0.02, 0.03
const THETA_MAX = 0.95 * pi / 2

p = Params(We=WE, Bo=BO, Oh=OH, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)

# --- advance in free flight until contact is detected, mirroring run_simulation ---
lvl0 = initial_level(p)
hist = SimHistory(lvl0, lvl0)
dt = 1e-3
theta_seed = nothing
for _ in 1:20_000
    th = onset_theta_c(hist, dt, p)
    if th !== nothing
        global theta_seed = th
        break
    end
    trial = free_flight_step(hist, dt, p)
    hist.prev = hist.curr
    hist.curr = trial
end
theta_seed === nothing && error("never detected onset")
@printf("onset detected at t = %.5f, seed theta_c = %.6f\n\n", hist.curr.t, theta_seed)

grid = vcat(range(1e-4, 0.05; length=40), range(0.06, THETA_MAX; length=110))

@printf("%-6s %-9s %-11s %-11s %-11s %-11s %-9s %-s\n",
        "step", "t", "sel theta_c", "true inf", "sup feas", "sel r_c", "ratio", "verdict")
theta_prev = theta_seed
chat_guess = zeros(p.N + 1); chat_guess[1] = 1e-3
for step in 1:16
    # brute-force feasibility over the whole admissible range
    feas = [feasible_at(hist, dt, th, chat_guess, p)[1] for th in grid]
    idx = findall(identity, feas)
    trueinf = isempty(idx) ? NaN : grid[first(idx)]
    truesup = isempty(idx) ? NaN : grid[last(idx)]

    best = select_theta_c(hist, dt, theta_prev, chat_guess, p)
    if best === nothing
        @printf("%-6d %-9.4f %-11s %-11.5f %-11.5f\n", step, hist.curr.t, "nothing", trueinf, truesup)
        break
    end
    sel = best.theta_c
    rc = forward_map_r(best.beta, sel, p.L)
    ratio = isnan(trueinf) ? NaN : sel / trueinf
    verdict = isnan(trueinf) ? "no feasible" :
              (abs(sel - trueinf) <= 2e-3 ? "selector FAITHFUL" : "selector LAGS true inf")
    @printf("%-6d %-9.4f %-11.5f %-11.5f %-11.5f %-11.5f %-9.3f %-s\n",
            step, hist.curr.t, sel, trueinf, truesup, rc, ratio, verdict)

    # advance one real contact step so the history evolves as in a run
    lvl, info = contact_step(hist, dt, theta_prev, chat_guess, p)
    lvl === nothing && (println("contact_step failed"); break)
    hist.prev = hist.curr; hist.curr = lvl
    global theta_prev = info.theta_c
    global chat_guess = lvl.X[1:p.N+1]
end

println("""

READING
  If 'true inf' tracks the selector closely, the search is faithful and the FEASIBILITY
  CRITERION is what permits a narrow patch -- the disagreement with experiment is then in the
  physics of non-intersection / monotone-r, not in the search.
  If 'true inf' is much LARGER than the selector's theta_c, the local walk is rate-limiting and
  the patch is dragged behind the physics.
  'sup feas' shows how wide the admissible band is: if it extends to ~1.49 while inf stays tiny,
  the band is broad and the choice of its LOWER edge is what pins the patch small.
""")
