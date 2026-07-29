using SpectralKM, Printf
# PHASE 1 (round 2). The b hypothesis is FALSIFIED: tc is flat in b (4.9402, 4.9613, 4.9614,
# 4.9614 for b = 6, 12, 20, 30), identical to 5 digits from b=20 on. Reflected waves are not
# the cause.
#
# tc is ALSO nearly flat in We (4.94-6.12 over We = 0.027..2.47) where experiment falls
# 4.02 -> 2.41. Flat in both suggests the duration is set by the model's CONTACT-EXIT DECISION
# rather than by the wave field or the impact energy.
#
# run_simulation ends contact when the net force f <= 0 for n_f_negative = 3 consecutive steps,
# or when no theta_c is admissible -- but theta_c_floor = 2e-3 makes the latter hard to reach.
# So this traces the exit decision directly: how f, theta_c and the trajectory behave through
# contact, and what actually terminates it.
#
# Reference: tc(InContact) ~ 4.24, against the l=2 oscillation period 2*pi/sqrt(8) = 2.22.
#
# MEASURED RESULT:
#   * f <= 0 on 0 of 221 steps. The net force decays asymptotically to 1.4e-8 from ABOVE and
#     never reverses, so the f<=0 exit criterion never fires; contact ends at the theta_c floor.
#   * v turns positive at t = 1.57 but contact ends at t = 4.24, so 63% of the contact duration
#     occurs while the drop is already RISING.
#   * z_cm bottoms at 0.207 (south pole 0.68R below the surface) and the climb back to z=1 at
#     v ~ 0.245 accounts for 3.2 of the 4.94 tau_cap threshold contact time on its own.

const WE, BO, OH = 0.998546, 0.047169, 0.048658

p = Params(We=WE, Bo=BO, Oh=OH, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)
levels, diag, phases = run_simulation(p; t_end=12.0)
ts = [l.t for l in levels]
rows = [d for d in diag if haskey(d, :theta_c)]

@printf("l=2 period 2pi/sqrt(8) = %.3f tau_cap\n", 2pi / sqrt(8.0))
@printf("contact_time(InContact) = %.4f   threshold = %s\n",
        contact_time(ts, phases),
        let v = threshold_contact_time(ts, levels); v === nothing ? "nothing" : @sprintf("%.4f", v) end)
ivs = contact_intervals(ts, phases)
@printf("contact intervals: %d\n", length(ivs))
for (i, iv) in enumerate(ivs)
    @printf("  %d: t %.4f -> %.4f  duration %.4f\n", i, iv.t_start, iv.t_end, iv.duration)
end

println("\n=== trace through contact: is f lingering near zero? ===")
@printf("%-9s %-10s %-12s %-11s %-10s %-10s\n", "t", "theta_c", "f", "z_cm", "v", "|T|")
n = length(rows)
idx = unique(round.(Int, range(1, n; length=min(n, 34))))
for i in idx
    d = rows[i]
    lvl = levels[findmin(abs.(ts .- d.t))[2]]
    @printf("%-9.4f %-10.5f %-12.4e %-11.5f %-10.5f %-10.2e\n",
            d.t, d.theta_c, d.f, lvl.com.z, lvl.com.v, abs(d.T))
end

println("\n=== how does contact END? ===")
fs = [d.f for d in rows]
neg = findall(<=(0), fs)
@printf("steps with f <= 0: %d of %d\n", length(neg), length(fs))
if !isempty(neg)
    @printf("first f<=0 at step %d, t = %.4f (contact ends after %d consecutive)\n",
            first(neg), rows[first(neg)].t, 3)
end
@printf("final theta_c = %.6f  (floor is 2e-3)\n", rows[end].theta_c)
@printf("min theta_c over contact = %.6f\n", minimum(d.theta_c for d in rows))
@printf("f at last 6 steps: %s\n", join([@sprintf("%.3e", d.f) for d in rows[max(1,end-5):end]], "  "))

println("\n=== when does the CoM actually start moving up, vs when contact ends? ===")
ic = findall(==(InContact), phases)
if !isempty(ic)
    i0, i1 = first(ic), last(ic)
    vs = [l.com.v for l in levels]
    iturn = i0 - 1 + findfirst(>(0), vs[i0:i1])
    if iturn !== nothing
        @printf("v turns positive at t = %.4f (index %d)\n", ts[iturn], iturn)
        @printf("contact ends at        t = %.4f (index %d)\n", ts[i1], i1)
        @printf("=> %.4f tau_cap of contact AFTER the drop starts rising\n", ts[i1] - ts[iturn])
        @printf("   (%.0f%% of the total contact duration)\n",
                100 * (ts[i1] - ts[iturn]) / (ts[i1] - ts[i0]))
    end
end
