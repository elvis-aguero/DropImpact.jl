using SpectralKM
using SpectralKM: forward_map_r
using DelimitedFiles, Printf

# Mode-by-mode comparison against the reference implementation
# (harrislab-brown/BouncingDroplets), run at OUR canonical reference impact:
#   We = 1.0958, Bo = 0.0166, Oh = 0.0062  <=>  R = 0.35 mm water, U = 0.4759 m/s.
#
# PURPOSE. Three hypotheses for the ~1.9x contact-time overprediction have now been eliminated by
# measurement -- the metric definition, the drop viscous closure, and the bath radius -- and a
# fourth (the contact-edge selection rule) was implemented to match the reference and changed
# nothing to three digits. The remaining evidence points at the trajectory excursion rather than
# contact bookkeeping: penetration reaches 0.684 with z_cm down to 0.207, and the climb back alone
# costs ~3.2 tau_cap. So the question is whether the FORCE or the MODE AMPLITUDES diverge, and
# this localises that by direct comparison instead of by proposing rules.
#
# READS ONLY. No source change, defaults untouched (selector = :feasible, the shipped rule), so
# the comparison measures the model as it actually is.
#
# Their trace comes from derivations/../<tmp>/BouncingDroplets/reference_trace.csv, produced by
# run_reference.m: columns tt (t/t_sigma), zc (z_cm/R), vv (v/U), ff (force, dimensional),
# rcn (rc/R), b2..b4 (drop modes / R), a1..a3 (bath modes / R).

const WE, BO, OH = 1.0958, 0.016614, 0.006166
const REF_CSV = get(ENV, "REF_TRACE",
                    joinpath(homedir(), ".claude", "jobs", "51a48cf9", "tmp",
                             "BouncingDroplets", "reference_trace.csv"))

p = Params(We=WE, Bo=BO, Oh=OH, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)
@printf("ours:      We=%.4f Bo=%.5f Oh=%.5f  b=%.1f M=%d L=%d N=%d selector=%s\n",
        p.We, p.Bo, p.Oh, p.b, p.M, p.L, p.N, p.selector)
@printf("reference: b=25R, M=151, L=55, poly6 pressure shape\n\n")

levels, diag, phases = run_simulation(p; t_end=14.0)
ts = [l.t for l in levels]
rows = [d for d in diag if haskey(d, :theta_c)]

tct = threshold_contact_time(ts, levels)
cor = coefficient_of_restitution(ts, levels, phases)
imax = isempty(rows) ? 0 : argmax([d.f for d in rows])

println("=== OUR metrics ===")
@printf("  tc(threshold)      %s\n", tct === nothing ? "n/a" : @sprintf("%.4f", tct))
@printf("  tc(InContact)      %.4f\n", contact_time(ts, phases))
@printf("  CoR                %s\n", cor === nothing ? "n/a" : @sprintf("%.4f", cor))
@printf("  min z_cm/R         %.4f\n", minimum(l.com.z for l in levels))
@printf("  max penetration    %.4f\n", max_penetration_depth(levels, p.L))
if imax > 0
    j = findmin(abs.(ts .- rows[imax].t))[2]
    @printf("  peak force         %.4f at t = %.4f\n", rows[imax].f, rows[imax].t)
    @printf("  max r_c/R          %.4f\n",
            maximum(forward_map_r(levels[findmin(abs.(ts .- d.t))[2]].drop.beta, d.theta_c, p.L)
                    for d in rows))
end
@printf("  peak |beta_2|      %.5f\n", maximum(abs(l.drop.beta[3]) for l in levels))
@printf("  peak |beta_3|      %.5f\n", maximum(abs(l.drop.beta[4]) for l in levels))
@printf("  peak |a_1|         %.5f\n", maximum(abs(l.bath.a[2]) for l in levels))
@printf("  peak |a_2|         %.5f\n", maximum(abs(l.bath.a[3]) for l in levels))

if !isfile(REF_CSV)
    println("\nreference trace not found at $REF_CSV -- run run_reference.m in MATLAB first.")
    exit(0)
end

raw = readdlm(REF_CSV, ','; header=true)
d, hdr = raw[1], vec(raw[2])
c(n) = findfirst(==(n), hdr)
tt = Float64.(d[:, c("tt")]); zc = Float64.(d[:, c("zc")])
ff = Float64.(d[:, c("ff")]); rcn = Float64.(d[:, c("rcn")])
b2r = Float64.(d[:, c("b2")]); b3r = Float64.(d[:, c("b3")])
a1r = Float64.(d[:, c("a1")]); a2r = Float64.(d[:, c("a2")])

println("\n=== REFERENCE metrics ===")
below = findall(<(1.0), zc)
@printf("  tc(threshold)      %.4f\n",
        isempty(below) ? NaN : tt[last(below)] - tt[first(below)])
pos = findall(>(0.0), ff)
@printf("  contact (f>0)      %.4f\n",
        isempty(pos) ? NaN : tt[last(pos)] - tt[first(pos)])
@printf("  min z_cm/R         %.4f\n", minimum(zc))
@printf("  max r_c/R          %.4f\n", maximum(rcn))
@printf("  peak |beta_2|      %.5f\n", maximum(abs, b2r))
@printf("  peak |beta_3|      %.5f\n", maximum(abs, b3r))
@printf("  peak |a_1|         %.5f\n", maximum(abs, a1r))
@printf("  peak |a_2|         %.5f\n", maximum(abs, a2r))

println("\n=== trajectory side by side, on the reference time grid ===")
@printf("%-9s %-11s %-11s %-11s %-11s %-11s\n",
        "t/tsig", "z_cm ours", "z_cm ref", "diff", "rc ours", "rc ref")
zs = [l.com.z for l in levels]
for k in 1:max(1, length(tt) ÷ 16):length(tt)
    t = tt[k]
    t > ts[end] && break
    j = findmin(abs.(ts .- t))[2]
    rcj = if isempty(rows)
        NaN
    else
        i = findmin(abs.([r.t for r in rows] .- t))[2]
        abs(rows[i].t - t) < 0.05 ? forward_map_r(levels[j].drop.beta, rows[i].theta_c, p.L) : 0.0
    end
    @printf("%-9.4f %-11.5f %-11.5f %+-11.5f %-11.5f %-11.5f\n",
            t, zs[j], zc[k], zs[j] - zc[k], rcj, rcn[k])
end

println("""

WHAT TO LOOK FOR
  * if z_cm tracks early and diverges only near maximum penetration, the FORCE is too weak;
  * if z_cm diverges from the first steps, the coupling or the initial condition differs;
  * if the mode amplitudes differ by a large factor while z_cm agrees, the projection or the
    normalisation of b_l / c_m differs rather than the dynamics.
""")
