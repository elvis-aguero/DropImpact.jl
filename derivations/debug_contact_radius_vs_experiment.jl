using SpectralKM
using SpectralKM: forward_map_r
using DelimitedFiles, Printf

# Time-resolved comparison of the contact radius against AlventosaEtAl2023 Figure 6(b), which
# gives rc/R against t/t_sigma for EXPERIMENT, DNS and their own KM model at exactly
# We = 0.0231, Oh = 0.03, Bo = 0.02. Scalar metrics cannot say WHEN we diverge; this can.
#
# WHY: our contact duration is ~50% long at low We and ~107% at high We, and our peak contact
# radius is 0.81R against 1.10-1.22R in all three references. Since their KM model matches
# experiment (3.730 vs 3.607 tau_sigma), this is not a limitation of the kinematic-match family.
#
# DEFINITION CAVEAT, stated because it bounds the comparison: their r_c is the geometric
# intersection of the reconstructed interfaces; ours is the kinematic-match patch extent chosen
# by the feasibility-edge selector, r_c = xi(theta_c) sin(theta_c). These are not the same
# construction. But the gap cannot be purely definitional: theirs exceeds R (1.10) while
# select_theta_c caps theta_c at 0.95*pi/2, so ours cannot represent a patch wider than the
# drop's equator at all.

const WE, BO, OH = 0.0231, 0.02, 0.03
const CSV = joinpath(@__DIR__, "..", "data", "experiments", "contact_radius_timeseries.csv")

function load_series(name)
    raw = readdlm(CSV, ','; header=true)
    data, hdr = raw[1], vec(raw[2])
    col(n) = findfirst(==(n), hdr)
    keep = findall(i -> String(strip(string(data[i, col("series")]))) == name, axes(data, 1))
    t = Float64.(data[keep, col("t_over_tsigma")])
    r = Float64.(data[keep, col("rc_over_R")])
    o = sortperm(t)
    return t[o], r[o]
end

println("running one simulation at We=$WE, Bo=$BO, Oh=$OH ...")
p = Params(We=WE, Bo=BO, Oh=OH, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)
levels, diag, phases = run_simulation(p; t_end=10.0)
ts = [l.t for l in levels]
rows = [d for d in diag if haskey(d, :theta_c)]

# our contact radius per contact step: r_c = xi(theta_c) sin(theta_c), via the forward map
t_sim = Float64[]; rc_sim = Float64[]
for d in rows
    i = findmin(abs.(ts .- d.t))[2]
    push!(t_sim, d.t)
    push!(rc_sim, forward_map_r(levels[i].drop.beta, d.theta_c, p.L))
end

@printf("\n%-22s %-12s %-14s %-14s\n", "series", "duration", "max rc/R", "t at max")
for nm in ("experiment_lowWe", "dns", "km_model")
    tt, rr = load_series(nm)
    nz = findall(>(1e-9), rr)
    isempty(nz) && continue
    imax = argmax(rr)
    @printf("%-22s %-12.3f %-14.4f %-14.3f\n", nm,
            tt[nz[end]] - tt[nz[1]], rr[imax], tt[imax])
end
imax = argmax(rc_sim)
@printf("%-22s %-12.3f %-14.4f %-14.3f\n", "SpectralKM (ours)",
        t_sim[end] - t_sim[1], rc_sim[imax], t_sim[imax])

# where do we diverge? compare on the experimental time grid
te, re = load_series("experiment_lowWe")
println("\nrc/R against the experimental series, on its own time grid:")
@printf("%-10s %-12s %-12s %-12s\n", "t/tsigma", "exp", "ours", "ours - exp")
for k in 1:max(1, length(te) ÷ 14):length(te)
    t = te[k]
    t > t_sim[end] && break
    j = findmin(abs.(t_sim .- t))[2]
    @printf("%-10.3f %-12.4f %-12.4f %+.4f\n", t, re[k], rc_sim[j], rc_sim[j] - re[k])
end

# --- SVG overlay ---
W, H, ml, mr, mt, mb = 880, 470, 85, 165, 46, 62
pw, ph = W - ml - mr, H - mt - mb
allt = vcat(te, t_sim); allr = vcat(re, rc_sim)
x1 = maximum(allt) * 1.02; y1 = maximum(allr) * 1.12
px(x) = ml + x / x1 * pw
py(y) = mt + (1 - y / y1) * ph
io = IOBuffer()
println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="Helvetica,Arial,sans-serif">""")
println(io, """<rect width="$W" height="$H" fill="white"/>""")
println(io, """<text x="$ml" y="26" font-size="15" font-weight="bold">Contact radius vs time, We=$WE Bo=$BO Oh=$OH (Alventosa et al. Fig 6b)</text>""")
println(io, """<rect x="$ml" y="$mt" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
for i in 0:4
    xv = i * x1 / 4; x = px(xv)
    println(io, """<line x1="$x" y1="$mt" x2="$x" y2="$(mt+ph)" stroke="#eee"/>""")
    println(io, """<text x="$x" y="$(mt+ph+20)" font-size="12" text-anchor="middle">$(round(xv,digits=2))</text>""")
end
for i in 0:5
    yv = i * y1 / 5; y = py(yv)
    println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y" stroke="#eee"/>""")
    println(io, """<text x="$(ml-9)" y="$(y+4)" font-size="12" text-anchor="end">$(round(yv,digits=2))</text>""")
end
println(io, """<line x1="$ml" y1="$(py(1.0))" x2="$(ml+pw)" y2="$(py(1.0))" stroke="#bbb" stroke-dasharray="4,3"/>""")
println(io, """<text x="$(ml+pw-4)" y="$(py(1.0)-5)" font-size="10.5" text-anchor="end" fill="#888">rc = R</text>""")
for (nm, col, wid) in (("km_model", "#7f7f7f", 1.4), ("dns", "#2e8b57", 1.8), ("experiment_lowWe", "#2e75b6", 2.4))
    tt, rr = load_series(nm)
    pts = join(["$(px(tt[i])),$(py(rr[i]))" for i in eachindex(tt)], " ")
    println(io, """<polyline points="$pts" fill="none" stroke="$col" stroke-width="$wid"/>""")
end
pts = join(["$(px(t_sim[i])),$(py(rc_sim[i]))" for i in eachindex(t_sim)], " ")
println(io, """<polyline points="$pts" fill="none" stroke="#c0392b" stroke-width="2.6"/>""")
println(io, """<text x="$(ml+pw/2)" y="$(H-16)" font-size="14" text-anchor="middle">t / t_sigma</text>""")
println(io, """<text x="24" y="$(mt+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 24 $(mt+ph/2))">r_c / R</text>""")
for (i, (lab, col)) in enumerate([("experiment", "#2e75b6"), ("DNS", "#2e8b57"),
                                  ("KM model (Alventosa)", "#7f7f7f"), ("SpectralKM (ours)", "#c0392b")])
    y = mt + 14 + (i - 1) * 19
    println(io, """<line x1="$(ml+pw+12)" y1="$y" x2="$(ml+pw+34)" y2="$y" stroke="$col" stroke-width="2.6"/>""")
    println(io, """<text x="$(ml+pw+39)" y="$(y+4)" font-size="11.5">$lab</text>""")
end
println(io, "</svg>")
out = joinpath(@__DIR__, "..", "data", "experiments", "contact_radius_overlay.svg")
write(out, String(take!(io)))
@printf("\nwrote %s\n", out)
