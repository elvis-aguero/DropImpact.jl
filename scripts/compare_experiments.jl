#!/usr/bin/env julia
# Overlay SpectralKM against the low-Weber rebound experiments.
#
#   (a) LAZY: each (We, Bo, Oh) simulation is cached to data/experiments/sim_cache.csv and
#       re-used on subsequent runs. Only genuinely new points cost anything, so this is cheap
#       to re-run after a model change and expensive only once.
#   (b) OVERLAY: coefficient of restitution against We, experiments and simulation together,
#       written as a standalone SVG (no plotting dependency).
#
# Usage
#   julia --project=. scripts/compare_experiments.jl              # default sweep
#   julia --project=. scripts/compare_experiments.jl --n 12       # more points
#   julia --project=. scripts/compare_experiments.jl --force      # ignore the cache
#   julia --project=. scripts/compare_experiments.jl --viscous lamb
#
# ---------------------------------------------------------------------------------------------
# TWO DEFINITIONAL CAVEATS. Both are unresolved in the source data, and the overlay is only as
# meaningful as they allow. Stated here rather than buried, because a plot silently comparing
# two different quantities is worse than no plot.
#
#  1. The experimental `cor` column is published as a coefficient of restitution and used that
#     way by the sister repo's lowWeberComparison.m, but it is NOT Vo/Vi (the ratio
#     cor/(Vo/Vi) spans -87..+34 across the 627 rows). Our `coefficient_of_restitution` is
#     specifically -V_exit/V_impact from the centre-of-mass velocity at contact entry/exit
#     (src/postprocessing.jl:105). These may differ by a reference-height/gravity correction.
#     The script therefore plots BOTH the published `cor` and the raw `Vo/Vi` for the
#     experiments, so the reader can see how much the choice matters.
#
#  2. The `ho_m` column is NOT plotted. Its meaning is undocumented, it ranges 0.26..5.73 times
#     R, and the sister repo's own experimental max-deflection scatter is commented out. Until
#     someone states what it is, comparing it to our max_penetration_depth would be guesswork.
# ---------------------------------------------------------------------------------------------

using SpectralKM
using DelimitedFiles
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const EXP_CSV = joinpath(ROOT, "data", "experiments", "rebound_low_weber.csv")
const CACHE = joinpath(ROOT, "data", "experiments", "sim_cache.csv")
const OUT_SVG = joinpath(ROOT, "data", "experiments", "cor_vs_we.svg")

# ---------------------------------------------------------------- args
function getarg(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && return default
    return ARGS[i+1]
end
const NPOINTS = parse(Int, getarg("--n", "8"))
const FORCE = "--force" in ARGS
const VISCOUS = Symbol(getarg("--viscous", "reid"))
const T_END = parse(Float64, getarg("--tend", "12.0"))

# ---------------------------------------------------------------- experiments
function load_experiments()
    raw = readdlm(EXP_CSV, ','; header=true)
    data, hdr = raw[1], vec(raw[2])
    col(name) = findfirst(==(name), hdr)
    getf(name) = Float64.(data[:, col(name)])
    return (We=getf("We"), Bo=getf("Bo"), Oh=getf("Oh"),
            cor=getf("cor"), cor_vv=getf("cor_from_velocities"))
end

# ---------------------------------------------------------------- cache
cache_key(We, Bo, Oh, viscous) = @sprintf("%.6f|%.6f|%.6f|%s", We, Bo, Oh, viscous)

function load_cache()
    d = Dict{String,NTuple{3,Float64}}()
    (FORCE || !isfile(CACHE)) && return d
    raw = readdlm(CACHE, ','; header=true)
    rows = raw[1]
    for i in axes(rows, 1)
        k = cache_key(rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4])
        d[k] = (Float64(rows[i, 5]), Float64(rows[i, 6]), Float64(rows[i, 7]))
    end
    return d
end

function append_cache(We, Bo, Oh, viscous, cor, tc, pen)
    isnew = !isfile(CACHE)
    open(CACHE, "a") do io
        isnew && println(io, "We,Bo,Oh,viscous,cor,contact_time,max_penetration")
        @printf(io, "%.6f,%.6f,%.6f,%s,%.8g,%.8g,%.8g\n", We, Bo, Oh, viscous, cor, tc, pen)
    end
end

# ---------------------------------------------------------------- one simulation
function simulate(We, Bo, Oh; viscous=VISCOUS, t_end=T_END)
    p = Params(We=We, Bo=Bo, Oh=Oh, b=6.0, h0=3.0, viscous=viscous)
    levels, diag, phases = run_simulation(p; t_end=t_end)
    ts = [l.t for l in levels]
    cor = coefficient_of_restitution(ts, levels, phases)
    cor === nothing && return nothing
    return (cor=cor,
            tc=contact_time(ts, phases),
            pen=max_penetration_depth(levels, p.L))
end

# ---------------------------------------------------------------- pick the sweep
"""
Choose `n` experimental points spanning We logarithmically, taking the median Bo/Oh within
each We bin so the simulated point is representative rather than an outlier.
"""
function choose_points(exp, n)
    lo, hi = log10(minimum(exp.We)), log10(maximum(exp.We))
    edges = range(lo, hi; length=n + 1)
    pts = NamedTuple[]
    for i in 1:n
        sel = findall(j -> edges[i] <= log10(exp.We[j]) <= edges[i+1], eachindex(exp.We))
        isempty(sel) && continue
        push!(pts, (We=median(exp.We[sel]), Bo=median(exp.Bo[sel]), Oh=median(exp.Oh[sel]),
                    nexp=length(sel)))
    end
    return pts
end

# ---------------------------------------------------------------- SVG
function write_svg(path, exp, sims)
    W, H, ml, mr, mt, mb = 900, 560, 90, 30, 60, 70
    pw, ph = W - ml - mr, H - mt - mb
    xs = log10.(exp.We)
    allc = vcat(exp.cor, [s.cor for s in sims])
    x0, x1 = minimum(xs) - 0.05, maximum(xs) + 0.05
    y0, y1 = 0.0, max(0.55, maximum(allc) * 1.1)
    px(x) = ml + (log10(x) - x0) / (x1 - x0) * pw
    py(y) = mt + (1 - (y - y0) / (y1 - y0)) * ph
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="Helvetica,Arial,sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<rect x="$ml" y="$mt" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    # gridlines / ticks
    for e in ceil(Int, x0):floor(Int, x1)
        x = px(10.0^e)
        println(io, """<line x1="$x" y1="$mt" x2="$x" y2="$(mt+ph)" stroke="#eee"/>""")
        println(io, """<text x="$x" y="$(mt+ph+22)" font-size="13" text-anchor="middle">10<tspan font-size="9" dy="-5">$e</tspan></text>""")
    end
    for yv in 0.0:0.1:y1
        y = py(yv)
        println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y" stroke="#eee"/>""")
        println(io, """<text x="$(ml-10)" y="$(y+4)" font-size="13" text-anchor="end">$(round(yv,digits=1))</text>""")
    end
    # experiments: published cor
    for i in eachindex(xs)
        println(io, """<circle cx="$(px(exp.We[i]))" cy="$(py(exp.cor[i]))" r="2.4" fill="#4a90d9" opacity="0.55"/>""")
    end
    # experiments: raw Vo/Vi, to expose how much the definition matters
    for i in eachindex(xs)
        c = exp.cor_vv[i]
        (0 <= c <= y1) || continue
        println(io, """<circle cx="$(px(exp.We[i]))" cy="$(py(c))" r="1.6" fill="none" stroke="#bbb" stroke-width="0.7"/>""")
    end
    # simulation
    ss = sort(sims; by=s -> s.We)
    pts = join(["$(px(s.We)),$(py(s.cor))" for s in ss], " ")
    println(io, """<polyline points="$pts" fill="none" stroke="#c0392b" stroke-width="2.2"/>""")
    for s in ss
        println(io, """<circle cx="$(px(s.We))" cy="$(py(s.cor))" r="6" fill="#c0392b" stroke="white" stroke-width="1.6"/>""")
    end
    # labels
    println(io, """<text x="$(ml+pw/2)" y="$(H-18)" font-size="16" text-anchor="middle">We</text>""")
    println(io, """<text x="24" y="$(mt+ph/2)" font-size="16" text-anchor="middle" transform="rotate(-90 24 $(mt+ph/2))">coefficient of restitution</text>""")
    println(io, """<text x="$ml" y="30" font-size="15" font-weight="bold">SpectralKM vs low-Weber rebound experiments (viscous = $(VISCOUS))</text>""")
    lx, ly = ml + pw - 250, mt + 16
    println(io, """<circle cx="$lx" cy="$ly" r="4" fill="#4a90d9" opacity="0.7"/><text x="$(lx+12)" y="$(ly+4)" font-size="12">experiment (published cor)</text>""")
    println(io, """<circle cx="$lx" cy="$(ly+18)" r="3" fill="none" stroke="#bbb"/><text x="$(lx+12)" y="$(ly+22)" font-size="12">experiment (raw Vo/Vi)</text>""")
    println(io, """<circle cx="$lx" cy="$(ly+36)" r="5" fill="#c0392b"/><text x="$(lx+12)" y="$(ly+40)" font-size="12">SpectralKM</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
end

# ---------------------------------------------------------------- main
exp = load_experiments()
@printf("experiments: %d points, We %.4g..%.4g, Bo %.4g..%.4g, Oh %.4g..%.4g\n",
        length(exp.We), minimum(exp.We), maximum(exp.We),
        minimum(exp.Bo), maximum(exp.Bo), minimum(exp.Oh), maximum(exp.Oh))

cache = load_cache()
@printf("cache: %d entries%s\n", length(cache), FORCE ? " (ignored, --force)" : "")

pts = choose_points(exp, NPOINTS)
@printf("\nsimulating %d representative points (viscous = %s, t_end = %.1f)\n", length(pts), VISCOUS, T_END)
@printf("%-4s %-10s %-10s %-10s %-6s %-10s %-10s %-9s %-s\n",
        "#", "We", "Bo", "Oh", "n_exp", "cor_sim", "t_c", "pen", "source")
flush(stdout)

sims = NamedTuple[]
for (i, pt) in enumerate(pts)
    k = cache_key(pt.We, pt.Bo, pt.Oh, VISCOUS)
    if haskey(cache, k)
        cor, tc, pen = cache[k]
        src = "cache"
    else
        r = simulate(pt.We, pt.Bo, pt.Oh)
        if r === nothing
            @printf("%-4d %-10.4g %-10.4g %-10.4g %-6d %-s\n", i, pt.We, pt.Bo, pt.Oh, pt.nexp,
                    "NO REBOUND (no contact, or still in contact at t_end)")
            flush(stdout)
            continue
        end
        cor, tc, pen = r.cor, r.tc, r.pen
        append_cache(pt.We, pt.Bo, pt.Oh, VISCOUS, cor, tc, pen)
        src = "ran"
    end
    push!(sims, (We=pt.We, Bo=pt.Bo, Oh=pt.Oh, cor=cor, tc=tc, pen=pen))
    @printf("%-4d %-10.4g %-10.4g %-10.4g %-6d %-10.4f %-10.4f %-9.4f %-s\n",
            i, pt.We, pt.Bo, pt.Oh, pt.nexp, cor, tc, pen, src)
    flush(stdout)
end

if isempty(sims)
    println("\nno simulated points; nothing to overlay.")
    exit(0)
end

write_svg(OUT_SVG, exp, sims)
@printf("\nwrote %s\n", relpath(OUT_SVG, ROOT))

# residuals against the nearest-in-We experimental band
println("\nsimulation vs experiment, per We band (experiment = published cor):")
@printf("%-10s %-12s %-12s %-12s %-s\n", "We", "cor_sim", "cor_exp_med", "cor_exp_iqr", "sim - med")
for s in sort(sims; by=x -> x.We)
    sel = findall(j -> abs(log10(exp.We[j]) - log10(s.We)) < 0.15, eachindex(exp.We))
    isempty(sel) && continue
    m = median(exp.cor[sel])
    q = quantile(exp.cor[sel], [0.25, 0.75])
    @printf("%-10.4g %-12.4f %-12.4f %-12s %+.4f\n", s.We, s.cor, m,
            @sprintf("%.3f-%.3f", q[1], q[2]), s.cor - m)
end

println("""

CAVEATS, restated because they bound what this plot means:
  * the experimental `cor` is published as a restitution coefficient but is NOT Vo/Vi, and our
    simulated value is specifically -V_exit/V_impact. The grey open circles show the raw Vo/Vi
    for comparison -- where blue and grey diverge, the definition matters more than the model.
  * `ho_m` is not plotted: its meaning is undocumented and the sister repo does not use it.
""")
