#!/usr/bin/env julia
# Overlay SpectralKM against the low-Weber rebound experiments.
#
#   (a) LAZY: each (We, Bo, Oh) simulation is cached to data/experiments/sim_cache.csv and
#       re-used on subsequent runs. Only genuinely new points cost anything, so this is cheap
#       to re-run after a model change and expensive only once.
#   (b) OVERLAY: CONTACT TIME tc/t_sigma against We -- the target variable -- with the
#       experimental error bars, plus CoR as a secondary panel. Standalone SVG, no plotting dep.
#
# THIS SCRIPT IS NOT RUN AUTOMATICALLY AND MUST NOT BE RUN ON A LAPTOP CASUALLY. Each new
# (We, Bo, Oh) point is a full impact simulation, ~2 min at the default truncations. The cache
# exists so the sweep is paid for once, deliberately, on hardware chosen for it.
#
# Usage
#   julia --project=. scripts/compare_experiments.jl              # default sweep
#   julia --project=. scripts/compare_experiments.jl --n 12       # more points
#   julia --project=. scripts/compare_experiments.jl --force      # ignore the cache
#   julia --project=. scripts/compare_experiments.jl --viscous lamb
#   julia --project=. scripts/compare_experiments.jl --plot-only   # NO simulation: plot the cache
#
# PARALLELISM. The points are completely independent, so the sweep is embarrassingly parallel and
# should not be run as a serial loop. Use --only to run one point into its own cache shard, fan
# out with xargs, then merge and plot:
#
#   seq 1 10 | xargs -P 6 -I{} julia --project=. scripts/compare_experiments.jl \
#       --n 10 --viscous reid --only {} --yes
#   julia --project=. scripts/compare_experiments.jl --merge-shards --viscous reid
#   julia --project=. scripts/compare_experiments.jl --plot-only --viscous reid
#
# Each shard is a separate file, so there is no concurrent-append race on the cache. Keep the
# fan-out below the core count if anything else is running on the machine.
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
const TC_CSV = joinpath(ROOT, "data", "experiments", "contact_time_vs_we.csv")
const COR_CSV = joinpath(ROOT, "data", "experiments", "cor_vs_we.csv")
# v2: `tc` now means threshold_contact_time (AlventosaEtAl2023's definition), NOT the
# InContact duration stored by v1. v1 entries are unusable, hence the new filename rather
# than silently mixing metrics.
const CACHE = joinpath(ROOT, "data", "experiments", "sim_cache_v2.csv")
const OUT_SVG = joinpath(ROOT, "data", "experiments", "comparison.svg")

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
const ONLY = let v = getarg("--only", ""); v == "" ? nothing : parse(Int, v) end
const MERGE = "--merge-shards" in ARGS
shard_path(i) = joinpath(ROOT, "data", "experiments",
                         "shard_$(VISCOUS)_$(lpad(i, 3, '0')).csv")

# ---------------------------------------------------------------- experiments
"""
Read one of the tidy figure-data CSVs, returning only rows from `src` ("experiment", "dns" or
"km_model"). `value` is the observable column, e.g. "tc_over_tsigma".
"""
function load_figure(path, value, src)
    raw = readdlm(path, ','; header=true)
    data, hdr = raw[1], vec(raw[2])
    col(n) = findfirst(==(n), hdr)
    keep = findall(i -> String(strip(string(data[i, col("source")]))) == src, axes(data, 1))
    num(n) = [x isa Number ? Float64(x) : NaN for x in data[keep, col(n)]]
    return (We=num("We"), val=num(value), val_unc=num("$(value)_unc"),
            Oh=num("Oh"), Bo=num("Bo"))
end

# ---------------------------------------------------------------- cache
cache_key(We, Bo, Oh, viscous) = @sprintf("%.6f|%.6f|%.6f|%s", We, Bo, Oh, viscous)

function load_cache()
    d = Dict{String,NTuple{4,Float64}}()
    (FORCE || !isfile(CACHE)) && return d
    raw = readdlm(CACHE, ','; header=true)
    rows = raw[1]
    for i in axes(rows, 1)
        k = cache_key(rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4])
        d[k] = (Float64(rows[i, 5]), Float64(rows[i, 6]), Float64(rows[i, 7]), Float64(rows[i, 8]))
    end
    return d
end

"""All cached results for `viscous`, as the plotting code expects them. Used by --plot-only,
which runs nothing: it plots exactly the points already paid for."""
function cached_sims(viscous)
    isfile(CACHE) || return NamedTuple[]
    raw = readdlm(CACHE, ','; header=true)
    rows = raw[1]
    out = NamedTuple[]
    for i in axes(rows, 1)
        String(strip(string(rows[i, 4]))) == String(viscous) || continue
        push!(out, (We=Float64(rows[i, 1]), Bo=Float64(rows[i, 2]), Oh=Float64(rows[i, 3]),
                    cor=Float64(rows[i, 5]), tc=Float64(rows[i, 6]),
                    tc_incontact=Float64(rows[i, 7]), pen=Float64(rows[i, 8])))
    end
    return out
end

const CACHE_HEADER = "We,Bo,Oh,viscous,cor,tc_threshold,tc_incontact,max_penetration"

"""Concatenate per-point shards into the main cache, de-duplicating on the cache key."""
function merge_shards()
    seen = Set{String}()
    rows = String[]
    if isfile(CACHE)
        for (i, ln) in enumerate(eachline(CACHE))
            i == 1 && continue
            f = split(ln, ',')
            k = cache_key(parse(Float64, f[1]), parse(Float64, f[2]), parse(Float64, f[3]), f[4])
            k in seen || (push!(seen, k); push!(rows, ln))
        end
    end
    shards = filter(f -> startswith(basename(f), "shard_"),
                    readdir(joinpath(ROOT, "data", "experiments"); join=true))
    nadd = 0
    for sf in shards
        for (i, ln) in enumerate(eachline(sf))
            i == 1 && continue
            isempty(strip(ln)) && continue
            f = split(ln, ',')
            k = cache_key(parse(Float64, f[1]), parse(Float64, f[2]), parse(Float64, f[3]), f[4])
            k in seen && continue
            push!(seen, k); push!(rows, ln); nadd += 1
        end
    end
    open(CACHE, "w") do io
        println(io, CACHE_HEADER)
        for r in rows; println(io, r); end
    end
    @printf("merged %d shard file(s): %d new rows, %d total in %s\n",
            length(shards), nadd, length(rows), relpath(CACHE, ROOT))
end

function append_cache(We, Bo, Oh, viscous, cor, tc, tci, pen)
    path = ONLY === nothing ? CACHE : shard_path(ONLY)
    isnew = !isfile(path)
    open(path, "a") do io
        isnew && println(io, CACHE_HEADER)
        @printf(io, "%.6f,%.6f,%.6f,%s,%.8g,%.8g,%.8g,%.8g\n",
                We, Bo, Oh, viscous, cor, tc, tci, pen)
    end
end

# ---------------------------------------------------------------- one simulation
function simulate(We, Bo, Oh; viscous=VISCOUS, t_end=T_END)
    p = Params(We=We, Bo=Bo, Oh=Oh, b=6.0, h0=3.0, viscous=viscous)
    levels, diag, phases = run_simulation(p; t_end=t_end)
    ts = [l.t for l in levels]
    cor = coefficient_of_restitution(ts, levels, phases)
    cor === nothing && return nothing
    # tc MUST use AlventosaEtAl2023's definition -- the time the centre of mass spends below
    # z=R -- not `contact_time`, which is the InContact duration. Comparing the InContact
    # duration against their published t_c produced a spurious 40-70% discrepancy.
    tc = threshold_contact_time(ts, levels)
    tc === nothing && return nothing
    # Units: our times are already in tau_cap = sqrt(rho R^3/sigma) = their t_sigma, so this is
    # directly tc/t_sigma with no rescaling.
    return (cor=cor,
            tc=tc,
            tc_incontact=contact_time(ts, phases),
            pen=max_penetration_depth(levels, p.L))
end

# ---------------------------------------------------------------- pick the sweep
"""
Choose `n` comparison points by clustering on (We, Bo, Oh) TOGETHER, not by binning on We alone.

WHY THIS IS NOT A DETAIL. The earlier version split log10(We) into equal intervals and took the
median We, Bo and Oh of whatever experimental points fell in each. The dataset spans
Oh = 0.0139..0.7865 and Bo = 0.0037..0.4197, so one We window mixes fluids differing by 33-54x in
Oh, and the experimental tc values WITHIN a window already differ by 1.35-1.48x. Comparing a
simulation at the window's median Bo/Oh against the window's median tc then manufactured a
"2x contact-time overprediction" that does not exist: run at matched parameters this model agrees
with the reference implementation to 0.65% on contact duration.

So: first restrict to a single fluid cluster in (Bo, Oh) -- `bo_tol`/`oh_tol` are RELATIVE
half-widths about the requested centre -- and only then spread points over We within it. Points
are reported with the actual Bo/Oh spread of their neighbourhood so a residual can never again be
quoted without it.
"""
function choose_points(exp, n; bo_centre=nothing, oh_centre=nothing,
                       bo_tol=0.25, oh_tol=0.25)
    # default cluster centre: the modal fluid, taken as the median of the densest Oh decade
    ohc = oh_centre === nothing ? median(exp.Oh) : oh_centre
    boc = bo_centre === nothing ? median(exp.Bo) : bo_centre
    keep = findall(i -> abs(exp.Oh[i] - ohc) <= oh_tol * ohc &&
                        abs(exp.Bo[i] - boc) <= bo_tol * boc, eachindex(exp.We))
    if length(keep) < 2 * n
        @warn """
            Only $(length(keep)) experimental points lie within the requested (Bo, Oh) cluster, so
            the comparison would be thin. Widen bo_tol/oh_tol, or pick a centre where the data
            actually lives -- but do NOT fall back to binning on We alone, which is what produced
            a spurious 2x discrepancy.
            """ ohc boc n_kept=length(keep)
    end
    isempty(keep) && return NamedTuple[]
    We_k, Bo_k, Oh_k = exp.We[keep], exp.Bo[keep], exp.Oh[keep]
    lo, hi = log10(minimum(We_k)), log10(maximum(We_k))
    edges = range(lo, hi; length=n + 1)
    pts = NamedTuple[]
    for i in 1:n
        sel = findall(j -> edges[i] <= log10(We_k[j]) <= edges[i+1], eachindex(We_k))
        isempty(sel) && continue
        push!(pts, (We=median(We_k[sel]), Bo=median(Bo_k[sel]), Oh=median(Oh_k[sel]),
                    nexp=length(sel),
                    oh_spread=maximum(Oh_k[sel]) / minimum(Oh_k[sel]),
                    bo_spread=maximum(Bo_k[sel]) / minimum(Bo_k[sel])))
    end
    return pts
end

# ---------------------------------------------------------------- SVG (two panels)
"""
Two stacked panels sharing a log-We axis: contact time on top (the target variable, with
experimental error bars), CoR beneath. Experiment, DNS, the sister repo's KM model, and this
model are overlaid so that model-vs-model and model-vs-experiment are both visible.
"""
function write_svg(path, panels, sims)
    W, ml, mr, mt, gap, phh, mb = 980, 95, 190, 46, 54, 220, 62
    H = mt + 2 * phh + gap + mb
    pw = W - ml - mr
    allWe = vcat([p.exp.We for p in panels]..., [s.We for s in sims])
    x0, x1 = log10(minimum(allWe)) - 0.05, log10(maximum(allWe)) + 0.05
    px(x) = ml + (log10(x) - x0) / (x1 - x0) * pw
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="Helvetica,Arial,sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$ml" y="26" font-size="15" font-weight="bold">SpectralKM vs Alventosa et al. (2023): experiment, DNS, KM model  [viscous = $(VISCOUS)]</text>""")

    for (k, P) in enumerate(panels)
        top = mt + (k - 1) * (phh + gap)
        vals = vcat(P.exp.val, P.dns.val, P.km.val, [P.simval(s) for s in sims])
        vals = filter(isfinite, vals)
        y0 = max(0.0, minimum(vals) - 0.08 * (maximum(vals) - minimum(vals)))
        y1 = maximum(vals) + 0.10 * (maximum(vals) - minimum(vals))
        py(y) = top + (1 - (y - y0) / (y1 - y0)) * phh
        println(io, """<rect x="$ml" y="$top" width="$pw" height="$phh" fill="none" stroke="#333"/>""")
        for e in ceil(Int, x0):floor(Int, x1)
            x = px(10.0^e)
            println(io, """<line x1="$x" y1="$top" x2="$x" y2="$(top+phh)" stroke="#eee"/>""")
            k == length(panels) && println(io, """<text x="$x" y="$(top+phh+22)" font-size="13" text-anchor="middle">10<tspan font-size="9" dy="-5">$e</tspan></text>""")
        end
        nt = 5
        for i in 0:nt
            yv = y0 + i * (y1 - y0) / nt
            y = py(yv)
            println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y" stroke="#eee"/>""")
            println(io, """<text x="$(ml-10)" y="$(y+4)" font-size="12" text-anchor="end">$(round(yv,digits=2))</text>""")
        end
        # experiment, with error bars where present
        for i in eachindex(P.exp.We)
            x, y = px(P.exp.We[i]), py(P.exp.val[i])
            u = P.exp.val_unc[i]
            if isfinite(u) && u > 0
                println(io, """<line x1="$x" y1="$(py(P.exp.val[i]-u))" x2="$x" y2="$(py(P.exp.val[i]+u))" stroke="#9dc3e6" stroke-width="0.7"/>""")
            end
            println(io, """<circle cx="$x" cy="$y" r="2.3" fill="#2e75b6" opacity="0.55"/>""")
        end
        for i in eachindex(P.km.We)
            println(io, """<circle cx="$(px(P.km.We[i]))" cy="$(py(P.km.val[i]))" r="1.3" fill="#7f7f7f" opacity="0.45"/>""")
        end
        for i in eachindex(P.dns.We)
            println(io, """<rect x="$(px(P.dns.We[i])-3)" y="$(py(P.dns.val[i])-3)" width="6" height="6" fill="none" stroke="#2e8b57" stroke-width="1.5"/>""")
        end
        ss = sort(sims; by=s -> s.We)
        pts = join(["$(px(s.We)),$(py(P.simval(s)))" for s in ss if isfinite(P.simval(s))], " ")
        println(io, """<polyline points="$pts" fill="none" stroke="#c0392b" stroke-width="2.4"/>""")
        for s in ss
            isfinite(P.simval(s)) || continue
            println(io, """<circle cx="$(px(s.We))" cy="$(py(P.simval(s)))" r="5.5" fill="#c0392b" stroke="white" stroke-width="1.5"/>""")
        end
        cy = top + phh / 2
        println(io, """<text x="26" y="$cy" font-size="14" text-anchor="middle" transform="rotate(-90 26 $cy)">$(P.label)</text>""")
    end
    println(io, """<text x="$(ml+pw/2)" y="$(H-16)" font-size="15" text-anchor="middle">We</text>""")
    lx, ly = ml + pw + 18, mt + 10
    for (i, (mk, lab)) in enumerate([("exp", "experiment (±unc)"), ("km", "KM model (Alventosa)"),
                                     ("dns", "DNS"), ("sim", "SpectralKM")])
        y = ly + (i - 1) * 20
        if mk == "exp";      println(io, """<circle cx="$lx" cy="$y" r="3.2" fill="#2e75b6"/>""")
        elseif mk == "km";   println(io, """<circle cx="$lx" cy="$y" r="2" fill="#7f7f7f"/>""")
        elseif mk == "dns";  println(io, """<rect x="$(lx-3)" y="$(y-3)" width="6" height="6" fill="none" stroke="#2e8b57" stroke-width="1.5"/>""")
        else                 println(io, """<circle cx="$lx" cy="$y" r="5" fill="#c0392b"/>""")
        end
        println(io, """<text x="$(lx+11)" y="$(y+4)" font-size="11.5">$lab</text>""")
    end
    println(io, "</svg>")
    write(path, String(take!(io)))
end

# ---------------------------------------------------------------- main
tc_exp = load_figure(TC_CSV, "tc_over_tsigma", "experiment")
tc_dns = load_figure(TC_CSV, "tc_over_tsigma", "dns")
tc_km  = load_figure(TC_CSV, "tc_over_tsigma", "km_model")
cor_exp = load_figure(COR_CSV, "cor", "experiment")
cor_dns = load_figure(COR_CSV, "cor", "dns")
cor_km  = load_figure(COR_CSV, "cor", "km_model")

@printf("TARGET VARIABLE  contact time tc/t_sigma : %d experimental points, We %.4g..%.4g, tc %.3f..%.3f\n",
        length(tc_exp.We), minimum(tc_exp.We), maximum(tc_exp.We), minimum(tc_exp.val), maximum(tc_exp.val))
@printf("secondary        CoR                     : %d experimental points\n", length(cor_exp.We))

if MERGE
    merge_shards()
    exit(0)
end

if "--plot-only" in ARGS
    sims = cached_sims(VISCOUS)
    if isempty(sims)
        println("\n--plot-only: no cached results for viscous = $(VISCOUS); nothing to plot.")
        exit(0)
    end
    @printf("\n--plot-only: %d cached points for viscous = %s. RUNNING NOTHING.\n",
            length(sims), VISCOUS)
    @printf("%-4s %-10s %-10s %-10s %-13s %-13s %-9s %-9s\n",
            "#","We","Bo","Oh","tc(threshold)","tc(InContact)","cor","pen")
    for (i, s_) in enumerate(sort(sims; by=x -> x.We))
        @printf("%-4d %-10.4g %-10.4g %-10.4g %-13.4f %-13.4f %-9.4f %-9.4f\n",
                i, s_.We, s_.Bo, s_.Oh, s_.tc, s_.tc_incontact, s_.cor, s_.pen)
    end
    panels = [(label="tc / tsigma  (TARGET)", exp=tc_exp, dns=tc_dns, km=tc_km, simval=x -> x.tc),
              (label="coefficient of restitution", exp=cor_exp, dns=cor_dns, km=cor_km, simval=x -> x.cor)]
    write_svg(OUT_SVG, panels, sims)
    @printf("\nwrote %s\n", relpath(OUT_SVG, ROOT))
    println("\nCONTACT TIME residuals vs experiment (median within +-0.15 dex in We):")
    @printf("%-10s %-12s %-12s %-14s %-12s %-s\n","We","tc_sim","tc_exp_med","tc_exp_iqr","sim - med","within unc?")
    for s_ in sort(sims; by=x -> x.We)
        sel = findall(j -> abs(log10(tc_exp.We[j]) - log10(s_.We)) < 0.15, eachindex(tc_exp.We))
        isempty(sel) && continue
        m = median(tc_exp.val[sel]); q = quantile(tc_exp.val[sel], [0.25, 0.75])
        u = median(filter(isfinite, tc_exp.val_unc[sel]))
        @printf("%-10.4g %-12.4f %-12.4f %-14s %+-12.4f %-s\n", s_.We, s_.tc, m,
                @sprintf("%.3f-%.3f", q[1], q[2]), s_.tc - m,
                (isfinite(u) && abs(s_.tc - m) <= u) ? "yes" : "NO")
    end
    exit(0)
end

cache = load_cache()
@printf("cache: %d entries%s\n", length(cache), FORCE ? " (ignored, --force)" : "")

pts = choose_points(tc_exp, NPOINTS)
if ONLY !== nothing
    (1 <= ONLY <= length(pts)) || (println("--only $ONLY out of range 1..$(length(pts))"); exit(1))
    pts = pts[ONLY:ONLY]
    @printf("--only %d: running a single point into %s\n", ONLY, relpath(shard_path(ONLY), ROOT))
end
nnew = count(pt -> !haskey(cache, cache_key(pt.We, pt.Bo, pt.Oh, VISCOUS)), pts)
@printf("\n%d representative points, %d already cached, %d to RUN (~2 min each)\n",
        length(pts), length(pts) - nnew, nnew)
if nnew > 0 && !("--yes" in ARGS)
    println("""
    Refusing to run $nnew new simulations without --yes. Each is a full impact; this is not
    something to trigger casually on a laptop. Re-invoke with --yes, or on CI, when you mean it.
    """)
    exit(0)
end

@printf("\n%-4s %-9s %-9s %-9s %-6s %-8s %-8s %-12s %-12s %-8s %-s\n",
        "#", "We", "Bo", "Oh", "n_exp", "Oh sprd", "Bo sprd", "tc(thresh)", "tc(InCont)", "cor", "source")
flush(stdout)
sims = NamedTuple[]
for (i, pt) in enumerate(pts)
    k = cache_key(pt.We, pt.Bo, pt.Oh, VISCOUS)
    if haskey(cache, k)
        cor, tc, tci, pen = cache[k]; src = "cache"
    else
        r = simulate(pt.We, pt.Bo, pt.Oh)
        if r === nothing
            @printf("%-4d %-10.4g %-10.4g %-10.4g %-6d %-s\n", i, pt.We, pt.Bo, pt.Oh, pt.nexp,
                    "NO REBOUND within t_end"); flush(stdout); continue
        end
        cor, tc, tci, pen = r.cor, r.tc, r.tc_incontact, r.pen
        append_cache(pt.We, pt.Bo, pt.Oh, VISCOUS, cor, tc, tci, pen); src = "ran"
    end
    push!(sims, (We=pt.We, Bo=pt.Bo, Oh=pt.Oh, cor=cor, tc=tc, tc_incontact=tci, pen=pen))
    @printf("%-4d %-9.4g %-9.4g %-9.4g %-6d %-8.2f %-8.2f %-12.4f %-12.4f %-8.4f %-s\n",
            i, pt.We, pt.Bo, pt.Oh, pt.nexp, pt.oh_spread, pt.bo_spread, tc, tci, cor, src)
    flush(stdout)
end
isempty(sims) && (println("\nnothing simulated; no overlay written."); exit(0))
if ONLY !== nothing
    println("\n--only mode: shard written, no overlay. Merge with --merge-shards, then --plot-only.")
    exit(0)
end

panels = [(label="tc / tsigma  (TARGET)", exp=tc_exp, dns=tc_dns, km=tc_km, simval=s -> s.tc),
          (label="coefficient of restitution", exp=cor_exp, dns=cor_dns, km=cor_km, simval=s -> s.cor)]
write_svg(OUT_SVG, panels, sims)
@printf("\nwrote %s\n", relpath(OUT_SVG, ROOT))

println("\nCONTACT TIME residuals vs experiment (median within +-0.15 dex in We):")
@printf("%-10s %-12s %-12s %-14s %-12s %-s\n", "We", "tc_sim", "tc_exp_med", "tc_exp_iqr", "sim - med", "within unc?")
for s in sort(sims; by=x -> x.We)
    sel = findall(j -> abs(log10(tc_exp.We[j]) - log10(s.We)) < 0.15, eachindex(tc_exp.We))
    isempty(sel) && continue
    m = median(tc_exp.val[sel]); q = quantile(tc_exp.val[sel], [0.25, 0.75])
    u = median(filter(isfinite, tc_exp.val_unc[sel]))
    @printf("%-10.4g %-12.4f %-12.4f %-14s %+-12.4f %-s\n", s.We, s.tc, m,
            @sprintf("%.3f-%.3f", q[1], q[2]), s.tc - m,
            (isfinite(u) && abs(s.tc - m) <= u) ? "yes" : "NO")
end

println("""

CAVEATS bounding what this shows:
  * contact time uses AlventosaEtAl2023's DEFINITION, not the InContact duration: their t_c is
    the interval between the droplet top crossing z=2R downward and returning to it (centre of
    mass crossing z=R for their model/DNS), which deliberately includes post-detachment free
    flight because detachment is not observable in their setup. Both are printed above; they are
    different quantities and an earlier revision of this script compared the wrong one, which
    manufactured a 40-70% "error".
  * units need no rescaling: our times are already in tau_cap = sqrt(rho R^3/sigma) = t_sigma.
  * the CoR panel's experimental values come from the same published figure, but note the
    separate raw dataset (rebound_low_weber_raw.csv) reports a `cor` that is NOT Vo/Vi, so
    restitution definitions differ between sources. Contact time has no such ambiguity, which
    is a further reason to treat it as the target.
  * max deflection is not compared: the only candidate experimental column (`ho`) has an
    undocumented meaning and is not used even by the sister repo.
""")
