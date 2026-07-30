# Three-panel scatter comparison for one working fluid: alpha, t_c/t_sigma, delta/R against We,
# with the measured points and their error bars, this model, and the published 1PKM and DNS curves.
#
#   julia --project=. scripts/plot_fluid_comparison.jl oil
#   julia --project=. scripts/plot_fluid_comparison.jl water oil        # both, one file each
#
# Reads data/experiments/model_vs_experiment_<fluid>.csv, written by compare_fluid_experiment.jl,
# and the published curves from bath_model_curves_<fluid>.csv. Writes
# data/experiments/comparison_<fluid>.svg.
#
# Dependency-free by construction: the SVG is emitted directly, so the package never acquires a
# plotting stack. Linear We axis, because unlike the earlier solid-substrate comparison these
# points span less than a decade and a log axis only crowds them.

using DelimitedFiles
using Printf

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data", "experiments")

read_mixed(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

const CURVE_ID = Dict(("oil", "pkm") => 2, ("oil", "dns") => 1,
                      ("water", "pkm") => 6, ("water", "dns") => 1)

function curve(which, who, metric)
    path = joinpath(DATA, "bath_model_curves_$(which).csv")
    isfile(path) || return (Float64[], Float64[])
    m = read_mixed(path)
    id = CURVE_ID[(which, who)]
    sel = findall(i -> m["metric"][i] == metric && round(Int, m["curve"][i]) == id,
                  eachindex(m["curve"]))
    xs, ys = Float64.(m["We"][sel]), Float64.(m["value"][sel])
    o = sortperm(xs)
    return (xs[o], ys[o])
end

const PANELS = [("cor", "α", "coefficient of restitution"),
                ("tc", "t_c / t_σ", "contact time"),
                ("delta", "δ / R", "maximum penetration depth")]

const C_EXP, C_OURS, C_PKM, C_DNS = "#111111", "#c0392b", "#2e75b6", "#7f7f7f"

fmt(x) = @sprintf("%.2f", x)

function plot_fluid(which)
    csv = joinpath(DATA, "model_vs_experiment_$(which).csv")
    isfile(csv) || (@warn "missing $csv -- run compare_fluid_experiment.jl $which first"; return)
    d = read_mixed(csv)
    metric = string.(d["metric"])

    W, ml, mr, mt, gap, phh, mb = 940, 92, 176, 62, 60, 190, 66
    H = mt + 3 * phh + 2 * gap + mb
    pw = W - ml - mr

    weall = Float64.(d["We"])
    x0, x1 = minimum(weall), maximum(weall)
    pad = 0.06 * (x1 - x0)
    x0 -= pad; x1 += pad
    px(x) = ml + (x - x0) / (x1 - x0) * pw

    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="Helvetica,Arial,sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$ml" y="28" font-size="16" font-weight="bold">SpectralKM vs Alventosa et al. (2023) — $(which)</text>""")
    ohbo = which == "water" ? "Bo = 0.017, Oh = 0.006" : "Bo = 0.056, Oh = 0.058"
    println(io, """<text x="$ml" y="48" font-size="12" fill="#555">$ohbo, R = 0.35 mm. Error bars are the standard deviation over at least 5 trials. Model at default truncation (M=60, L=120, N=3, nq=200).</text>""")

    for (k, (met, ylab, title)) in enumerate(PANELS)
        top = mt + (k - 1) * (phh + gap)
        idx = findall(==(met), metric)
        we = Float64.(d["We"][idx])
        ours = Float64.(d["model"][idx])
        ex = Float64.(d["exp"][idx])
        sd = Float64.(d["exp_sd"][idx])
        xp, yp = curve(which, "pkm", met)
        xd, yd = curve(which, "dns", met)

        inrange(xs, ys) = [(xs[i], ys[i]) for i in eachindex(xs) if x0 <= xs[i] <= x1]
        vals = filter(isfinite, vcat(ours, ex .+ sd, ex .- sd,
                                     [v for (_, v) in inrange(xp, yp)],
                                     [v for (_, v) in inrange(xd, yd)]))
        y0, y1 = minimum(vals), maximum(vals)
        m = 0.12 * (y1 - y0 + eps())
        y0 -= m; y1 += m
        py(y) = top + (1 - (y - y0) / (y1 - y0)) * phh

        println(io, """<text x="$ml" y="$(top-10)" font-size="13" font-weight="bold">($(('a'+k-1))) $title</text>""")
        println(io, """<rect x="$ml" y="$top" width="$pw" height="$phh" fill="none" stroke="#333"/>""")
        println(io, """<text x="$(ml-64)" y="$(top+phh/2)" font-size="13" text-anchor="middle" transform="rotate(-90 $(ml-64) $(top+phh/2))">$ylab</text>""")

        # x grid + ticks
        for xv in ceil(x0):1.0:floor(x1)
            x = px(xv)
            println(io, """<line x1="$x" y1="$top" x2="$x" y2="$(top+phh)" stroke="#eee"/>""")
            println(io, """<text x="$x" y="$(top+phh+20)" font-size="12" text-anchor="middle">$(Int(xv))</text>""")
        end
        println(io, """<text x="$(ml+pw/2)" y="$(top+phh+44)" font-size="13" text-anchor="middle">We</text>""")
        # y grid
        for i in 0:4
            yv = y0 + i * (y1 - y0) / 4
            y = py(yv)
            println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y" stroke="#eee"/>""")
            println(io, """<text x="$(ml-10)" y="$(y+4)" font-size="11" text-anchor="end">$(fmt(yv))</text>""")
        end

        polyline(pts, col, dash) = length(pts) >= 2 && println(io,
            """<polyline points="$(join(["$(px(a)),$(py(b))" for (a,b) in pts], " "))" fill="none" stroke="$col" stroke-width="2.1"$dash/>""")
        polyline(inrange(xd, yd), C_DNS, """ stroke-dasharray="7,5" """)
        polyline(inrange(xp, yp), C_PKM, "")

        # experiment: error bar then marker
        for i in eachindex(we)
            x = px(we[i])
            if isfinite(sd[i]) && sd[i] > 0
                println(io, """<line x1="$x" y1="$(py(ex[i]-sd[i]))" x2="$x" y2="$(py(ex[i]+sd[i]))" stroke="$C_EXP" stroke-width="1.3"/>""")
                for s in (-1, 1)
                    println(io, """<line x1="$(x-4)" y1="$(py(ex[i]+s*sd[i]))" x2="$(x+4)" y2="$(py(ex[i]+s*sd[i]))" stroke="$C_EXP" stroke-width="1.3"/>""")
                end
            end
            println(io, """<circle cx="$x" cy="$(py(ex[i]))" r="3.4" fill="$C_EXP"/>""")
        end
        # this model
        for i in eachindex(we)
            isfinite(ours[i]) || continue
            println(io, """<circle cx="$(px(we[i]))" cy="$(py(ours[i]))" r="5" fill="$C_OURS" stroke="white" stroke-width="1.4"/>""")
        end
        # points where the model produced nothing: mark on the axis, never silently omitted
        for i in eachindex(we)
            isfinite(ours[i]) && continue
            x = px(we[i])
            println(io, """<text x="$x" y="$(top+phh-6)" font-size="15" fill="$C_OURS" text-anchor="middle">✗</text>""")
        end
    end

    lx = ml + pw + 26
    ly = mt + 6
    for (col, lab, kind) in ((C_EXP, "experiment ±1 sd", :pt), (C_OURS, "SpectralKM", :pt),
                             (C_PKM, "1PKM (quasi-potential)", :line), (C_DNS, "DNS", :dash))
        if kind === :pt
            println(io, """<circle cx="$(lx+9)" cy="$ly" r="4.6" fill="$col"/>""")
        elseif kind === :line
            println(io, """<line x1="$lx" y1="$ly" x2="$(lx+18)" y2="$ly" stroke="$col" stroke-width="2.4"/>""")
        else
            println(io, """<line x1="$lx" y1="$ly" x2="$(lx+18)" y2="$ly" stroke="$col" stroke-width="2.4" stroke-dasharray="7,5"/>""")
        end
        println(io, """<text x="$(lx+26)" y="$(ly+4)" font-size="12">$lab</text>""")
        ly += 24
    end
    ly += 8
    println(io, """<text x="$lx" y="$(ly+4)" font-size="11" fill="#777">✗ = no contact time</text>""")
    println(io, """<text x="$lx" y="$(ly+20)" font-size="11" fill="#777">metric offset: experiment</text>""")
    println(io, """<text x="$lx" y="$(ly+34)" font-size="11" fill="#777">times the north pole at</text>""")
    println(io, """<text x="$lx" y="$(ly+48)" font-size="11" fill="#777">z=2R, the model the CoM</text>""")
    println(io, """<text x="$lx" y="$(ly+62)" font-size="11" fill="#777">at z=R: ~$(which=="water" ? 5 : 2)% on α and t_c.</text>""")
    println(io, """<text x="$lx" y="$(ly+84)" font-size="11" fill="#777">Provenance caveat:</text>""")
    println(io, """<text x="$lx" y="$(ly+98)" font-size="11" fill="#777">data/experiments/</text>""")
    println(io, """<text x="$lx" y="$(ly+112)" font-size="11" fill="#777">PROVENANCE.md</text>""")
    println(io, "</svg>")

    out = joinpath(DATA, "comparison_$(which).svg")
    write(out, String(take!(io)))
    println("wrote $out")
end

fluids = filter(a -> !startswith(a, "--"), ARGS)
isempty(fluids) && (fluids = ["water", "oil"])
foreach(plot_fluid, fluids)
