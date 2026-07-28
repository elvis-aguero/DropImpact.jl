# Validation against EXPERIMENT only.
#
# The reference is `data/reference/experimental/` -- the measured droplet top and bottom
# trajectories, with error bars, from the experiments that supersede AlventosaEtAl2023.
# See that directory's PROVENANCE.md. Nothing here compares against another model.
#
# Note deliberately what cannot be validated this way: there is NO experimental contact
# radius and NO experimental droplet width in the dataset. Both exist only as DNS and
# 1PKM series, so a contact-radius comparison is model-to-simulation, and additionally
# definition-sensitive (DNS thresholds on the trapped gas film; this model's theta_c is
# the boundary of the kinematically matched region). `validate_trajectory.jl` still
# reports those for information; this script does not treat them as validation.
using SpectralKM, Printf, DelimitedFiles

const EXPDIR = joinpath(@__DIR__, "..", "data", "reference", "experimental")
readexp(f) = sortslices(readdlm(joinpath(EXPDIR, f), ','), dims=1)

"""Linear interpolation of a model series (t, y) onto time `t`."""
function interp(T, Y, t)
    (t < first(T) || t > last(T)) && return nothing
    j = searchsortedfirst(T, t)
    j <= 1 && return Y[1]
    j > length(T) && return Y[end]
    T[j] == T[j-1] && return Y[j]
    w = (t - T[j-1]) / (T[j] - T[j-1])
    return (1 - w) * Y[j-1] + w * Y[j]
end

"""
Measurement uncertainty, estimated format-agnostically.

`topbotom_errorbars_exp.csv` is a digitisation of the plotted error bars, but its layout
is not self-describing: the rows are neither one-per-measurement nor cleanly two-per-bar
(172 rows cover 60 top-series measurements), and the endpoint times do not line up with
the measurement times, so consecutive rows are not the two caps of a single bar. Two
earlier attempts to reconstruct bars by pairing rows gave uncertainties of 0.64 and 0.093
-- the first of order the entire trajectory range, the second inconsistent with the
visibly small bars in the source data.

Rather than guess the layout, take each digitised cap, interpolate the MEASURED series at
that cap's time, and call the absolute deviation the bar half-length. That is exactly what
a cap is, needs no assumption about row ordering, and degrades gracefully if some rows are
centres rather than caps (those simply contribute near-zero).

Reported as a MEDIAN with quartiles, not a mean: about 2% of the caps sit further than
0.3 from the measured curve (max 1.55), which is digitisation debris rather than
uncertainty, and it drags a mean badly. The surviving distribution is tight -- quartiles
0.081 and 0.118 about a median of 0.100 -- so the bar scale is well determined despite
the unreadable layout.
"""
function error_scale(Ttop, Ytop, Tbot, Ybot)
    E = readexp("topbotom_errorbars_exp.csv")
    bars = Float64[]
    for k in axes(E, 1)
        t, z = E[k, 1], E[k, 2]
        T, Y = z > 1.0 ? (Ttop, Ytop) : (Tbot, Ybot)
        ym = interp(T, Y, t)
        ym === nothing && continue
        push!(bars, abs(z - ym))
    end
    sort!(bars)
    n = length(bars)
    med = bars[cld(n, 2)]
    return med, bars[cld(n, 4)], bars[cld(3n, 4)], n
end

function main()
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=60, L=60, N=3, b=6.0, h0=3.0, nq=40)
    levels, diag = run_simulation(p; t_end=8.0, dt_init=1e-3)
    T = [lv.t for lv in levels]
    ztop = [lv.com.z + xi_of_theta(lv.drop.beta, π, p.L) for lv in levels]
    zbot = [lv.com.z - xi_of_theta(lv.drop.beta, 0.0, p.L) for lv in levels]

    Dtop = readexp("top_exp.csv"); Dbot = readexp("bottom_exp.csv")
    mean_bar, q1, q3, nbar = error_scale(Dtop[:, 1], Dtop[:, 2], Dbot[:, 1], Dbot[:, 2])
    @printf("experimental uncertainty from %d digitised bar caps: median %.4f (quartiles %.4f, %.4f)\n\n",
            nbar, mean_bar, q1, q3)

    @printf("%-14s %-5s %-10s %-10s %-10s %-12s\n",
            "series", "n", "mean|err|", "max|err|", "RMS", "RMS/bar")
    total = Float64[]
    for (label, file, Y) in (("droplet top", "top_exp.csv", ztop),
                             ("droplet bottom", "bottom_exp.csv", zbot))
        D = readexp(file)
        errs = Float64[]
        for k in axes(D, 1)
            ym = interp(T, Y, D[k, 1])
            ym === nothing && continue
            push!(errs, ym - D[k, 2])
        end
        rms = sqrt(sum(abs2, errs) / length(errs))
        append!(total, errs)
        @printf("%-14s %-5d %-10.4f %-10.4f %-10.4f %-12.2f\n", label, length(errs),
                sum(abs, errs) / length(errs), maximum(abs, errs), rms, rms / mean_bar)
    end
    rms = sqrt(sum(abs2, total) / length(total))
    @printf("%-14s %-5d %-10.4f %-10.4f %-10.4f %-12.2f\n", "combined", length(total),
            sum(abs, total) / length(total), maximum(abs, total), rms, rms / mean_bar)

    println("\nRMS/bar is the figure of merit: 1 would mean the model sits inside the")
    println("experimental error bars on average. Larger means the discrepancy is real")
    println("physics or model error, not measurement scatter.")
end

main()
