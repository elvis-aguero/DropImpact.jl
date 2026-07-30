# Compare this model's contact time against the BATH EXPERIMENTS of AlventosaEtAl2023, for either
# working fluid, at the experiment's own We values.
#
#   julia --project=. scripts/compare_fluid_experiment.jl oil
#   julia --project=. scripts/compare_fluid_experiment.jl water --M 40 --L 60 --nq 120
#
# WHAT THIS COMPARES AGAINST. data/experiments/bath_experiment_{water,oil}.csv, read out of the
# authors' own .fig files by scripts/extract_bath_experiment.m -- measured points with a standard
# deviation over at least 5 trials each. NOT lowWeberComparison.csv, which is model output with no
# experiment column, and NOT any of the SOLID_SUBSTRATE_* files, which are a different physical
# problem entirely (rigid wall, tc = 1.96..3.36, disjoint from the bath's 4.44..8.09).
#
# THE METRIC DIFFERS BETWEEN THE TWO SIDES, and it is not a small effect next to the residuals
# below. The experiment times the NORTH POLE crossing z = 2R, because the detachment instant was
# not optically resolvable; threshold_contact_time implements the paper's MODEL convention, the
# CENTRE OF MASS crossing z = R. The authors quote a typical difference between the conventions of
# 5% for the water experiments and 2% for the oil. Any residual below that size is not resolved by
# this comparison.
#
# EACH POINT IS ONE SIMULATION AT PRODUCTION TRUNCATION, roughly 3.5 minutes, so the full 12-point
# oil set is about 40 minutes. Run it in CI, not on a laptop. --M/--L/--N/--nq are provided for
# reducing that at a cost in resolution, and the reduction is printed so a cheap run cannot be
# mistaken for a converged one.

using SpectralKM
using DelimitedFiles
using Printf

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data", "experiments")
const R_EXP = 3.5e-4      # AlventosaEtAl2023 table 1: R = 0.35 mm for both fluids

# `parse(String, s)` does not exist, so a String-valued flag has to be returned verbatim rather
# than parsed. Getting this wrong made --selector/--viscous throw a MethodError on the first CI
# dispatch, before a single simulation had run.
function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    v = ARGS[i+1]
    return default isa AbstractString ? v : parse(typeof(default), v)
end

read_csv(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => Float64.(d[:, i]) for (i, n) in enumerate(h))
end

function main()
    which = isempty(ARGS) || startswith(ARGS[1], "--") ? "" : ARGS[1]
    which in ("water", "oil") ||
        error("name the fluid: `water` or `oil`. Got $(isempty(which) ? "nothing" : which).")

    csv = joinpath(DATA, "bath_experiment_$(which).csv")
    isfile(csv) || error("missing $csv -- regenerate with scripts/extract_bath_experiment.m")
    e = read_csv(csv)

    M  = argval("--M", 60)
    L  = argval("--L", 120)
    N  = argval("--N", 3)
    nq = argval("--nq", 200)
    b  = argval("--b", 6.0)
    h0 = argval("--h0", 3.0)
    tend = argval("--t_end", 14.0)
    sel = Symbol(argval("--selector", "feasible"))
    vis = Symbol(argval("--viscous", "reid"))

    f = which == "water" ? WATER : OIL_5CST
    # The dimensional route, so the groups are derived once from the fluid properties rather than
    # retyped per point: We fixes V0, and Bo/Oh follow from (rho, sigma, nu, R) alone.
    metric_pct = which == "water" ? 5 : 2

    @printf("\n%s, %s (rho = %.0f kg/m^3, sigma = %.4g N/m, nu = %.4g cSt), R = %.3g mm\n",
            uppercase(which), f.name, f.rho, f.sigma, f.nu * 1e6, R_EXP * 1e3)
    c0 = conditions(drop=f, R=R_EXP, V0=1.0)
    @printf("Bo = %.6f, Oh = %.6f, t_sigma = %.4f ms   (published: Bo = %.3f, Oh = %.3f)\n",
            c0.Bo, c0.Oh, t_sigma(f, R_EXP) * 1e3, e["Bo"][1], e["Oh"][1])
    @printf("truncation M = %d, L = %d, N = %d, nq = %d, b = %g, h0 = %g, selector = %s, viscous = %s\n",
            M, L, N, nq, b, h0, sel, vis)
    (M == 60 && L == 120 && nq == 200) ||
        @printf("NOTE: reduced from the production truncation (M=60, L=120, nq=200) -- not a converged run.\n")
    @printf("metric offset: experiment times the north pole at z = 2R, model the CoM at z = R;\n")
    @printf("               the authors quote ~%d%% between the conventions for %s.\n\n", metric_pct, which)

    @printf("%-4s %-10s %-9s %-9s %-9s %-9s %-8s\n",
            "#", "We", "tc_model", "tc_exp", "sd_exp", "resid_%", "n_sd")
    rows = Any[]
    for (i, We) in enumerate(e["We"])
        V0 = sqrt(We * f.sigma / (f.rho * R_EXP))       # invert We = rho R V0^2 / sigma
        c = conditions(drop=f, R=R_EXP, V0=V0)
        p = Params(c; b=b, h0=h0, M=M, L=L, N=N, nq=nq, selector=sel, viscous=vis)
        levels, _, _ = run_simulation(p; t_end=tend)
        tct = threshold_contact_time([l.t for l in levels], levels)
        tce, sd = e["tc_over_tsigma"][i], e["tc_sd"][i]
        if tct === nothing
            @printf("%-4d %-10.4f %-9s %-9.4f %-9.4f %-9s %-8s\n", i, We, "none", tce, sd, "-", "-")
            push!(rows, (We, NaN, tce, sd, NaN, NaN))
        else
            rel = 100 * (tct - tce) / tce
            nsd = (tct - tce) / sd
            @printf("%-4d %-10.4f %-9.4f %-9.4f %-9.4f %+-9.2f %+-8.2f\n",
                    i, We, tct, tce, sd, rel, nsd)
            push!(rows, (We, tct, tce, sd, rel, nsd))
        end
        flush(stdout)
    end

    ok = [r for r in rows if !isnan(r[2])]
    if !isempty(ok)
        rels = [r[5] for r in ok]
        nsds = [abs(r[6]) for r in ok]
        @printf("\n%d/%d points completed. residual: mean %+.2f%%, range %+.2f%% .. %+.2f%%\n",
                length(ok), length(rows), sum(rels) / length(rels), minimum(rels), maximum(rels))
        @printf("within 1 sd: %d/%d;  within 2 sd: %d/%d;  max |n_sd| = %.2f\n",
                count(<=(1.0), nsds), length(nsds), count(<=(2.0), nsds), length(nsds),
                maximum(nsds))
        @printf("for scale, the metric-definition offset alone is ~%d%% for %s.\n", metric_pct, which)
    end

    out = joinpath(DATA, "model_vs_experiment_$(which).csv")
    open(out, "w") do io
        println(io, "We,tc_model,tc_exp,tc_sd,residual_pct,n_sd")
        for r in rows
            @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", r...)
        end
    end
    @printf("\nwrote %s\n", out)
end

main()
