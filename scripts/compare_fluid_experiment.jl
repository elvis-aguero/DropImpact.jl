# Compare ALL THREE rebound metrics of this model against the BATH EXPERIMENTS of
# AlventosaEtAl2023, at the experiment's own We values, for either working fluid -- and against the
# published quasi-potential (1PKM) and DNS curves on the same axes.
#
#   julia --project=. -t auto scripts/compare_fluid_experiment.jl oil
#   julia --project=. -t auto scripts/compare_fluid_experiment.jl water --M 40 --L 60 --nq 120
#
# METRICS, all three read off the SAME geometric event, as the paper does (its §4.2 defines t_c as
# the interval between the two instants the north pole crosses z = 2R and alpha as minus the ratio
# of the vertical velocities AT THOSE TIMES; for the model and DNS both move to the centre of mass
# crossing z = R):
#
#   t_c    threshold_contact_time                  -- CoM below z = R
#   alpha  threshold_coefficient_of_restitution    -- -v_exit/v_enter at those same two crossings,
#                                                      NOT the phase-based coefficient_of_restitution,
#                                                      which uses the first/last InContact step and,
#                                                      with multi-bounce trajectories, the exit of
#                                                      the LAST bounce rather than the measured one
#   delta  max_penetration_depth                   -- -min(z_cm - xi(theta=0)); independent of any
#                                                      contact-time definition and needs no window,
#                                                      since each rebound is weaker than the last
#
# THE METRIC OFFSET IS NOT NEGLIGIBLE next to the residuals: experiment times the north pole at
# z = 2R, this model the CoM at z = R, and the authors quote a typical 5% difference for water and
# 2% for oil on both t_c and alpha. Residuals below that size are not resolved by this comparison.
#
# PROVENANCE OF THE REFERENCE VALUES: see data/experiments/PROVENANCE.md. In particular the .fig
# files these come from are NOT confirmed to be the published FINAL2 figures.
#
# COST: one simulation per point at production truncation -- minutes at low We, up to an hour at
# We ~ 7. Runs points in parallel across threads; use -t auto. Not a laptop job.

using SpectralKM
using DelimitedFiles
using Printf
using Base.Threads

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data", "experiments")
const R_EXP = 3.5e-4      # AlventosaEtAl2023 table 1: R = 0.35 mm for both fluids

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    v = ARGS[i+1]
    return default isa AbstractString ? v : parse(typeof(default), v)
end

read_num(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => Float64.(d[:, i]) for (i, n) in enumerate(h))
end
read_mixed(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

# Curve 2 in the oil figure and curve 6 in the water figure are the blue solid [0 0.45 0.74] lines
# the captions identify as the quasi-potential model; curve 1 is the black dashed DNS.
const CURVE_ID = Dict(("oil", "pkm") => 2, ("oil", "dns") => 1,
                      ("water", "pkm") => 6, ("water", "dns") => 1)

function ref_curve(which, who, metric)
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

function interp_at(xs, ys, x)
    isempty(xs) && return NaN
    (x < first(xs) || x > last(xs)) && return NaN      # never extrapolate a published curve
    j = searchsortedfirst(xs, x)
    j <= 1 && return ys[1]
    x0, x1 = xs[j-1], xs[j]
    return x1 == x0 ? ys[j-1] : ys[j-1] + (ys[j] - ys[j-1]) * (x - x0) / (x1 - x0)
end

function main()
    which = isempty(ARGS) || startswith(ARGS[1], "--") ? "" : ARGS[1]
    which in ("water", "oil") || error("name the fluid: `water` or `oil`")

    e = read_num(joinpath(DATA, "bath_experiment_$(which).csv"))
    M, L, N, nq = argval("--M", 60), argval("--L", 120), argval("--N", 3), argval("--nq", 200)
    b, h0, tend = argval("--b", 6.0), argval("--h0", 3.0), argval("--t_end", 14.0)
    sel = Symbol(argval("--selector", string(SpectralKM.DEFAULT_SELECTOR)))
    vis = Symbol(argval("--viscous", string(SpectralKM.DEFAULT_VISCOUS)))

    f = which == "water" ? WATER : OIL_5CST
    off = which == "water" ? 5 : 2
    c0 = conditions(drop=f, R=R_EXP, V0=1.0)
    @printf("\n%s (%s): rho=%.0f, sigma=%.4g N/m, nu=%.3g cSt, R=%.3g mm\n",
            uppercase(which), f.name, f.rho, f.sigma, f.nu*1e6, R_EXP*1e3)
    @printf("Bo=%.6f Oh=%.6f t_sigma=%.4f ms   (published Bo=%.3f Oh=%.3f)\n",
            c0.Bo, c0.Oh, t_sigma(f, R_EXP)*1e3, e["Bo"][1], e["Oh"][1])
    @printf("M=%d L=%d N=%d nq=%d b=%g h0=%g selector=%s viscous=%s threads=%d\n",
            M, L, N, nq, b, h0, sel, vis, nthreads())
    (M == 60 && L == 120 && nq == 200) ||
        @printf("NOTE: below production truncation (M=60 L=120 nq=200) -- not converged.\n")
    @printf("metric offset experiment(north pole @2R) vs model(CoM @R): ~%d%% for %s\n\n", off, which)

    n = length(e["We"])
    tc = fill(NaN, n); cor = fill(NaN, n); dl = fill(NaN, n); wall = fill(NaN, n)
    @threads for i in 1:n
        We = e["We"][i]
        V0 = sqrt(We * f.sigma / (f.rho * R_EXP))     # invert We = rho R V0^2 / sigma
        c = conditions(drop=f, R=R_EXP, V0=V0)
        p = Params(c; b=b, h0=h0, M=M, L=L, N=N, nq=nq, selector=sel, viscous=vis)
        t0 = time()
        levels, _, _ = run_simulation(p; t_end=tend)
        wall[i] = time() - t0
        ts = [l.t for l in levels]
        v = threshold_contact_time(ts, levels);                  v !== nothing && (tc[i] = v)
        v = threshold_coefficient_of_restitution(ts, levels);    v !== nothing && (cor[i] = v)
        dl[i] = max_penetration_depth(levels, p.L)
        @printf("  [%2d/%2d] We=%.4f done in %.0f s\n", i, n, We, wall[i]); flush(stdout)
    end

    out = joinpath(DATA, "model_vs_experiment_$(which).csv")
    open(out, "w") do io
        println(io, "fluid,We,metric,model,exp,exp_sd,pkm,dns,n_sd,wall_s")
        for (metric, mine, ex, sd) in (("tc", tc, e["tc_over_tsigma"], e["tc_sd"]),
                                       ("cor", cor, e["cor"], e["cor_sd"]),
                                       ("delta", dl, e["delta_over_R"], e["delta_sd"]))
            xp, yp = ref_curve(which, "pkm", metric)
            xd, yd = ref_curve(which, "dns", metric)
            for i in 1:n
                @printf(io, "%s,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f\n",
                        which, e["We"][i], metric, mine[i], ex[i], sd[i],
                        interp_at(xp, yp, e["We"][i]), interp_at(xd, yd, e["We"][i]),
                        (mine[i] - ex[i]) / sd[i], wall[i])
            end
        end
    end

    for (metric, mine, ex, sd, lbl) in (("tc", tc, e["tc_over_tsigma"], e["tc_sd"], "t_c/t_sigma"),
                                        ("cor", cor, e["cor"], e["cor_sd"], "alpha"),
                                        ("delta", dl, e["delta_over_R"], e["delta_sd"], "delta/R"))
        xp, yp = ref_curve(which, "pkm", metric)
        xd, yd = ref_curve(which, "dns", metric)
        @printf("\n%s\n%-9s %-9s %-9s %-8s %-9s %-8s %-9s %-8s\n", lbl,
                "We", "ours", "exp", "sd", "n_sd", "1PKM", "pkm_nsd", "DNS")
        wins = 0; cmp = 0
        for i in 1:n
            q = interp_at(xp, yp, e["We"][i]); d = interp_at(xd, yd, e["We"][i])
            ns = (mine[i] - ex[i]) / sd[i]; qs = (q - ex[i]) / sd[i]
            @printf("%-9.4f %-9s %-9.4f %-8.4f %-+9.2f %-8.4f %-+9.2f %-8.4f\n",
                    e["We"][i], isnan(mine[i]) ? "none" : @sprintf("%.4f", mine[i]),
                    ex[i], sd[i], ns, q, qs, d)
            if !isnan(mine[i]) && !isnan(q)
                cmp += 1; abs(ns) < abs(qs) && (wins += 1)
            end
        end
        ok = [(mine[i]-ex[i])/sd[i] for i in 1:n if !isnan(mine[i])]
        if !isempty(ok)
            @printf("  %d/%d points ran; within 1 sd %d, within 2 sd %d; mean n_sd %+.2f\n",
                    length(ok), n, count(x->abs(x)<=1, ok), count(x->abs(x)<=2, ok),
                    sum(ok)/length(ok))
            @printf("  closer to experiment than 1PKM at %d of %d comparable points\n", wins, cmp)
        end
    end
    @printf("\nwrote %s\n", out)
    @printf("total wall %.0f s over %d threads\n", sum(filter(!isnan, wall)), nthreads())
end

main()
