# THE MEASURED SWEEP, PINNED. All 17 experimental points of AlventosaEtAl2023, both working fluids,
# all three rebound metrics, at the default truncation (M=60, L=120, N=3, nq=200, b=6, h0=3), run on
# a 12-core machine in 507 s (oil) and 198 s (water). Produced by
#
#     julia --project=. -t auto scripts/compare_fluid_experiment.jl {oil,water}
#
# and stored in data/experiments/model_vs_experiment_{oil,water}.csv.
#
# These tests do NOT re-run the sweep -- they assert the CONCLUSIONS drawn from it, against the
# stored results. That is the point: the numbers in the README and in derivations/ are quoted from
# this file, so if the stored results are ever regenerated and a conclusion no longer holds, the
# suite fails instead of the prose quietly going stale.
#
# WHY RELATIVE ERROR AND NOT SIGMA. Both are reported by the sweep, but the headline claims are in
# relative error, because the experimental sd is 5-trial repeatability, not accuracy: at oil
# We = 5.1431 the alpha sd is 2.5% and at water We = 1.9387 the t_c sd is 1.4%, so a model can sit
# 3% away and read as 17 sd. The documented metric-definition offset alone -- experiment times the
# north pole at z = 2R, the model the centre of mass at z = R, ~5% for water and ~2% for oil on both
# t_c and alpha -- is larger than several of those sd. Sigma still appears below where the claim is
# about agreement WITHIN the measurement.

using Test
using DelimitedFiles

const SWEEP = joinpath(@__DIR__, "..", "data", "experiments")

sweep_read(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

function sweep_metric(fluid, metric)
    d = sweep_read(joinpath(SWEEP, "model_vs_experiment_$(fluid).csv"))
    sel = findall(==(metric), string.(d["metric"]))
    return (We = Float64.(d["We"][sel]), model = Float64.(d["model"][sel]),
            exp = Float64.(d["exp"][sel]), sd = Float64.(d["exp_sd"][sel]),
            pkm = Float64.(d["pkm"][sel]), dns = Float64.(d["dns"][sel]))
end

relerr(a, b) = @. 100 * (a - b) / b
meanabs(v) = let f = filter(isfinite, v); sum(abs, f) / length(f) end
meansigned(v) = let f = filter(isfinite, v); sum(f) / length(f) end

@testset "the sweep covered every experimental point" begin
    for (fluid, n) in (("oil", 12), ("water", 5))
        for m in ("tc", "cor", "delta")
            g = sweep_metric(fluid, m)
            @test length(g.We) == n
            @test all(isfinite, g.model)      # no point failed to produce a value
            @test all(g.sd .> 0)              # every point carries a real uncertainty
        end
    end
end

@testset "HEADLINE: oil contact time, 2.7% and unbiased, better than 1PKM and than DNS" begin
    g = sweep_metric("oil", "tc")
    r_ours, r_pkm, r_dns = relerr(g.model, g.exp), relerr(g.pkm, g.exp), relerr(g.dns, g.exp)

    # Measured: 2.7% mean absolute, -0.2% signed. The bias bound is the strong statement -- this is
    # not a model that happens to average out, it tracks the measurement across the whole range.
    @test meanabs(r_ours) < 3.5
    @test abs(meansigned(r_ours)) < 1.5

    # Better than the published quasi-potential model (11.0%) by a factor of about four...
    @test meanabs(r_ours) < 0.5 * meanabs(r_pkm)
    # ...and better than the fully resolved DNS (3.3%), which is the part worth not losing.
    @test meanabs(r_ours) < meanabs(r_dns)

    # At EVERY point, not on average. 1PKM decays monotonically across the range while the measured
    # contact time stays flat; this does not.
    @test all(abs.(r_ours) .< abs.(r_pkm))
    @test count(i -> abs(g.model[i] - g.exp[i]) <= g.sd[i], eachindex(g.We)) >= 10   # 10/12 within 1 sd
    @test all(abs.(g.model .- g.exp) .<= 2 .* g.sd)                                  # 12/12 within 2 sd

    # The flatness itself: ours spans 5.02..5.08 over We = 1.22..7.31, against a measured 4.66..5.26.
    @test maximum(g.model) - minimum(g.model) < 0.12
    @test 1.2 < minimum(g.We) < 1.3 && 7.3 < maximum(g.We) < 7.4
end

@testset "OPEN: alpha is sound at low Oh and reversed at high Oh" begin
    w, o = sweep_metric("water", "cor"), sweep_metric("oil", "cor")
    rw, ro = relerr(w.model, w.exp), relerr(o.model, o.exp)

    # Water, Oh = 0.006: 4.9% mean, and correctly DECREASING with We -- the machinery is sound.
    @test meanabs(rw) < 7
    @test w.model[end] < w.model[1]
    @test w.exp[end] < w.exp[1]

    # Oil, Oh = 0.058 (9.4x): 36.8% mean, and the TREND IS BACKWARDS. The paper singles out the
    # rising alpha as the distinctive oil behaviour; 1PKM reproduces it at 5.3% and this does not.
    @test 30 < meanabs(ro) < 45
    @test o.exp[end] > o.exp[1]        # measured alpha RISES across the oil range
    @test o.model[end] < o.model[1]    # ours FALLS -- the defect, asserted so it cannot vanish quietly
    @test meanabs(ro) > 4 * meanabs(relerr(o.pkm, o.exp))

    # Not a definitional error: the same code is within 5% in water. Whatever this is, it scales
    # with Oh. Recorded as the sharpest open question this sweep produced.
    @test meanabs(rw) < 0.25 * meanabs(ro)
end

@testset "OPEN: delta under-predicts, and it worsens with Oh alongside alpha" begin
    w, o = sweep_metric("water", "delta"), sweep_metric("oil", "delta")
    rw, ro = relerr(w.model, w.exp), relerr(o.model, o.exp)

    # Both under-predict -- every point, both fluids, one sign.
    @test all(r -> r < 0, filter(isfinite, rw))
    @test all(r -> r < 0, filter(isfinite, ro))

    # Water 8.9%, and BETTER than 1PKM's 11.9% at 4 of 5 points.
    @test meanabs(rw) < 11
    @test meanabs(rw) < meanabs(relerr(w.pkm, w.exp))

    # Oil 25.4%, about twice 1PKM's 12.3%. Same direction as the alpha failure and the same Oh
    # dependence: both AMPLITUDE metrics degrade at high Oh while the contact TIMESCALE does not.
    @test 20 < meanabs(ro) < 30
    @test meanabs(ro) > 1.5 * meanabs(relerr(o.pkm, o.exp))
    @test meanabs(rw) < meanabs(ro)
end

@testset "the scoreboard, stated so it cannot drift in the prose" begin
    ours = pkm = 0
    for fluid in ("oil", "water"), m in ("tc", "cor", "delta")
        g = sweep_metric(fluid, m)
        for i in eachindex(g.We)
            (isfinite(g.pkm[i]) && isfinite(g.model[i])) || continue
            if abs(g.model[i] - g.exp[i]) < abs(g.pkm[i] - g.exp[i])
                ours += 1
            else
                pkm += 1
            end
        end
    end
    # 16 to 35 over 51 comparisons. This model does NOT supersede 1PKM overall; it wins the
    # contact-time comparison in oil outright and loses both energy metrics there.
    @test ours + pkm == 51
    @test ours == 16
    @test pkm == 35
end
