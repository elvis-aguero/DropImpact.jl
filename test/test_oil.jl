# OIL: the second working fluid, and the first test of this model outside water.
#
# 5 cSt silicone oil sits at Bo = 0.0563, Oh = 0.0578 -- a factor 3.4 in Bo and 9.4 in Oh above the
# water case. That is the whole point of running it: water alone cannot distinguish a model that has
# the physics right from one tuned at a single (Bo, Oh).
#
# WHAT WAS MEASURED, at the lowest experimental point (We = 1.215765), production truncation
# M = 60, L = 120, N = 3, nq = 200, b = 6, h0 = 3, t_end = 14:
#
#   selector    closure   tc         vs experiment 4.659815 +/- 0.219633
#   :crossing   :reid     5.080336   +9.0%,  1.91 sd      <- the default since the selector switch
#   :feasible   :reid     5.078683   +9.0%,  1.91 sd      selector: 0.03% -- not the cause
#   :feasible   :lamb     5.091248   +9.3%,  1.96 sd      closure:  0.24% -- not the cause
#
# The selector makes no difference AT THIS We, which is why it is listed here as not the cause. It
# makes all the difference at We = 7.307, where :feasible loses contact after 0.194 and the drop
# sinks; that is why the default changed. See DEFAULT_SELECTOR in src/types.jl.
#
# SO THE OIL RESIDUAL IS NOT EXPLAINED BY EITHER KNOB, and it points the OPPOSITE way from water,
# where the same code at the same truncation gives -3.4% (0.84 sd). Three things this is not:
#
#   * not the contact-edge rule: :crossing, the reference implementation's own rule, moves tc by
#     0.03%, three orders of magnitude short of the residual;
#   * not the drop viscous closure: at Oh = 0.058, where Lamb's damping RATE is wrong by tens of
#     percent at high l, the effect on the contact TIME is 0.24%. Worth recording, because the
#     large damping-rate error at moderate Oh does NOT translate into a large error in this
#     observable, and an argument from the rate alone would have overstated it;
#   * not the metric definition: the authors quote ~2% between the experiment's north-pole-at-2R
#     convention and the model's CoM-at-R convention for the oil (against 5% for water), which is a
#     quarter of the residual.
#
# It also disagrees in SIGN with the reference model, which the paper reports as UNDERpredicting
# tc for the oil at intermediate We. That makes the oil residual a genuine open finding rather than
# a documented offset, and it is asserted here loosely but with its sign, so that a future change
# which moves it must say so.
#
# ONLY ONE We IS PINNED HERE. tc is nearly flat across the experimental oil range (4.66 to 5.26),
# and this is its LOWEST point; whether +9% holds across We is the job of the 12-point sweep in the
# fluid-experiment CI job (`gh workflow run CI -f fluid=oil`), not of a unit test.

using Test
using SpectralKM
using DelimitedFiles

const OILDATA = joinpath(@__DIR__, "..", "data", "experiments")

oread(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => Float64.(d[:, i]) for (i, n) in enumerate(h))
end

@testset "oil: the experimental file is the oil one, and it is a different regime" begin
    e = oread(joinpath(OILDATA, "bath_experiment_oil.csv"))
    w = oread(joinpath(OILDATA, "bath_experiment_water.csv"))
    @test length(e["We"]) == 12
    @test all(e["Bo"] .== 0.056) && all(e["Oh"] .== 0.058)
    # The regime separation that makes oil worth running at all.
    @test e["Bo"][1] / w["Bo"][1] > 3
    @test e["Oh"][1] / w["Oh"][1] > 9
    # Oil reaches much higher We than the water set, and its contact time is flat across it.
    @test maximum(e["We"]) > 7 && maximum(w["We"]) < 2
    @test 4.6 < minimum(e["tc_over_tsigma"]) && maximum(e["tc_over_tsigma"]) < 5.3
    # Still the BATH problem, not the rigid wall: every point is far above the 3.4 ceiling of the
    # solid-substrate band. Guards the same confusion test_bath_reference.jl guards for water.
    @test minimum(e["tc_over_tsigma"]) > 3.4
    # Every point carries a real uncertainty (>= 5 trials each), so "within n sd" means something.
    @test all(e["tc_sd"] .> 0)
end

@testset "oil: preset -> (Bo, Oh) -> Params round-trips to the published values" begin
    c = conditions(drop=:oil, R=3.5e-4, V0=0.2724)
    @test round(c.Bo, digits=3) == 0.056
    @test round(c.Oh, digits=3) == 0.058
    p = Params(c; b=6.0, h0=3.0, M=20, L=20, N=3, nq=60)
    @test p.Bo == c.Bo && p.Oh == c.Oh
end

# One simulation. At production truncation this is ~3.5 minutes, which is why it is one point and
# not twelve.
@testset "oil: contact time at the lowest experimental We" begin
    e = oread(joinpath(OILDATA, "bath_experiment_oil.csv"))
    i = 1
    We, tce, sd = e["We"][i], e["tc_over_tsigma"][i], e["tc_sd"][i]
    @test We ≈ 1.215765 atol = 1e-6
    @test tce ≈ 4.659815 atol = 1e-6

    f = OIL_5CST
    R = 3.5e-4
    V0 = sqrt(We * f.sigma / (f.rho * R))          # invert We = rho R V0^2 / sigma
    @test 0.27 < V0 < 0.28                         # 27 cm/s, inside the tabulated 20-100 cm/s
    c = conditions(drop=f, R=R, V0=V0)
    @test c.We ≈ We rtol = 1e-12                   # the inversion is exact

    p = Params(c; b=6.0, h0=3.0)
    levels, _, _ = run_simulation(p; t_end=14.0)
    tct = threshold_contact_time([l.t for l in levels], levels)
    @test tct !== nothing

    @info "oil: model vs experiment" We model=tct experiment=tce sd n_sd=(tct - tce)/sd

    # In the BATH band, not the rigid-wall one -- the assertion that would have caught the
    # wrong-problem comparison immediately.
    @test tct > 3.4

    # Measured 5.080336 under the default :crossing (5.078683 under :feasible -- 0.03% apart).
    # Pinned to 1% so a real change in the closure has to declare itself, while step-controller
    # noise does not.
    @test tct ≈ 5.080336 rtol = 0.01

    # AGAINST EXPERIMENT. Deliberately asserted as OVERPREDICTING by 5-15%: that sign is the open
    # finding, it is the opposite of the water case (-3.4%) and of the reference model's reported
    # direction for oil, and it must not be allowed to change silently.
    @test tct > tce
    @test 0.05 < (tct - tce) / tce < 0.15
    @test 1.0 < (tct - tce) / sd < 3.0             # measured 1.91 sd -- outside 1 sd, inside 2
end
