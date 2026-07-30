using Test
using SpectralKM
using DelimitedFiles

# Guard the problem identity itself.
#
# A spurious factor-of-two contact-time "overprediction" was chased as a model defect for
# several rounds. Its actual cause was that the comparison dataset is the DROPLET-ONTO-SOLID-
# SUBSTRATE problem, not the bath problem this package models. The evidence is laid out in
# scripts/convert_experiments.py; what belongs in the testbed is the falsifiable consequence:
#
#   the two problems occupy DISJOINT contact-time bands, so the datasets can never be mixed
#   and no parameter matching reconciles them.
#
# On a rigid wall the drop rebounds on its own l = 2 free oscillation. A bath deforms and
# carries impact energy away as interfacial waves, so the drop rides down and back up and
# contact lasts substantially longer. The bands below are measured from the shipped data.

const DATA = joinpath(@__DIR__, "..", "data", "experiments")

read_csv(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

# 64 of the bath reference's 153 rows carry NaN contact time -- runs where the metric was not
# measured. Filter rather than propagate, but assert the surviving count so a future data swap
# that silently drops rows cannot pass unnoticed.
finite_only(v) = filter(isfinite, Float64.([x for x in v if isa(x, Number)]))

@testset "bath vs solid-substrate problem identity" begin
    bath = read_csv(joinpath(DATA, "bath_km_contact_time.csv"))
    tc_bath = finite_only(bath["contact_time"])

    # The bath band, from lowWeberComparison.csv (km-dropplet-onto-bath, 153 runs at N = 60,
    # of which 89 carry a measured contact time).
    @test length(bath["contact_time"]) == 153
    @test length(tc_bath) == 89
    @test minimum(tc_bath) > 4.4          # measured 4.4408
    @test maximum(tc_bath) < 8.1          # measured 8.0946

    solid = read_csv(joinpath(DATA, "SOLID_SUBSTRATE_contact_time_vs_we.csv"))
    src = string.(solid["source"])
    tc_solid = solid["tc_over_tsigma"]

    # The rigid-wall model/DNS band. Anchored physically by the drop's l = 2 free oscillation:
    # omega_2 = sqrt(l(l-1)(l+2)) = sqrt(8), full period 2*pi/sqrt(8) = 2.2214 t_sigma, and the
    # classical superhydrophobic-rebound value ~2.6 t_sigma (Richard, Clanet & Quere 2002).
    tc_dns = finite_only(tc_solid[src .== "dns"])
    @test !isempty(tc_dns)
    @test maximum(tc_dns) < 3.4           # measured 3.36
    @test 2.2 < 2π / sqrt(8) < 2.3        # the l = 2 period this band brackets

    # THE POINT: disjoint. Nothing in the bath set is as short as the longest rigid-wall run.
    @test minimum(tc_bath) > maximum(tc_dns)

    # The solid-substrate km_model column is the full KM on a RIGID WALL, not the 1PKM bath
    # model: its Bo is a single fixed value (hardcoded Bo = 0.0189 in that repo's sweeper) and
    # it reaches down to the Bo = Oh = 0 inviscid, zero-gravity limit.
    km = src .== "km_model"
    @test minimum(Float64.(solid["Oh"])[km]) == 0.0
    @test maximum(Float64.(solid["Bo"])[km]) ≈ 0.0189 atol = 1e-4

    # The bath paper has exactly TWO experimental (Bo, Oh) points (water, oil); this file's
    # experiment block is a broad continuum, which is how it fails to be that paper's data.
    exp_ = src .== "experiment"
    @test length(unique(round.(Float64.(solid["Oh"])[exp_], digits=3))) > 10
end

@testset "bath reference contains the water point" begin
    bath = read_csv(joinpath(DATA, "bath_km_contact_time.csv"))
    bo, oh = Float64.(bath["Bo"]), Float64.(bath["Oh"])
    tc = Float64.(bath["contact_time"])
    k = findfirst(i -> abs(oh[i] - 0.006165) < 1e-5 && abs(bo[i] - 0.016644) < 1e-5, eachindex(oh))
    @test k !== nothing
    # Independently reproduced by running the reference MATLAB directly: 4.5297.
    @test tc[k] ≈ 4.5286 atol = 1e-3
end

# The live check: this model, at the bath reference's own water point, on the SAME metric
# (threshold_contact_time = time the centre of mass spends below z = R). One simulation.
# Deliberately loose at 5%: the reference runs b = 25R, M = 151, L = 55 and a poly6 pressure
# shape against our b = 6, M = 60, L = 120 shifted-Legendre patch, so exact agreement is not
# the claim -- being in the BATH band rather than the rigid-wall one is.
@testset "this model lands in the bath band, not the rigid-wall band" begin
    p = Params(We=0.7, Bo=0.016644, Oh=0.006165, b=6.0, h0=3.0)
    levels, _, _ = run_simulation(p; t_end=14.0)
    ts = [l.t for l in levels]
    tct = threshold_contact_time(ts, levels)
    @test tct !== nothing
    @test tct > 3.4                       # above the entire rigid-wall band
    @test tct ≈ 4.5286 rtol = 0.05        # within 5% of the bath reference
end
