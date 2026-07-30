# FLUID API TESTS. The point of src/fluids.jl is to remove a hand transcription from between the
# bench and the solver, so these tests check the two things that could make it worse than the
# transcription it replaces: (i) that the presets reproduce the (Bo, Oh) actually printed on the
# published figures and recorded in data/experiments/, and (ii) that the group definitions obey
# their own dimensional scalings, which is what catches a factor dropped from eq:nondim.

using Test
using SpectralKM
using SpectralKM: Fluid, WATER, OIL_5CST, fluid, conditions, dynamic_viscosity, t_sigma,
    STANDARD_GRAVITY, ImpactConditions
using DelimitedFiles

const FDATA = joinpath(@__DIR__, "..", "data", "experiments")
fread(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

# AlventosaEtAl2023 table 1: R = 0.35 mm for both working fluids.
const R_EXP = 3.5e-4

@testset "presets reproduce the published (Bo, Oh)" begin
    # THE CENTRAL ASSERTION. If the presets, the definitions in eq:nondim, and the experimental
    # CSVs do not agree, one of the three is wrong and every fluid comparison is built on it.
    for (name, preset, csv) in (("water", :water, "bath_experiment_water.csv"),
                                ("oil",   :oil,   "bath_experiment_oil.csv"))
        c = conditions(drop=preset, R=R_EXP, V0=0.5)
        e = fread(joinpath(FDATA, csv))
        Bo_csv, Oh_csv = Float64(e["Bo"][1]), Float64(e["Oh"][1])
        # The CSVs carry the figure captions' printed values, 2-3 significant figures, so compare
        # at the rounding they were printed to rather than pretending to more precision.
        @test round(c.Bo, digits=3) == Bo_csv
        @test round(c.Oh, digits=3) == Oh_csv
        @info "preset vs published" fluid=name Bo=c.Bo Bo_published=Bo_csv Oh=c.Oh Oh_published=Oh_csv
    end
end

@testset "presets reproduce the reference sweep's own (Bo, Oh) to 1%" begin
    # The bath reference file carries a water row at higher precision than the figure captions:
    # Bo = 0.016644, Oh = 0.006165. Our table-1-derived values are 0.016611 and 0.006146. They must
    # agree to a fraction of a percent -- the residual is the authors' rounding of R and sigma, and
    # anything larger would mean a different definition, not a different rounding.
    c = conditions(drop=:water, R=R_EXP, V0=0.5)
    @test isapprox(c.Bo, 0.016644; rtol=0.01)
    @test isapprox(c.Oh, 0.006165; rtol=0.01)
end

@testset "the groups obey their dimensional scalings" begin
    # eq:nondim: We ~ V0^2 R, Bo ~ R^2, Oh ~ R^{-1/2}. These catch an exponent slip that a
    # single-point check against one published number cannot.
    a = conditions(drop=:water, R=R_EXP, V0=0.4)
    b = conditions(drop=:water, R=R_EXP, V0=0.8)
    @test isapprox(b.We / a.We, 4.0; rtol=1e-12)          # doubling V0 quadruples We
    @test isapprox(b.Bo, a.Bo; rtol=1e-12)                # ...and leaves Bo alone
    @test isapprox(b.Oh, a.Oh; rtol=1e-12)                # ...and Oh
    d = conditions(drop=:water, R=2 * R_EXP, V0=0.4)
    @test isapprox(d.We / a.We, 2.0; rtol=1e-12)          # We ~ R
    @test isapprox(d.Bo / a.Bo, 4.0; rtol=1e-12)          # Bo ~ R^2
    @test isapprox(d.Oh / a.Oh, 1 / sqrt(2); rtol=1e-12)  # Oh ~ 1/sqrt(R)
    z = conditions(drop=:water, R=R_EXP, V0=0.4, g=0.0)
    @test z.Bo == 0.0                                     # Bo is the ONLY group carrying g
    @test isapprox(z.We, a.We; rtol=1e-12)
    @test isapprox(z.Oh, a.Oh; rtol=1e-12)
end

@testset "Bo = g t_sigma^2 / R is the same number as rho g R^2 / sigma" begin
    # The paper writes Bo both ways. They are algebraically identical, so agreement here is a check
    # that t_sigma is the inertio-capillary time it claims to be, which the contact-time
    # nondimensionalisation depends on entirely.
    for f in (WATER, OIL_5CST), R in (2e-4, R_EXP, 6e-4)
        c = conditions(drop=f, R=R, V0=0.5)
        @test isapprox(c.Bo, STANDARD_GRAVITY * t_sigma(f, R)^2 / R; rtol=1e-12)
    end
    # And t_sigma must land where a millimetric drop actually lives. Measured: 0.7698 ms for water
    # at R = 0.35 mm, so the experiment's tc = 4.6628 t_sigma is 3.59 ms of contact -- which is why
    # this measurement needs a high-speed camera, and is the sanity check that the whole
    # nondimensionalisation is anchored to the right physical scale.
    ts = t_sigma(WATER, R_EXP)
    @test isapprox(ts, 7.698e-4; rtol=1e-3)
    @test isapprox(4.6628 * ts, 3.59e-3; rtol=0.01)      # contact time in seconds
    # Oil is denser-per-surface-tension, so it rings SLOWER at the same radius: sigma is 3.5x
    # smaller while rho is nearly unchanged, giving t_sigma a factor sqrt(0.0722/0.0205) = 1.88.
    @test isapprox(t_sigma(OIL_5CST, R_EXP) / ts, sqrt((960 / 998) * (0.0722 / 0.0205)); rtol=1e-12)
    @test t_sigma(OIL_5CST, R_EXP) > ts
end

@testset "either side may be given, and they agree" begin
    kw = (R=R_EXP, V0=0.45)
    cd = conditions(; drop=:oil, kw...)
    cb = conditions(; bath=:oil, kw...)
    cboth = conditions(; drop=:oil, bath=:oil, kw...)
    for f in (:We, :Bo, :Oh)
        @test getfield(cd, f) == getfield(cb, f) == getfield(cboth, f)
    end
    # A Fluid and its preset name are interchangeable, and aliases resolve to the same object.
    @test fluid(:oil) === fluid(:silicone_oil_5cSt) === OIL_5CST
    @test fluid(WATER) === WATER
    @test conditions(; drop=OIL_5CST, kw...).Oh == cd.Oh
end

@testset "underspecification and two-fluid specification are refused, not guessed" begin
    kw = (R=R_EXP, V0=0.45)
    # Nothing to infer from.
    @test_throws ArgumentError conditions(; kw...)
    # require_both is the opt-in strict mode: inferring the second side is fine by default, and
    # refusable when the caller would rather state it.
    @test_throws ArgumentError conditions(; drop=:water, require_both=true, kw...)
    @test_throws ArgumentError conditions(; bath=:water, require_both=true, kw...)
    @test conditions(; drop=:water, bath=:water, require_both=true, kw...).Oh > 0
    # A genuine two-fluid impact is OUTSIDE this model: eq:nondim has one rho, sigma, nu. Refusing
    # matters more than it looks -- silently taking the drop's properties would produce a plausible
    # number for a problem the package cannot represent.
    @test_throws ArgumentError conditions(; drop=:water, bath=:oil, kw...)
    # ...but a relabelled identical liquid is not a two-fluid case.
    twin = Fluid(:water_relabelled, WATER.rho, WATER.sigma, WATER.nu)
    @test conditions(; drop=:water, bath=twin, kw...).Oh == conditions(; drop=:water, kw...).Oh
    # Unphysical dimensional inputs.
    @test_throws ArgumentError conditions(; drop=:water, R=0.0, V0=0.45)
    @test_throws ArgumentError conditions(; drop=:water, R=R_EXP, V0=-0.45)
    @test_throws ArgumentError fluid(:mercury)
end

@testset "dimensional bath geometry becomes b and h0 in units of R" begin
    c = conditions(drop=:water, R=R_EXP, V0=0.45,
                   bath_radius=25 * R_EXP, bath_depth=3 * R_EXP)
    @test isapprox(c.b, 25.0; rtol=1e-12)
    @test isapprox(c.h0, 3.0; rtol=1e-12)
    @test conditions(drop=:water, R=R_EXP, V0=0.45).b === nothing
end

@testset "conditions feeds Params without a transcription step" begin
    c = conditions(drop=:oil, R=R_EXP, V0=0.6, bath_radius=6 * R_EXP, bath_depth=3 * R_EXP)
    p = Params(c; M=20, L=20, N=3, nq=60)
    @test p.We == c.We
    @test p.Bo == c.Bo
    @test p.Oh == c.Oh
    @test p.b == c.b
    @test p.h0 == c.h0
    # Oil sits at Oh = 0.058, where Lamb's asymptotics are wrong by tens of percent, so the default
    # closure had better be the arbitrary-Oh one.
    @test p.viscous === :reid
    @test dynamic_viscosity(OIL_5CST) ≈ 960.0 * 5.0e-6
end
