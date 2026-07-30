# THE REGRESSION GATE ON THE CONTACT-EDGE RULE.
#
# WHY THIS FILE EXISTS. :crossing was about to be made the default on the strength of one
# observation: at high-We oil the incumbent :feasible rule holds contact for 21 steps instead of a
# physical ~5 time units, detaches, and the drop then free-falls to z_cm = -84 without ever
# re-contacting, so no contact time exists at all. :crossing returns a physical trajectory there.
# But "fixes the case that was broken" is not "does not break the cases that worked", and the only
# way to know that is to check BOTH observables against experiment across the range, at the
# DEFAULT numerical settings rather than at some cheaper truncation chosen to make it pass.
#
# So: contact time AND maximum penetration depth, each within TWO standard deviations of the
# measured value, at M = 60, L = 120, N = 3, nq = 200, b = 6, h0 = 3 -- whatever Params gives when
# you ask it for nothing.
#
# 2 sd is the bar because these are 5-trial standard deviations on a linear model of a nonlinear
# deformation, and because the two sides do not even measure contact time the same way: the
# experiment times the north pole across z = 2R, this model the centre of mass across z = R, which
# the authors put at ~5% for water and ~2% for oil. A 1 sd bar would be asserting agreement finer
# than the definitional offset.
#
# NOT PART OF THE DEFAULT SUITE. Each point is one simulation at production truncation -- minutes
# at low We, up to an hour at We ~ 7 -- so this runs only when asked:
#
#   SPECTRALKM_EXPERIMENT_REGRESSION=1 SPECTRALKM_SELECTOR=crossing julia --project=. -e 'using Pkg; Pkg.test()'
#
# and in CI as the `Selector regression vs experiment` job, which runs it under BOTH selectors so
# the comparison is like-for-like on the same runner.

using Test
using SpectralKM
using DelimitedFiles
using Printf

const REG_ON = get(ENV, "SPECTRALKM_EXPERIMENT_REGRESSION", "0") == "1"
const REG_SELECTOR = Symbol(get(ENV, "SPECTRALKM_SELECTOR", string(SpectralKM.DEFAULT_SELECTOR)))
const REGDATA = joinpath(@__DIR__, "..", "data", "experiments")

rread(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => Float64.(d[:, i]) for (i, n) in enumerate(h))
end

# The anchor points: both fluids, spanning We from 0.73 to 7.31, i.e. from where :feasible is fine
# to where it fails outright. Chosen before any of them was run under :crossing, so this is a
# prediction being tested rather than a fit.
const ANCHORS = [
    (:water, 0.725074), (:water, 1.938695),
    (:oil, 1.215765), (:oil, 3.946288), (:oil, 7.307001),
]

if !REG_ON
    @info "skipping the experiment regression gate (set SPECTRALKM_EXPERIMENT_REGRESSION=1)" selector=REG_SELECTOR
else
    @testset "selector regression vs experiment ($(REG_SELECTOR), default truncation)" begin
        for (which, We_t) in ANCHORS
            e = rread(joinpath(REGDATA, "bath_experiment_$(which).csv"))
            i = argmin(abs.(e["We"] .- We_t))
            @test e["We"][i] ≈ We_t atol = 1e-6
            tce, tcsd = e["tc_over_tsigma"][i], e["tc_sd"][i]
            dme, dmsd = e["delta_over_R"][i], e["delta_sd"][i]

            f = which == :water ? WATER : OIL_5CST
            R = 3.5e-4
            c = conditions(drop=f, R=R, V0=sqrt(We_t * f.sigma / (f.rho * R)))
            # DEFAULT numerical settings: only the physics and the bath geometry are named.
            p = Params(c; b=6.0, h0=3.0, selector=REG_SELECTOR)
            levels, _, phases = run_simulation(p; t_end=14.0)
            ts = [l.t for l in levels]

            tct = threshold_contact_time(ts, levels)
            # Penetration depth over the PRIMARY contact interval only. Taken over the whole run it
            # would pick up a later bounce, which is not what the experiment's delta is.
            ivs = contact_intervals(ts, phases)
            prim = isempty(ivs) ? nothing : argmax(iv -> iv.duration, ivs)
            dmm = prim === nothing ? nothing :
                  max_penetration_depth([l for l in levels if prim.t_start <= l.t <= prim.t_end], p.L)

            @info("regression point",
                  fluid=which, We=We_t, selector=REG_SELECTOR,
                  tc_model=tct, tc_exp=tce, tc_sd=tcsd,
                  tc_nsd=(tct === nothing ? NaN : (tct - tce) / tcsd),
                  delta_model=dmm, delta_exp=dme, delta_sd=dmsd,
                  delta_nsd=(dmm === nothing ? NaN : (dmm - dme) / dmsd))

            # A rule that loses contact and lets the drop sink returns `nothing` here. That is the
            # failure this gate exists to catch, so it is an assertion and not a skip.
            @test tct !== nothing
            @test dmm !== nothing
            tct === nothing && continue

            # Still the BATH problem, not a rigid wall.
            @test tct > 3.4

            # THE GATE.
            @test abs(tct - tce) <= 2 * tcsd
            @test abs(dmm - dme) <= 2 * dmsd
        end
    end
end
