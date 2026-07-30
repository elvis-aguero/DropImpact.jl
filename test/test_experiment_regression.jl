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
# The published quasi-potential (1PKM) curve at a given We, read from the same figure the
# experimental points came from. Curve 2 in the oil figure and curve 6 in the water figure are the
# blue solid `[0.00 0.45 0.74]` lines the captions identify as the quasi-potential model.
const PKM_CURVE = Dict(:oil => 2, :water => 6)

function pkm_reference(which::Symbol, metric::AbstractString, We_t::Real)
    path = joinpath(REGDATA, "bath_model_curves_$(which).csv")
    isfile(path) || return nothing
    m = rread_str(path)
    sel = findall(i -> m["metric"][i] == metric &&
                       round(Int, m["curve"][i]) == PKM_CURVE[which], eachindex(m["curve"]))
    isempty(sel) && return nothing
    xs, ys = Float64.(m["We"][sel]), Float64.(m["value"][sel])
    o = sortperm(xs); xs, ys = xs[o], ys[o]
    # Refuse to extrapolate: outside the published curve's own range there is no reference.
    (We_t < first(xs) || We_t > last(xs)) && return nothing
    j = searchsortedfirst(xs, We_t)
    j <= 1 && return ys[1]
    x0, x1, y0, y1 = xs[j-1], xs[j], ys[j-1], ys[j]
    return x1 == x0 ? y0 : y0 + (y1 - y0) * (We_t - x0) / (x1 - x0)
end

# The model-curve file has a string column, so it cannot go through `rread`'s Float64 conversion.
rread_str(p) = let raw = readdlm(p, ','; header=true)
    d, h = raw[1], vec(raw[2])
    Dict(string(n) => d[:, i] for (i, n) in enumerate(h))
end

# (fluid, metric) pairs carrying a documented, SELECTOR-INDEPENDENT deficit, marked @test_broken
# below rather than left failing. Only oil delta, and only where 1PKM itself misses 2 sd -- i.e. at
# We >= 3.9. At oil We = 1.216 1PKM is within 2 sd, so that point takes the strict branch and must
# pass; water delta at We = 1.939 is within 25% of 1PKM's own shortfall and must also pass.
const KNOWN_DELTA_DEFICIT = Set([(:oil, "delta")])

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
            # Penetration depth over the FIRST contact interval, which is the impact the experiment
            # measured. Not the whole run, which picks up later bounces; and NOT the longest
            # interval, which is what an earlier version of this file used -- at oil We = 1.216 the
            # intervals are [(0, 4.389), (4.39, 4.415), (9.101, 14.0)] and the longest is the SECOND
            # bounce, clipped by t_end. That reported delta = 0.2389 against a true 0.7231 and read
            # as a 9.3 sd model defect. It was a defect in this test.
            ivs = contact_intervals(ts, phases)
            first_iv = isempty(ivs) ? nothing : ivs[1]
            dmm = first_iv === nothing ? nothing :
                  max_penetration_depth([l for l in levels if l.t <= first_iv.t_end], p.L)

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
            (tct === nothing || dmm === nothing) && continue

            # Still the BATH problem, not a rigid wall.
            @test tct > 3.4

            # THE GATE, with the published reduced model as the reference for what is achievable.
            #
            # 2 sd of the measurement is the bar WHERE THE MODEL CLASS CAN MEET IT. It cannot
            # everywhere: at water We = 1.9387 the experimental sd is 1.4% on tc and 2.8% on delta,
            # and 1PKM itself lands -8.25 sd and -3.25 sd there. Demanding 2 sd of this package at a
            # point where the published quasi-potential model is off by eight would not be a
            # regression gate, it would be a permanent failure that teaches nothing.
            #
            # So: where 1PKM is within 2 sd, so must we be. Where it is not, we must be no worse
            # than it by more than a quarter -- which still catches a regression, because :crossing
            # rescuing high-We oil must not be paid for by degrading anywhere else.
            for (label, ours, exp, sd, curve) in
                (("tc", tct, tce, tcsd, "tc"), ("delta", dmm, dme, dmsd, "delta"))
                ref = pkm_reference(which, curve, We_t)
                if ref === nothing
                    @test abs(ours - exp) <= 2 * sd
                elseif abs(ref - exp) <= 2 * sd
                    @info "  gate: 2 sd (1PKM meets it)" label ours_nsd=(ours-exp)/sd pkm_nsd=(ref-exp)/sd
                    @test abs(ours - exp) <= 2 * sd
                elseif (which, curve) in KNOWN_DELTA_DEFICIT
                    # SELECTOR-INDEPENDENT DEFICIT, recorded rather than tolerated silently. delta at
                    # oil We >= 3.9 is about twice 1PKM's own shortfall, and it is identical to four
                    # figures under BOTH rules (1.07417 :crossing vs 1.07481 :feasible at We = 3.946),
                    # so it is not what this gate is for and must not hold the gate red forever.
                    # @test_broken so that FIXING it fails the suite and forces this to be revisited.
                    @info "  KNOWN DELTA DEFICIT (both selectors, ~2x 1PKM)" label ours_nsd=(ours-exp)/sd pkm_nsd=(ref-exp)/sd
                    @test_broken abs(ours - exp) <= 1.25 * abs(ref - exp)
                else
                    @info "  gate: no worse than 1PKM +25% (1PKM misses 2 sd)" label ours_nsd=(ours-exp)/sd pkm_nsd=(ref-exp)/sd
                    @test abs(ours - exp) <= 1.25 * abs(ref - exp)
                end
            end
        end
    end
end
