# Rebound metrics (coefficient of restitution, contact time, max penetration depth)
# and trajectory/shape reconstruction, matching the quantities reported in
# Alventosa et al. (2023) and Agüero et al. (2026) for comparison.

"""South-pole height above the undisturbed bath: `z_cm - ξ(θ=0)`."""
south_pole_height(lvl::Level, L::Integer) = lvl.com.z - xi_of_theta(lvl.drop.beta, 0.0, L)

"""
    max_penetration_depth(levels, L) -> value

`-min(south_pole_height)` over the run (design doc's `δ`, matching Alventosa et al.'s
maximum-penetration-depth metric): how far below the undisturbed bath surface the
droplet's lowest point reaches.
"""
function max_penetration_depth(levels::Vector{Level}, L::Integer)
    return -minimum(south_pole_height(lvl, L) for lvl in levels)
end

"""
    contact_intervals(times, phases) -> Vector{NamedTuple}

The maximal contiguous runs of `InContact`, each as `(t_start, t_end, duration, nsteps)`.

This exists because a single impact does NOT produce a single interval, and the difference
matters for the headline metric. Measured at We=1.0958, M=L=60, N=3, dt_max=0.02, one impact
yields five intervals: the physical contact, 0 -> 3.6293 over 187 steps, followed by four
re-entries of 3 to 8 steps each. Three of the four are separated from their predecessor by a
gap of exactly `dt_init`, which is the signature of the `just_left` guard in
`run_simulation`: it forces one advancing free-flight step after contact ends, after which
onset is immediately re-detected. That is stepper chatter at detachment, not bouncing.

Three different scalars can therefore be called "the contact time", and they disagree by
17%: the first interval's duration (3.629), the sum over all intervals (3.815), and the
first-to-last span (4.263, which also swallows a 0.446-long free-flight excursion). Report
`primary_contact_time` and say so; the span is the one number that should never be used.
"""
function contact_intervals(times::Vector{Float64}, phases::Vector{Phase})
    ivs = NamedTuple{(:t_start, :t_end, :duration, :nsteps),
                     Tuple{Float64,Float64,Float64,Int}}[]
    i = 1
    while i <= length(phases)
        if phases[i] == InContact
            j = i
            while j + 1 <= length(phases) && phases[j+1] == InContact
                j += 1
            end
            # Contact began between step i-1 and i, so the interval opens at times[i-1].
            t0 = times[max(i - 1, 1)]
            push!(ivs, (; t_start=t0, t_end=times[j], duration=times[j] - t0, nsteps=j - i + 1))
            i = j + 1
        else
            i += 1
        end
    end
    return ivs
end

"""
    primary_contact_time(times, phases) -> value

Duration of the LONGEST contiguous contact interval: the physical impact, excluding both
the detachment chatter and any later re-contact. This is the headline contact time and the
metric to converge in `M`, `L`, `N` and `dt_max`. Returns `0.0` if contact never occurred.

Preferred over [`contact_time`](@ref) because it is the one of the three candidate
definitions (see [`contact_intervals`](@ref)) that measures a single physical event.

The longest interval rather than the *first*, which is a distinction that only matters when
something is wrong -- and then matters a lot. At converged settings the physical impact IS
the first interval and the two agree exactly. But an under-resolved run chatters from the
very beginning: at `M = L = 20`, `nq = 16` the first interval is 0.0081 long out of eight
intervals, so "first" reports a contact time three orders of magnitude too small while
"longest" still finds the impact. Taking the longest also makes the diagnostic
self-checking: if [`contact_intervals`](@ref) shows the longest interval is not the first,
the run is chattering at onset and the truncation is too coarse to trust.
"""
function primary_contact_time(times::Vector{Float64}, phases::Vector{Phase})
    ivs = contact_intervals(times, phases)
    return isempty(ivs) ? 0.0 : maximum(iv.duration for iv in ivs)
end

"""
    threshold_contact_time(times, levels; z_threshold=1.0) -> value or nothing

Contact time as defined by AlventosaEtAl2023, so that a comparison against their published
data compares like with like: **the duration for which the droplet's centre of mass lies below
`z_threshold`**, with the crossing instants linearly interpolated.

WHY THIS EXISTS, AND WHY IT IS NOT [`contact_time`](@ref). These are different quantities and
conflating them produced a spurious 40-70% "model error" before it was caught:

  * `contact_time` is the duration actually spent with `phase == InContact` -- the physical
    contact duration, the obvious definition.
  * AlventosaEtAl2023 measure something else, out of experimental necessity: their §2 states
    `t_c` is "the time duration from which the top of the droplet crosses the height `z=2R` to
    the time the top of the droplet returns to that height", because "it was impossible to
    determine precisely when the droplets lost physical contact with the fluid; however, this
    always occurred BEFORE the top of the drop returned to the level `z=2R`". For their model
    and DNS they use the equivalent centre-of-mass threshold `z=R`, reporting a 2-5% difference
    between the two choices.

So their `t_c` deliberately INCLUDES post-detachment free flight, and is a trajectory metric
rather than a contact metric. `z_threshold=1.0` is `z=R` in this package's units (lengths are
scaled by `R`), and note `initial_level` starts the drop at exactly `z_cm=1.0`, so the
descending crossing is `t=0` by construction.

Returns `nothing` if the centre of mass never returns above the threshold within the run.
"""
function threshold_contact_time(times::Vector{Float64}, levels::Vector{Level};
                                z_threshold::Float64=1.0)
    length(levels) == length(times) || throw(ArgumentError("times and levels must align"))
    zs = [l.com.z for l in levels]
    # descending crossing: last index at or above threshold before first going below
    i_down = findfirst(i -> zs[i] < z_threshold, eachindex(zs))
    i_down === nothing && return nothing
    t_enter = if i_down == 1
        times[1]                        # started at the threshold, as initial_level does
    else
        z0, z1 = zs[i_down-1], zs[i_down]
        times[i_down-1] + (z0 - z_threshold) / (z0 - z1) * (times[i_down] - times[i_down-1])
    end
    # ascending return: first index after i_down back at or above threshold
    i_up = findfirst(i -> zs[i] >= z_threshold, i_down:length(zs))
    i_up === nothing && return nothing
    i_up += i_down - 1
    i_up == 1 && return nothing
    z0, z1 = zs[i_up-1], zs[i_up]
    t_exit = times[i_up-1] + (z_threshold - z0) / (z1 - z0) * (times[i_up] - times[i_up-1])
    return t_exit - t_enter
end

"""
    threshold_crossings(times, levels; z_threshold=1.0) -> (t_enter, v_enter, t_exit, v_exit) or nothing

The two instants at which the centre of mass crosses `z_threshold` -- downward, then back upward --
with the crossing times AND the vertical velocities there, both linearly interpolated. This is the
single geometric event that AlventosaEtAl2023's §4.2 defines both of its rebound metrics from, so
both are derived from it here rather than from separate conventions.

Returns `nothing` if the centre of mass never returns above the threshold within the run.
"""
function threshold_crossings(times::Vector{Float64}, levels::Vector{Level};
                             z_threshold::Float64=1.0)
    length(levels) == length(times) || throw(ArgumentError("times and levels must align"))
    zs = [l.com.z for l in levels]
    vs = [l.com.v for l in levels]
    i_down = findfirst(i -> zs[i] < z_threshold, eachindex(zs))
    i_down === nothing && return nothing
    t_enter, v_enter = if i_down == 1
        times[1], vs[1]                 # started at the threshold, as initial_level does
    else
        z0, z1 = zs[i_down-1], zs[i_down]
        f = (z0 - z_threshold) / (z0 - z1)
        (times[i_down-1] + f * (times[i_down] - times[i_down-1]),
         vs[i_down-1] + f * (vs[i_down] - vs[i_down-1]))
    end
    i_up = findfirst(i -> zs[i] >= z_threshold, i_down:length(zs))
    i_up === nothing && return nothing
    i_up += i_down - 1
    i_up == 1 && return nothing
    z0, z1 = zs[i_up-1], zs[i_up]
    f = (z_threshold - z0) / (z1 - z0)
    t_exit = times[i_up-1] + f * (times[i_up] - times[i_up-1])
    v_exit = vs[i_up-1] + f * (vs[i_up] - vs[i_up-1])
    return (t_enter, v_enter, t_exit, v_exit)
end

"""
    threshold_coefficient_of_restitution(times, levels; z_threshold=1.0) -> value or nothing

`alpha = -v_exit / v_enter`, with both velocities taken AT THE TWO INSTANTS THE CENTRE OF MASS
CROSSES `z_threshold` -- the paper's own definition of alpha, and the partner of
[`threshold_contact_time`](@ref).

WHY THIS IS NOT [`coefficient_of_restitution`](@ref), and why the distinction matters. That
function takes the velocities at the first and last `InContact` STEP, which is a different event:

  * AlventosaEtAl2023 §4.2 defines `t_c` as the interval between the two instants the north pole
    crosses `z=2R`, and alpha as "minus the ratio of the vertical velocities at those times" --
    i.e. alpha and `t_c` are read off the SAME pair of crossings. For their model and DNS they move
    both to the centre of mass crossing `z=R`. So the matching alpha here must use the `z_cm=R`
    crossings, not the contact phases.
  * The phase-based version is additionally fragile once a run rebounds more than once: it uses
    `findlast(==(InContact), phases)`, which is the exit of the LAST bounce in the run, not of the
    impact the experiment measured. With `selector=:crossing` producing genuine multi-bounce
    trajectories, that is no longer a subtle difference.

Both remain available; this is the one to compare against the published alpha.
"""
function threshold_coefficient_of_restitution(times::Vector{Float64}, levels::Vector{Level};
                                              z_threshold::Float64=1.0)
    c = threshold_crossings(times, levels; z_threshold=z_threshold)
    c === nothing && return nothing
    _, v_enter, _, v_exit = c
    v_enter == 0 && return nothing
    return -v_exit / v_enter
end

"""
    contact_time(times, phases) -> value

Total duration spent with `phase==InContact`, summed over contiguous contact intervals.

Note this sums the detachment chatter documented in [`contact_intervals`](@ref) along with
the physical contact, so it overestimates the impact duration (3.815 against 3.629 in the
reference case). Use [`primary_contact_time`](@ref) for the headline metric; this remains
the right quantity when the total time under load is what is wanted, e.g. for a dissipation
budget."""
function contact_time(times::Vector{Float64}, phases::Vector{Phase})
    total = 0.0
    for i in 2:length(times)
        if phases[i] == InContact
            total += times[i] - times[i-1]
        end
    end
    return total
end

"""
    coefficient_of_restitution(times, levels, L) -> value or nothing

`α = -V_exit/V_impact` (south-pole vertical velocity at rebound over at impact),
using `ż_cm` as a proxy for the south-pole velocity (consistent to leading order in
the small-deformation regime the whole model is built for). Returns `nothing` if the
run never both entered and exited contact (e.g. still mid-bounce at `t_end`).
"""
function coefficient_of_restitution(times::Vector{Float64}, levels::Vector{Level}, phases::Vector{Phase})
    first_contact = findfirst(==(InContact), phases)
    (first_contact === nothing || first_contact == 1) && return nothing
    last_contact = findlast(==(InContact), phases)
    (last_contact === nothing || last_contact == length(phases)) && return nothing
    v_impact = levels[first_contact-1].com.v
    v_exit = levels[last_contact+1].com.v
    return -v_exit / v_impact
end

"""
    reconstruct_bath(lvl, r_grid, p) -> Vector

`η(r,τ)` (design doc eq:eta-xi) evaluated on `r_grid`, for plotting."""
function reconstruct_bath(lvl::Level, r_grid::AbstractVector{Float64}, p::Params)
    return [sum(lvl.bath.a[m+1] * besselj0(p.k[m+1] * r) for m in 0:p.M) for r in r_grid]
end

"""
    reconstruct_drop(lvl, theta_grid, L) -> (r, z)

Droplet surface `(r(θ,τ), z_d(θ,τ))` (design doc eq:forward-map) evaluated on
`theta_grid`, for plotting the meridional cross-section."""
function reconstruct_drop(lvl::Level, theta_grid::AbstractVector{Float64}, L::Integer)
    r = [forward_map_r(lvl.drop.beta, th, L) for th in theta_grid]
    z = [lvl.com.z - forward_map_zd(lvl.drop.beta, th, L) for th in theta_grid]
    return r, z
end
