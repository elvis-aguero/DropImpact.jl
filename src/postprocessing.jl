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
