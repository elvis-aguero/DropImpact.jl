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
    contact_time(times, phases) -> value

Total duration spent with `phase==InContact`, summed over contiguous contact
intervals (there may be more than one bounce in a longer run)."""
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
