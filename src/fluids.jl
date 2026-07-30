# DIMENSIONAL FLUID PROPERTIES, and the map from them to (We, Bo, Oh).
#
# WHY THIS EXISTS. `Params` takes the nondimensional groups, which is the right interface for the
# solver and the wrong one for the bench. An experimentalist measures rho, sigma, nu, the droplet
# radius and the impact speed; asking them to hand-compute We, Bo, Oh puts a transcription error
# between the measurement and the model, and there is nowhere in the repo that such an error would
# be caught. This module closes that gap: physical constants in, groups out, with the groups
# cross-checked in the testbed against the (Bo, Oh) recorded in data/experiments/bath_experiment_*.csv.
#
# ONE FLUID, NOT TWO. eq:nondim of the paper has a single rho, sigma, nu: the drop and the bath are
# the same liquid ("normally impact a bath of the same fluid", AlventosaEtAl2023 abstract). So the
# API takes whichever side you characterised and applies it to both, and REFUSES a genuinely
# two-fluid specification rather than silently using the drop's properties for the bath -- that
# would be a different physical problem, and this package has no density or viscosity ratio to
# describe it with. Pass `require_both=true` if you would rather be forced to state both sides
# explicitly than to have the second one inferred.

"""Gravitational acceleration in m/s^2. 9.81, the value tabulated by AlventosaEtAl2023 (981 cm/s^2)."""
const STANDARD_GRAVITY = 9.81

"""
    Fluid(name, rho, sigma, nu)

Dimensional properties of a liquid, in SI: density `rho` (kg/m^3), surface tension against air
`sigma` (N/m), kinematic viscosity `nu` (m^2/s). Dynamic viscosity is `dynamic_viscosity(f) = rho*nu` (Pa s).
"""
struct Fluid
    name::Symbol
    rho::Float64
    sigma::Float64
    nu::Float64
end

# Named in full rather than `mu`: affine.jl already uses `mu` for the BDF2 history term of
# eq:kappa-cm, and two unrelated `mu`s in one package is how sign errors get written.
"""Dynamic viscosity `rho*nu`, in Pa s."""
dynamic_viscosity(f::Fluid) = f.rho * f.nu

# The two working fluids of AlventosaEtAl2023, values from its table 1 verbatim (converted from
# CGS): water 0.998 g/cm^3, 72.2 dynes/cm, 0.978 cSt; oil 0.96 g/cm^3, 20.5 dynes/cm, 5 cSt.
# These are the authors' own numbers, not textbook values, and at R = 0.35 mm they reproduce the
# (Bo, Oh) printed on both experimental figures -- asserted in test/test_fluids.jl.
const WATER = Fluid(:deionized_water, 998.0, 0.0722, 0.978e-6)
const OIL_5CST = Fluid(:silicone_oil_5cSt, 960.0, 0.0205, 5.0e-6)

const FLUID_ALIASES = Dict(
    :water => WATER, :deionized_water => WATER, :di_water => WATER,
    :oil => OIL_5CST, :silicone_oil => OIL_5CST, :silicone_oil_5cSt => OIL_5CST,
    :oil_5cSt => OIL_5CST,
)

"""
    fluid(x) -> Fluid

Resolve a `Fluid` or one of the preset names (`:water`, `:oil`, ...) to a `Fluid`.
"""
fluid(f::Fluid) = f
function fluid(s::Symbol)
    haskey(FLUID_ALIASES, s) && return FLUID_ALIASES[s]
    throw(ArgumentError("unknown fluid $s; known presets are " *
                        join(sort(string.(collect(keys(FLUID_ALIASES)))), ", ") *
                        ", or pass a Fluid(name, rho, sigma, nu) directly"))
end

"""
    ImpactConditions

The nondimensional groups of eq:nondim, together with the dimensional inputs they came from, so a
run can be traced back to the measurement. Build with [`conditions`](@ref); feed to `Params`.
"""
struct ImpactConditions
    liquid::Fluid
    R::Float64          # undeformed droplet radius, m
    V0::Float64         # impact speed, m/s
    g::Float64          # m/s^2
    We::Float64
    Bo::Float64
    Oh::Float64
    b::Union{Nothing,Float64}     # bath radius / R, if a dimensional one was given
    h0::Union{Nothing,Float64}    # bath depth  / R, if a dimensional one was given
end

"""
    conditions(; drop=nothing, bath=nothing, R, V0, g=STANDARD_GRAVITY,
                 bath_radius=nothing, bath_depth=nothing, require_both=false)

Nondimensionalise a measured impact. Give the liquid on either side -- `drop` or `bath`, as a
`Fluid` or a preset name -- and the other is taken to be the SAME liquid, which is the physical
setup this model describes. `R` is the undeformed droplet radius and `V0` the impact speed, both
in SI; `bath_radius` and `bath_depth`, if given in metres, are returned as the nondimensional `b`
and `h0` that `Params` wants.

Per eq:nondim, with `t_sigma = sqrt(rho R^3/sigma)`,

    We = rho R V0^2 / sigma,    Bo = g t_sigma^2 / R = rho g R^2 / sigma,    Oh = nu / sqrt(sigma R/rho).

Errors, deliberately, when:

  * neither `drop` nor `bath` is given -- there is nothing to infer from;
  * `require_both=true` and only one is given -- opt in when you would rather state both sides
    than have one inferred;
  * both are given and they are different liquids -- the model carries a single `(rho, sigma, nu)`
    and cannot represent a density or viscosity ratio, so this is refused rather than resolved by
    quietly preferring one side.

# Examples
```julia
conditions(drop=:water, R=3.5e-4, V0=0.4)                    # Bo = 0.0166, Oh = 0.00615
conditions(bath=:oil,   R=3.5e-4, V0=0.6)                    # Bo = 0.0563, Oh = 0.0578
conditions(drop=Fluid(:glycerol_mix, 1150.0, 0.065, 5e-5), R=4e-4, V0=0.5)
```
"""
function conditions(; drop=nothing, bath=nothing, R::Real, V0::Real,
                    g::Real=STANDARD_GRAVITY,
                    bath_radius=nothing, bath_depth=nothing,
                    require_both::Bool=false)
    if drop === nothing && bath === nothing
        throw(ArgumentError("give the liquid on at least one side: conditions(drop=:water, ...) " *
                            "or conditions(bath=:oil, ...). The other side is taken to be the " *
                            "same liquid, which is the setup this model describes."))
    end
    if require_both && (drop === nothing || bath === nothing)
        side = drop === nothing ? "drop" : "bath"
        throw(ArgumentError("require_both=true, so name both sides explicitly; `$side` is missing. " *
                            "Drop require_both to have it inferred as the same liquid."))
    end
    fd = drop === nothing ? nothing : fluid(drop)
    fb = bath === nothing ? nothing : fluid(bath)
    if fd !== nothing && fb !== nothing && !same_liquid(fd, fb)
        throw(ArgumentError("drop and bath are different liquids ($(fd.name) vs $(fb.name)). This " *
                            "model has a single (rho, sigma, nu) -- eq:nondim -- and no density or " *
                            "viscosity ratio, so a two-fluid impact is outside it. Refusing rather " *
                            "than silently using the drop's properties for the bath."))
    end
    f = fd === nothing ? fb : fd
    R > 0 || throw(ArgumentError("R must be positive, got $R m"))
    V0 > 0 || throw(ArgumentError("V0 must be positive, got $V0 m/s (impact speed, not velocity)"))
    We = f.rho * R * V0^2 / f.sigma
    Bo = f.rho * g * R^2 / f.sigma
    Oh = f.nu / sqrt(f.sigma * R / f.rho)
    return ImpactConditions(f, float(R), float(V0), float(g), We, Bo, Oh,
                            bath_radius === nothing ? nothing : float(bath_radius) / R,
                            bath_depth === nothing ? nothing : float(bath_depth) / R)
end

# Two Fluid values describe the same liquid when their PROPERTIES agree; the name is a label and is
# not compared, so Fluid(:oil, ...) and Fluid(:silicone_oil, ...) with equal rho/sigma/nu pass.
same_liquid(a::Fluid, b::Fluid) =
    isapprox(a.rho, b.rho; rtol=1e-8) && isapprox(a.sigma, b.sigma; rtol=1e-8) &&
    isapprox(a.nu, b.nu; rtol=1e-8)

"""
    t_sigma(f::Fluid, R) -> seconds

The inertio-capillary time `sqrt(rho R^3/sigma)` that nondimensionalises time. Needed to convert a
model contact time back into milliseconds for comparison with a high-speed camera.
"""
t_sigma(f::Fluid, R::Real) = sqrt(f.rho * R^3 / f.sigma)
t_sigma(c::ImpactConditions) = t_sigma(c.liquid, c.R)
