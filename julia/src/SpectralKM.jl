module SpectralKM

using LinearAlgebra
using SpecialFunctions
using ForwardDiff

include("legendre.jl")
include("bessel.jl")
include("quadrature.jl")
include("types.jl")
include("pressure.jl")
include("geometry.jl")
include("bessel_moments.jl")
include("affine.jl")
include("accel_closure.jl")
include("residual.jl")
include("newton.jl")
include("timestepper.jl")
include("postprocessing.jl")

export Params, Level, SimHistory, Phase, FreeFlight, InContact
export legendre_P, legendre_dP, bonnet_H
export bessel_zeros_J1
export gauss_legendre_nodes
export pressure_poly, b_l_all, drop_galerkin_term, com_force_closed
export forward_map_r, forward_map_zd, xi_of_theta
export c_m_all, W_nm_all
export bath_affine, drop_affine, com_affine, bdf2_coeffs
export residual
export newton_solve, NewtonResult, NewtonStatus, Converged, MaxIterExceeded, Stalled
export run_simulation
export coefficient_of_restitution, contact_time, max_penetration_depth

end # module SpectralKM
