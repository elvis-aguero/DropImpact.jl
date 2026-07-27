using Test
using SpectralKM
using SpectralKM: legendre_P, legendre_dP, legendre_P_table, legendre_dP_table, bonnet_H,
    mcmahon_seed_J1, refine_zero_J1, gauss_quad, bdf2_coeffs, bath_affine, drop_affine,
    com_affine, BathModeState, DropModeState, COMState, Level, SimHistory,
    min_nq_for_exact_com, residual, unpack_state,
    Converged, MaxIterExceeded, Stalled, initial_level, gap_at_pole, free_flight_step,
    build_residual, contact_step, warm_start, warm_start_extrapolated,
    xi_tau_of_x, outer_bracket_of_x, K_of_x, Pi_of_x, accel_galerkin_term,
    xi_of_x, r_of_x, forward_map_r, forward_map_zd, b_l_all, drop_galerkin_term,
    com_force_closed, pressure_poly_raw
using SpecialFunctions
using ForwardDiff
using LinearAlgebra

include("test_legendre.jl")
include("test_bessel.jl")
include("test_quadrature.jl")
include("test_affine.jl")
include("test_conditioning.jl")
include("test_residual.jl")
include("test_newton.jl")
include("test_timestepper.jl")
include("test_reductions.jl")
