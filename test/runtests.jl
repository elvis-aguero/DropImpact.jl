using Test
using SpectralKM
using SpectralKM: legendre_P, legendre_dP, legendre_P_table, legendre_dP_table, bonnet_H,
    mcmahon_seed_J1, refine_zero_J1, gauss_quad, bdf2_coeffs, bdf_derivative, bath_affine, drop_affine,
    com_affine, BathModeState, DropModeState, COMState, Level, SimHistory,
    min_nq_for_exact_com, bessel_zeros_J0, residual, unpack_state, contact_quad, step_affine,
    C_at_theta, tangency_residual, check_nonintersect, check_monotone_r, check_positivity,
    Converged, MaxIterExceeded, Stalled, initial_level, gap_at_pole, free_flight_step,
    contact_step, select_theta_c, feasible_at, onset_theta_c,
    xi_of_x, r_of_x, zd_of_x, w_of_x, forward_map_r, forward_map_zd,
    mapped_nodes, legendre_tables, geom_at_nodes, b_l_all,
    com_force_closed, pressure_poly_raw, c_m_all, contact_intervals, primary_contact_time,
    apply_clamp, first_bessel_zero_half, reid_char_residual
using SpecialFunctions
using SpecialFunctions: besseljx
using ForwardDiff
using LinearAlgebra

include("test_legendre.jl")
include("test_bessel.jl")
include("test_quadrature.jl")
include("test_affine.jl")
include("test_conditioning.jl")
include("test_rank_law.jl")
include("test_reid.jl")
include("test_residual.jl")
include("test_newton.jl")
include("test_timestepper.jl")
include("test_reductions.jl")
include("test_postprocessing.jl")
include("test_physics.jl")
include("test_wall_clamped.jl")
