# COMPARISON (not validation) against Alventosa et al. (2023)'s own 1PKM predictions, digitized
# from km-dropplet-onto-bath's water_QPExp_DNS_3panel_FFF figure (julia/data/reference/).
# NOT a bit-for-bit match target (this is a genuinely different closure — see design doc's
# own framing, §subsec:accel-motivation) — a plausibility check that trends (max
# penetration depth, coefficient of restitution, contact time vs. We) are in the right
# ballpark, at the SAME (Bo,Oh) slice the reference data was generated at: water,
# Bo=0.017, Oh=0.006 (rho=1 g/cm^3, sigma=72.20 dyn/cm, nu=0.98e-2 cm^2/s, R=0.035 cm,
# g=981 cm/s^2 — confirmed against km-dropplet-onto-bath/matlab/1_code/ctime_maxdef_alfa.m
# and water_alventosa_simulation.m).

using SpectralKM

"""Minimal two-column CSV reader (We,Y / We,alpha / We,tc) — no external CSV dependency
needed for this small, fixed-format reference data."""
function read_two_col_csv(path::String)
    lines = readlines(path)
    We = Float64[]
    val = Float64[]
    for line in lines[2:end]  # skip header
        isempty(strip(line)) && continue
        parts = split(line, ',')
        push!(We, parse(Float64, parts[1]))
        push!(val, parse(Float64, parts[2]))
    end
    return We, val
end

function main()
    data_dir = joinpath(@__DIR__, "..", "data", "reference")
    We_maxdef, Y_ref = read_two_col_csv(joinpath(data_dir, "1PKM_maxdef.csv"))
    We_cor, alpha_ref = read_two_col_csv(joinpath(data_dir, "1PKM_coefres.csv"))
    We_tc, tc_ref = read_two_col_csv(joinpath(data_dir, "1PKM_contacttime.csv"))

    Bo, Oh = 0.017, 0.006
    M, L, N = 12, 12, 1

    # A handful of We values spanning the reference sweep (not the full 29-point sweep,
    # to keep this script's runtime reasonable — widen once the model is validated).
    We_test = [0.106, 0.575, 1.096, 2.188, 4.839]

    println("We      Y_model   Y_1PKM   alpha_model  alpha_1PKM   tc_model  tc_1PKM")
    for We in We_test
        p = Params(We=We, Bo=Bo, Oh=Oh, M=M, L=L, N=N, b=6.0, h0=3.0, nq=20)
        levels, diag, phases = run_simulation(p; t_end=8.0, dt_init=1e-3, dt_min=1e-9, dt_max=0.05)
        times = [lv.t for lv in levels]
        Y_model = max_penetration_depth(levels, p.L)
        tc_model = primary_contact_time(times, phases)
        cor_model = coefficient_of_restitution(times, levels, phases)

        i_Y = argmin(abs.(We_maxdef .- We))
        i_a = argmin(abs.(We_cor .- We))
        i_t = argmin(abs.(We_tc .- We))

        cor_str = cor_model === nothing ? "n/a" : string(round(cor_model, digits=4))
        println("$We   $(round(Y_model,digits=4))   $(round(Y_ref[i_Y],digits=4))   " *
                "$cor_str   $(round(alpha_ref[i_a],digits=4))   " *
                "$(round(tc_model,digits=4))   $(round(tc_ref[i_t],digits=4))")
    end
end

main()
