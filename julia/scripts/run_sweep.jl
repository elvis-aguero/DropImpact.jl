# Parameter sweep over impact Weber number at fixed (Bo,Oh), mirroring Alventosa et al.'s
# own sweep drivers (km-dropplet-onto-bath/matlab/1_code/driver_3B.m-style usage).
# Usage: julia --project=. scripts/run_sweep.jl [Bo] [Oh] [M] [L] [N]

using SpectralKM

function main()
    Bo = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.017
    Oh = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.006
    M = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
    L = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 12
    N = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1

    We_values = [0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.5]

    println("We    max_depth    contact_time    coeff_of_restitution")
    for We in We_values
        p = Params(We=We, Bo=Bo, Oh=Oh, M=M, L=L, N=N, b=6.0, h0=3.0, nq=20)
        times, levels, phases = run_simulation(p; t_end=8.0, dt_init=1e-3, dt_min=1e-9, dt_max=0.05)
        Y = max_penetration_depth(levels, p.L)
        tc = contact_time(times, phases)
        cor = coefficient_of_restitution(times, levels, phases)
        cor_str = cor === nothing ? "n/a" : string(round(cor, digits=4))
        println("$We   $(round(Y,digits=4))   $(round(tc,digits=4))   $cor_str")
    end
end

main()
