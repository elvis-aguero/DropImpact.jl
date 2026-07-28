# Single-case driver: run one impact simulation and report rebound metrics.
# Usage: julia --project=. scripts/run_impact.jl [We] [Bo] [Oh] [M] [L] [N] [t_end] [wall]

using SpectralKM

function main()
    We = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0958
    Bo = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.017
    Oh = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.006
    M = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 60
    L = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 60
    N = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
    t_end = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 8.0
    wall = length(ARGS) >= 8 ? Symbol(ARGS[8]) : :free

    p = Params(; We, Bo, Oh, M, L, N, b=6.0, h0=3.0, nq=40, wall)
    println("Running We=$We Bo=$Bo Oh=$Oh M=$M L=$L N=$N t_end=$t_end wall=$wall")

    levels, diag, phases = run_simulation(p; t_end=t_end, dt_init=1e-3, dt_min=1e-9, dt_max=0.05)
    times = [lv.t for lv in levels]

    println("steps recorded: ", length(times), "   t_final: ", times[end])
    println("contact steps: ", length(diag))
    println("max penetration depth (δ/R): ", max_penetration_depth(levels, p.L))
    println("contact time (t_c/T_s): ", contact_time(times, phases))
    cor = coefficient_of_restitution(times, levels, phases)
    println("coefficient of restitution: ", cor === nothing ? "n/a (no clean rebound resolved)" : cor)
end

main()
