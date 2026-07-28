# Validation sweep against the published reference curves (data/reference/*.csv):
# coefficient of restitution, contact time, and maximum deformation vs Weber number,
# at the water parameters Bo=0.017, Oh=0.006 of AlventosaEtAl2023.
using SpectralKM, Printf, DelimitedFiles

const BO = 0.017
const OH = 0.006

"""Rebound metrics from one run: (CoR, contact time, max contact radius, max width)."""
function metrics(We; M=60, L=60, N=3, nq=40, t_end=14.0)
    p = Params(We=We, Bo=BO, Oh=OH, M=M, L=L, N=N, b=6.0, h0=3.0, nq=nq)
    levels, diag = run_simulation(p; t_end=t_end, dt_init=1e-3)
    isempty(diag) && return (nothing, nothing, nothing, nothing)
    t_in = minimum(d.t for d in diag); t_out = maximum(d.t for d in diag)
    iout = argmax([d.t for d in diag])
    cor = abs(diag[iout].v / -sqrt(We))
    rc = sin(maximum(d.theta_c for d in diag))
    # maximum droplet width: max over time of max over theta of r(theta) = xi sin(theta)
    wmax = 0.0
    for lv in levels
        for th in range(0.0, π; length=200)
            wmax = max(wmax, forward_map_r(lv.drop.beta, th, p.L))
        end
    end
    return (cor, t_out - t_in, rc, wmax)
end

ref(f) = readdlm(joinpath(@__DIR__, "..", "data", "reference", f), ','; skipstart=1)
Rcor = ref("1PKM_coefres.csv"); Rtc = ref("1PKM_contacttime.csv"); Rmd = ref("1PKM_maxdef.csv")

@printf("%-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "We", "CoR", "CoR_ref", "tc", "tc_ref", "r_c", "width", "md_ref")
rows = Any[]
for i in 1:size(Rcor, 1)
    We = Rcor[i, 1]
    cor, tc, rc, wd = metrics(We)
    cor === nothing && (println("We=$We : NO CONTACT"); continue)
    @printf("%-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f\n",
            We, cor, Rcor[i,2], tc, Rtc[i,2], rc, wd, Rmd[i,2])
    push!(rows, (We, cor, Rcor[i,2], tc, Rtc[i,2], rc, wd, Rmd[i,2]))
    flush(stdout)
end
if !isempty(rows)
    ec = [abs(r[2]-r[3])/r[3] for r in rows]; et = [abs(r[4]-r[5])/r[5] for r in rows]
    @printf("\nCoR    : mean rel err %.1f%%  max %.1f%%  over %d points\n", 100*sum(ec)/length(ec), 100*maximum(ec), length(ec))
    @printf("t_cont : mean rel err %.1f%%  max %.1f%%\n", 100*sum(et)/length(et), 100*maximum(et))
    writedlm(joinpath(@__DIR__, "..", "data", "sweep_results.csv"),
             vcat(["We" "CoR" "CoR_ref" "tc" "tc_ref" "rc" "width" "md_ref"], reduce(vcat, [collect(r)' for r in rows])), ',')
end
