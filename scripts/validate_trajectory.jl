# Trajectory validation against the experiments that superseded AlventosaEtAl2023,
# archived in the sister repo km-dropplet-onto-bath (Luke Paper Raw Data):
#   top_exp.csv / bottom_exp.csv   experimental droplet top and bottom vs time
#   c_radius_DNS.csv               DNS contact radius vs time
#   width_DNS.csv                  DNS maximum droplet width vs time
#
# Top and bottom are the two poles of the spectral droplet shape:
#   z_bottom = z_cm - xi(theta=0)      (theta=0 is the pole nearest the bath)
#   z_top    = z_cm + xi(theta=pi)
using SpectralKM, Printf, DelimitedFiles

const REF = joinpath(homedir(), "Documents", "Github", "km-dropplet-onto-bath",
                     "matlab", "0_data", "manual", "Luke Paper Raw Data")
readref(f) = readdlm(joinpath(REF, f), ',')

"""Linear interpolation of a scattered (t,y) reference onto time `t`."""
function interp(T, Y, t)
    o = sortperm(T); Ts = T[o]; Ys = Y[o]
    (t < Ts[1] || t > Ts[end]) && return nothing
    j = searchsortedfirst(Ts, t)
    j <= 1 && return Ys[1]
    j > length(Ts) && return Ys[end]
    Ts[j] == Ts[j-1] && return Ys[j]
    w = (t - Ts[j-1]) / (Ts[j] - Ts[j-1])
    return (1 - w) * Ys[j-1] + w * Ys[j]
end

p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=60, L=60, N=3, b=6.0, h0=3.0, nq=40)
levels, diag = run_simulation(p; t_end=8.0, dt_init=1e-3)
@printf("run: %d levels, %d contact steps, contact time %.4f\n", length(levels), length(diag),
        isempty(diag) ? 0.0 : maximum(d.t for d in diag) - minimum(d.t for d in diag))

t_mod = [lv.t for lv in levels]
ztop = [lv.com.z + xi_of_theta(lv.drop.beta, π, p.L) for lv in levels]
zbot = [lv.com.z - xi_of_theta(lv.drop.beta, 0.0, p.L) for lv in levels]
wid  = [maximum(forward_map_r(lv.drop.beta, th, p.L) for th in range(0, π; length=200)) for lv in levels]

for (label, file, my, mt) in (("top (exp)","top_exp.csv",ztop,t_mod),
                              ("bottom (exp)","bottom_exp.csv",zbot,t_mod),
                              ("width (DNS)","width_DNS.csv",wid,t_mod))
    D = readref(file); T = Float64.(D[:,1]); Y = Float64.(D[:,2])
    errs = Float64[]
    for (t,y) in zip(T,Y)
        ym = interp(mt, my, t); ym === nothing && continue
        push!(errs, abs(ym - y))
    end
    isempty(errs) ? @printf("%-14s no overlap\n", label) :
        @printf("%-14s n=%3d  mean |err|=%.4f  max |err|=%.4f   (data range %.3f..%.3f)\n",
                label, length(errs), sum(errs)/length(errs), maximum(errs), minimum(Y), maximum(Y))
end

D = readref("c_radius_DNS.csv"); Td = Float64.(D[:,1]); Rd = Float64.(D[:,2])
if !isempty(diag)
    tt = [d.t for d in diag]; rr = [sin(d.theta_c) for d in diag]
    errs = Float64[]
    for (t,r) in zip(Td,Rd)
        rm = interp(tt, rr, t); rm === nothing && continue
        push!(errs, abs(rm - r))
    end
    @printf("%-14s n=%3d  mean |err|=%.4f  max |err|=%.4f\n", "r_c (DNS)",
            length(errs), isempty(errs) ? NaN : sum(errs)/length(errs), maximum(errs; init=0.0))
    @printf("%-14s model peak %.4f at t=%.3f ; DNS peak %.4f at t=%.3f\n", "r_c peak",
            maximum(rr), tt[argmax(rr)], maximum(Rd), Td[argmax(Rd)])
    @printf("%-14s model %.4f ; DNS last %.4f\n", "contact end", maximum(tt), maximum(Td))
end
