# Render an impact to an mp4.
#
# Deliberately dependency-free: SpectralKM is a library and should not acquire a plotting
# stack, so this writes raw PPM (P6) frames -- a header plus RGB bytes, about forty lines
# of rasteriser -- and pipes them to ffmpeg. No packages to install; `ffmpeg` on PATH is
# the only requirement.
#
# Usage:
#   julia --project=. scripts/make_video.jl [We] [outfile.mp4] [M] [L] [N] [nq]
#
# What is drawn, per frame:
#   * the bath free surface eta(r) = sum_m a_m J0(k_m r), mirrored about the axis;
#   * the droplet outline from the spectral shape, r = xi sin(theta), z = z_cm - xi cos(theta);
#   * the contact patch theta in [0, theta_c] picked out in red on the droplet, which is
#     exactly the region over which the kinematic match is imposed;
#   * the pressure profile p(x) over the patch, in an inset panel, on its own scale;
#   * a readout of t, theta_c, r_c, the net contact force f, and the COM velocity.
using SpectralKM
using SpectralKM: pressure_poly_raw
using SpecialFunctions: besselj0
using Printf

const W, H = 900, 620
const RMIN, RMAX = -2.2, 2.2
const ZMIN, ZMAX = -1.3, 2.9

mutable struct Canvas
    px::Array{UInt8,3}   # H x W x 3
end
Canvas() = Canvas(fill(UInt8(255), H, W, 3))

@inline function px_of(r, z)
    i = round(Int, (ZMAX - z) / (ZMAX - ZMIN) * (H - 1)) + 1
    j = round(Int, (r - RMIN) / (RMAX - RMIN) * (W - 1)) + 1
    return i, j
end

@inline function dot!(c::Canvas, i, j, col, rad=1)
    for di in -rad:rad, dj in -rad:rad
        ii, jj = i + di, j + dj
        (1 <= ii <= H && 1 <= jj <= W) || continue
        c.px[ii, jj, 1] = col[1]; c.px[ii, jj, 2] = col[2]; c.px[ii, jj, 3] = col[3]
    end
end

"""Draw a straight segment in WORLD coordinates by walking it in pixel steps."""
function segment!(c::Canvas, r1, z1, r2, z2, col; rad=1)
    i1, j1 = px_of(r1, z1); i2, j2 = px_of(r2, z2)
    n = max(abs(i2 - i1), abs(j2 - j1), 1)
    for s in 0:n
        dot!(c, round(Int, i1 + (i2 - i1) * s / n), round(Int, j1 + (j2 - j1) * s / n), col, rad)
    end
end

function polyline!(c::Canvas, rs, zs, col; rad=1)
    for k in 1:length(rs)-1
        segment!(c, rs[k], zs[k], rs[k+1], zs[k+1], col; rad=rad)
    end
end

"""Fill a closed polygon by even-odd scanline test (used for the droplet interior)."""
function fillpoly!(c::Canvas, rs, zs, col)
    imin, imax = H, 1
    for k in eachindex(rs)
        i, _ = px_of(rs[k], zs[k]); imin = min(imin, i); imax = max(imax, i)
    end
    for i in max(imin, 1):min(imax, H)
        z = ZMAX - (i - 1) / (H - 1) * (ZMAX - ZMIN)
        xs = Float64[]
        for k in 1:length(rs)-1
            z1, z2 = zs[k], zs[k+1]
            ((z1 <= z) != (z2 <= z)) || continue
            t = (z - z1) / (z2 - z1)
            push!(xs, rs[k] + t * (rs[k+1] - rs[k]))
        end
        sort!(xs)
        for m in 1:2:length(xs)-1
            _, j1 = px_of(xs[m], z); _, j2 = px_of(xs[m+1], z)
            for j in max(j1, 1):min(j2, W)
                c.px[i, j, 1] = col[1]; c.px[i, j, 2] = col[2]; c.px[i, j, 3] = col[3]
            end
        end
    end
end

# --- 5x7 bitmap digits/letters, enough for the readout ---
const GLYPHS = Dict{Char,Vector{String}}(
 '0'=>[" ### ","#   #","#  ##","# # #","##  #","#   #"," ### "],
 '1'=>["  #  "," ##  ","  #  ","  #  ","  #  ","  #  "," ### "],
 '2'=>[" ### ","#   #","    #","   # ","  #  "," #   ","#####"],
 '3'=>["#####","   # ","  #  ","   # ","    #","#   #"," ### "],
 '4'=>["   # ","  ## "," # # ","#  # ","#####","   # ","   # "],
 '5'=>["#####","#    ","#### ","    #","    #","#   #"," ### "],
 '6'=>["  ## "," #   ","#    ","#### ","#   #","#   #"," ### "],
 '7'=>["#####","    #","   # ","  #  "," #   "," #   "," #   "],
 '8'=>[" ### ","#   #","#   #"," ### ","#   #","#   #"," ### "],
 '9'=>[" ### ","#   #","#   #"," ####","    #","   # "," ##  "],
 '.'=>["     ","     ","     ","     ","     ","  ## ","  ## "],
 '-'=>["     ","     ","     ","#####","     ","     ","     "],
 '='=>["     ","     ","#####","     ","#####","     ","     "],
 ' '=>["     ","     ","     ","     ","     ","     ","     "],
 't'=>["  #  ","  #  "," ####","  #  ","  #  ","  #  ","   ##"],
 'h'=>["#    ","#    ","# ## ","##  #","#   #","#   #","#   #"],
 'c'=>["     ","     "," ####","#    ","#    ","#    "," ####"],
 'r'=>["     ","     ","# ## ","##  #","#    ","#    ","#    "],
 'f'=>["  ###"," #   ","#### "," #   "," #   "," #   "," #   "],
 'v'=>["     ","     ","#   #","#   #","#   #"," # # ","  #  "],
 '_'=>["     ","     ","     ","     ","     ","     ","#####"],
 'p'=>["     ","     ","#### ","#   #","#### ","#    ","#    "],
 'e'=>["     ","     "," ### ","#   #","#####","#    "," ####"],
 's'=>["     ","     "," ####","#    "," ### ","    #","#### "],
 'u'=>["     ","     ","#   #","#   #","#   #","#  ##"," ## #"],
 'a'=>["     ","     "," ### ","    #"," ####","#   #"," ####"],
 'z'=>["     ","     ","#####","   # ","  #  "," #   ","#####"],
 'W'=>["#   #","#   #","#   #","# # #","# # #","## ##","#   #"],
 '0'=>[" ### ","#   #","#  ##","# # #","##  #","#   #"," ### "],
)
function text!(c::Canvas, i0, j0, str, col; scale=2)
    j = j0
    for ch in str
        g = get(GLYPHS, ch, GLYPHS[' '])
        for (rr, row) in enumerate(g), (cc, chr) in enumerate(row)
            chr == '#' || continue
            for a in 0:scale-1, b in 0:scale-1
                ii = i0 + (rr - 1) * scale + a; jj = j + (cc - 1) * scale + b
                (1 <= ii <= H && 1 <= jj <= W) || continue
                c.px[ii, jj, 1] = col[1]; c.px[ii, jj, 2] = col[2]; c.px[ii, jj, 3] = col[3]
            end
        end
        j += 6 * scale
    end
end

const BLACK = (0x00, 0x00, 0x00)
const GREY = (0x90, 0x90, 0x90)
const BATHFILL = (0x0d, 0x3b, 0x66)     # the bath body, dark blue
const BATHLINE = (0x08, 0x24, 0x40)     # its free surface, darker still
const DROPFILL = (0xe8, 0xf4, 0xff)     # the droplet, very light blue
const DROPLINE = (0x2a, 0x6f, 0xb0)
const CONTACT = (0xdd, 0x22, 0x22)

"""Bath surface height at radius `r`."""
@inline eta_at(lvl, p, r) = sum(lvl.bath.a[m+1] * besselj0(p.k[m+1] * abs(r)) for m in 0:p.M)

"""Fill the bath body: every column from its free surface down to the bottom of frame."""
function fillbath!(c::Canvas, lvl, p::Params)
    for j in 1:W
        r = RMIN + (j - 1) / (W - 1) * (RMAX - RMIN)
        itop, _ = px_of(r, eta_at(lvl, p, r))
        for i in max(itop, 1):H
            c.px[i, j, 1] = BATHFILL[1]; c.px[i, j, 2] = BATHFILL[2]; c.px[i, j, 3] = BATHFILL[3]
        end
    end
end

# Inset panel (pressure profile over the contact patch), anchored to the TOP-RIGHT corner.
const IX0, IZ0 = 1.20, 2.02     # axis origin, world coordinates
const IW, IH = 0.92, 0.66       # axis lengths

"""Draw the inset axes with ticks and labels; returns the p-value -> world-z map."""
function inset_axes!(c::Canvas, rc_max, pmin, pmax)
    span = max(pmax - pmin, 1e-12)
    z_of_p(v) = IZ0 + (v - pmin) / span * IH
    # y axis (pressure) and x axis (radius)
    segment!(c, IX0, IZ0, IX0, IZ0 + IH, BLACK; rad=0)
    segment!(c, IX0, IZ0, IX0 + IW, IZ0, BLACK; rad=0)
    # ticks: three on each axis
    for k in 0:2
        zt = IZ0 + k * IH / 2
        segment!(c, IX0 - 0.03, zt, IX0, zt, BLACK; rad=0)
        rt = IX0 + k * IW / 2
        segment!(c, rt, IZ0 - 0.03, rt, IZ0, BLACK; rad=0)
    end
    # zero-pressure line, if the range straddles it
    if pmin < 0 < pmax
        segment!(c, IX0, z_of_p(0.0), IX0 + IW, z_of_p(0.0), GREY; rad=0)
    end
    i0, j0 = px_of(IX0, IZ0 + IH)
    text!(c, i0 - 16, j0 - 4, "p", BLACK; scale=2)
    text!(c, i0 - 16, j0 + 16, @sprintf("%.2f", pmax), BLACK; scale=1)
    ib, jb = px_of(IX0 + IW, IZ0)
    text!(c, ib + 6, jb - 30, "r", BLACK; scale=2)
    text!(c, ib + 6, jb - 96, @sprintf("%.2f", rc_max), BLACK; scale=1)
    ia, ja = px_of(IX0, IZ0)
    text!(c, ia + 6, ja - 12, "0", BLACK; scale=1)
    return z_of_p
end

function frame(lvl, p::Params, f)
    c = Canvas()
    fillbath!(c, lvl, p)

    # bath free surface, drawn on top of its own fill
    rs = range(0.0, RMAX; length=460)
    eta = [eta_at(lvl, p, r) for r in rs]
    polyline!(c, collect(rs), eta, BATHLINE; rad=1)
    polyline!(c, collect(-rs), eta, BATHLINE; rad=1)
    segment!(c, RMIN, 0.0, RMAX, 0.0, GREY; rad=0)     # undisturbed level

    # droplet: very light fill, then outline
    ths = range(0, π; length=280)
    rr = [forward_map_r(lvl.drop.beta, th, p.L) for th in ths]
    zz = [lvl.com.z - forward_map_zd(lvl.drop.beta, th, p.L) for th in ths]
    poly_r = vcat(rr, reverse(-rr), [rr[1]])
    poly_z = vcat(zz, reverse(zz), [zz[1]])
    fillpoly!(c, poly_r, poly_z, DROPFILL)
    polyline!(c, poly_r, poly_z, DROPLINE; rad=1)

    # contact patch and the pressure inset
    theta_c = lvl.X === nothing ? 0.0 : lvl.X[end]
    if theta_c > 0
        thc = range(0, theta_c; length=90)
        rc = [forward_map_r(lvl.drop.beta, th, p.L) for th in thc]
        zc = [lvl.com.z - forward_map_zd(lvl.drop.beta, th, p.L) for th in thc]
        polyline!(c, rc, zc, CONTACT; rad=2); polyline!(c, -rc, zc, CONTACT; rad=2)

        chat = lvl.X[1:p.N+1]; xc = cos(theta_c)
        xs = range(xc, 1.0; length=110)
        pv = [pressure_poly_raw(chat, xc, x) for x in xs]
        prs = [forward_map_r(lvl.drop.beta, acos(clamp(x, -1, 1)), p.L) for x in xs]
        rcm = max(maximum(prs), 1e-9)
        pmin, pmax = min(minimum(pv), 0.0), max(maximum(pv), 1e-12)
        z_of_p = inset_axes!(c, rcm, pmin, pmax)
        polyline!(c, IX0 .+ IW .* prs ./ rcm, [z_of_p(v) for v in pv], CONTACT; rad=1)
        ia, ja = px_of(IX0, IZ0)
        text!(c, ia + 26, ja, "f=" * @sprintf("%.4f", f), CONTACT; scale=2)
    end

    text!(c, 16, 24, "t=" * @sprintf("%.3f", lvl.t), BLACK; scale=3)
    text!(c, 56, 24, "th_c=" * @sprintf("%.4f", theta_c), BLACK; scale=2)
    text!(c, 80, 24, "r_c=" * @sprintf("%.4f", sin(theta_c)), BLACK; scale=2)
    text!(c, 104, 24, "v=" * @sprintf("%.4f", lvl.com.v), BLACK; scale=2)
    return c
end

function main()
    We = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0958
    M = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 120
    L = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 120
    N = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 6
    nq = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 60
    mediadir = joinpath(@__DIR__, "..", "output", "media")
    mkpath(mediadir)
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(mediadir, "impact_We$(We).mp4")
    p = Params(We=We, Bo=0.017, Oh=0.006, M=M, L=L, N=N, b=6.0, h0=3.0, nq=nq)
    println("running We=$We at M=$M L=$L N=$N nq=$nq ...")
    levels, diag = run_simulation(p; t_end=8.0, dt_init=1e-3)
    fmap = Dict(round(d.t, digits=12) => d.f for d in diag)
    println("  $(length(levels)) levels, $(length(diag)) contact steps")

    # resample uniformly in time so the video plays at constant physical rate
    nframes = 400
    tgrid = range(levels[1].t, levels[end].t; length=nframes)
    tl = [lv.t for lv in levels]

    open(`ffmpeg -y -loglevel error -f image2pipe -vcodec ppm -r 30 -i - -vcodec libx264
          -pix_fmt yuv420p -crf 20 $out`, "w") do io
        for (n, t) in enumerate(tgrid)
            lvl = levels[argmin(abs.(tl .- t))]
            f = get(fmap, round(lvl.t, digits=12), 0.0)
            c = frame(lvl, p, f)
            write(io, "P6\n$W $H\n255\n")
            buf = Vector{UInt8}(undef, W * H * 3)
            k = 1
            for i in 1:H, j in 1:W, ch in 1:3
                buf[k] = c.px[i, j, ch]; k += 1
            end
            write(io, buf)
            n % 50 == 0 && println("  frame $n/$nframes")
        end
    end
    println("wrote $out")
end

main()
