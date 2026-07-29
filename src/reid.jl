# Reid (1960) exact linear viscous drop oscillation: damping rates and frequencies valid at
# ARBITRARY Ohnesorge number, replacing Lamb's (1881) small-viscosity asymptotics.
#
# WHY. The drop mode equation used throughout this package (and, as it happens, in
# DropRebound.jl / DropSolver's `residual.jl` too) is
#
#     beta_ddot_l + 2*lambda_l*beta_dot_l + omega_l^2*beta_l = F_l * b_l,
#     lambda_l = Oh*(l-1)*(2l+1),        omega_l^2 = l*(l-1)*(l+2),
#
# whose damping is Lamb's small-viscosity correction. It is asymptotic in Oh, and its error
# grows with BOTH Oh and l. Measured against Reid's exact roots (verified below in
# test/test_reid.jl), Lamb OVERPREDICTS the damping rate by:
#
#     Oh = 0.006 (production):   4.1% at l=2,  11.9% at l=16,  18.2% at l=60,  22.9% at l=120
#     Oh = 0.1:                 22.9% at l=2,  41.4% at l=4,   64.3% at l=8,   97.2% at l=16
#     Oh = 0.3:                 36.7% at l=2
#
# The frequency error is much smaller (<= 2.3% at Oh = 0.006 up to l = 120) but reaches
# 12% at Oh = 0.3 and 37% at Oh = 0.5. So even at OUR OWN small Oh the high-l modes are
# over-damped by roughly a fifth, and at Oh ~ 0.1 upward Lamb is qualitatively wrong.
#
# THEORY. Reid reduces the linear viscous drop to a single transcendental characteristic
# equation (equivalent to Chandrasekhar's 1959 self-gravitating globe; the restoring force
# enters only through the inviscid frequency). With `sigma` a DECAY rate, i.e. perturbations
# ~ exp(-sigma*t),
#
#     alpha^4/q^4 + 1 = (2(l-1)/q^2) * [ l + (l+1)*(q - 2l*Q(q))/(q - 2*Q(q)) ],
#     q^2 = sigma*R^2/nu,   alpha^2 = omega_{l,0}*R^2/nu,   Q(q) = j_{l+1}(q)/j_l(q),
#
# and in this package's nondimensionalisation (lengths by R, times by sqrt(rho R^3/sigma),
# so that nu -> Oh) this is `q^2 = sigma/Oh`, `alpha^2 = omega_{l,0}/Oh`, with
# `omega_{l,0} = sqrt(l(l-1)(l+2))`. Derivation worked out at length in
# km-viscous-drop/docs/reid1960_expanded-3.tex; Molacek & Bush (2012) draw their A_m, D_m
# from the same equation.
#
# WHAT THIS DOES AND DOES NOT GIVE. Reid's equation has infinitely many roots per `l`: a
# least-damped oscillatory pair plus a sequence of more strongly damped (eventually
# non-oscillatory) viscous roots. We track the least-damped pair and report it as
# `(lambda_l, omega_l)`. Two honest limitations follow:
#
#   1. At arbitrary Oh the FORCED response is not a second-order ODE -- it carries memory
#      from the discarded viscous spectrum. Substituting exact eigenvalues into a
#      second-order oscillator is exact for FREE decay and an approximation under forcing.
#      This is the same approximation Molacek & Bush make, and it is a genuine improvement
#      over Lamb (right eigenvalues instead of asymptotic ones) but not the exact kernel.
#   2. Once a mode is overdamped the oscillatory pair merges onto the real axis. `reid_root`
#      reports `omega = 0` there; `reid_pole_pair` then scans the real axis and applies Vieta
#      to the TWO SLOWEST roots, reproducing both rates exactly. Derivation and the numbers
#      justifying it: derivations/reid-viscous-closure.tex, verified symbolically in
#      derivations/cas_reid_two_pole.py.

using SpecialFunctions: besselj, besseljx

"""
    sph_bessel_ratio(l, q) -> Complex

`Q_l(q) = j_{l+1}(q)/j_l(q)` by DOWNWARD recurrence, which never forms `j_l` itself.
From `j_{l-1} + j_{l+1} = ((2l+1)/q) j_l`, dividing through by `j_l`,

    Q_{l-1} = 1 / ( (2l+1)/q - Q_l ),

seeded at high order with the small-argument limit `Q_n -> q/(2n+3)`.

This replaces a scaled-Bessel evaluation (`besseljx(l+3/2,q)/besseljx(l+1/2,q)`) and is
strictly more robust: `besselj` OVERFLOWS at small `Oh` (where `|q| ~ 50` with large
imaginary part) and UNDERFLOWS at large order with small argument -- the latter silently
produced a spurious `q* = 0.271` at `l = 120` during development. The recurrence agrees with
the scaled-Bessel reference to 1e-13..1e-16 wherever that reference works at all.
"""
function sph_bessel_ratio(l::Integer, q::Number)
    pad = max(60, l ÷ 2 + ceil(Int, abs(q)))
    n0 = l + pad + ceil(Int, abs(q))
    Q = q / (2 * n0 + 3)
    for n in n0:-1:(l+1)
        Q = 1 / ((2n + 1) / q - Q)
    end
    return Q
end

"""
    first_bessel_zero_half(l) -> Float64

First positive zero of `J_{l+1/2}`. Bisected on `besselj` directly, bracketed by
`[nu, nu + 3 nu^{1/3} + 3]` -- `J_nu` has no zero below its order, and the first zero lies
within the McMahon offset. Evaluated near `q ~ l`, where `besselj(l+1/2, l) ~ l^{-1/3}` is
well scaled, so this avoids the underflow that makes an asymptotic-only estimate unsafe.
"""
function first_bessel_zero_half(l::Integer)
    nu = l + 0.5
    lo, hi = max(nu, 1.0), nu + 3 * cbrt(nu) + 3
    fa = besselj(nu, lo)
    for _ in 1:300
        m = (lo + hi) / 2
        fm = besselj(nu, m)
        if sign(fm) == sign(fa)
            lo, fa = m, fm
        else
            hi = m
        end
    end
    return (lo + hi) / 2
end

const _QSTAR_CACHE = Dict{Int,Float64}()

"""
    reid_first_singularity(l) -> Float64

`q*(l)`: the first singularity of the characteristic equation's right-hand side, in the
`q` variable. Depends on `l` ALONE, not on `Oh`, so it is memoised.

Note which singularity this is. The poles of `Q` (zeros of `j_l`) are REMOVABLE in

    rhs = (2(l-1)/q^2) [ l + (l+1)(q - 2l Q)/(q - 2 Q) ],

since `(q-2lQ)/(q-2Q) -> l` as `Q -> infinity`. The true singularities are the zeros of
`g(q) = q - 2Q(q)`. There is exactly one below the first zero of `j_l`: `g > 0` as `q -> 0+`
(because `Q ~ q/(2l+3)`, so `g ~ q(1 - 2/(2l+3))`), `g -> -infinity` at that zero, and `g`
rises to a single maximum in between -- so bisection on `(0, jz_1)` is rigorous.

Measured: `q*/l` = 2.656, 1.972, 1.577, 1.346, 1.209, 1.134, 1.082, 1.051 for
`l = 2,4,8,16,32,60,120,240`, always just below `jz_1`.
"""
function reid_first_singularity(l::Integer)
    return get!(_QSTAR_CACHE, l) do
        jz1 = first_bessel_zero_half(l)
        hi = jz1 * (1 - 1e-10)
        gg(q) = real(complex(q, 0.0) - 2 * sph_bessel_ratio(l, complex(q, 0.0)))
        gg(hi) < 0 || return NaN
        lo, fa = 1e-8, gg(1e-8)
        for _ in 1:300
            m = (lo + hi) / 2
            fm = gg(m)
            if sign(fm) == sign(fa)
                lo, fa = m, fm
            else
                hi = m
            end
        end
        (lo + hi) / 2
    end
end

"""
    reid_char_residual(s, l, Oh) -> Complex

Residual of Reid's characteristic equation at complex decay rate `s`. Zero at an
eigenvalue. Uses the exponentially scaled `besseljx` because plain `besselj` overflows:
at small `Oh` the argument `q = sqrt(s/Oh)` has `|q| ~ 50` with a large imaginary part, so
`J_nu(q)` blows up like `exp(|Im q|)` while the RATIO stays O(1).
"""
function reid_char_residual(s::Complex, l::Integer, Oh::Real)
    om0 = sqrt(float(l) * (l - 1) * (l + 2))
    q2 = s / Oh
    q = sqrt(q2)
    a2 = om0 / Oh
    Qv = sph_bessel_ratio(l, q)
    rhs = (2 * (l - 1) / q2) * (l + (l + 1) * (q - 2l * Qv) / (q - 2 * Qv))
    return a2^2 / q2^2 + 1 - rhs
end

# Guarded evaluation: AMOS fails outside its analytic range, and an unguarded Newton step
# walks straight into it. Returns `nothing` rather than throwing.
function _reid_resid_safe(s::Complex, l::Integer, Oh::Real)
    try
        v = reid_char_residual(s, l, Oh)
        return isfinite(abs(v)) ? v : nothing
    catch
        return nothing
    end
end

"""
    reid_root(l, Oh; ...) -> (lambda, omega, resid)

Least-damped Reid eigenvalue for drop mode `l` at Ohnesorge number `Oh`, returned as decay
rate `lambda = Re(sigma) >= 0`, frequency `omega = |Im(sigma)|`, and the achieved
characteristic-equation residual (for the caller to assert on).

Seeded from Lamb, then a DAMPED, STEP-LIMITED complex Newton: the step is capped at a
fraction of `|s|` and backtracked until the residual decreases. Plain Newton overshoots into
arguments where the Bessel ratio cannot be evaluated -- this is not defensive padding, it was
the observed failure mode.
"""
function reid_root(l::Integer, Oh::Real; maxit::Int=400, tol::Float64=1e-13,
                   step_cap_frac::Float64=0.25, seed::Union{Nothing,Complex}=nothing)
    l >= 2 || throw(ArgumentError("Reid roots are defined for l >= 2, got l = $l"))
    Oh > 0 || throw(ArgumentError("Oh must be positive, got $Oh"))
    om0 = sqrt(float(l) * (l - 1) * (l + 2))
    s = seed === nothing ? complex(Oh * (l - 1) * (2l + 1), om0) : seed
    f = _reid_resid_safe(s, l, Oh)
    f === nothing && return (NaN, NaN, NaN)
    for _ in 1:maxit
        h = 1e-7 * max(abs(s), 1.0)
        fp = _reid_resid_safe(s + h, l, Oh)
        fp === nothing && break
        df = (fp - f) / h
        abs(df) < 1e-30 && break
        step = f / df
        cap = step_cap_frac * max(abs(s), 1e-3)
        abs(step) > cap && (step *= cap / abs(step))
        accepted = false
        t = 1.0
        for _ in 1:40
            cand = s - t * step
            fc = _reid_resid_safe(cand, l, Oh)
            if fc !== nothing && abs(fc) < abs(f)
                s, f, accepted = cand, fc, true
                break
            end
            t /= 2
        end
        accepted || break
        abs(step) < tol * max(abs(s), 1.0) && break
    end
    return (real(s), abs(imag(s)), abs(f))
end

"""
    lamb_eigenvalue(l, Oh) -> (lambda, omega)

Least-damped EIGENVALUE of the Lamb oscillator, which is not the same thing as its damping
coefficient. `beta_ddot + 2*c*beta_dot + w0^2*beta = 0` with `c = Oh(l-1)(2l+1)` and
`w0^2 = l(l-1)(l+2)` has roots `-c +- sqrt(c^2 - w0^2)`: for `c < w0` the eigenvalue decay
rate is `c` itself, but for `c > w0` it is `c - sqrt(c^2 - w0^2)`, which is SMALLER than `c`.

This matters because it flips the direction of the error against Reid. Comparing `c` to
Reid's eigenvalue is only fair while the mode is underdamped.
"""
function lamb_eigenvalue(l::Integer, Oh::Real)
    c = Oh * (l - 1) * (2l + 1)
    w02 = float(l) * (l - 1) * (l + 2)
    if c * c <= w02
        return (c, sqrt(w02 - c * c))
    else
        return (c - sqrt(c * c - w02), 0.0)
    end
end

"""
    reid_root_tracked(l, Oh; oh_start=1e-4, nsteps=24) -> (lambda, omega, resid)

Least-damped Reid eigenvalue obtained by CONTINUATION in `Oh`, which is what makes the
result trustworthy past the underdamping transition.

A single Lamb-seeded Newton is only reliable while the mode is comfortably underdamped.
Once `lambda` approaches `omega` the least-damped branch stops being near the Lamb point and
Newton converges to a different, more strongly damped root -- measured at `l = 16`,
`Oh = 0.3`, where a direct solve returns a purely real `lambda = 288.6` against Lamb's
`148.5`, i.e. the wrong branch.

Continuation removes the guesswork instead of guarding against it: start at an `Oh` small
enough that Lamb is genuinely asymptotic, then walk geometrically up to the target, seeding
each solve with the previous root. The branch is followed continuously, including through the
transition where the complex pair merges onto the real axis.

Returns `omega = 0` for a genuinely overdamped mode. Note that a real pair cannot be
represented exactly by our second-order oscillator: `omega2 = lambda^2` reproduces the
least-damped rate as a double root but not the second, more strongly damped real root.
"""
function reid_root_tracked(l::Integer, Oh::Real; oh_start::Real=1e-4, nsteps::Int=24)
    # If the target is already tiny, Lamb is asymptotic there and one solve suffices.
    Oh <= oh_start && return reid_root(l, Oh)
    lam, om, resid = reid_root(l, oh_start)
    isfinite(lam) || return (NaN, NaN, NaN)
    s = complex(lam, om)
    ladder = exp.(range(log(oh_start), log(Oh); length=nsteps + 1))[2:end]
    for ohv in ladder
        lam, om, resid = reid_root(l, ohv; seed=s)
        isfinite(lam) || return (NaN, NaN, NaN)
        s = complex(lam, om)
    end
    return (lam, om, resid)
end

"""
    reid_real_roots(l, Oh) -> Vector{Float64}

The real decay rates of the DOMINANT pair, ascending, or empty if the mode is still
oscillatory.

This is a bracketed search, not a scan. The theoretical structure (roots interlace with the
singularity ladder; the two slowest are the fundamental surface mode -- see
`derivations/reid-viscous-closure.tex`, and the discussion in
`km-viscous-drop/docs/reid1960_expanded-3.tex`) confines them to a single bounded interval:

    0 < s < s*(l, Oh) = Oh * q*(l)^2,

`q*` being the first singularity from `reid_first_singularity`. Verified over
`l = 2..32` and `Oh = 0.05..3`: that interval holds EXACTLY 0 real roots while the mode
oscillates and EXACTLY 2 once the pair has merged -- never 1 or 3. Every further root sits
just above its own singularity and is not wanted here.

For real `s > 0` the residual is real (`q = sqrt(s/Oh)` real, hence `Q` real), so roots come
from sign changes plus geometric bisection. An earlier revision scanned 40 000 log-spaced
points over a guessed range; this needs a few dozen evaluations in one interval whose
endpoints are known analytically.
"""
function reid_real_roots(l::Integer, Oh::Real; nsub::Int=400)
    qs = reid_first_singularity(l)
    isfinite(qs) || return Float64[]
    sstar = Oh * qs^2
    smin = 1e-9 * sstar
    ss = exp.(range(log(smin), log(sstar * (1 - 1e-9)); length=nsub))
    roots = Float64[]
    prev, prevs = nothing, 0.0
    for sv in ss
        v = _reid_resid_real(sv, l, Oh)
        if v === nothing
            prev = nothing
            continue
        end
        if prev !== nothing && sign(v) != sign(prev)
            a, b, fa = prevs, sv, prev
            for _ in 1:100
                m = sqrt(a * b)
                fm = _reid_resid_real(m, l, Oh)
                fm === nothing && break
                if sign(fm) == sign(fa)
                    a, fa = m, fm
                else
                    b = m
                end
            end
            r = sqrt(a * b)
            fr = _reid_resid_real(r, l, Oh)
            fr !== nothing && abs(fr) < 1e-6 && push!(roots, r)
        end
        prev, prevs = v, sv
    end
    return sort(roots)
end

function _reid_resid_real(sv::Real, l::Integer, Oh::Real)
    v = _reid_resid_safe(complex(float(sv), 0.0), l, Oh)
    v === nothing && return nothing
    # Off the real axis by roundoff only; if not, this point is not usable.
    abs(imag(v)) > 1e-6 * max(abs(real(v)), 1.0) && return nothing
    return real(v)
end

"""
    reid_pole_pair(l, Oh) -> (lambda, omega2, info)

The faithful two-pole coefficients for mode `l`, per
`derivations/reid-viscous-closure.tex`. Vieta on the two least-damped eigenvalues:

    lambda = -(s1 + s2)/2,     omega2 = s1*s2

which needs no case split. `info` is `:underdamped`, `:overdamped`, or `:failed`.

Underdamped: the pair is `sigma, conj(sigma)`, giving `lambda = Re(sigma)` and
`omega2 = Re^2 + Im^2`. Overdamped: the pair has merged, so the real axis is scanned and the
TWO SLOWEST roots are used -- the continued root is not reliably the slowest.

Emergent check: `omega2` should come out near `omega_{l,0}^2 = l(l-1)(l+2)`, since viscosity
damps but does not change what restores. Measured at `l=2`: `omega2 = 7.487, 7.485, 7.480` at
`Oh = 0.8, 1.0, 3.0` against `omega_{l,0}^2 = 8`.
"""
function reid_pole_pair(l::Integer, Oh::Real)
    lam, om, resid = reid_root_tracked(l, Oh)
    (isfinite(lam) && isfinite(resid) && resid < 1e-6 && lam > 0) ||
        return (NaN, NaN, :failed)
    # A merged pair leaves a residual imaginary part of order roundoff (measured ~1e-15),
    # so `om > 0` is NOT the right test -- it silently took the underdamped branch and
    # reproduced exactly the omega2 = lambda^2 error this function exists to remove. The
    # comparison must be relative to the decay rate.
    if om > 1e-8 * max(lam, 1.0)
        return (lam, lam^2 + om^2, :underdamped)          # Vieta on a conjugate pair
    end
    rs = reid_real_roots(l, Oh)
    length(rs) >= 2 || return (NaN, NaN, :failed)
    g1, g2 = rs[1], rs[2]                                 # two SLOWEST, ascending
    return ((g1 + g2) / 2, g1 * g2, :overdamped)          # Vieta on a real pair
end

"""
    drop_viscous_coeffs(L, Oh, model) -> (lambda, omega2)

Per-mode damping and squared frequency for `l = 0..L`, 1-indexed as `lambda[l+1]`, matching
the storage convention used everywhere else. Entries for `l = 0,1` are zero (never evolved).

- `model = :lamb` -- `lambda_l = Oh(l-1)(2l+1)`, `omega_l^2 = l(l-1)(l+2)`. The published
  model; asymptotic in `Oh`.
- `model = :reid` -- exact roots of Reid's characteristic equation. Falls back to Lamb for
  any mode whose root fails to converge, and warns naming the modes, rather than silently
  returning a wrong coefficient.
"""
function drop_viscous_coeffs(L::Integer, Oh::Real, model::Symbol)
    lambda = zeros(L + 1)
    omega2 = zeros(L + 1)
    fallback = Int[]
    overdamped = Int[]
    suspect = Tuple{Int,Float64}[]
    for l in 2:L
        lam_lamb = Oh * (l - 1) * (2l + 1)
        om2_lamb = float(l) * (l - 1) * (l + 2)
        if model === :lamb
            lambda[l+1], omega2[l+1] = lam_lamb, om2_lamb
        elseif model === :reid
            lam, om2, info = reid_pole_pair(l, Oh)
            if info === :failed
                lambda[l+1], omega2[l+1] = lam_lamb, om2_lamb
                push!(fallback, l)
            else
                lambda[l+1], omega2[l+1] = lam, om2
                info === :overdamped && push!(overdamped, l)
                # Emergent sanity check (see reid-viscous-closure.tex sec. 3): viscosity
                # damps but does not change the restoring force, so omega2 must stay near
                # omega_{l,0}^2. A large drift means a bad root pair, not physics.
                if !(0.2 * om2_lamb < om2 < 5 * om2_lamb)
                    push!(suspect, (l, om2 / om2_lamb))
                end
            end
        else
            throw(ArgumentError("viscous model must be :lamb or :reid, got $model"))
        end
    end
    if !isempty(fallback)
        # No maxlog: this changes which physical model a mode is using, so it should be
        # loud every time rather than once per session.
        @warn """
            Reid continuation failed for some drop modes; fell back to Lamb there. Those
            modes are NOT using the arbitrary-Oh model. Worth investigating rather than
            ignoring -- continuation is expected to succeed across the overdamping
            transition, so a failure here means something outside the tested range.
            """ modes=fallback Oh=Oh
    end
    if !isempty(overdamped)
        @info """
            Some drop modes are OVERDAMPED at this Oh (Reid pair has merged onto the real
            axis). They use Vieta on the two slowest real roots, which reproduces BOTH rates
            exactly -- not the critically damped approximation this replaced.
            """ modes=overdamped Oh=Oh
    end
    if !isempty(suspect)
        @warn """
            omega2 drifted far from the inviscid omega_{l,0}^2 for some modes. Viscosity
            should not change the restoring force, so this points at a wrong root pair rather
            than at physics. Reported as (mode, omega2/omega_{l,0}^2).
            """ suspect=suspect Oh=Oh
    end
    return lambda, omega2
end
