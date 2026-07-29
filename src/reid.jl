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
#   2. Once a mode is overdamped the oscillatory pair merges onto the real axis and
#      "omega_l" ceases to mean a frequency. `reid_root` reports `omega = 0` there and the
#      caller must treat the mode as non-oscillatory.

using SpecialFunctions: besseljx

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
    Qv = besseljx(l + 3 / 2, q) / besseljx(l + 1 / 2, q)
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
                   step_cap_frac::Float64=0.25)
    l >= 2 || throw(ArgumentError("Reid roots are defined for l >= 2, got l = $l"))
    Oh > 0 || throw(ArgumentError("Oh must be positive, got $Oh"))
    om0 = sqrt(float(l) * (l - 1) * (l + 2))
    s = complex(Oh * (l - 1) * (2l + 1), om0)          # Lamb seed
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
    for l in 2:L
        lam_lamb = Oh * (l - 1) * (2l + 1)
        om2_lamb = float(l) * (l - 1) * (l + 2)
        if model === :lamb
            lambda[l+1], omega2[l+1] = lam_lamb, om2_lamb
        elseif model === :reid
            lam, om, resid = reid_root(l, Oh)
            # The oscillator beta_ddot + 2*lambda*beta_dot + omega2*beta has roots
            # -lambda +- i*sqrt(omega2 - lambda^2). To reproduce Reid's eigenvalue
            # -lam +- i*om we therefore need omega2 = lam^2 + om^2, NOT om^2. (The
            # published Lamb path has the same O(lambda^2/omega^2) slip, but there it sits
            # inside its own asymptotic error; here it would be a needless one.)
            ok = isfinite(lam) && isfinite(om) && isfinite(resid) &&
                 resid < 1e-6 && lam > 0 && om > 0 &&
                 # Branch guard: in the underdamped regime Reid damping is ALWAYS below
                 # Lamb's (verified across Oh = 0.006..0.3, l = 2..16). A root that damps
                 # harder than Lamb means Newton left the least-damped branch, which is
                 # what happens once the mode approaches overdamping.
                 lam < lam_lamb
            if ok
                lambda[l+1], omega2[l+1] = lam, lam^2 + om^2
            else
                lambda[l+1], omega2[l+1] = lam_lamb, om2_lamb
                push!(fallback, l)
            end
        else
            throw(ArgumentError("viscous model must be :lamb or :reid, got $model"))
        end
    end
    if !isempty(fallback)
        @warn """
            Reid root-tracking rejected for some drop modes; fell back to Lamb there.
            This is expected for modes at or beyond overdamping, where the least-damped
            branch leaves the complex pair and the Lamb-seeded Newton no longer follows it.
            Lamb predicts overdamping once Oh > sqrt(l(l-1)(l+2)) / ((l-1)(2l+1)), i.e.
            Oh > 0.31 at l=2 but only Oh > 0.13 at l=16 -- so high modes hit this first.
            Underdamped modes are unaffected and use exact Reid coefficients.
            """ modes=fallback Oh=Oh maxlog=1
    end
    return lambda, omega2
end
