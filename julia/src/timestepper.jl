# The nested time stepper (design doc §subsec:newton, §subsec:summary).
#
# Each contact step is a ONE-DIMENSIONAL outer search over theta_c wrapping an inner
# Newton solve for the N+1 pressure coefficients:
#
#   OUTER  theta_c^{k+1} = argmin |T(theta)| over admissible candidates bracketing
#          theta_c^k, where T = dC/dtheta at the contact edge (eq:theta-c-argmin).
#          Candidates failing eq:check-nonintersect or eq:check-monotone-r are
#          discarded before the selector is consulted, exactly as AgueroEtAl2026 set
#          e(q) = infinity on infeasible candidates.
#   INNER  for each candidate, solve the square (N+1) Galerkin system at that fixed
#          theta_c (residual.jl). Conditioning is order unity and flat in delta.
#
# Why argmin and not bisection: T is measured to be non-monotone, so a bracket can
# straddle two roots and bisection has no guarantee of finding the physical one; argmin
# needs no sign change. Why a POINTWISE edge residual and not an integrated one: a
# selector minimising an integral over the contact patch is biased toward vanishing
# contact, since shrinking the patch shrinks the domain the residual is measured over --
# measured, its argmin is the smallest admissible theta_c, reproducing the onset
# degeneracy rather than curing it (design doc §subsubsec:contact-angle).
#
# ONSET is the one step this construction does not cover, and the treatment here is an
# initialisation choice rather than a derived rule (design doc's "Onset -- not specified
# here"): at first contact theta_c^k = 0, where eq:tangency-degenerate makes T identically
# zero for every state, so the first theta_c is seeded from the geometric crossing
# C(theta_c) = 0 instead -- non-degenerate there -- and the argmin selector takes over
# from the second contact step on.

"""`z_{cm} - ξ(θ=0) - η(r=0)`: droplet south-pole height minus bath height at the axis.
Negative means the pole has penetrated, i.e. contact."""
function gap_at_pole(bath::BathModeState, drop::DropModeState, com::COMState, L::Integer)
    return com.z - xi_of_x(drop.beta, 1.0, L) - sum(bath.a)
end

"""Advance one BDF2 step with zero pressure forcing (`c_m=b_l=f=0`): the free-flight
update. No Newton solve — `a_m,β_l,z_cm` are pure history recursions."""
function free_flight_step(hist::SimHistory, dt::Float64, p::Params)
    dtprev = hist.curr.dt
    kappa, alpha = bath_affine(hist.curr.bath, hist.prev.bath, p, dt, dtprev)
    lambda, gam = drop_affine(hist.curr.drop, hist.prev.drop, p, dt, dtprev)
    kappa_cm, mu = com_affine(hist.curr.com, hist.prev.com, p, dt, dtprev)

    a_new = alpha  # c_m ≡ 0
    beta_new = gam # b_l ≡ 0
    z_new = mu     # f ≡ 0

    adot_new = [bdf_derivative(a_new[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, dtprev) for m in 0:p.M]
    betadot_new = [bdf_derivative(beta_new[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, dtprev) for l in 0:p.L]
    v_new = bdf_derivative(z_new, hist.curr.com.z, hist.prev.com.z, dt, dtprev)

    return Level(BathModeState(a_new, adot_new), DropModeState(beta_new, betadot_new),
                 COMState(z_new, v_new), hist.curr.t + dt, dt, nothing)
end

"""`Params -> Level`: the state at `t=0`, one radius above the undisturbed bath,
falling at the nondimensional impact speed `√We` (design doc eq:com's `v(0)=-√We`)."""
function initial_level(p::Params)
    return Level(BathModeState(p.M), DropModeState(p.L), COMState(1.0, -sqrt(p.We)),
                 0.0, 0.0, nothing)
end

"""Affine coefficients for one trial `dt` (design doc §subsec:affine), computed once per
step and never re-differentiated."""
function step_affine(hist::SimHistory, dt::Float64, p::Params)
    dtprev = hist.curr.dt
    kappa, alpha = bath_affine(hist.curr.bath, hist.prev.bath, p, dt, dtprev)
    lambda, gam = drop_affine(hist.curr.drop, hist.prev.drop, p, dt, dtprev)
    kappa_cm, mu = com_affine(hist.curr.com, hist.prev.com, p, dt, dtprev)
    return kappa, alpha, lambda, gam, kappa_cm, mu
end

"""
    inner_solve(hist, dt, theta_c, chat_guess, p) -> (chat, result, am, beta, zcm, f)

Solve the square `N+1` Galerkin system (design doc eq:summary-galerkin) at FIXED
`theta_c`. Returns the state the solved pressure induces, so the caller can evaluate the
outer selector and the feasibility filters without recomputing anything.
"""
function inner_solve(hist::SimHistory, dt::Float64, theta_c::Float64,
    chat_guess::Vector{Float64}, p::Params)
    kappa, alpha, lambda, gam, kappa_cm, mu = step_affine(hist, dt, p)
    xc = cos(theta_c)
    q = contact_quad(xc, p)
    R = chat -> residual(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    result = newton_solve(R, chat_guess)
    am, beta, zcm, f = unpack_state(result.X, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    return result.X, result, am, beta, zcm, f
end

"""
    feasible_at(hist, dt, theta_c, chat_guess, p) -> (ok, payload)

Solve the inner system at `theta_c` and test the two hard feasibility conditions:
eq:check-nonintersect (surfaces must not overlap beyond the contact edge) and
eq:check-monotone-r (`r_c < r_M`). Returns the solved state so the caller need not
recompute it.
"""
function feasible_at(hist::SimHistory, dt::Float64, theta_c::Float64,
    chat_guess::Vector{Float64}, p::Params; practical_resid_tol::Float64=1e-4)
    chat, result, am, beta, zcm, f = inner_solve(hist, dt, theta_c, chat_guess, p)
    usable = result.status == Converged || result.resid_norm_hist[end] < practical_resid_tol
    usable || return false, nothing
    xc = cos(theta_c)
    check_monotone_r(beta, xc, p.L) || return false, nothing
    check_nonintersect(am, beta, zcm, theta_c, p) || return false, nothing
    return true, (; chat, result, am, beta, zcm, f, theta_c)
end

"""
    select_theta_c(hist, dt, theta_c_prev, chat_guess, p) -> payload | nothing

The outer closure: `theta_c` is the SMALLEST contact angle at which the surfaces do not
overlap beyond the contact edge,

    theta_c = inf { theta : eq:check-nonintersect and eq:check-monotone-r hold } .

This is the operational form of the Signorini complementarity pair, and the evidence for
using it in place of a tangency root is direct. Scanning candidates at a real contact
step, the non-intersection filter is violated below a threshold and satisfied above it,
and the net force `f` and the pointwise pressure BOTH change sign across that same
threshold (measured at the first contact step: `f = +3.7e-3`, `min p = +1.2` at
`theta_c = 0.05`, against `f = -1.7e-2`, `min p = -3.7` at `theta_c = 0.075`, with the
feasibility boundary between 0.03 and 0.05). Above the boundary one is forcing contact
where the surfaces would separate, which requires suction; below it they interpenetrate.
The physical solution sits exactly ON the constraint boundary -- which is what
complementarity says. Unlike the tangency residual `T`, the predicate is non-degenerate at
onset, so no special first-step rule is needed.

IMPLEMENTATION -- the feasible set is a BAND, not a half-line, so this is a bracketed
edge search and NOT a plain bisection on a monotone predicate. Measured: candidates are
infeasible at small `theta_c` (the surfaces interpenetrate beyond the edge) and infeasible
again at large `theta_c` (eq:check-monotone-r fails once the patch reaches the droplet's
widest point), with a feasible band between. An earlier version assumed monotonicity and
expanded the upper bracket geometrically on failure, which walked straight past the band
and reported no admissible angle at steps where one plainly existed. The search therefore
starts at `theta_c_prev` -- where continuity says the answer is nearby -- and walks
outward to bracket the LOWER edge of the band before refining it.
"""
function select_theta_c(hist::SimHistory, dt::Float64, theta_c_prev::Float64,
    chat_guess::Vector{Float64}, p::Params; dt_ref::Float64=1e-3,
    tol::Float64=2e-4, nwalk::Int=14, maxit::Int=30)
    theta_max = 0.95 * π / 2
    start = clamp(theta_c_prev <= 0 ? 1e-3 : theta_c_prev, 1e-4, theta_max)
    step = max(0.15 * start, 5e-3)

    ok_start, pay_start = feasible_at(hist, dt, start, chat_guess, p)

    lo_infeasible = -1.0     # largest angle known infeasible, below a feasible one
    hi_feasible = -1.0
    best = nothing

    if ok_start
        # Walk DOWN to find where feasibility is lost: that is the band's lower edge.
        hi_feasible = start; best = pay_start
        th = start
        for _ in 1:nwalk
            th -= step
            th <= 1e-4 && (lo_infeasible = 1e-4; break)
            ok, pay = feasible_at(hist, dt, th, chat_guess, p)
            if ok
                hi_feasible = th; best = pay
            else
                lo_infeasible = th; break
            end
        end
        lo_infeasible < 0 && return best      # feasible all the way down to the floor
    else
        # Walk UP to find the band at all.
        th = start
        for _ in 1:nwalk
            lo_infeasible = th
            th += step
            th > theta_max && break
            ok, pay = feasible_at(hist, dt, th, chat_guess, p)
            if ok
                hi_feasible = th; best = pay; break
            end
        end
        hi_feasible < 0 && return nothing     # no feasible angle anywhere below theta_max
    end

    # Refine the lower edge of the feasible band.
    lo, hi = lo_infeasible, hi_feasible
    for _ in 1:maxit
        hi - lo < tol && break
        mid = (lo + hi) / 2
        ok, pay = feasible_at(hist, dt, mid, chat_guess, p)
        if ok
            hi = mid; best = pay
        else
            lo = mid
        end
    end
    return best
end

"""
    onset_theta_c(hist, dt, p; nsample=400) -> Union{Nothing,Float64}

Seed for the FIRST contact step only. `T` vanishes identically at `theta_c = 0`
(eq:tangency-degenerate), so the argmin selector is uninformative there; instead locate
the geometric crossing `C(theta) = 0` of the two free-flight-advanced surfaces, which is
non-degenerate at onset. Returns `nothing` if no crossing exists (not yet in contact).
"""
function onset_theta_c(hist::SimHistory, dt::Float64, p::Params; nsample::Int=400)
    trial = free_flight_step(hist, dt, p)
    am, beta, zcm = trial.bath.a, trial.drop.beta, trial.com.z
    thetas = range(1e-4, π / 2; length=nsample)
    Cprev = C_at_theta(am, beta, zcm, first(thetas), p)
    Cprev <= 0 && return nothing          # pole not penetrating: no contact yet
    for theta in Iterators.drop(thetas, 1)
        Cnow = C_at_theta(am, beta, zcm, theta, p)
        Cnow <= 0 && return theta
        Cprev = Cnow
    end
    return nothing
end

"""
    contact_step(hist, dt, theta_c_prev, chat_guess, p) -> (level, info) | (nothing, info)

One full contact step: the outer selection of `theta_c` (design doc eq:theta-c-argmin and
the feasibility-boundary discussion in `select_theta_c`) wrapping the inner Newton solve.

Returns `nothing` if no admissible angle exists at this `dt`, which the caller treats as
a signal to reduce `dt` and retry.
"""
function contact_step(hist::SimHistory, dt::Float64, theta_c_prev::Float64,
    chat_guess::Vector{Float64}, p::Params; dt_ref::Float64=1e-3)
    best = select_theta_c(hist, dt, theta_c_prev, chat_guess, p; dt_ref=dt_ref)
    best === nothing && return nothing, (; nadmissible=0)

    dtprev = hist.curr.dt
    adot = [bdf_derivative(best.am[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, dtprev) for m in 0:p.M]
    betadot = [bdf_derivative(best.beta[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, dtprev) for l in 0:p.L]
    v = bdf_derivative(best.zcm, hist.curr.com.z, hist.prev.com.z, dt, dtprev)
    level = Level(BathModeState(best.am, adot), DropModeState(best.beta, betadot),
                  COMState(best.zcm, v), hist.curr.t + dt, dt,
                  vcat(best.chat, best.theta_c))
    T = tangency_residual(best.am, best.beta, best.zcm, best.theta_c, p)
    return level, (; nadmissible=1, theta_c=best.theta_c, T=T, f=best.f,
                   status=best.result.status, resid=best.result.resid_norm_hist[end])
end

"""
    run_simulation(p; t_end, dt_init, ...) -> (levels, diagnostics)

Free flight until the pole penetrates, then contact steps until contact is lost. Contact
is deemed lost when the net contact force `f` turns non-positive (AlventosaEtAl2023's own
criterion — they monitor `sign f`) or when no candidate `theta_c` is admissible at the
minimum step size.
"""
function run_simulation(p::Params; t_end::Float64, dt_init::Float64=1e-3,
    dt_min::Float64=1e-8, dt_max::Float64=0.02, theta_c_floor::Float64=2e-3,
    verbose::Bool=false)
    lvl0 = initial_level(p)
    hist = SimHistory(lvl0, lvl0)
    levels = Level[lvl0]
    diag = NamedTuple[]
    dt = dt_init
    phase = FreeFlight
    theta_c_prev = 0.0
    chat_guess = zeros(p.N + 1)
    nsteps = 0
    just_left = false
    while hist.curr.t < t_end && nsteps < 200_000
        nsteps += 1
        dt = min(dt, dt_max, t_end - hist.curr.t)
        dt <= 0 && break

        if phase == FreeFlight
            # `just_left` forces at least one advancing free-flight step after contact
            # ends. Without it, a step that fails admissibility flips to FreeFlight,
            # onset is immediately re-detected at the SAME t, and the loop spins with no
            # time advance -- observed directly before this guard was added.
            th = just_left ? nothing : onset_theta_c(hist, dt, p)
            just_left = false
            if th === nothing
                trial = free_flight_step(hist, dt, p)
                hist.prev = hist.curr; hist.curr = trial
                push!(levels, trial)
                dt = min(dt * 1.2, dt_max)
            else
                # Contact begins. Bisect on dt so the first contact step starts as close
                # to the true onset instant as the step floor allows, then hand the
                # geometric seed to the argmin selector from the next step on.
                theta_c_prev = th
                chat_guess = zeros(p.N + 1)
                chat_guess[1] = 1e-3
                phase = InContact
                verbose && println("  onset at t=$(round(hist.curr.t,digits=5)) theta_c=$(round(th,digits=5))")
            end
            continue
        end

        level, info = contact_step(hist, dt, theta_c_prev, chat_guess, p; dt_ref=dt_init)
        if level === nothing
            if dt > dt_min
                dt /= 2
                continue
            end
            phase = FreeFlight
            dt = dt_init
            just_left = true
            verbose && println("  contact lost (no admissible theta_c) at t=$(round(hist.curr.t,digits=5))")
            continue
        end
        # LOSS OF CONTACT. AlventosaEtAl2023 monitor sign(f), which is necessary but not
        # sufficient here: as the patch collapses, f -> 0 from ABOVE and never changes
        # sign, so an f-only test leaves the droplet nominally in contact while it climbs
        # away (observed: theta_c pinned at the search floor with f ~ 1e-7 for thousands
        # of steps). The patch collapsing IS the loss of contact, so the primary test is
        # on theta_c, with sign(f) retained as the secondary one.
        if info.f <= 0 || info.theta_c <= theta_c_floor
            phase = FreeFlight
            dt = dt_init
            just_left = true
            verbose && println("  contact lost (" * (info.f <= 0 ? "f<=0" : "patch collapsed") *
                               ") at t=$(round(hist.curr.t, digits=5))")
            continue
        end
        hist.prev = hist.curr; hist.curr = level
        push!(levels, level)
        push!(diag, merge(info, (; t=level.t, dt=dt, z=level.com.z, v=level.com.v)))
        theta_c_prev = info.theta_c
        chat_guess = level.X[1:p.N+1]
        verbose && println("  t=$(round(level.t,digits=5)) dt=$(round(dt,sigdigits=2)) " *
                           "th_c=$(round(info.theta_c,digits=4)) f=$(round(info.f,sigdigits=3)) " *
                           "v=$(round(level.com.v,digits=4))")
        dt = min(dt * 1.5, dt_max)
    end
    return levels, diag
end
