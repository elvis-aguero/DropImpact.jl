# The Phase (FreeFlight/InContact) state machine: free-flight BDF2 advance with zero
# pressure forcing, event-located onset/loss transitions, and the Newton-based
# in-contact step, unified behind a single halve-dt-and-retry loop for all three
# failure modes (non-convergence, inadmissible, discontinuous) per the design's
# architecture review.

"""`z_{cm} - ξ(θ=0) - η(r=0)`: droplet south-pole height minus bath height at the
axis. Contact begins when this crosses zero (matching how Alventosa et al. detect
contact from the geometric gap, not from a pressure/force sign, since there is no
pressure to speak of yet during free flight)."""
function gap_at_pole(bath::BathModeState, drop::DropModeState, com::COMState, L::Integer)
    eta0 = sum(bath.a)                      # Σ_m a_m J0(0), J0(0)=1
    xi0 = xi_of_theta(drop.beta, 0.0, L)    # ξ(θ=0)
    return com.z - xi0 - eta0
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

    bath_new = BathModeState(a_new, adot_new)
    drop_new = DropModeState(beta_new, betadot_new)
    com_new = COMState(z_new, v_new)
    return Level(bath_new, drop_new, com_new, hist.curr.t + dt, dt, nothing)
end

"""`Params -> Level`: the state at `t=0`, one radius above the undisturbed bath,
falling at the nondimensional impact speed `√We` (design doc eq:com's initial
condition `v(0)=-√We`)."""
function initial_level(p::Params)
    bath = BathModeState(p.M)
    drop = DropModeState(p.L)
    com = COMState(1.0, -sqrt(p.We))
    return Level(bath, drop, com, 0.0, 0.0, nothing)  # dt=0 sentinel: no prior step
end

"""Residual closure capturing the (plain Float64) affine coefficients for one trial
`dt`, per design doc §subsec:affine/subsec:newton: these are computed once per
timestep and never re-differentiated. Also captures `hist.curr.bath`/`hist.curr.drop`
as the FROZEN τ^k state the acceleration-level Galerkin rows extrapolate from
(design doc §subsec:accel-galerkin) — the previous converged step, i.e. zeroth-order
extrapolation; see accel_closure.jl's module note."""
function build_residual(hist::SimHistory, dt::Float64, p::Params)
    dtprev = hist.curr.dt
    kappa, alpha = bath_affine(hist.curr.bath, hist.prev.bath, p, dt, dtprev)
    lambda, gam = drop_affine(hist.curr.drop, hist.prev.drop, p, dt, dtprev)
    kappa_cm, mu = com_affine(hist.curr.com, hist.prev.com, p, dt, dtprev)
    R = X -> residual(X, kappa, alpha, lambda, gam, kappa_cm, mu, hist.curr.bath, hist.curr.drop, p)
    return R, (kappa, alpha, lambda, gam, kappa_cm, mu)
end

"""Positivity/non-intersection admissibility (design doc eq:check-positivity/
eq:check-nonintersect), checked post-hoc against a converged Newton solution — a
pragmatic, sampled version (a handful of points across the pressed/unpressed regions)
rather than an exhaustive symbolic check, consistent with the design doc's own
framing that this is a property to examine numerically, not enforce structurally."""
function check_admissible(X::Vector{Float64}, am::Vector{Float64}, beta::Vector{Float64}, zcm::Float64, p::Params; nsample::Int=9)
    N = p.N
    chat = X[1:N+1]
    theta_c = X[end]
    xc = cos(theta_c)
    positivity_ok = true
    for s in range(0.0, 1.0; length=nsample)
        x = xc + s * (1 - xc)
        if pressure_poly_raw(chat, xc, x) < -1e-8
            positivity_ok = false
            break
        end
    end
    nonintersect_ok = true
    for theta in range(theta_c, pi; length=nsample)[2:end]
        eta_val = sum(am[m+1] * besselj0(p.k[m+1] * forward_map_r(beta, theta, p.L)) for m in 0:p.M)
        gap = eta_val - (zcm + forward_map_zd(beta, theta, p.L))
        if gap > 1e-8
            nonintersect_ok = false
            break
        end
    end
    return positivity_ok, nonintersect_ok
end

"""
    contact_step(hist, dt, X_guess, p) -> (Level_or_nothing, NewtonResult, admissible)

One in-contact step: build the residual at trial `dt`, Newton-solve for `X`, and
(if converged) unpack the resulting state and check admissibility. Returns
`(nothing, result, (false,false))` if Newton itself failed to converge.
"""
function contact_step(hist::SimHistory, dt::Float64, X_guess::Vector{Float64}, p::Params;
    practical_resid_tol::Float64=1e-4)
    R, (kappa, alpha, lambda, gam, kappa_cm, mu) = build_residual(hist, dt, p)
    result = newton_solve(R, X_guess)
    # A `Stalled` result with small residual is accepted as practically usable rather
    # than rejected outright: given the confirmed severe Jacobian ill-conditioning
    # (newton.jl module note), demanding formal `Converged` status is unrealistic,
    # most acutely right at contact onset where the true contact patch has zero width.
    # This is a disclosed, logged relaxation, not a silent one — callers can inspect
    # `result.status` and `result.resid_norm_hist` directly.
    usable = result.status == Converged ||
             (result.status == Stalled && result.resid_norm_hist[end] < practical_resid_tol)
    if !usable
        return nothing, result, (false, false)
    end
    am, beta, zcm = unpack_state(result.X, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    pos_ok, nonint_ok = check_admissible(result.X, am, beta, zcm, p)

    adot = [bdf_derivative(am[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, hist.curr.dt) for m in 0:p.M]
    betadot = zeros(p.L + 1)
    for l in 2:p.L
        betadot[l+1] = bdf_derivative(beta[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, hist.curr.dt)
    end
    v = bdf_derivative(zcm, hist.curr.com.z, hist.prev.com.z, dt, hist.curr.dt)

    lvl = Level(BathModeState(am, adot), DropModeState(beta, betadot), COMState(zcm, v), hist.curr.t + dt, dt, result.X)
    return lvl, result, (pos_ok, nonint_ok)
end

"""
    edge_contact_step(hist, dt, Y_guess, p) -> (Level_or_nothing, NewtonResult, admissible)

PRIMARY in-contact step (2026-07-27), replacing both the original joint `contact_step`
(broken tangency) and the interim `kinematic_contact_step` (decoupled θ_c, which sidestepped
tangency's degeneracy but never let real pressure history build up — see `STATUS.md`).
Jointly Newton-solves `Y=(ĉ_1,...,ĉ_N,θ_c)` via `residual_edge` — the pressure-vanishing
edge condition `ĉ_0≡0` in place of tangency — restoring a genuinely simultaneous,
non-degenerate solve for pressure and contact radius together, as originally intended.
"""
function edge_contact_step(hist::SimHistory, dt::Float64, Y_guess::Vector{Float64}, p::Params;
    practical_resid_tol::Float64=1e-4)
    R, (kappa, alpha, lambda, gam, kappa_cm, mu) = build_residual(hist, dt, p)
    Redge(Y) = residual_edge(Y, kappa, alpha, lambda, gam, kappa_cm, mu, hist.curr.bath, hist.curr.drop, p)
    result = newton_solve(Redge, Y_guess)
    usable = result.status == Converged ||
             (result.status == Stalled && result.resid_norm_hist[end] < practical_resid_tol)
    if !usable
        return nothing, result, (false, false)
    end
    X = unpack_Y_edge(result.X, p.N)
    am, beta, zcm = unpack_state(X, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    pos_ok, nonint_ok = check_admissible(X, am, beta, zcm, p)

    adot = [bdf_derivative(am[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, hist.curr.dt) for m in 0:p.M]
    betadot = zeros(p.L + 1)
    for l in 2:p.L
        betadot[l+1] = bdf_derivative(beta[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, hist.curr.dt)
    end
    v = bdf_derivative(zcm, hist.curr.com.z, hist.prev.com.z, dt, hist.curr.dt)

    lvl = Level(BathModeState(am, adot), DropModeState(beta, betadot), COMState(zcm, v), hist.curr.t + dt, dt, X)
    return lvl, result, (pos_ok, nonint_ok)
end

"""Warm start for `edge_contact_step`'s reduced `Y=(ĉ_1,...,ĉ_N,θ_c)` unknown. Falls back
to a small generic seed if no prior contact solution exists yet."""
function warm_start_edge(prev_X::Union{Nothing,Vector{Float64}}, N::Integer)
    if prev_X !== nothing
        return [prev_X[2:N+1]; prev_X[end]]
    end
    return [zeros(N); 0.05]
end

function warm_start_edge_extrapolated(t_target::Float64, X_a::Union{Nothing,Vector{Float64}}, t_a::Float64,
    X_b::Union{Nothing,Vector{Float64}}, t_b::Float64, N::Integer)
    if X_b === nothing
        return warm_start_edge(nothing, N)
    end
    if X_a === nothing || t_b <= t_a
        return warm_start_edge(X_b, N)
    end
    Yb = warm_start_edge(X_b, N)
    Ya = warm_start_edge(X_a, N)
    frac = (t_target - t_b) / (t_b - t_a)
    Y = Yb .+ frac .* (Yb .- Ya)
    Y[end] = max(Y[end], 1e-8)
    return Y
end

"""
    gap_at_theta(theta, z_cm_est, bath_frozen, drop_frozen, p) -> value

Pointwise penetration gap at polar angle `θ`, using the FROZEN (previous-step) bath/drop
shape: `(z_cm_est - z_d(θ;β_frozen)) - η(r(θ);a_frozen)`. Negative where the droplet
surface has penetrated below the bath (in contact), positive where it hasn't — the same
sign convention as `gap_at_pole`/`check_admissible`. `θ_c` is the (assumed unique) zero
crossing from negative to positive as `θ` increases from `0`.
"""
function gap_at_theta(theta, z_cm_est, bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params)
    zd = forward_map_zd(drop_frozen.beta, theta, p.L)
    r = forward_map_r(drop_frozen.beta, theta, p.L)
    eta = sum(bath_frozen.a[m+1] * besselj0(p.k[m+1] * r) for m in 0:p.M)
    return (z_cm_est - zd) - eta
end

"""
    theta_c_kinematic(z_cm_est, bath_frozen, drop_frozen, p; ngrid=200) -> Float64 or nothing

Contact-radius determination DECOUPLED from the pressure solve, replacing the joint
`(N+2)`-dim Newton solve for `(ĉ,θ_c)`: found to have `θ_c=0` as an exact, degenerate
root of the tangency condition for ANY pressure (see module note on
`is_cold_start`/history below), a coordinate-singularity artifact (`∂_θC` is odd in `θ`
near the axis for any smooth axisymmetric state — `C` itself is even, since `r(θ)` is odd
and `η` is even in `r`, and `ξ(θ)cosθ` is even — so it vanishes at `θ=0` by symmetry,
independent of amplitude, and no amount of further `τ`- or `θ`-differentiation in the
style of §4/`accel_closure.jl` removes this; it is a different, deeper obstruction than
the `O(δ²)` conditioning issue Section 4 fixes).

Root-finds the purely KINEMATIC gap condition `gap_at_theta(θ,z_cm_est,...)=0` via
bisection over a fine grid, using the FROZEN spectral bath/drop state (still genuinely
spectral — `a_m,β_l` are real Fourier–Bessel/Legendre coefficients, not a mesh) rather
than an undeformed-sphere assumption: as real surface deformation accumulates in
`bath_frozen,drop_frozen`, this gives a self-limiting contact patch, unlike a naive
undeformed-geometry estimate (found empirically to let `θ_c` grow past the small-
deformation regime and diverge). Returns `nothing` if no sign change is found on
`(0,π)` (not currently in geometric contact at all, given this shape/estimate).
"""
function theta_c_kinematic(z_cm_est, bath_frozen::BathModeState, drop_frozen::DropModeState, p::Params;
    theta_guess::Union{Nothing,Float64}=nothing, ngrid::Int=200)
    bisect(lo, hi) = begin
        for _ in 1:30  # 2^-30 of the bracket width is far below any physically meaningful θ_c precision
            mid = (lo + hi) / 2
            gm = gap_at_theta(mid, z_cm_est, bath_frozen, drop_frozen, p)
            gm <= 0 ? (lo = mid) : (hi = mid)
        end
        return (lo + hi) / 2
    end

    # Fast path: bracket locally around the previous step's θ_c first (it barely moves
    # step to step outside genuine onset/liftoff transients) — avoids paying for a full
    # ngrid-point scan on every Newton trial once a good guess is available.
    if theta_guess !== nothing && theta_guess > 1e-6
        window = max(0.05, 0.5 * theta_guess)
        lo0 = max(1e-6, theta_guess - window)
        hi0 = min(pi - 1e-6, theta_guess + window)
        g_lo = gap_at_theta(lo0, z_cm_est, bath_frozen, drop_frozen, p)
        g_hi = gap_at_theta(hi0, z_cm_est, bath_frozen, drop_frozen, p)
        if g_lo <= 0 && g_hi > 0
            return bisect(lo0, hi0)
        end
    end

    thetas = range(1e-6, pi - 1e-6; length=ngrid)
    g_prev = gap_at_theta(thetas[1], z_cm_est, bath_frozen, drop_frozen, p)
    g_prev > 0 && return nothing  # not even penetrating at the pole
    for i in 2:ngrid
        g = gap_at_theta(thetas[i], z_cm_est, bath_frozen, drop_frozen, p)
        if g > 0
            return bisect(thetas[i-1], thetas[i])
        end
        g_prev = g
    end
    return thetas[end]  # fully engulfed within this grid's resolution
end

"""
    rate_limit_theta_c(theta_new, theta_prev, dt; max_step=0.05) -> Float64

STOPGAP REGULARIZATION, not a derived or validated part of the model: caps how much
`θ_c` can change in a single step. Added because `theta_c_kinematic`, even with the
fixed-point closure in `kinematic_contact_step`, was still found to let `θ_c` grow
unboundedly (past `π`) before the underlying pressure/force chain could catch up — the
COM force `f` stayed small (often even wrong-signed) throughout, well past the small-
deformation regime this model assumes, for reasons not fully diagnosed (see
`STATUS.md`). This bound does not fix that; it only prevents the resulting blow-up from
crashing the solver, so a bounded (if not fully validated) trajectory can be produced.
`max_step=0.05` is an arbitrary, undisclosed-elsewhere constant — tune or remove once
the underlying force-magnitude issue is actually root-caused.
"""
function rate_limit_theta_c(theta_new, theta_prev::Union{Nothing,Float64}, dt::Float64;
    max_step::Float64=0.05, theta_c_max::Float64=1.2)
    # theta_c_max: a hard ceiling at the outer edge of this model's own stated small-
    # deformation regime (design doc §subsec:formulation) — found necessary because the
    # gradual march toward θ_c=π (full engulfment) that motivated this function was NOT
    # a discontinuous jump (rate-limiting alone does not catch it: ~0.0016 rad/step
    # average, far under max_step) but a steady drift past where this model's own
    # premises still apply. Also a stopgap, same caveats as the module note above.
    theta_prev === nothing && return min(theta_new, theta_c_max)
    limited = clamp(theta_new, theta_prev - max_step, theta_prev + max_step)
    return min(limited, theta_c_max)
end

"""
    kinematic_contact_step(hist, dt, chat_guess, p) -> (Level_or_nothing, NewtonResult_or_nothing, admissible)

PRIMARY in-contact step, replacing the joint `(N+2)`-dim Newton solve entirely (not just
as a cold-start bootstrap — see `theta_c_kinematic`'s module note for why the joint solve
is not salvageable near θ_c=0, a coordinate-singularity triviality rather than a
conditioning defect). `θ_c` is fixed via `theta_c_kinematic`, then the well-posed, reduced
`(N+1)`-dim Galerkin subsystem is solved for `ĉ` — pressure remains a genuine spectral
unknown, not an assumed shape; only the free boundary's determination is decoupled from
it, which the tangency-degeneracy finding shows is unavoidable in this formulation.

The two are then iterated to a FIXED POINT (a handful of passes, not a new Newton
unknown): using the pressure-free projection `μ` for `z_cm` in `theta_c_kinematic` gives
`θ_c` no feedback at all from the pressure it's about to induce, so it just tracks an
unresisted free fall — found empirically to grow unboundedly (past `π`, fully engulfing
the drop, then diverging) since the deceleration a real contact patch would produce never
enters the estimate. Recomputing `θ_c` from the ACTUAL `z_cm=μ+κ_{cm}f(ĉ)` after each
pressure solve, and re-solving `ĉ` at the updated `θ_c`, closes this loop.
"""
function kinematic_contact_step(hist::SimHistory, dt::Float64, chat_guess::Vector{Float64}, p::Params;
    max_fixedpoint_iter::Int=8, fixedpoint_tol::Float64=1e-6)
    R, (kappa, alpha, lambda, gam, kappa_cm, mu) = build_residual(hist, dt, p)
    theta_prev = hist.curr.X === nothing ? nothing : hist.curr.X[end]
    theta_c = theta_c_kinematic(mu, hist.curr.bath, hist.curr.drop, p; theta_guess=theta_prev)
    if theta_c === nothing || theta_c < 1e-8
        return nothing, nothing, (false, false)
    end
    theta_c = rate_limit_theta_c(theta_c, theta_prev, dt)
    local result
    chat = chat_guess
    for _ in 1:max_fixedpoint_iter
        Rgal(c) = R([c; theta_c])[1:p.N+1]
        result = newton_solve(Rgal, chat)
        usable = result.status == Converged ||
                 (result.status == Stalled && result.resid_norm_hist[end] < 1e-4)
        !usable && return nothing, result, (false, false)
        chat = result.X
        xc = cos(theta_c)
        bl = b_l_all(chat, xc, p.L)
        beta = gam .+ lambda .* bl
        f = com_force_closed(chat, xc, beta, p.L, p.com_nodes, p.com_weights)
        zcm = mu + kappa_cm * f
        theta_c_new = theta_c_kinematic(zcm, hist.curr.bath, hist.curr.drop, p; theta_guess=theta_c)
        (theta_c_new === nothing || theta_c_new < 1e-8) && return nothing, result, (false, false)
        theta_c_new = rate_limit_theta_c(theta_c_new, theta_prev, dt)
        abs(theta_c_new - theta_c) < fixedpoint_tol && (theta_c = theta_c_new; break)
        theta_c = theta_c_new
    end
    Rgal_final(c) = R([c; theta_c])[1:p.N+1]
    result = newton_solve(Rgal_final, chat)
    usable = result.status == Converged ||
             (result.status == Stalled && result.resid_norm_hist[end] < 1e-4)
    if !usable
        return nothing, result, (false, false)
    end
    Xfull = [result.X; theta_c]
    am, beta, zcm = unpack_state(Xfull, kappa, alpha, lambda, gam, kappa_cm, mu, p)
    pos_ok, nonint_ok = check_admissible(Xfull, am, beta, zcm, p)

    adot = [bdf_derivative(am[m+1], hist.curr.bath.a[m+1], hist.prev.bath.a[m+1], dt, hist.curr.dt) for m in 0:p.M]
    betadot = zeros(p.L + 1)
    for l in 2:p.L
        betadot[l+1] = bdf_derivative(beta[l+1], hist.curr.drop.beta[l+1], hist.prev.drop.beta[l+1], dt, hist.curr.dt)
    end
    v = bdf_derivative(zcm, hist.curr.com.z, hist.prev.com.z, dt, hist.curr.dt)

    lvl = Level(BathModeState(am, adot), DropModeState(beta, betadot), COMState(zcm, v), hist.curr.t + dt, dt, Xfull)
    return lvl, result, (pos_ok, nonint_ok)
end

"""`true` while the joint Newton solve for `θ_c` would still be degenerate: history too
small for the BDF2 intercepts `α_m,γ_l` to give tangency genuine `O(1)` sensitivity to
`a_m`. Kept for diagnostic/reference use; `run_simulation` no longer switches on this
since it uses `kinematic_contact_step` throughout (see that function's module note —
the degeneracy turned out to be a coordinate-singularity triviality, not something a
cold-start/full-solve handoff can outgrow)."""
function is_cold_start(bath::BathModeState, drop::DropModeState; tol::Float64=1e-8)
    return sum(abs, bath.a) + sum(abs, drop.beta) < tol
end

"""Warm start for the Newton solve. With two prior converged solutions and their
time levels, linearly extrapolate in `t` rather than reusing the last `X` verbatim —
important near contact onset, where `θ_c` grows rapidly (Wagner-theory-type `√t`
growth) and even a modest step-size change can leave the stale `X` a poor match,
exactly the refinement the architecture review flagged. Falls back to reusing the
last `X`, or (no prior contact solution at all) a small, generic seed."""
function warm_start(prev_X::Union{Nothing,Vector{Float64}}, N::Integer)
    if prev_X !== nothing
        return copy(prev_X)
    end
    chat0 = zeros(N + 1)
    chat0[1] = 0.05
    return [chat0; 0.05]
end

function warm_start_extrapolated(t_target::Float64, X_a::Union{Nothing,Vector{Float64}}, t_a::Float64,
    X_b::Union{Nothing,Vector{Float64}}, t_b::Float64, N::Integer)
    if X_b === nothing
        return warm_start(nothing, N)
    end
    if X_a === nothing || t_b <= t_a
        return copy(X_b)
    end
    frac = (t_target - t_b) / (t_b - t_a)
    X = X_b .+ frac .* (X_b .- X_a)
    X[end] = max(X[end], 1e-8)  # theta_c must stay positive
    return X
end

"""Warm start for `kinematic_contact_step`'s reduced `(N+1)`-dim pressure solve
(`chat` only — `θ_c` is no longer a Newton unknown, so there is nothing analogous to
`warm_start_extrapolated`'s `θ_c`-extrapolation to do here)."""
function warm_start_chat(prev_chat::Union{Nothing,Vector{Float64}}, N::Integer)
    return prev_chat !== nothing ? copy(prev_chat) : zeros(N + 1)
end

"""
    run_simulation(p; t_end, dt_init, dt_min=1e-10, dt_max=Inf, maxsteps=200000) -> (times, levels, phases)

The main driver: free flight until the geometric gap (`gap_at_pole`) crosses zero
(located by bisection so the transition step lands near the true onset time), then
Newton-solved contact steps until the post-hoc positivity check fails (located the
same way for liftoff), alternating for as many bounces as `t_end` allows. All three
failure modes (Newton non-convergence, inadmissible solution, and — as a placeholder
for the disclosed spurious-root risk — a crude continuity check on `θ_c`) get the
same response: halve `dt` and retry, down to `dt_min` before giving up on the step.
"""
function run_simulation(p::Params; t_end::Float64, dt_init::Float64, dt_min::Float64=1e-10,
    dt_max::Float64=Inf, maxsteps::Int=200_000)
    lvl0 = initial_level(p)
    hist = SimHistory(lvl0, lvl0)
    times = Float64[0.0]
    levels = Level[lvl0]
    phases = Phase[InContact]

    dt = dt_init
    # z_cm(0)=1, ξ(0)≈1: the south pole starts exactly at the undisturbed bath surface
    # (design doc eq:com's initial condition IS the moment of first touch, not a height
    # to fall from), so the simulation begins in contact, not free flight.
    phase = InContact
    nsteps = 0
    stuck_count = 0  # consecutive InContact-total-failure -> FreeFlight-immediate-flip-back cycles

    while hist.curr.t < t_end && nsteps < maxsteps
        nsteps += 1
        dt = min(dt, dt_max, t_end - hist.curr.t)

        if phase == FreeFlight
            g_prev = gap_at_pole(hist.curr.bath, hist.curr.drop, hist.curr.com, p.L)
            if g_prev <= 0
                stuck_count += 1
                if stuck_count > 20
                    # Safety valve: InContact has completely failed to make progress
                    # this many cycles in a row with no genuine liftoff resolved in
                    # between. NOTE: unlike Agüero et al.'s own adaptive scheme (which is
                    # refinement-ONLY — it starts from a fixed 5000-point time vector and
                    # only ever INSERTS a finer intermediate step when r_c jumps by more
                    # than δr, Deformable_impactors.tex Algorithm 1/§Numerical
                    # implementation — it never grows dt, and the paper says nothing
                    # about detachment being favored/disfavored by step size), this is an
                    # original escalation heuristic for THIS solver's specific failure
                    # mode: force a GROWN free-flight step rather than a tiny dt_min one,
                    # on the empirical (not paper-derived) reasoning that a larger step
                    # gives a genuine liftoff more room to actually clear the gap, instead
                    # of repeating the same near-threshold probe.
                    trial = free_flight_step(hist, dt, p)
                    hist.prev = hist.curr
                    hist.curr = trial
                    push!(times, trial.t); push!(levels, trial); push!(phases, FreeFlight)
                    stuck_count = 0
                    continue
                end
                # Already at/below contact height (e.g. a numerical contact failure
                # just fell back to free flight, not a genuine liftoff): re-attempt
                # contact immediately from the current state, no time advance needed.
                # dt escalates geometrically with consecutive stuck cycles instead of
                # resetting to dt_init every cycle, which otherwise traps the solver
                # retrying the identical small step forever at a marginal θ_c≈0
                # threshold (empirically confirmed, and fixed by this escalation: cf.
                # task #21). This escalation is an original heuristic for this solver,
                # NOT drawn from Agüero et al.'s scheme, which is refinement-only (see
                # the module note above and the safety-valve comment below it).
                phase = InContact
                dt = min(dt_init * 2.0^stuck_count, dt_max)
                continue
            end
            trial = free_flight_step(hist, dt, p)
            g_new = gap_at_pole(trial.bath, trial.drop, trial.com, p.L)
            if g_new <= 0
                # locate the crossing by bisection on dt
                lo, hi = 0.0, dt
                for _ in 1:40
                    mid = (lo + hi) / 2
                    tmid = free_flight_step(hist, mid, p)
                    gm = gap_at_pole(tmid.bath, tmid.drop, tmid.com, p.L)
                    if gm > 0
                        lo = mid
                    else
                        hi = mid
                    end
                end
                trial = free_flight_step(hist, hi, p)
                hist.prev = hist.curr
                hist.curr = trial
                push!(times, trial.t); push!(levels, trial); push!(phases, FreeFlight)
                phase = InContact
                dt = dt_init
            else
                hist.prev = hist.curr
                hist.curr = trial
                push!(times, trial.t); push!(levels, trial); push!(phases, FreeFlight)
                dt = min(dt * 1.1, dt_max)
            end
        else # InContact
            dt_try = dt
            accepted = false
            best_lvl = nothing        # best attempt seen (newton_ok && admis[1]), for a forced-progress fallback
            best_dt_try = dt_try
            local lvl, result, admis
            while dt_try >= dt_min
                Y_guess = warm_start_edge_extrapolated(hist.curr.t + dt_try, hist.prev.X, hist.prev.t, hist.curr.X, hist.curr.t, p.N)
                lvl, result, admis = edge_contact_step(hist, dt_try, Y_guess, p)
                newton_ok = lvl !== nothing
                # theta_c IS again a genuine joint Newton unknown (edge_contact_step),
                # and residual_edge was found to have multiple roots (like the design
                # doc's own disclosed J0-multi-rootedness risk) — a step-to-step
                # continuity check on θ_c (as the ORIGINAL joint solve used) guards
                # against jumping to a spurious, discontinuous root. Positivity is the
                # blocking admissibility gate; non-intersection is monitored only.
                theta_c_ok = newton_ok && (hist.curr.X === nothing ||
                                           abs(lvl.X[end] - hist.curr.X[end]) < 0.3 + 0.3 * dt_try / dt_init)
                if newton_ok && admis[1] && theta_c_ok
                    accepted = true
                    break
                end
                if newton_ok && admis[1] && best_lvl === nothing
                    best_lvl = lvl
                    best_dt_try = dt_try
                end
                dt_try /= 2
            end
            if accepted
                hist.prev = hist.curr
                hist.curr = lvl
                push!(times, lvl.t); push!(levels, lvl); push!(phases, InContact)
                dt = min(dt_try * 1.1, dt_max)
                stuck_count = 0
            elseif best_lvl !== nothing
                # Never met the (somewhat arbitrary) continuity check, but DID find an
                # admissible (positive-pressure) Newton solution at some dt_try: force
                # progress with it rather than falling back to free flight, which would
                # otherwise immediately re-detect g≤0 and ping-pong without advancing.
                hist.prev = hist.curr
                hist.curr = best_lvl
                push!(times, best_lvl.t); push!(levels, best_lvl); push!(phases, InContact)
                dt = best_dt_try
                stuck_count = 0
            else
                # No admissible Newton solution at any dt down to dt_min: genuine
                # breakdown or true liftoff. Fall back to free flight; the g_prev<=0
                # check above will immediately re-attempt contact next iteration if
                # the gap says we're still touching, so this is not a silent escape.
                phase = FreeFlight
                dt = dt_init
            end
        end
    end

    return times, levels, phases
end
