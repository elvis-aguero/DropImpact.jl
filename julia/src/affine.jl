# The three BDF2-affine relations (design doc §subsec:affine, eq:kappa-general and its
# three substitutions eq:kappa-m/lambda-l/kappa-cm), computed ONCE per timestep from
# Level history as plain Float64 — deliberately OUTSIDE the ForwardDiff-differentiated
# residual closure, since they must never be re-differentiated w.r.t. the pressure
# unknowns X (design doc's own emphasis in subsec:affine/subsec:newton).
#
# `bdf2_coeffs(dt, dt_prev)` reduces to BDF1 (𝔞=1,b=-1,c=0) at `dt_prev ≤ 0` (the s→0
# limit of the same formula) — used for the very first step of a run, or the first
# step after a free-flight→contact transition where only one prior level exists.

"""`(𝔞, b, c)` variable-step BDF2 coefficients (design doc eq:bdf2), with `s=dt/dt_prev`.
`dt_prev ≤ 0` gives the BDF1 limit `s=0` (𝔞=1, b=-1, c=0)."""
function bdf2_coeffs(dt::Float64, dt_prev::Float64)
    s = dt_prev > 0 ? dt / dt_prev : 0.0
    a = (1 + 2s) / (1 + s)
    b = -(1 + s)
    c = s^2 / (1 + s)
    return (a, b, c)
end

"""`y^{k+1} = (𝔞x^{k+1}+bx^k+cx^{k-1})/δ` (design doc eq:bdf2's first equation, solved
for `y^{k+1}`), recovering the derivative once `x^{k+1}` is known from
`bath_affine`/`drop_affine`/`com_affine` — needed to store the full state
(value + derivative) for the NEXT step's history."""
function bdf_derivative(x_new::Float64, xk::Float64, xkm1::Float64, dt::Float64, dt_prev::Float64)
    a, b, c = bdf2_coeffs(dt, dt_prev)
    return (a * x_new + b * xk + c * xkm1) / dt
end

"""
    bath_affine(bath_curr, bath_prev, p, dt, dt_prev) -> (kappa, alpha)

`a_m^{k+1} = alpha[m+1] + kappa[m+1] * c_m^{k+1}` (eq:kappa-m), for `m=0..M`. The
`m=0` piston mode (`k_0=0`) naturally gives `kappa[1]=0` (no branch needed — the
forcing coefficient `F = -2 k_0 tanh(k_0 h0)` is itself zero, and the denominator
reduces to `𝔞²`, nonzero).
"""
function bath_affine(bath_curr::BathModeState, bath_prev::BathModeState, p::Params, dt::Float64, dt_prev::Float64)
    a, b, c = bdf2_coeffs(dt, dt_prev)
    M = length(p.k) - 1
    kappa = zeros(M + 1)
    alpha = zeros(M + 1)
    for m in 0:M
        km = p.k[m+1]
        th = tanh(km * p.h0)
        gamma = 2 * p.Oh * km^2
        omega2 = (km^2 + p.Bo) * km * th
        F = -2 * km * th
        denom = a^2 + 2 * gamma * a * dt + dt^2 * omega2
        xk, xkm1 = bath_curr.a[m+1], bath_prev.a[m+1]
        yk, ykm1 = bath_curr.adot[m+1], bath_prev.adot[m+1]
        hist_num = -(a + 2 * gamma * dt) * (b * xk + c * xkm1) - dt * (b * yk + c * ykm1)
        kappa[m+1] = dt^2 * F / denom
        alpha[m+1] = hist_num / denom
    end
    return kappa, alpha
end

"""
    drop_affine(drop_curr, drop_prev, p, dt, dt_prev) -> (lambda, gam)

`β_l^{k+1} = gam[l+1] + lambda[l+1] * b_l^{k+1}` (eq:lambda-l), for `l=2..L`
(`l=0,1` entries left at zero — those modes are never evolved, design doc §subsec:drop).
"""
function drop_affine(drop_curr::DropModeState, drop_prev::DropModeState, p::Params, dt::Float64, dt_prev::Float64)
    a, b, c = bdf2_coeffs(dt, dt_prev)
    lambda = zeros(p.L + 1)
    gam = zeros(p.L + 1)
    for l in 2:p.L
        damp = p.Oh * (2l + 1) * (l - 1)
        omega2 = l * (l - 1) * (l + 2)
        F = -(2l + 1) * l
        denom = a^2 + 2 * damp * a * dt + dt^2 * omega2
        xk, xkm1 = drop_curr.beta[l+1], drop_prev.beta[l+1]
        yk, ykm1 = drop_curr.betadot[l+1], drop_prev.betadot[l+1]
        hist_num = -(a + 2 * damp * dt) * (b * xk + c * xkm1) - dt * (b * yk + c * ykm1)
        lambda[l+1] = dt^2 * F / denom
        gam[l+1] = hist_num / denom
    end
    return lambda, gam
end

"""
    com_affine(com_curr, com_prev, p, dt, dt_prev) -> (kappa_cm, mu)

`z_cm^{k+1} = mu + kappa_cm * f^{k+1}` (eq:kappa-cm). No stiffness/damping
(`γ=ω²=0`), plus the constant gravitational forcing `-Bo` (eq:com) folded into `mu`.
"""
function com_affine(com_curr::COMState, com_prev::COMState, p::Params, dt::Float64, dt_prev::Float64)
    a, b, c = bdf2_coeffs(dt, dt_prev)
    denom = a^2
    xk, xkm1 = com_curr.z, com_prev.z
    yk, ykm1 = com_curr.v, com_prev.v
    hist_num = -a * (b * xk + c * xkm1) - dt * (b * yk + c * ykm1)
    kappa_cm = dt^2 * 1.5 / denom
    mu = hist_num / denom + dt^2 * (-p.Bo) / denom
    return kappa_cm, mu
end
