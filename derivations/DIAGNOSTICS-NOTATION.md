# Diagnostics notation

Symbols used in the numerical diagnosis of the contact-pressure closure (the
conditioning / pressure-convergence investigation). **Model** notation — the symbols
that appear in the formulation itself — lives in the notation table at the top of
`paper-formulation.tex`; this file covers the *diagnostic* quantities invented for the
investigation, which appear nowhere in the paper and are easy to lose track of.

Nothing here is a physical parameter. These are all properties of the discretization.

---

## 1. Discretization sizes (these ARE in the paper, repeated for convenience)

| Symbol | Meaning | Typical |
|---|---|---|
| `M` | number of bath Fourier–Bessel modes, `m = 0..M` | 60–640 |
| `L` | number of droplet Legendre modes, `l = 2..L` | 60–640 |
| `N` | polynomial degree of the contact pressure | 3–32 |
| `nq` | Gauss–Legendre quadrature nodes for the `c_m` / `W_n^{(m)}` projections | 200–400 |
| `b` | bath radius (in droplet radii) | 6.0 |
| `h0` | bath depth (in droplet radii) | 3.0 |
| `δ` | BDF2 time-step parameter used when freezing the geometry for an operator probe | 1e-3 |
| `θ_c` | contact angle — edge of the contact patch | 0.05–0.8 |
| `x_c` | `cos θ_c`, the patch edge in the `x = cos θ` variable | — |
| `r_c` | `sin θ_c` at the undeformed geometry — patch radius in bath coordinates | ≈0.3 |

## 2. The patch coordinate and the pressure basis

| Symbol | Definition | Notes |
|---|---|---|
| `x` | `cos θ` | the droplet-side integration variable |
| `ψ` | `(x - x_c)/(1 - x_c)` | **patch-local coordinate**: `ψ=0` at the contact line, `ψ=1` at the pole. Rescales to the current patch size every step — the "multiple scales" property |
| `P̃_n(ψ)` | `P_n(2ψ - 1)` | shifted Legendre polynomial, the current pressure basis |
| `ĉ_n` | coefficients in `p = Σ_n ĉ_n P̃_n(ψ)` | the pressure unknowns |
| `α` | assumed edge exponent in a trial basis `ψ^α P̃_n(ψ)` | `α=0` is the current basis; `α=1/2` is the proposed `√ψ` basis |
| `c_s` | coefficient of an added `ψ^{-1/2}` basis function | the **singular / flat-punch edge mode**; see §5 |

## 3. The compliance operator

| Symbol | Meaning |
|---|---|
| `𝒜` | **compliance operator**: maps contact pressure to gap displacement, `C = C_free + 𝒜p` (`eq:compliance-def`) |
| `𝒦` | its integral kernel (`eq:compliance-kernel`) — a smooth sum of bath-Bessel, droplet-Legendre, and centre-of-mass terms |
| `C` | the gap between the two surfaces. `C = 0` on the patch is the contact condition |
| `C_free` | the gap that *would* exist with no contact pressure — the data of the problem |
| `κ_m`, `λ_l`, `κ_cm` | the per-mode affine coefficients that make `C` linear in `p` at frozen geometry (bath, droplet, COM respectively) |
| `T` | **tangency residual** `T = ∂C/∂θ` at `θ = θ_c` (`eq:tangency-selector`). Note: computed at `timestepper.jl:243` as a *reported diagnostic only* — it does not select `θ_c` in the current code |

## 4. Spectral / conditioning diagnostics — the ones I kept using without defining

| Symbol | Definition | What it tells you |
|---|---|---|
| `σ_n` | the `n`-th largest `\|eigenvalue\|` of `-𝒜` in the weighted inner product, sorted descending | how strongly the interfaces respond to the `n`-th most-visible pressure direction |
| **`n*`** | the first index `n` with `σ_n/σ_1 < 10⁻¹⁰` | **the numerical rank of the compliance operator** — how many pressure directions the truncated bath+droplet can respond to at all. Push `N+1` past it and you are adding modes the discretization is blind to. The `10⁻¹⁰` cutoff is a *choice*; `n*` shifts an index or two under another. Grows with `M`, `L`, `θ_c` |
| `J_N` | the `(N+1)×(N+1)` Galerkin matrix of `𝒜` in the pressure basis | the matrix actually inverted for `ĉ_n` |
| `cond(J_N)` | `σ_max/σ_min` of `J_N` | error amplification. `> 1e15` is past double precision (unit roundoff ≈ 2.2e-16) and should be read as "unreliable", not as a number |
| "ideal cond" | `σ_1/σ_{N+1}` of `𝒜`'s own spectrum | the **best conditioning any `N`-dimensional trial=test subspace can achieve** for a fixed operator/domain/equations (Rayleigh–Ritz / Courant–Fischer). Attained only by the operator's own top-`N` eigenvector subspace. Basis-independent *only* under that fixed setup — it says nothing about a reformulation that changes the unknowns |
| "cliff" | the sharp fall in `σ_n` at `n ≈ n*` | measured ≈5 orders of magnitude per index. Identified as a **Slepian** tail (below) |
| `k_*` | `(𝔞/δ)^{2/3}`, the wavenumber where the compliance symbol turns over from growing to decaying | ≈131 at `δ=1e-3`. Since `k_M ≈ 42` at `M=80`, the operator is *anti-smoothing* over the whole retained spectrum at production step sizes |
| `𝔞` | the BDF2 leading coefficient (3/2) | appears only inside `k_*` |
| `ρ` | safety factor in the proposed budget `N+1 ≤ ρ·n*` | ≈0.6 keeps `cond ≲ 1e6` |

## 5. Error measures used in the basis stress tests

| Symbol | Definition |
|---|---|
| `L²` error | `‖p_rec - p_true‖_w / ‖p_true‖_w`, weighted by the quadrature weights — measures *average* accuracy |
| `L^∞` error | `max\|p_rec - p_true\| / max\|p_true\|` — measures *worst-point* accuracy. Diverges while `L²` and the force stay fine when the edge behaviour is wrong |
| "Cauchy distance" | `‖p_N - p_{N-2}‖_w / ‖p_N‖_w` — self-consistency between successive truncations. Used when no true solution is available, so convergence can be judged without manufacturing one (which is what made the review's original `√ψ` demo circular) |
| `min p` | minimum of `p` over the patch. **Negative values are unphysical** (contact cannot pull) — used as the primary pressure-quality diagnostic; `check_positivity` in `residual.jl` |
| `f` | net contact force — a smooth linear functional of `p`, hence well-conditioned even when pointwise `p` is not |
| `c_1`, `b_2` | low-order bath and droplet pressure moments — likewise smooth functionals, likewise well-behaved |

## 6. Concepts / jargon

- **Compact operator** — roughly, an operator that smooths: its singular values decay to
  zero. Inverting one amplifies high-frequency error without bound.
- **First-kind Fredholm equation** — `𝒜p = data` with `𝒜` compact and *no* identity
  term. The textbook ill-posed inverse problem. Contrast "identity + compact"
  (`(I + 𝒦)p = data`), which is classically well-posed via the Fredholm alternative.
  *This was my original diagnosis of the pressure problem, and it was wrong* — see below.
- **Order of an operator** — how the symbol scales with wavenumber. Negative order =
  smoothing (`~1/k`), positive order = differentiating (`~k`). The compliance operator is
  order **+1** (bath) / **+2** (droplet) in the production regime, i.e. anti-smoothing,
  which is why the compactness framing does not apply to what the model actually retains.
- **Slepian / prolate concentration** — restricting a band-limited function space
  (`span{P_l : l ≤ L}`) to a subinterval (`θ ∈ [0,θ_c]`) yields eigenvalues on a plateau of
  size ≈ `Lθ_c/π` followed by super-exponential decay. This is what `n*` counts and what
  the "cliff" is.
- **Flat-punch / rigid-die singularity** — for a contact problem with the edge location
  *imposed* rather than solved for, the pressure carries a `ψ^{-1/2}` component whose
  coefficient (`c_s`) vanishes if and only if the edge is correctly placed (smooth
  pasting). With smooth pasting, the physical pressure vanishes like `ψ^{1/2}` at the edge.
- **Signorini / complementarity** — the contact conditions `gap ≥ 0`, `p ≥ 0`,
  `p·gap = 0`. Note that `select_theta_c` in `timestepper.jl:142` already implements an
  operational form of this (`θ_c = inf{θ : non-intersection and monotone-r hold}`), which
  is **not** what `paper-formulation.tex:312` describes (`argmin |T|`).

---

## Summary of what these diagnostics established

1. **`cond(J_N) → ∞` under `N`-refinement is bandwidth truncation, not operator
   compactness.** At fixed `N=12` and fixed basis, refining `M=L` from 80 to 640 drops
   `cond(J_12)` from 5.03e10 to 3.24e1. The cliff index moves with `M`, `L`, `θ_c` and is
   invariant across four decades of `δ` (13 every time) while `σ_1` itself moves five
   orders. So `N`-refinement at fixed `M, L` is the wrong experiment — refining the trial
   space without refining what can respond to it.
2. **Production negative pressure is explained by the bandwidth budget, not by `T`.**
   At `M=L=60`, `t_end=12`: `N=3` → 2% of contact steps have `min p < 0`; `N=6` → 2%;
   `N=12` → **90%**, median `min p = -1.34`. Meanwhile median `|T|` *improves* with `N`
   (1.6e-2 → 1.8e-3), and at `N=12` the negative-pressure steps have *smaller* `|T|`
   than the positive ones — opposite to the punch-mode prediction. `n*(L=60, θ_c=0.3) ≈ 11`
   makes the budget `N ≲ 5.6`, matching the 2/2/90 pattern.
3. **The `√ψ` basis is a real but conditional gain.** Spectral on free-contact-line
   shapes (`L²` 3.3e-3 → 2.0e-15 at `N=12`); but non-convergent with `L^∞ ≈ 0.95` and
   negative pressure whenever the true edge value is finite and nonzero, since every basis
   function vanishes at `ψ=0`. Adding `ψ^{-1/2}` does not fix that. It hard-codes a free,
   smoothly-pasted contact line.
4. **`c_s` is a valid closure residual but not obviously a root-findable one.** With
   consistent data it converges spectrally to zero (`-9.4e-4 → -5.3e-14` over `N=2..12`);
   with inconsistent data it parks at `O(10⁻²)`. But sweeping `θ_c ∈ [0.1, 0.6]` at fixed
   data it stays strictly negative — smooth, no sign change, no root to bisect. (Caveat:
   that sweep held the data fixed while `θ_c` varied; real `C_free` co-varies.)
