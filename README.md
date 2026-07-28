# A fully spectral kinematic match for droplet–bath impact

[![CI](https://github.com/elvis-aguero/DropImpact.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/elvis-aguero/DropImpact.jl/actions/workflows/ci.yml)

This repository is the numerical companion to a model of non-coalescing droplet
impact on a liquid bath in which **both** interfaces are spectral and the contact
pressure is solved for rather than assumed. The bath is a Fourier–Bessel series,
the droplet a Legendre series, and the contact pressure a shifted-Legendre series
supported exactly on the contact patch. Nothing sits on a mesh; the free boundary
is determined by a complementarity condition rather than by a prescribed pressure
shape or a discrete tangency search over mesh points.

![Impact simulation](assets/impact_demo.gif)

*(Full-resolution MP4: `output/media/impact_We1.0958_hires.mp4`.)
Output of `run_simulation` for the water case `We = 1.0958`, `Bo = 0.017`,
`Oh = 0.006` at `M = L = 120`, `N = 6`. Dark blue is the bath, pale blue the
droplet, and the red arc is the contact patch `θ ∈ [0, θ_c]`. The inset is the
solved pressure profile over that patch — an output of the model rather than a
prescribed shape, though not a converged one; see "Choosing `M`, `L`, `N`".*

It sits between two published models and takes something from each. From
Alventosa, Cimpeanu & Harris (2023, *JFM* **958**, A24) — the "1PKM" — it takes the
spectral representation of both interfaces and their modal oscillator equations.
From Agüero et al. (2026) — the "full KM" — it takes contact enforced over the
whole pressed region with the pressure as a genuine unknown, and the practice of
selecting the contact extent by a feasibility-filtered residual rather than a
root-find. What is new here is that the pressure is spectral rather than
mesh-resolved, and that the contact angle is a parameter of an outer scalar
search rather than an unknown in a joint algebraic system.

| Path | Contents |
|---|---|
| `src/`, `test/` | The `SpectralKM` package (root `Project.toml`): model, time stepper, test suite. |
| `scripts/` | Validation, rendering, and sweep scripts. |
| `notebooks/tutorial.ipynb` | Guided tour: run an impact, read the diagnostics, compare wall conditions. Dependency-free (SVG figures). |
| `scripts/sweep.jl` | Parallel parameter sweep; picks its own worker count by measuring, and is resumable. |
| `docs/next-gen-KM-model.tex` | The model derivation — the physics ground truth for this package. Includes a section recording arguments from earlier revisions that were wrong, and why. |
| `docs/BouncingDroplets.tex`, `docs/Deformable_impactors.tex` | The two parent papers, for reference and for the claims the derivation makes about them. |
| `derivations/` | CAS and numerical audit scripts backing specific claims in the `.tex`. Every measurement the document quotes is produced by one of them. |

## Setup

Requires Julia 1.12+ and, for the renderer only, `ffmpeg` on `PATH`. The package
itself has no plotting dependency.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running a simulation

```julia
using SpectralKM

p = Params(We = 1.0958, Bo = 0.017, Oh = 0.006,   # water, R = 0.35 mm
           M = 60, L = 60, N = 3,                  # bath / droplet / pressure modes
           b = 6.0, h0 = 3.0, nq = 40)             # bath radius, depth, quadrature order

levels, diag, phases = run_simulation(p; t_end = 8.0, dt_init = 1e-3)
```

New here? Start with **`notebooks/tutorial.ipynb`** — a guided tour of the model,
the diagnostics, the contact-time trap below, and the wall conditions. It has no
plotting dependency either: figures are emitted as SVG.

`levels` is the trajectory: one `Level` per accepted step, holding the bath modes
`a_m`, droplet modes `β_l`, centre-of-mass height and velocity, and the solved
`X = (ĉ_0…ĉ_N, θ_c)`. `diag` carries per-step contact diagnostics — contact angle,
net force `f`, tangency residual, Newton status.

### Choosing `M`, `L`, `N`

`M` and `L` are set by convergence; `N` is a different kind of parameter. Judge
convergence on **contact time** rather than the coefficient of restitution — CoR
settles long before anything else does. At `We = 1.0958`:

| `(M,L,N,nq)` | wall (s) | contact time | CoR | max width |
|---|---|---|---|---|
| `(30,30,3,30)` | 11 | 3.6093 | 0.2997 | 1.1225 |
| `(60,60,3,40)` | 25 | 3.6293 | 0.2996 | 1.1225 |
| `(120,120,6,60)` | 96 | 3.6293 | 0.2995 | 1.1225 |

Convergence is quick: `M = L = 30` is already within 0.55 % of the settled contact
time, and going from `M = L = 60` to `120` moves it not at all — identical to five
digits — while costing four times the wall clock. CoR and maximum width are flat
across the whole range. `M = L = 60` is the recommendation because it is settled
rather than nearly so, not because coarser is unusable.

(An earlier revision of this table reported `M = L = 30` as badly unconverged, with a
contact time of 2.152 and an unphysical width of 1.795. That was the span metric
below, not the physics: at the corrected metric the coarse case is fine. Wall times
here were measured with three cases running concurrently, so they are indicative.)

> **Which contact time?** A single impact does not produce a single contact interval:
> the stepper detaches and immediately re-attaches a few times, separated by gaps of
> exactly one `dt_init`, which is the signature of the guard that forces one advancing
> free-flight step after contact ends. Three scalars therefore compete, and at the
> reference case they spread 17 %: the first interval (3.629), the sum over intervals
> (3.815), and the first-to-last span (4.263). **`primary_contact_time` — the first
> interval — is the metric**; it is the only one measuring a single physical event, and
> the only one monotone in `We`. Earlier revisions of this README reported the span, which
> is why several numbers here changed. `contact_intervals` exposes the full structure.
>
> `primary_contact_time` takes the **longest** interval, not the first. They coincide
> whenever a run is clean, but an under-resolved case chatters from onset — at
> `M = L = 20`, `nq = 16` the first interval is 0.008 long against a physical 3.56 — so
> taking the first would report a contact time three orders of magnitude too small. That
> also makes the metric self-checking: if `contact_intervals` shows the longest interval
> is not the first, the truncation is too coarse to trust.

**`N` does not converge the pressure.** Sweeping `N ∈ {2,3,6,10,16}` along a real
trajectory at `M = L = 60`, contact time is **identical to six significant figures
at every `N`**, and CoR moves only in its fifth digit (0.29959 → 0.29962). The one
quantity that drifts is the maximum contact radius, from 0.8192 at `N = 2` to
0.7999 at `N = 16` — 2.4 %, and in the direction *away* from DNS.

Contact time is thus flat across the range, while the
solved pressure *profile* at `N = 3` differs from `N = 16` by 46–96 % in relative
L2 over the patch. Both are consistent: the compliance operator is compact, so
pressure modes beyond a few sit in its numerical nullspace, and the pressure
reaches the dynamics *only* through the moments `c_m`, `b_l`, `f`, which agree to
between `1e-5` and `1e-2`. Refining `N` buys a different-looking pressure profile
and the same physics.

Use `M = L = 60`, `N = 3`, `nq = 40` for trajectories, and do not quote the
pointwise pressure as a converged output at any `N`.

### Wall condition

`Params(...; wall = :free | :pinned | :clamped)` selects the free-surface condition
at the container wall `r = b`.

| | `:free` (default) | `:pinned` | `:clamped` |
|---|---|---|---|
| condition at `r = b` | `∂η/∂r = 0` | `η = 0` | `η = 0` |
| contact line | 90°, free to slide | pinned, wall slope free | pinned (route B) |
| eigenvalues | zeros of `J_1`, plus `k_0 = 0` | zeros of `J_0`, no `k_0 = 0` | same as `:free` |
| Fourier–Bessel weight | `2/(b J_0(k_m b))²` | `2/(b J_1(k_m b))²` | same as `:free` |

`:free` is the configuration of both parent papers. `:pinned` implements route (A)
of `derivations/feasibility_pinned_contact_line.jl`: the surface is pinned exactly
and by construction — `η(b)` sits at roundoff for a whole impact, because pinning is
an identity of the basis rather than a constraint being enforced. There is no
`k_0 = 0` piston mode, correctly: a surface pinned on its whole boundary cannot
translate uniformly.

> ⚠️ **`:pinned` does not conserve bath volume. Treat it as a diagnostic, not physics.**
> A mode displaces volume `(b/k_m) J_1(k_m b)`. Under `:free` that is *zero* for every
> non-piston mode by definition of the eigenvalues, and the one volume-carrying mode
> (the piston) has `κ_0 = 0` and can never be driven — so volume conservation is
> structural, measured at `5e-15`. Under `:pinned` every mode carries volume and
> nothing constrains the sum: `|∫η r dr|` reaches **2.79, i.e. 17.5 R³ created from
> nothing, four times the droplet's own volume**. `:pinned` also breaks no-flux at
> leading order, and none of the closure diagnostics (self-adjointness, conditioning,
> the tangency root) have been re-measured under it.

`:clamped` implements route (B): the `:free` basis and weight, unchanged, with pinning
imposed as a scalar constraint `Σ a_m J_0(k_m b) = 0` carried by a Lagrange multiplier —
physically the line force the rim exerts on the contact line. Applied at every step,
including free flight (pinning constrains `a_m` itself, not a byproduct of contact
pressure). It is the recommended formulation, since `κ_0 = 0` means the multiplier can
never drive the piston mode — pinning by constraint does not reopen `:pinned`'s volume
defect. Over the reference impact, `:clamped` holds *both* `|η(b,τ)|` and `|∫η r dr|` at
roundoff simultaneously (`7e-17` and `2e-14`), where `:pinned` traded one for the other.
The algebra is verified at operator level in `derivations/feasibility_pinned_contact_line.jl`
§5 (constraint exact to `5e-22`, self-adjoint to `1.1e-17`, sign preserved for any step,
not just the one measured) — but the step-level closure diagnostics of §subsec:contact and
§subsubsec:compliance in the design doc (conditioning, resolvable pressure dimension, the
tangency root) are still `:free` measurements, not yet repeated for `:clamped` at a real
contact step. `:free` vs `:clamped` is negligible at the production default `b = 6`
(`ΔCoR = -0.001`) but far outside step-controller noise once the bath is smaller —
and not monotone in `b` (`+0.26` at `b=4`, `+0.11` at `b=3`, `+0.07` at `b=2`, `-0.21` at
`b=1.5`), consistent with a container resonance crossing the impact timescale as `k_m`
scales with `1/b`. The wall condition matters for a small bath and not for `b = 6`; this
is a gate, not a `b`-convergence study.

There is also a continuum result worth knowing before choosing either: with no-flux
walls in a rigid container, `∂_τ[∂_rη(b)] = O(Oh)`, so **the wall slope is frozen**.
"Pinned with a freely time-varying wall slope" is over-determined, for any basis. The
admissible pinned configuration is the clamped edge, and a constant nonzero wall slope
is obtained by adding a static meniscus. An earlier revision of this README argued the
multiplier route "cannot represent a nonzero wall slope, converging at only `O(1/M)`";
that came from a diagnostic measuring its own grid spacing — computed analytically the
convergence is `M^(-3/2)` and the slope is recovered at every `r < b`.

## Validation

```bash
julia --project=. scripts/validate_trajectory.jl   # vs experiment + DNS
julia --project=. scripts/validate_experimental.jl  # vs measured trajectories + error bars
```

Validation is against **measurement**, not against another model. The reference
is `data/reference/experimental/` — the droplet top and bottom trajectories with
error bars, from the experiments that supersede Alventosa et al. (2023). See that
directory's `PROVENANCE.md`.

```
experimental uncertainty from 293 digitised bar caps: median 0.1004 (quartiles 0.0810, 0.1179)

series         n     mean|err|  max|err|   RMS        RMS/bar
droplet top    60    0.1345     0.2503     0.1526     1.52
droplet bottom 39    0.1055     0.2580     0.1221     1.22
combined       99    0.1231     0.2580     0.1414     1.41
```

The model tracks the measured trajectory at about 1.4 times the experimental
error bar — the same order as the measurement uncertainty, but outside it, so the
residual is model error rather than scatter.

**What cannot be validated against experiment.** The dataset contains no measured
contact radius and no measured droplet width;
both exist only as DNS and 1PKM series (verified by reading the WebPlotDigitizer
projects — each contains exactly two datasets, `*_DNS` and `*_1PKM`). Comparisons
of those quantities are model-to-simulation, and additionally definition-sensitive:
the DNS contact radius is thresholded on the trapped gas film, whereas this
model's `θ_c` bounds the kinematically matched region. `validate_trajectory.jl`
still reports them for information; they are not validation.

The contact radius peaks early: `τ ≈ 0.89` against the DNS `1.46`, and doubling
`M` and `L` moves that by 0.02, so it is not a resolution effect. The bias is
partly generic to reduced models of this class — 1PKM peaks at `τ ≈ 1.22`, also
early against the same DNS. Normalised by contact time the peak falls at 0.20 of
the way through contact here, 0.26 for 1PKM, 0.33 for DNS: same direction, more
skewed. With no measured contact radius available there is nothing to adjudicate
between a closure defect and a definition mismatch, so this stays open.

### Sweeps

```bash
julia --project=. -t auto scripts/sweep.jl --wall=free 0.2 0.4 1.0958 3.0
```

Cases run concurrently at a worker count the script **measures** rather than assumes:

1. **Calibrate.** Time one cheap proxy case and record its peak-RSS growth. A single
   impact's marginal footprint is only about 12 MiB — the ~450 MiB the process holds is
   the Julia runtime plus the package, paid once — because `case_metrics` reduces each
   run to scalars and drops `levels`/`diag` before returning. Keeping full histories is
   what would make a long sweep grow; refusing to is what keeps it flat.
2. **Cap by memory.** Budget against `Sys.total_memory()`, deliberately *not*
   `Sys.free_memory()`: on macOS the latter counts only genuinely free pages and reads
   as ~0 GiB on a warm machine, which would silently cap every sweep at one worker. With
   a 12 MiB footprint the cap is in the hundreds, so it only binds at large `M`, `L`, `nq`.
3. **Ablate.** Time `W = 1, 2, 4, …` (running `2W` cases each, so every worker gets at
   least two) and take the knee — the last `W` that improved throughput by >15 %.

Measured on 8 cores: 0.82 → 1.63 → 2.76 → 2.92 cases/s at `W = 1, 2, 4, 8`. Scaling is
linear to `W = 2`, sublinear by 4, and **flat from 4 to 8** — doubling the workers buys
6 %, because these runs are allocation-heavy and contend on memory bandwidth and the GC.
So the knee lands at `W = 4`, and `Sys.CPU_THREADS` would have been the wrong answer.

Results append to a CSV; re-running skips Weber numbers already present, so an
interrupted sweep resumes. `--workers=N` overrides the selection, `--no-ablation` keeps
the memory cap only. A case that throws is recorded as failed without abandoning the rest.

### Performance

A typical impact runs in about 25 s at production settings, roughly flat in `We`:

| `We` | wall (s) | steps | ms/step | contact time | CoR |
|---|---|---|---|---|---|
| 0.0646 | 32 | 519 | 61 | 4.769 | 0.5173 |
| 0.3000 | 24 | 545 | 44 | 3.869 | 0.4023 |
| 1.0958 | 23 | 543 | 42 | 3.629 | 0.2996 |
| 3.0000 | 24 | 546 | 44 | 3.529 | 0.2331 |

Loss of contact is detected from the contact patch collapsing (`θ_c` reaching its
search floor), with a sustained non-positive net force as a secondary test. The
force test requires several consecutive steps rather than a single one: `f` can
dip transiently negative during the early transient without contact ending, and
acting on one such dip returns the stepper to free flight, where onset is
immediately re-detected and a fresh onset search is paid. Repeated, that cycle
cost `We ≈ 0.3` about twenty times its neighbours' wall time.

## Rendering

```bash
julia --project=. scripts/make_video.jl [We] [outfile.mp4] [M] [L] [N] [nq]
```

Writes an MP4 to `output/media/`. Dependency-free by construction: it
rasterises PPM frames itself and pipes them to `ffmpeg`, so the library never
acquires a plotting stack.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the same command on every push to `main` and `dev`
(`.github/workflows/ci.yml`). Coverage includes the Legendre and Bessel
primitives, Gauss–Legendre exactness on the projection integrands, the BDF2
affine reduction, the inner Galerkin residual, and the contact-step machinery.

## Derivation audits

`derivations/` holds the scripts that back specific quantitative claims in the
`.tex`, and they are meant to be run, not trusted:

| Script | Establishes |
|---|---|
| `verify_legendre_pressure_basis.jl` | The shifted-Legendre pressure basis is exactly degree `n` in `x`, and the projection integrands have the degrees the quadrature bookkeeping assumes — exactly, in `Rational{BigInt}`, through `n = 80`. |
| `audit_compliance_operator.jl` | Self-adjointness, definiteness, `δ`-scaling and resolvable rank of the pressure→gap compliance operator. |
| `audit_nested_closure.jl` | The inner system's conditioning is flat in `δ` and of order unity, against `~1e18` for the joint system it replaces. |
| `audit_weak_determinacy.jl` | Which functionals converge in `N` and which do not, and the identifiability of the self-adjointness diagnosis. |
| `feasibility_pinned_contact_line.jl` | The pinned-wall study: why the no-flux basis converges to a pinned profile after all (`M^(-3/2)`, analytically), why the Dirichlet basis loses volume conservation, the retracted grid artifact that argued otherwise, and (§5) the operator-level verification behind `wall=:clamped`. |
| `probe_global_basis_argmin.jl` | Records a formulation that **fails**: a global pressure basis with weakly-imposed zero pressure off the patch, selected by minimising an integrated residual. |

Scripts marked `SUPERSEDED` in their header verify claims that have since been
withdrawn; they are retained for the record and should not be read as support for
the current model.
