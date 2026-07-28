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

*(Full-resolution MP4: `julia/output/media/impact_We1.0958_hires.mp4`.)
Output of `run_simulation` for the water case `We = 1.0958`, `Bo = 0.017`,
`Oh = 0.006` at `M = L = 120`, `N = 6`. Dark blue is the bath, pale blue the
droplet, and the red arc is the contact patch `θ ∈ [0, θ_c]`. The inset is the
solved pressure profile over that patch — an output of the model, not an input.*

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
| `julia/` | The `SpectralKM` package: model, time stepper, validation and rendering scripts. |
| `docs/next-gen-KM-model.tex` | The model derivation — the physics ground truth for `julia/`. Includes a section recording arguments from earlier revisions that were wrong, and why. |
| `docs/BouncingDroplets.tex`, `docs/Deformable_impactors.tex` | The two parent papers, for reference and for the claims the derivation makes about them. |
| `derivations/` | CAS and numerical audit scripts backing specific claims in the `.tex`. Every measurement the document quotes is produced by one of them. |

## Setup

Requires Julia 1.10+ and, for the renderer only, `ffmpeg` on `PATH`. The package
itself has no plotting dependency.

```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running a simulation

```julia
using SpectralKM

p = Params(We = 1.0958, Bo = 0.017, Oh = 0.006,   # water, R = 0.35 mm
           M = 60, L = 60, N = 3,                  # bath / droplet / pressure modes
           b = 6.0, h0 = 3.0, nq = 40)             # bath radius, depth, quadrature order

levels, diag = run_simulation(p; t_end = 8.0, dt_init = 1e-3)
```

`levels` is the trajectory: one `Level` per accepted step, holding the bath modes
`a_m`, droplet modes `β_l`, centre-of-mass height and velocity, and the solved
`X = (ĉ_0…ĉ_N, θ_c)`. `diag` carries per-step contact diagnostics — contact angle,
net force `f`, tangency residual, Newton status.

### Choosing `M`, `L`, `N`

`M` and `L` are set by convergence and `N` is not the important parameter, which
is worth stating because it is counterintuitive. At `We = 1.0958`:

| `(M,L,N,nq)` | wall (s) | CoR | contact time | max width |
|---|---|---|---|---|
| `(30,30,3,30)` | 68 | 0.5399 | 2.152 | 1.795 |
| `(60,60,3,40)` | 24 | 0.2996 | 4.2624 | 1.1225 |
| `(60,60,6,60)` | 52 | 0.2996 | 4.2624 | 1.1225 |
| `(120,120,6,60)` | 99 | 0.2995 | 4.2646 | 1.1225 |
| `(120,120,10,80)` | 177 | 0.2996 | 4.2646 | 1.1225 |

`M = L = 30` is badly unconverged (a coefficient of restitution and width that
are not physical). From `M = L = 60` the rebound metrics agree to four digits and
refining further changes nothing. Raising `N` from 3 to 10 changes nothing at
all: the compliance operator is compact, so pressure modes beyond a few sit in
its numerical nullspace and do not affect the moments through which the pressure
acts. Use `M = L = 60`, `N = 3`, `nq = 40` for production and treat larger `N`
purely as a rendering nicety.

## Validation

```bash
cd julia
julia --project=. scripts/validate_trajectory.jl   # vs experiment + DNS
julia --project=. scripts/validate_sweep.jl        # rebound metrics vs We
```

`validate_trajectory.jl` compares against the experimental and DNS records that
accompany the full-KM paper (archived in the sister repository
`km-dropplet-onto-bath`), which supersede the reduced curves of the 1PKM paper.
At `M = L = 60`, `N = 3`:

| | model | reference | error |
|---|---|---|---|
| coefficient of restitution | 0.2996 | 0.3043 | 1.5 % |
| contact time | 4.2624 | 4.4074 (DNS) | 3.3 % |
| maximum droplet width | 1.113 | 1.117 (DNS) | 0.4 % |
| maximum contact radius | 0.8183 | 0.8821 (DNS) | 7.2 % |

Pointwise over the trajectory the mean absolute deviation is 0.0104 for the
droplet width against DNS, 0.1345 for the position of the droplet's top and
0.1055 for its bottom against experiment.

One discrepancy is open and is not a resolution effect: the contact radius
reaches its maximum at `τ ≈ 0.89` against the DNS value `τ ≈ 1.46`, and doubling
`M` and `L` moves that by 0.02. The patch grows and recedes too quickly even
though the total contact time is accurate to 3 %.

## Rendering

```bash
julia --project=. scripts/make_video.jl [We] [outfile.mp4] [M] [L] [N] [nq]
```

Writes an MP4 to `julia/output/media/`. Dependency-free by construction: it
rasterises PPM frames itself and pipes them to `ffmpeg`, so the library never
acquires a plotting stack.

## Tests

```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
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
| `probe_global_basis_argmin.jl` | Records a formulation that **fails**: a global pressure basis with weakly-imposed zero pressure off the patch, selected by minimising an integrated residual. |

Scripts marked `SUPERSEDED` in their header verify claims that have since been
withdrawn; they are retained for the record and should not be read as support for
the current model.
