# What the contact pressure actually looks like, and why `N`-refinement cannot recover it

Literature basis for the shape of the contact pressure in droplet/bath impact, and the
consequence for this model's pressure representation. Written because the question "if
raising `N` enlarges the trial space, why doesn't accuracy improve?" has a physical answer,
not a numerical one, and because the answer changes what should be claimed in the paper.

Sources are in `references/`: `BouncingDroplets.tex` (Alventosa et al. 2023),
`Deformable_impactors.tex` (Agüero et al. 2026), `LubricationMediated.tex` (Phillips &
Milewski 2024, arXiv:2406.17138).

---

## 1. Alventosa et al.'s ansatz, and its physical grounding

They posit (`BouncingDroplets.tex:272`)

    p_s(r,t) = F(t)/(pi R^2) * H_r(r/r_c(t)),     H_r(s) = C (1 - s^6) for s <= 1, else 0

i.e. **a smoothed top hat**, with `C` fixed by requiring the pressure integrate to `F(t)`.
The 6th-order polynomial is a deliberate approximation to a top hat, not a parabola: the
parabolic form of Blanchette et al. is explicitly rejected as too peaked.

Four independent lines of evidence are cited, three of them experimental:

| source | evidence |
|---|---|
| Galeano-Ríos et al. | DNS: pressure under a non-wetting sphere on a bath is "flatter and more similar to a top-hat function for most times" |
| de Ruiter et al. 2015 | air-film thickness by **interferometry**, pressure inferred via lubrication: "approximately uniform, with deviations from uniformity only near the edge of the film" |
| Tang et al. 2019 | measured film thickness for drops bouncing on a **deep pool** at comparable `We`: "significantly more uniform", attributed to the deformability of both impactor and substrate |
| Alventosa et al.'s own DNS | "the pressure in the air gap across the evolving contact radius may thus be robustly approximated by a **top-hat function at every flow stage** investigated" |

**Mechanism.** Lubrication pressure is set by the gap profile. When both interfaces are free
to deform they relax toward a near-parallel gap, giving a near-uniform pressure. The
contrast case establishes this by negation: for a drop on a **wettable solid** (Mandre &
Mani), the pressure "increases sharply near the contact line", because a rigid substrate
cannot adjust and the drop's curvature pinches a thin annular neck. Deformability of *both*
surfaces is what flattens the profile.

**They also already hit our numerical problem.** `BouncingDroplets.tex:301`:

> "The reconstruction of the top-hat function in Fourier--Bessel space converges too slowly
> to be of practical use (Storey 1968), also noted by Blanchette... we tested higher order
> polynomials (corresponding to a larger flat region), and found increasingly poor
> convergence behaviour, similar to that of the top hat."

The 6th-order choice is called "a practical compromise". So *a near-top-hat pressure
converges badly in a smooth spectral basis* is a known result in this literature. Note also
their Appendix B finding: the trajectory is **largely insensitive to the shape function**,
but a **time-dependent `r_c`** is essential. The shape is not what the dynamics needs.

## 2. Phillips & Milewski 2024: the pressure resolved rather than assumed

Their lubrication-mediated model solves a free-boundary elliptic problem for the air-layer
pressure instead of positing a shape, so it predicts the profile. Findings
(`LubricationMediated.tex:308`, nondimensionalised by `2 sigma / R_0`):

* interior pressure in the range **0.9--1.5** over "the vast majority of the region" -- the
  top hat, recovered from first principles rather than assumed;
* air layer **0.5--2 um in the bulk**, "narrowing on its boundary to **0.15--0.5 um**". "In
  effect, for the bulk of the impact, the air layer forms a quasi-static cushion constricted
  on its circular boundary";
* "**A pressure spike occurs where the layer thins at its edge**", consistent with Hendrix
  et al. 2016;
* extremes of **7.6** at impact and **-3.8** at liftoff (genuine suction at detachment);
* and critically: "Most of the work on the drop is due to the quasi-uniform pressure at
  intermediate times... these features do not substantially affect the overall force."

So the physical target is

    p*  ~=  (top hat over the patch)  +  (a spike at the rim, from film thinning)

## 3. Consequences for this model

### 3.1 Why refining `N` does not improve pointwise pressure

Enlarging the trial space *does* strictly improve best-approximation error -- that part is a
theorem and needs no qualification. The Galerkin error splits as

    ||p* - p_N||  <=  (1/gamma_N) * inf_{v in V_N} ||p* - v||

with `gamma_N` the discrete stability constant. Two failure modes, now separated:

1. `gamma_N` degrading. This was the resolvable-rank/bandwidth problem. Real, diagnosed,
   fixed by refining `M, L` (`cond` 5.0e10 -> 32); pinned in CI by `test/test_rank_law.jl`.
2. The `inf` term shrinking slowly, because **`p*` is not smooth**. A top hat is
   near-discontinuous at the patch edge. Polynomial approximation of a near-discontinuity
   exhibits **Gibbs phenomenon**: `L^inf` error does not decay at all, `L^2` decays only
   algebraically, and there is a systematic **undershoot adjacent to the jump whose
   amplitude does not decrease with `N`**.

Mode 2 accounts for every measurement we have:

| observation | Gibbs prediction |
|---|---|
| `L^2` deviation improves only 1.65x over `N = 1 -> 7` | algebraic, not spectral |
| `min p` pinned at `~ -3.8`, flat in `N` | undershoot amplitude is `N`-independent |
| `c_m`, `b_l`, `f`, `t_c`, CoR converge to 1e-4..1e-6 | oscillations are localised and cancel under smooth-weighted integration |
| `alpha = +1/2` made everything worse | it forces `p(x_c) = 0`, but physically the pressure is at its **maximum** at the rim |

### 3.2 The harder point: the rim feature is excluded by a modelling assumption

`paper-formulation.tex:47` assumes the air layer "transmits pressure between the two
interfaces without otherwise participating". The edge spike exists **because** the layer
participates -- it is produced by the film thinning to 0.15--0.5 um at the rim. A model with
no air-film dynamics therefore **cannot represent the rim feature at any `N`, in any basis**.

That is a modelling limitation, not a discretisation one, and it is the honest form of
"moments converge, pointwise pressure does not":

* the unresolvable part of `p*` is a lubrication-scale feature this model deliberately
  excludes;
* it carries little net force (Phillips & Milewski), which is *why* the smooth functionals
  converge cleanly and why Alventosa et al.'s crude ansatz still predicts trajectories well.

The earlier justification given for this in the codebase -- that the compliance operator is
compact -- was never verified and is false; see `DIAGNOSTICS-NOTATION.md`.

### 3.3 What would actually help, in increasing order of cost

1. **Report honestly.** State that the pointwise pressure converges algebraically because
   `p*` is near-discontinuous at the rim, and that the rim spike is outside the model's
   assumptions. Report functionals as the converged outputs. No code change.
2. **A rim-adapted basis.** Gibbs is defeated by putting the near-discontinuity into the
   basis -- a smoothed step / boundary-layer function localised at `psi = 0`, not an edge
   exponent `psi^alpha`. Note this fixes the *top-hat* edge, not the spike, which is absent
   from the model regardless.
3. **Resolve the air layer** (the Phillips & Milewski route). Then the spike appears for the
   right reason, at the cost of a free-boundary elliptic solve per step -- a different model,
   not a better discretisation of this one.

An edge exponent `alpha` is *not* on this list: `alpha > 0` contradicts the flat top and was
measured to be worse, and `alpha < 0` mimics the spike's blow-up but for the wrong reason
(an imposed-edge algebraic singularity rather than film thinning), so it cannot be expected
to carry the right physics even if it fits better.

## 4. Further reading located on arXiv

Air-film structure and pressure in impact, with experimental basis:

* **arXiv:2104.09229** Lakshman et al., *Deformation and relaxation of viscous thin films
  under bouncing drops* -- digital holographic microscopy to 20 nm, film deformation from
  air-pressure buildup, with a matching lubrication analysis.
* **arXiv:2111.13832** Roy et al., *Droplet impact on immiscible liquid pool: ... entrapped
  air cushion* -- reflection interferometry of the entrapped film on a **liquid pool**.
* **arXiv:2303.00444** Roy et al., *Insights into air cushion dynamics ...* -- resolves the
  film into a central dimple plus a peripheral disc with distinct rim physics.
* **arXiv:2106.09551** Jain et al., *Air-cushioning effect and Kelvin--Helmholtz instability
  before the slamming of a disk on water* -- central depression from stagnation pressure with
  a **lift-up under the edge**.
* **arXiv:2101.01794** Moore, *Introducing pre-impact air-cushioning effects into the Wagner
  model* -- leading-order pressure and force with air cushioning.

Primary sources cited by the papers above but not on arXiv, worth pulling by DOI:
de Ruiter et al. 2015 (interferometry, uniform film pressure), Tang et al. 2019 (deep pool),
Hendrix et al. 2016 (the rim spike), Mandre & Mani (solid-substrate contrast case),
Storey 1968 (slow Fourier--Bessel convergence of a top hat).
