# Audit: Phillips & Milewski (2024) lubrication-mediated model

Physics-faithfulness audit of the lubrication-mediated (LM) model of Phillips, Cimpeanu &
Milewski, ahead of deciding how to incorporate lubrication physics into SpectralKM. Covers
both papers: `references/LubricationMediated.tex` (arXiv:2406.17138, 3D, spheres) and
`references/LubricationMediated2D.tex` + `LubricationMediated2D_Sections/`
(arXiv:2406.16750, 2D, deformable drops).

**Short verdict: not a toy model. The one substantive physics objection is rarefaction --
its most distinctive output sits outside the validity of its own closure.**

The derivation is careful and standard, the one free parameter is justified with a
sensitivity study and a scaling law, and results are validated against both DNS and
experiment.

> **CORRECTION (this section supersedes an earlier draft of this audit).** An earlier version
> of this document listed "the deformable droplet is derived but never solved" as the leading
> limitation. **That was wrong**, and the sphere paper itself points to why at `:77`: the
> deformable-droplet lubrication-mediated model *is* published and DNS-validated in
> **Phillips, Cimpeanu & Milewski, arXiv:2406.16750**, "Modelling two-dimensional droplet
> rebound off deep fluid baths" (companion paper, same day), now in `references/` as
> `LubricationMediated2D.tex` + `LubricationMediated2D_Sections/`. That paper couples bath,
> air and *deformable* drop, validates against dedicated Navier-Stokes DNS, and additionally
> computes multiple rebounds and long-time dynamics on a vibrating bath. The remaining gap is
> therefore much narrower than claimed: only the **3D/axisymmetric deformable numerics** are
> unrun, and that is a gap in computation, not in derivation or in validation-in-principle.

---

## 1. What the model is

Air layer treated as a genuine fluid region rather than an assumed pressure shape:

* lubrication balance `partial_z p_a = 0`, `mu_a partial_z^2 u_a^H = grad_H p_a`, with
  velocity continuity at *both* interfaces (`LubricationMediated.tex:110-118`);
* flux `Q = -(h^3/12 mu_a) grad_H P + (h/2)(grad_H phi_b + u_d^H)` -- Poiseuille driven by
  the pressure gradient plus Couette dragged by the two moving surfaces (`:126`);
* thin-film equation `partial_t h + div Q = 0`, which the authors correctly characterise as
  **a nonlinear free-boundary elliptic problem for `P`** with `-h_t` as forcing and `P = 0`
  on the moving boundary `partial Omega_L` (`:132`);
* bath and droplet are quasi-potential with weak viscous corrections, i.e. the same lineage
  as SpectralKM;
* lubrication region footprint `Omega_L = {r <= r*(theta,t)}` with `r*` set by
  `h(r*,theta,t) = epsilon` (`:109`).

This is the right structure. The pressure is *solved for*, not posited, which is exactly
what we lack.

## 2. Scope of what has actually been computed (revised)

| paper | geometry | impactor | validated against |
|---|---|---|---|
| arXiv:2406.16750 (`LubricationMediated2D`) | 2D / cylindrical | **deformable droplet** | dedicated Navier-Stokes DNS; also multiple rebounds, vibrating bath |
| arXiv:2406.17138 (`LubricationMediated`) | 3D axisymmetric | rigid sphere | KM model, DNS, experiment (`galeano2021capillary`) |

So the deformable coupling is done and tested, in 2D. The 3D paper derives the general
three-dimensional model including the spherical-harmonic droplet equations (`:138-160`) and
then specialises to spheres for its computations, "eliminating droplet oscillations"
(`:197`), with 3D drop deformation listed as future work (`:335`).

**Consequence for us.** Adopting this machinery is not adopting something untested -- the
deformable lubrication coupling has been exercised against DNS. What is untested is the
*combination* we want: 3D axisymmetric **and** deformable. Two things to keep in view:

* 2D (cylindrical) is not a mild reduction. There is no azimuthal curvature, the radial
  metric factors are absent from the thin-film operator, and the droplet's modal structure
  differs (Fourier vs. spherical harmonics), so the 2D validation does not transfer
  quantitatively to the axisymmetric case -- it establishes that the coupling works, not that
  our numbers will.
* Their DNS reference for the sphere case is itself a proxy: in `galeano2021capillary` the
  solid sphere is modelled in Gerris "by using artificially high viscosity and surface tension
  coefficients" (`:274`). The 2D deformable paper uses dedicated DNS instead, which is the
  stronger comparison.

## 3. The substantive objection: continuum lubrication is invalid where the headline result lives

The paper's distinctive claim is the rim pressure spike: "A pressure spike occurs where the
layer thins at its edge" (`:308`), with the film "0.5-2 um in the bulk of the layer and a
narrowing on its boundary to **0.15-0.5 um**".

Air mean free path at STP is `lambda ~ 68 nm`. Hence `Kn = lambda/h`:

| region | `h` | `Kn` | regime |
|---|---|---|---|
| bulk, thickest | 2.00 um | 0.034 | slip correction already needed |
| bulk, thinnest | 0.50 um | 0.136 | transition -- continuum invalid |
| rim, thickest | 0.50 um | 0.136 | transition -- continuum invalid |
| rim, thinnest | 0.15 um | **0.453** | transition -- continuum invalid |

No-slip continuum lubrication requires `Kn <~ 0.01`; a first-order slip correction extends
usable range to `Kn ~ 0.1`; beyond that the Poiseuille flow rate must come from kinetic
theory. **The rim -- and even the thin end of the "bulk" -- is in the transition regime.**

The model solves no-slip continuum lubrication throughout, so:

* the rim spike is computed with a closure that does not apply there;
* the error has a **known sign**: rarefaction increases the Poiseuille flow rate for a given
  pressure gradient (equivalently lowers the effective viscosity), so a no-slip treatment
  **overpredicts** the pressure needed to drive the same flux. The spike is therefore
  expected to be too large, by a factor that grows as the film thins.

The same objection applies to the 2D deformable paper, which reports the central region at
"near constant thickness of `O(1) um`" but films "as small as `0.5 um`" at the edges of the
lubrication region and during the pinch before lift-off
(`LubricationMediated2D_Sections/Section4.tex:86,131`) -- i.e. `Kn ~ 0.14`, again transition
regime, again solved with continuum no-slip.

Two independent corroborations:

* Roy et al. (arXiv:2303.00444), measuring the air film under a drop impacting a solid, find
  the central dimple in the continuum Stokes regime but the **peripheral disc in the
  slip/transition regime at high `Kn`**, and state that "a unified treatment, including
  continuum and non-continuum mechanics, is required".
* More pointedly, the 2D paper itself **cites Sprittles (2024), *Gas microfilms in droplet
  dynamics: When do drops bounce?*, Annu. Rev. Fluid Mech. 56, 91-118** -- the review devoted
  to precisely this question, in which non-continuum gas-film behaviour is central to whether
  and when drops bounce. It is cited as corroboration for the `O(1) um` film thickness
  (`Section4.tex:131`) while the model retains continuum no-slip. So the relevant literature
  is known to the authors and simply not incorporated; this is an acknowledged frontier rather
  than an oversight, but it does bound what the rim predictions are worth.

**This is a fixable flaw, cheaply.** The gas-bearing / magnetic-storage literature handles
exactly this `Kn` range routinely. Minimum viable fix is a first-order Maxwell slip
correction, replacing `h^3/12 mu_a` by `h^3 (1 + 6 Kn)/12 mu_a` (or the Fukui--Kaneko
database correction for higher `Kn`). It is a one-line change to the flux and it removes a
qualitative objection.

## 3b. Reading the model derivation itself (`LubricationMediated2D/Sections/ModelSubsections/`)

Four points that only appear in the derivation, not the results.

**(i) The rim is doubly suspect, not just Knudsen-suspect.** The lubrication reduction is
made "assuming weak impacts in which the lubrication layer will not deviate substantially
from the horizontal" (`Subsection2.tex:14`). That is the standard slowly-varying-gap
requirement, `|partial_x h| << 1`. It is weakest exactly where the film pinches at
`l*` -- the same place `Kn` is largest and the same place the pressure spike is reported.
So two independent approximations degrade together at the rim: the kinetic one (Sec. 3) and
the geometric one underpinning lubrication scaling itself. This strengthens the conclusion
that the rim spike should be treated as qualitative, and it is *not* repaired by the slip
correction -- slip fixes the constitutive closure, not the aspect-ratio assumption.

**(ii) The air-to-bath shear coupling is deliberately one-way.** The bath boundary
conditions drop the air's tangential stress at `O(mu_a/mu_b)` (`Subsection1.tex:32`,
"Disregarding terms of `O(mu_a/mu_b)`"), which for their fluids is
`1.825e-5 / 9.78e-4 ~ 1.9%`. Meanwhile the bath's own surface velocity *does* drag the air,
appearing as the Couette term `(partial_x phi_b + u_d) h / 2` in the flux. So momentum
transfer is asymmetric by construction: liquid drives air, air does not shear back. This is
consistent at the stated order and almost certainly harmless, but it should be inherited
knowingly rather than accidentally. Viscous normal stresses in the air are likewise dropped
(`Subsection1.tex:21`), leaving pressure as the sole air loading -- the same assumption
SpectralKM makes.

**(iii) THE KEY STRUCTURAL DIFFERENCE, and it dissolves our Gibbs problem rather than
fighting it.** They never expand the contact pressure in a truncated smooth basis on a moving
patch. `P` is obtained by *solving* the thin-film equation
`partial_t h + partial_x[-(h^3/12 mu_a) partial_x P + (partial_x phi_b + u_d)h/2] = 0`
as a free-boundary elliptic problem on a grid, with `P = 0` at `x = +-l*`
(`Subsection2.tex:50-54`). The drop and bath then receive it only through *projections*
(`p_hat_n`, `Subsection3.tex:29`), which is structurally the same role our `b_l` and `c_m`
play. This is precisely why they can carry a top-hat-with-a-rim-spike and we cannot: their
pressure lives on a grid where a near-discontinuity is representable, ours lives in a
degree-`N` shifted-Legendre expansion on `[x_c,1]` where it provokes Gibbs oscillation and
negative pressure. **Adopting the lubrication layer therefore replaces the object that causes
our pointwise-pressure problem, rather than requiring us to solve it.**

**(iv) A new ingredient we would have to build.** The Couette term needs the *tangential
surface velocity* of the droplet, `u_d|_{eta_d}`. In 2D they get it to leading order as
`u_d(R_0,theta,t) = -sum_n c_dot_n sin(n theta)` (`Subsection3.tex:52`). SpectralKM currently
computes no analogous quantity -- we carry `beta_l` and `beta_l_dot` but never the surface
tangential velocity field. The axisymmetric analogue (a `dP_l/dtheta`-weighted sum of
`beta_l_dot`) is straightforward from the existing Legendre machinery, but it is genuinely new
code and a new term in the coupling, not a rearrangement of what exists.

For completeness, their droplet deformation equation is
`c_ddot_n + 2 lambda_n c_dot_n + omega_n^2 c_n = -(n / rho_d R_0) p_hat_n` with
`lambda_n = 2 mu_d n(n-1)/(rho_d R_0^2)`, `omega_n^2 = sigma n(n^2-1)/(rho_d R_0^3)`
(`Subsection3.tex:42-48`) -- the 2D Fourier counterpart of our Legendre drop modes, forced by
the pressure projection exactly as ours are, and matching Lamb/Aalilija damping rates.

## 4. Shared limitation: small `Oh`, linear interfaces

Explicitly truncated at linear order in Ohnesorge, with `Oh^2` dropped
(`:156`, and `:356`: "the system has already been truncated at leading order in `Oh` when
applying stress balances... we may therefore formally eliminate any terms `Oh^2`"), plus
"small surface deformations (resulting in linear wave systems)". The droplet's lower surface
`S` must be single-valued with bounded gradients (`:106`), the same no-overturning
restriction SpectralKM's forward map has.

These are the *same* limits as ours, so adopting this model does not worsen them -- but it
does mean a lubrication upgrade buys nothing toward arbitrary `Oh` (tracked separately).

## 5. Checked and cleared -- these are NOT flaws

**The free parameter `epsilon`.** Set to 40 um, and the paper's insensitivity claim holds up:
they tested 1%--10% of `R_0` and report insensitivity, justified because "the fast decay in
pressure as `h` increases renders the solution insensitive to larger values of the cutoff"
(`:312`). They also derive `epsilon/R_0 ~ (mu_a/sqrt(rho_b R_0 sigma))^{1/3} ~ 0.05`
consistent with Moore (2021)'s inertial-cushioning law (`:360`). This is a well-handled
cutoff, not a fudge. Note `epsilon` bounds the *outer* edge of the lubrication region where
pressure has already decayed; it is not the contact-zone film thickness.

**Air compressibility.** They solve incompressible air (`div u_a = 0`). Peak pressure is
`7.6 x 2 sigma/R_0`, i.e. 1319 Pa at `R_0 = 0.83 mm` and 4378 Pa at `R_0 = 0.25 mm` --
**1.3% and 4.3% of atmospheric**. Incompressible is amply justified at these impact speeds.
It would NOT be at higher `We`, where Mandre & Mani's compressible cushioning matters.

## 6. Minor issues

* **Probable typo.** `:286` gives `R_0 = 0.83 x 10^-4 m` for the detailed case. The table
  (`:245`) ranges `2.5-8.3 x 10^-4 m`, and `:312` states `epsilon = 40 um` is "approximately
  5% the radius". 40 um is 4.8% of 0.83 **mm** but 48% of 0.83 x 10^-4 m. So `:286` should
  read `0.83 x 10^-3 m`. Worth confirming before we quote any of their numbers.
* **Pressure profile not fully converged.** "Small oscillations observed in the pressure
  profile at two intermediate times, we believe are numerical artifacts, as they are lessened
  by increased mesh resolution" (`:308`). Honest, but it means the fine structure of the
  reported profile is partly numerical -- relevant if we ever benchmark our pressure against
  theirs.
* **Cost.** Longest runs `O(10 hours)` on a personal computer (`:312`), though they note an
  adaptive timestep would help. Cheaper than DNS, far more expensive than a kinematic match.

## 7. What to expect if we adopt lubrication physics

**Trustworthy:** the net force and trajectory, which are dominated by the quasi-uniform
interior pressure -- "Most of the work on the drop is due to the quasi-uniform pressure at
intermediate times" (`:310`) -- and validated against experiment for spheres. Also the
qualitative three-phase structure (rapid expansion, slow contraction, suction at detachment)
and the `O(1 um)` film scale, which matches Tang et al. 2019 measurements.

**Not trustworthy without repair:** the rim pressure spike, quantitatively. It is the
model's most novel output and it sits at `Kn ~ 0.14-0.45`.

**Recommended shape of the work, if we proceed:**

1. Port the axisymmetric lubrication layer as a free-boundary elliptic solve for `P`, keeping
   our existing spectral bath and droplet -- this replaces the assumed pressure shape and the
   `theta_c` selector at once, since `Omega_L` emerges from `h = epsilon` instead of from a
   feasibility edge.
2. **Include the slip correction from the start** (`h^3 -> h^3(1 + 6 Kn)`). It is nearly free
   and without it the rim physics is not defensible.
3. Validate on the **axisymmetric deformable** droplet against Alventosa et al.'s DNS and
   experiments, plus our own group's `Deformable_impactors.tex` (arXiv:2509.22826). The 2D
   deformable case is validated upstream; the axisymmetric deformable case is not, and that is
   the one we need.
4. Expect this to resolve the pressure-shape question honestly: the top hat plus rim
   structure would then be an *output*, not an ansatz, and the Gibbs problem in the current
   pressure basis becomes moot because we would no longer be expanding a near-discontinuous
   `p` in smooth polynomials on a patch (see 3b(iii)).
5. Budget for the new coupling term: the droplet's tangential surface velocity, which we do
   not currently compute (3b(iv)).
6. Do not expect slip alone to make the rim quantitative -- the slowly-varying-gap assumption
   also degrades there (3b(i)). If the rim matters for a result we intend to publish, it needs
   either a higher-order (non-lubrication) treatment of the pinch or an explicit statement of
   what is qualitative.
