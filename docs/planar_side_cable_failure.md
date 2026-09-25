# Planar side-cable failure in the canonical 3-strut prism

## Summary

This note documents a reproducible failure mode of the canonical prestressed
3-strut prism implemented in this repository.

Removing one of the three side cables from the bilateral axial-spring prism
causes unconstrained energy relaxation to approach a geometrically singular
planar state.

The limiting planar state can be reconstructed directly and satisfies the full
three-dimensional force-balance equations to machine precision.

At that state:

- the rigidity rank is 9,
- the internal mechanism dimension is 3,
- the self-stress dimension is 2,
- all three exact mechanisms are out-of-plane,
- the physical spring-force state occupies one direction in the 2D
  self-stress space,
- all three mechanisms have positive second-order energy curvature,
- direct nonlinear energy sweeps verify the predicted quadratic stabilization.

The result is therefore an example of a first-order flexible framework that is
prestress-stabilized at second order within the current bilateral spring model.

---

## 1. Model

The reference structure is the canonical six-node, twelve-member 3-strut prism
defined in:

```text
src/tensegrity/examples.py
```

It contains:

- 3 compression struts,
- 9 tension cables,
- 18 Cartesian degrees of freedom,
- rigidity rank 11,
- 1 infinitesimal mechanism,
- 1 state of self-stress.

Rest lengths are constructed from the unique state of self-stress so that the
reference geometry is a true force-equilibrium configuration.

The energy model is currently bilateral:

\[
E =
\frac{1}{2}
\sum_e
k_e
\left(
L_e-L_{0,e}
\right)^2 .
\]

The corresponding force-density value for member \(e\) is

\[
q_e =
k_e
\frac{L_e-L_{0,e}}{L_e}.
\]

At equilibrium,

\[
R^T q = 0,
\]

where \(R\) is the rigidity matrix.

---

## 2. Side-cable ablation

The representative side cable considered here is member 9:

```text
(0, 3)
```

After removing this cable, eleven members remain.

The damaged structure therefore has:

```text
R shape = (11, 18)
```

and after rigid-body motions are removed:

```text
R_internal shape = (11, 12)
```

At a generic rank-11 configuration, the damaged framework has one exact
internal infinitesimal mechanism.

---

## 3. Unconstrained relaxation

The damaged structure was relaxed using the analytical spring-energy gradient
and Armijo backtracking.

The relaxation did not satisfy the nominal residual threshold of \(10^{-9}\)
before the line search reached numerical limits.

However, the residual decreased from approximately

\[
4.49\times 10^{-1}
\]

to

\[
1.80\times 10^{-8}.
\]

The final elastic energy approached

\[
E =
0.1488746997651.
\]

The spring-force-density norm remained finite:

\[
\|q\|
\approx
0.298764735497.
\]

This showed that the small equilibrium residual was not caused by disappearance
of the internal forces.

---

## 4. Approach to singularity

The two smallest rigidity singular values decreased toward zero during
relaxation.

At the final unconstrained iterate:

\[
\sigma_1
\approx
6.04\times10^{-8},
\]

\[
\sigma_2
\approx
2.00\times10^{-7}.
\]

The next singular value remained of order unity:

\[
\sigma_3
\approx
1.139.
\]

The equilibrium residual scaled approximately linearly with the smallest
singular value:

\[
\|\nabla E\|
\propto
\sigma_1.
\]

The spring-force vector aligned almost exactly with the left singular vector
associated with the smallest singular value:

\[
\cos(q,u_1)
\approx
1.
\]

In contrast,

\[
\cos(q,u_2)
\approx
0.
\]

Thus two kinematic directions became soft, while the actual prestress state
selected by relaxation approached only one of the corresponding
near-self-stress directions.

---

## 5. Geometric collapse to a plane

The centered node-coordinate matrix of the relaxed geometry had singular
values

\[
3.353665043506,
\]

\[
1.616665966645,
\]

\[
2.456898194655\times10^{-7}.
\]

Therefore,

\[
\frac{s_2}{s_1}
\approx
0.482,
\]

while

\[
\frac{s_3}{s_1}
\approx
7.33\times10^{-8}.
\]

The structure is therefore approaching a planar, rather than line-like,
configuration.

The maximum node distance from the best-fit plane was approximately

\[
1.40\times10^{-7},
\]

with RMS distance

\[
1.00\times10^{-7}.
\]

Projecting the relaxed geometry exactly onto this nearest plane caused only a
very small geometric perturbation.

At the exactly planar projection, the two smallest rigidity singular values
fell to machine precision:

\[
4.74\times10^{-16},
\]

\[
1.03\times10^{-16}.
\]

The projected geometry had:

```text
rigidity rank = 9
internal mechanisms = 3
self-stress dimension = 2
```

over numerical tolerances ranging from \(10^{-6}\) through \(10^{-12}\).

---

## 6. Exact planar equilibrium

The approximately planar relaxed state was used as the starting point for a
constrained planar nonlinear equilibrium solve.

The nodes were represented in the best-fit 2D plane basis and the free
in-plane coordinates were solved using nonlinear least squares.

The final state satisfied:

\[
\|\nabla_{\mathrm{plane}}E\|
=
1.43\times10^{-15},
\]

and, crucially,

\[
\|\nabla_{\mathrm{3D}}E\|
=
1.44\times10^{-15}.
\]

Thus the state is not merely an in-plane constrained solution.

It is a full three-dimensional force equilibrium that happens to be exactly
planar.

Its energy remained

\[
E =
0.1488746997651,
\]

matching the plateau reached by unconstrained relaxation.

The exact planar coordinate spectrum was

\[
3.353665043506,
\]

\[
1.616665966645,
\]

\[
8.79\times10^{-17}.
\]

The corresponding rigidity spectrum contained two additional zeros at machine
precision.

---

## 7. Exact self-stress structure

At the exact planar state,

\[
\dim \ker(R^T)=2.
\]

The physical spring force-density vector satisfies

\[
\|R^Tq\|
\approx
1.47\times10^{-15}.
\]

Projecting the physical force vector into the computed self-stress basis gave

\[
\|q-QQ^Tq\|
\approx
4.51\times10^{-16}.
\]

Therefore the physical force state lies entirely inside the exact
two-dimensional self-stress space.

A physically meaningful basis was then constructed:

\[
s_{\mathrm{loaded}}
=
\frac{q}{\|q\|},
\]

with a second orthogonal self-stress vector
\(s_{\mathrm{unused}}\).

The projections were

\[
q\cdot s_{\mathrm{loaded}}
=
0.298764735497,
\]

and

\[
q\cdot s_{\mathrm{unused}}
\approx
4.64\times10^{-17}.
\]

Thus the equilibrium uses one specific self-stress direction while the second
exact self-stress direction is geometrically available but unloaded.

---

## 8. Exact mechanism space

After rigid-body modes are removed, the planar framework has three exact
internal infinitesimal mechanisms.

Their rigidity residuals were all approximately \(10^{-15}\).

All three mechanisms were almost perfectly normal to the plane.

For each mechanism,

```text
in-plane norm  ~ 10^-16
out-of-plane norm = 1
```

to numerical precision.

Thus the three first-order mechanisms correspond to out-of-plane deformation
of the planar framework.

---

## 9. Second-order stability

The full energy Hessian was computed numerically and projected onto the
non-rigid subspace.

All twelve non-rigid Hessian eigenvalues were positive.

The smallest was

\[
\lambda_{\min}
=
0.073425017023.
\]

There were:

```text
negative modes = 0
near-zero modes = 0
positive modes = 12
```

The Hessian restricted specifically to the three-dimensional exact mechanism
subspace had eigenvalues

\[
0.073425017023,
\]

\[
0.329190983609,
\]

\[
0.465030956237.
\]

Therefore every exact first-order mechanism has positive quadratic energy
curvature.

Within the current bilateral spring model, the planar rank-9 equilibrium is
prestress-stabilized at second order.

---

## 10. Direct nonlinear verification

The second-order result was independently checked by perturbing the exact
equilibrium along each of the three mechanism-Hessian eigenvectors.

For each normalized mechanism mode \(v\), the actual nonlinear energy

\[
\Delta E(a)
=
E(X+av)-E(X)
\]

was compared against

\[
\Delta E_{\mathrm{quad}}(a)
=
\frac{1}{2}
\lambda a^2.
\]

Amplitudes from

\[
10^{-4}
\]

through

\[
10^{-2}
\]

were tested in both positive and negative directions.

For all three modes:

- every nonzero tested perturbation increased the energy,
- no negative-energy perturbations were observed,
- the positive and negative branches were symmetric to numerical precision,
- the ratio

\[
\frac{\Delta E_{\mathrm{actual}}}
{\Delta E_{\mathrm{quad}}}
\]

approached 1 for small amplitude.

For the softest mode,

\[
\lambda
=
0.073425017023,
\]

and at

\[
a=10^{-3},
\]

the measured energy increase was

\[
3.67126517\times10^{-8},
\]

compared with the quadratic prediction

\[
3.67125085\times10^{-8}.
\]

The ratio was approximately

\[
1.0000039.
\]

This provides a direct nonlinear confirmation of the positive quadratic
curvature.

---

## 11. Interpretation

The side-cable failure sequence is therefore:

\[
\text{prestressed 3D prism}
\]

\[
\downarrow
\]

\[
\text{remove side cable}
\]

\[
\downarrow
\]

\[
\text{large geometric relaxation}
\]

\[
\downarrow
\]

\[
\text{approach planar singular geometry}
\]

\[
\downarrow
\]

\[
\text{exact planar rank-9 equilibrium}.
\]

At the limiting equilibrium:

\[
\operatorname{rank}(R)=9,
\]

\[
\text{internal mechanism dimension}=3,
\]

\[
\text{self-stress dimension}=2.
\]

Despite possessing three first-order mechanisms, the structure is a local
quadratic energy minimum because prestress supplies positive second-order
stiffness along all three mechanism directions.

This is an explicit computational example of the distinction between
first-order rigidity and prestress stability.

---

## 12. Comparison with perimeter-cable removal

The behavior differs qualitatively from removal of a perimeter cable such as
member 3.

For perimeter-cable removal:

- only one additional rigidity singular value approaches zero,
- the second-smallest singular value remains approximately \(0.991\),
- the relaxed geometry remains fully three-dimensional,
- the centered coordinate singular-value ratios remain of order unity.

For representative member 3:

\[
\frac{s_2}{s_1}
\approx
0.676,
\]

\[
\frac{s_3}{s_1}
\approx
0.329.
\]

Thus the planar rank-9 structure obtained by manually projecting the
perimeter-cable case is not a nearby continuation of its relaxed geometry.

The current evidence is instead consistent with:

```text
perimeter-cable failure:
    3D approach toward rank 10
    one additional soft mode

side-cable failure:
    planar approach toward rank 9
    two additional soft modes
```

---

## 13. Limitations

Several limitations are important.

### Bilateral constitutive law

The present energy model treats all members as bilateral axial springs.

Therefore:

- cables can carry compression,
- struts can carry tension.

This is not yet a complete unilateral tensegrity constitutive model.

The same experiments should eventually be repeated using:

- tension-only cables,
- compression-only struts.

### Small benchmark

The current result concerns one canonical six-node prism.

It should not yet be generalized to arbitrary tensegrity structures.

### Numerical construction

The planar equilibrium is established numerically to machine precision rather
than symbolically.

An analytic construction of the limiting geometry would strengthen the result.

### Biological interpretation

No claim is made here that this particular prism failure mode directly models
a biological tissue process.

The result is currently a mechanics benchmark.

---

## 14. Reproduction

Relevant experiments are:

```text
experiments/prism_relaxed_ablation.py
experiments/prism_relaxed_ablation_diagnostic.py
experiments/prism_singularity_trajectory.py
experiments/prism_soft_modes.py
experiments/prism_projected_modes.py
experiments/prism_limiting_geometry.py
experiments/prism_planar_equilibrium.py
experiments/prism_planar_self_stress.py
experiments/prism_planar_stability.py
experiments/prism_planar_energy_sweep.py
```

Representative output directories include:

```text
outputs/prism_singularity_trajectory/
outputs/prism_soft_modes/
outputs/prism_projected_modes/
outputs/prism_limiting_geometry/
outputs/prism_planar_energy_sweep/
```

---

## 15. Next research direction

The canonical prism now provides a validated mechanics benchmark for:

- rigidity,
- mechanisms,
- self-stress,
- prestress stabilization,
- member ablation,
- singular relaxation,
- exact limiting equilibrium construction.

The next structural extension is to reproduce a biologically motivated
attachment architecture based on Susan Crawford-Young's epithelial tensegrity
models.

The target architecture is:

```text
central T4 trijunction
+
T3 attachment structures connected toward posts/supports
+
eventually multiple connected epithelial-cell units
```

The same analysis pipeline can then be applied to:

- attachment removal,
- prestress changes,
- substrate stiffness,
- imposed tissue stretch,
- deformation-mode changes,
- stability transitions.

This provides a route from the canonical prism benchmark toward computational
models of epithelial tissue stretching.
