# Quantum Entanglement Imaging / Computational Tensegrity Research

This repository preserves an exploratory research history spanning quantum-imaging prototypes, X-ray simulation experiments, and Richard Gordon-inspired tensegrity questions. The current scientifically developed component is the **computational tensegrity mechanics core** under `src/tensegrity/`.

The immediate research goal is deliberately classical and testable: characterize rigidity, self-stress, prestress stabilization, and structural failure in tensegrity frameworks. Quantum/atomic interpretations remain hypotheses and are not assumed by the mechanics code.

## Current tensegrity capabilities

- 3D rigidity-matrix construction
- numerical rigidity rank and mechanism counting
- self-stress dimension and self-stress basis (`ker(R.T)`)
- bilateral axial-spring energy model
- analytical energy gradient / equilibrium residual
- finite-difference energy Hessian
- explicit removal of rigid-body modes before stability classification
- stable / marginal / unstable Hessian classification
- single-member perturbation scans
- canonical equilibrated 3-strut tensegrity prism benchmark
- equilibrium prestress sweep
- single-cable ablation diagnostics

## Install / run

The package currently requires Python 3.9+, NumPy, and Matplotlib.

```bash
python -m pip install -e .
pytest -q
```

Without editable installation:

```bash
PYTHONPATH=src pytest -q
```

## Canonical prism benchmark

```bash
PYTHONPATH=src python experiments/prism_benchmark.py
```

The benchmark uses a six-node, twelve-member 3-strut prism with:

- 3 compression struts
- 9 tension cables
- rigidity rank 11
- 1 infinitesimal mechanism
- 1 state of self-stress

Crucially, the spring rest lengths are derived from a vector in `ker(R.T)` rather than assigned by arbitrary percentage offsets. This makes the reference geometry an actual force-equilibrium configuration (up to floating-point error), so the non-rigid Hessian spectrum can be interpreted as a local-stability test.

A typical run gives an equilibrium residual around `1e-15` and a positive minimum non-rigid Hessian eigenvalue, demonstrating prestress stabilization of a first-order mechanism.

## Prestress sweep

```bash
PYTHONPATH=src python experiments/prism_prestress_sweep.py
```

This writes:

```text
outputs/prism_prestress_sweep.csv
```

At zero prestress the canonical prism is marginal. Positive equilibrium prestress lifts the mechanism and produces a positive non-rigid stiffness mode over the tested range.

## Cable ablation

```bash
PYTHONPATH=src python experiments/prism_member_ablation.py
```

This writes:

```text
outputs/prism_member_ablation.csv
```

The ablation experiment reports rigidity, loss of self-stress, equilibrium residual, and the instantaneous Hessian curvature after each cable is removed.

**Important:** removing a prestressed cable generally destroys force equilibrium at the original geometry. Therefore the post-ablation Hessian is labeled an *instantaneous curvature diagnostic*, not the stability of a relaxed post-failure equilibrium. A future milestone is to solve for the new equilibrium before assigning post-failure stability.

## Repository layout

```text
src/tensegrity/
    __init__.py
    energy.py
    rigidity.py
    examples.py

experiments/
    prism_benchmark.py
    prism_prestress_sweep.py
    prism_member_ablation.py
    cable_removal/
    stability_map/

tests/
    test_energy.py
    test_rigidity.py
    test_examples.py

docs/literature/
    README.md
    2012_fraternali_tensegrity_systems.md

algorithms/
    quantum_noise_reduction.py
    quantum_reconstruction.py
    cnn_image_reconstruction.py

simulations/
    entangled_photon_simulation.py
    rbyrct_simulation.py
    xray_interaction_with_tissue.py
```

## Scientific scope

The tensegrity package currently models all members as bilateral axial springs in the energy layer. Member labels distinguish bars/struts from cables for topology and interpretation, but **tension-only cable and compression-only strut constitutive behavior is not yet implemented**.

Accordingly, the current prism work should be described as a validated prestressed axial-network tensegrity benchmark, not yet a complete unilateral tensegrity solver.

The older quantum, entangled-photon, RBYRCT, and image-reconstruction scripts are retained for provenance but are not evidence that tensegrity applies to quantum physics or that the included classical image filters are quantum algorithms.

## Recommended next milestone

1. Add equilibrium relaxation after member removal.
2. Add unilateral cable/strut constitutive laws.
3. Validate against additional canonical tensegrity structures and published benchmark values.
4. Build robustness metrics only after relaxed post-failure states are available.
5. Keep atomic/quantum applications as explicitly testable downstream hypotheses.
