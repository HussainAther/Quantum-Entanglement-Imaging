"""Viscoelastic support models used by tensegrity experiments.

The main abstraction is a Kelvin-Voigt basement/foundation connection:
a linear spring and dashpot in parallel.  The tensegrity members themselves
remain tension-only cables / compression-only struts; this module represents
only the compliant external attachment.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _indices(support_nodes: Iterable[int]) -> np.ndarray:
    idx = np.asarray(list(support_nodes), dtype=int)
    if idx.ndim != 1:
        raise ValueError("support_nodes must be one-dimensional")
    return idx


def kelvin_voigt_foundation_force(
    nodes: np.ndarray,
    velocities: np.ndarray,
    reference_nodes: np.ndarray,
    support_nodes: Iterable[int],
    stiffness: float,
    damping: float,
) -> np.ndarray:
    """Return restoring force exerted by a Kelvin-Voigt foundation.

    At each support node the force is

        F = -k (x - x_ref) - c v.
    """
    X = np.asarray(nodes, dtype=float)
    V = np.asarray(velocities, dtype=float)
    X0 = np.asarray(reference_nodes, dtype=float)
    if X.shape != X0.shape or V.shape != X.shape:
        raise ValueError("nodes, velocities, and reference_nodes must match")
    if stiffness < 0.0 or damping < 0.0:
        raise ValueError("stiffness and damping must be non-negative")

    idx = _indices(support_nodes)
    force = np.zeros_like(X)
    force[idx] = (
        -stiffness * (X[idx] - X0[idx])
        - damping * V[idx]
    )
    return force


def foundation_incremental_potential_and_gradient(
    nodes: np.ndarray,
    previous_nodes: np.ndarray,
    reference_nodes: np.ndarray,
    support_nodes: Iterable[int],
    stiffness: float,
    damping: float,
    dt: float,
) -> Tuple[float, np.ndarray]:
    """Backward-Euler incremental potential for a Kelvin-Voigt foundation.

    Minimizing this potential together with the structural potential gives the
    force balance

        grad(E_structure)
        + k (x_s - x_ref)
        + c (x_s - x_s_prev) / dt
        - F_external = 0

    at the support nodes.  Internal tensegrity nodes are still treated
    quasi-statically, while the basement attachment contributes rate dependence.
    """
    X = np.asarray(nodes, dtype=float)
    Xprev = np.asarray(previous_nodes, dtype=float)
    X0 = np.asarray(reference_nodes, dtype=float)
    if X.shape != Xprev.shape or X.shape != X0.shape:
        raise ValueError("nodes, previous_nodes, and reference_nodes must match")
    if stiffness < 0.0 or damping < 0.0:
        raise ValueError("stiffness and damping must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    idx = _indices(support_nodes)
    elastic_disp = X[idx] - X0[idx]
    step_disp = X[idx] - Xprev[idx]

    potential = (
        0.5 * stiffness * float(np.sum(elastic_disp * elastic_disp))
        + 0.5 * damping / dt * float(np.sum(step_disp * step_disp))
    )

    gradient = np.zeros_like(X)
    gradient[idx] = (
        stiffness * elastic_disp
        + damping / dt * step_disp
    )
    return float(potential), gradient
