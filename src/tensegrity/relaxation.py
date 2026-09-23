"""Energy relaxation utilities for free tensegrity frameworks.

The mechanics energy is invariant under rigid translation and rotation. Direct
optimization in Cartesian coordinates therefore contains rigid-body flat
directions. This module removes that numerical drift by rigidly aligning every
accepted iterate to the initial reference geometry.

The current energy layer is bilateral; unilateral cable/strut laws remain a
separate future milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .energy import (
    SpringMember,
    energy_gradient,
    equilibrium_residual_norm,
    total_energy,
)


@dataclass(frozen=True)
class RelaxationResult:
    """Result of an energy-minimization relaxation."""

    nodes: np.ndarray
    converged: bool
    iterations: int
    energy_initial: float
    energy_final: float
    residual_initial: float
    residual_final: float
    max_displacement: float
    accepted_steps: int
    final_step_size: float
    message: str


def rigid_align_to_reference(
    nodes: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return nodes optimally rigid-aligned to reference.

    A proper Kabsch rotation is used, so reflections are not introduced.
    Translation is removed by matching centroids.
    """
    X = np.asarray(nodes, dtype=float)
    Y = np.asarray(reference, dtype=float)

    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(
            "nodes and reference must both have shape (n, 3)"
        )

    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = Y.mean(axis=0, keepdims=True)

    Xc = X - x_mean
    Yc = Y - y_mean

    U, _, Vt = np.linalg.svd(Xc.T @ Yc)

    rotation = U @ Vt

    # Prevent accidental reflection.
    if np.linalg.det(rotation) < 0.0:
        U[:, -1] *= -1.0
        rotation = U @ Vt

    return Xc @ rotation + y_mean


def max_aligned_displacement(
    nodes: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return the largest node displacement after rigid alignment."""
    aligned = rigid_align_to_reference(nodes, reference)

    displacement = np.linalg.norm(
        aligned - np.asarray(reference, dtype=float),
        axis=1,
    )

    if displacement.size == 0:
        return 0.0

    return float(np.max(displacement))


def relax_framework(
    nodes: np.ndarray,
    members: Iterable[SpringMember],
    *,
    residual_tol: float = 1e-9,
    max_iterations: int = 5000,
    initial_step_size: float = 0.2,
    min_step_size: float = 1e-12,
    armijo: float = 1e-4,
    backtrack_factor: float = 0.5,
) -> RelaxationResult:
    """Relax a free framework toward a local energy minimum.

    Gradient descent with Armijo backtracking is used because the repository
    already provides an analytical gradient and this keeps the mechanics core
    NumPy-only.

    After each accepted step, the geometry is rigidly aligned to the initial
    configuration to suppress physically irrelevant rigid-body drift.

    Convergence means the Euclidean gradient norm is <= residual_tol.
    Failure to converge is reported explicitly rather than silently treated
    as an equilibrium.
    """
    X0 = np.asarray(nodes, dtype=float)

    if X0.ndim != 2 or X0.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")

    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    if initial_step_size <= 0.0 or min_step_size <= 0.0:
        raise ValueError("step sizes must be positive")

    if min_step_size > initial_step_size:
        raise ValueError(
            "min_step_size cannot exceed initial_step_size"
        )

    if not (0.0 < backtrack_factor < 1.0):
        raise ValueError(
            "backtrack_factor must lie in (0, 1)"
        )

    if not (0.0 < armijo < 1.0):
        raise ValueError("armijo must lie in (0, 1)")

    spring_list: List[SpringMember] = list(members)

    X = X0.copy()

    energy_initial = total_energy(X, spring_list)
    residual_initial = equilibrium_residual_norm(
        X,
        spring_list,
    )

    if residual_initial <= residual_tol:
        return RelaxationResult(
            nodes=X,
            converged=True,
            iterations=0,
            energy_initial=energy_initial,
            energy_final=energy_initial,
            residual_initial=residual_initial,
            residual_final=residual_initial,
            max_displacement=0.0,
            accepted_steps=0,
            final_step_size=initial_step_size,
            message=(
                "initial geometry already satisfies "
                "the equilibrium tolerance"
            ),
        )

    step_size = float(initial_step_size)
    accepted_steps = 0
    message = "maximum iterations reached"
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        grad = energy_gradient(
            X,
            spring_list,
        )

        residual = float(np.linalg.norm(grad))

        if residual <= residual_tol:
            message = "equilibrium residual reached tolerance"

            return RelaxationResult(
                nodes=X,
                converged=True,
                iterations=iteration - 1,
                energy_initial=energy_initial,
                energy_final=total_energy(
                    X,
                    spring_list,
                ),
                residual_initial=residual_initial,
                residual_final=residual,
                max_displacement=max_aligned_displacement(
                    X,
                    X0,
                ),
                accepted_steps=accepted_steps,
                final_step_size=step_size,
                message=message,
            )

        energy_here = total_energy(
            X,
            spring_list,
        )

        grad_sq = residual * residual

        trial_step = step_size
        accepted = False

        while trial_step >= min_step_size:
            candidate = X - trial_step * grad

            # Remove rigid-body drift.
            candidate = rigid_align_to_reference(
                candidate,
                X0,
            )

            candidate_energy = total_energy(
                candidate,
                spring_list,
            )

            armijo_rhs = (
                energy_here
                - armijo * trial_step * grad_sq
            )

            if candidate_energy <= armijo_rhs:
                X = candidate
                accepted_steps += 1

                # Allow step size to grow back after a successful step.
                step_size = min(
                    initial_step_size,
                    trial_step / backtrack_factor,
                )

                accepted = True
                break

            trial_step *= backtrack_factor

        if not accepted:
            step_size = trial_step
            message = (
                "line search reached minimum step size"
            )
            break

    residual_final = equilibrium_residual_norm(
        X,
        spring_list,
    )

    return RelaxationResult(
        nodes=X,
        converged=residual_final <= residual_tol,
        iterations=iteration,
        energy_initial=energy_initial,
        energy_final=total_energy(
            X,
            spring_list,
        ),
        residual_initial=residual_initial,
        residual_final=residual_final,
        max_displacement=max_aligned_displacement(
            X,
            X0,
        ),
        accepted_steps=accepted_steps,
        final_step_size=step_size,
        message=message,
    )
