"""Energy-based local stability utilities for tensegrity-like frameworks.

Members are currently modeled as bilateral axial springs with energy

    E = 0.5 * k * (L - L0)^2

This is intentionally a small mechanics core. Cable tension-only and strut
compression-only behavior should be added later, after the bilateral model is
validated against canonical benchmark structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal

import numpy as np

MemberType = Literal["bar", "cable"]
StabilityClass = Literal["stable", "marginal", "unstable"]


@dataclass(frozen=True)
class SpringMember:
    """Axial spring member joining nodes ``i`` and ``j``."""

    i: int
    j: int
    kind: MemberType
    k: float
    L0: float


@dataclass(frozen=True)
class StabilityResult:
    """Summary of the Hessian spectrum on the non-rigid subspace."""

    lambda_min: float
    negative_modes: int
    zero_modes: int
    positive_modes: int
    classification: StabilityClass
    eigenvalues: np.ndarray


def _validate_nodes(nodes: np.ndarray) -> np.ndarray:
    X = np.asarray(nodes, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")
    return X


def total_energy(nodes: np.ndarray, members: Iterable[SpringMember]) -> float:
    """Return total bilateral-spring energy for a framework."""
    X = _validate_nodes(nodes)
    energy = 0.0

    for member in members:
        n = X.shape[0]
        if not (0 <= member.i < n and 0 <= member.j < n) or member.i == member.j:
            raise ValueError(f"Invalid member endpoints: {member}")
        if member.k < 0.0:
            raise ValueError("spring stiffness k must be non-negative")
        if member.L0 < 0.0:
            raise ValueError("rest length L0 must be non-negative")

        d = X[member.i] - X[member.j]
        length = float(np.linalg.norm(d))
        extension = length - member.L0
        energy += 0.5 * member.k * extension * extension

    return float(energy)


def energy_gradient(
    nodes: np.ndarray,
    members: Iterable[SpringMember],
) -> np.ndarray:
    """Return the analytical gradient ``dE/dx`` with shape ``(n, 3)``."""
    X = _validate_nodes(nodes)
    grad = np.zeros_like(X, dtype=float)

    for member in members:
        n = X.shape[0]
        if not (0 <= member.i < n and 0 <= member.j < n) or member.i == member.j:
            raise ValueError(f"Invalid member endpoints: {member}")
        if member.k < 0.0:
            raise ValueError("spring stiffness k must be non-negative")
        if member.L0 < 0.0:
            raise ValueError("rest length L0 must be non-negative")

        d = X[member.i] - X[member.j]
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError("zero-length members are not supported")

        g = member.k * (L - member.L0) * (d / L)
        grad[member.i] += g
        grad[member.j] -= g

    return grad


def equilibrium_residual_norm(
    nodes: np.ndarray,
    members: Iterable[SpringMember],
) -> float:
    """Return ``||dE/dx||_2``; equilibrium requires this to be near zero."""
    return float(np.linalg.norm(energy_gradient(nodes, members)))


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Return a central-finite-difference Hessian for scalar ``f(x)``."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")

    n = x.size
    H = np.zeros((n, n), dtype=float)
    fx = float(f(x))

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = 1.0

        for j in range(i, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = 1.0

            if i == j:
                fpp = float(f(x + eps * ei))
                fmm = float(f(x - eps * ei))
                value = (fpp - 2.0 * fx + fmm) / (eps * eps)
            else:
                fpp = float(f(x + eps * ei + eps * ej))
                fpm = float(f(x + eps * ei - eps * ej))
                fmp = float(f(x - eps * ei + eps * ej))
                fmm = float(f(x - eps * ei - eps * ej))
                value = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)

            H[i, j] = value
            H[j, i] = value

    return 0.5 * (H + H.T)


def rigid_body_basis(nodes: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for independent rigid-body motions.

    For a generic non-collinear 3D framework this has six columns: three
    translations and three rotations. Degenerate geometries can have fewer
    independent rotational modes, so rank is determined numerically.
    """
    X = _validate_nodes(nodes)
    n = X.shape[0]
    dof = 3 * n
    if dof == 0:
        return np.empty((0, 0), dtype=float)

    centered = X - X.mean(axis=0, keepdims=True)
    B = np.zeros((dof, 6), dtype=float)

    for a in range(n):
        B[3 * a + 0, 0] = 1.0
        B[3 * a + 1, 1] = 1.0
        B[3 * a + 2, 2] = 1.0

        x, y, z = centered[a]
        B[3 * a : 3 * a + 3, 3] = (0.0, -z, y)
        B[3 * a : 3 * a + 3, 4] = (z, 0.0, -x)
        B[3 * a : 3 * a + 3, 5] = (-y, x, 0.0)

    U, singular_values, _ = np.linalg.svd(B, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    return U[:, :rank]


def nonrigid_basis(nodes: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the complement of rigid motions."""
    X = _validate_nodes(nodes)
    dof = 3 * X.shape[0]
    rigid = rigid_body_basis(X, tol=tol)

    if dof == 0:
        return np.empty((0, 0), dtype=float)
    if rigid.shape[1] == 0:
        return np.eye(dof, dtype=float)

    Q, _ = np.linalg.qr(rigid, mode="complete")
    return Q[:, rigid.shape[1] :]


def reduced_nonrigid_hessian(
    H: np.ndarray,
    nodes: np.ndarray,
    rigid_tol: float = 1e-12,
) -> np.ndarray:
    """Express Hessian directly in non-rigid coordinates: ``Z.T @ H @ Z``."""
    X = _validate_nodes(nodes)
    H = np.asarray(H, dtype=float)
    dof = 3 * X.shape[0]

    if H.shape != (dof, dof):
        raise ValueError(f"H must have shape ({dof}, {dof})")

    Z = nonrigid_basis(X, tol=rigid_tol)
    return Z.T @ H @ Z


def stability_index_energy_hessian(
    nodes: np.ndarray,
    members: List[SpringMember],
    eps: float = 1e-5,
    eig_tol: float = 1e-7,
    rigid_tol: float = 1e-12,
) -> StabilityResult:
    """Classify local stability from the non-rigid Hessian spectrum.

    stable: every non-rigid eigenvalue > eig_tol
    marginal: no significantly negative eigenvalues, but >=1 near-zero mode
    unstable: at least one eigenvalue < -eig_tol
    """
    if eig_tol <= 0.0:
        raise ValueError("eig_tol must be positive")

    X = _validate_nodes(nodes)

    def energy_from_flat(flat: np.ndarray) -> float:
        return total_energy(flat.reshape((-1, 3)), members)

    H = numerical_hessian(energy_from_flat, X.reshape(-1), eps=eps)
    H_nonrigid = reduced_nonrigid_hessian(H, X, rigid_tol=rigid_tol)

    if H_nonrigid.size == 0:
        eigenvalues = np.empty(0, dtype=float)
        return StabilityResult(
            lambda_min=float("inf"),
            negative_modes=0,
            zero_modes=0,
            positive_modes=0,
            classification="stable",
            eigenvalues=eigenvalues,
        )

    eigenvalues = np.sort(np.linalg.eigvalsh(H_nonrigid))
    negative_modes = int(np.sum(eigenvalues < -eig_tol))
    zero_modes = int(np.sum(np.abs(eigenvalues) <= eig_tol))
    positive_modes = int(np.sum(eigenvalues > eig_tol))

    if negative_modes:
        classification: StabilityClass = "unstable"
    elif zero_modes:
        classification = "marginal"
    else:
        classification = "stable"

    return StabilityResult(
        lambda_min=float(eigenvalues[0]),
        negative_modes=negative_modes,
        zero_modes=zero_modes,
        positive_modes=positive_modes,
        classification=classification,
        eigenvalues=eigenvalues,
    )
