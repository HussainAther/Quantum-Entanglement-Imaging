"""Rigidity-matrix utilities for 3D tensegrity-like frameworks.

These routines treat every member as a bilateral distance constraint. They
therefore answer first-order bar-joint rigidity questions only. Unilateral
cable/strut behavior and prestress stability belong to the energy/stiffness
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import numpy as np

MemberType = Literal["bar", "cable"]


@dataclass(frozen=True)
class Member:
    """Distance-constraint member joining nodes ``i`` and ``j``."""

    i: int
    j: int
    kind: MemberType = "cable"


@dataclass(frozen=True)
class RigidityResult:
    """Rank/nullity summary for a 3D framework."""

    rank: int
    nullity: int
    rigid_body_modes: int
    mechanisms: int
    self_stress_dimension: int
    singular_values: np.ndarray


def _validate_nodes(nodes: np.ndarray) -> np.ndarray:
    X = np.asarray(nodes, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")
    return X


def _rigid_body_mode_dimension(nodes: np.ndarray, tol: float = 1e-12) -> int:
    """Return the number of independent infinitesimal rigid motions."""
    X = _validate_nodes(nodes)
    n = X.shape[0]
    dof = 3 * n
    if dof == 0:
        return 0

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

    singular_values = np.linalg.svd(B, compute_uv=False)
    return int(np.sum(singular_values > tol))


def rigidity_matrix(nodes: np.ndarray, members: Iterable[Member]) -> np.ndarray:
    """Build the standard 3D bar-joint rigidity matrix R."""
    X = _validate_nodes(nodes)
    n = X.shape[0]
    member_list = list(members)
    R = np.zeros((len(member_list), 3 * n), dtype=float)

    for row, member in enumerate(member_list):
        i, j = member.i, member.j
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError(f"Invalid member endpoints: {member}")

        d = X[i] - X[j]
        R[row, 3 * i : 3 * i + 3] = d
        R[row, 3 * j : 3 * j + 3] = -d

    return R


def singular_values_of_rigidity(
    nodes: np.ndarray,
    members: Iterable[Member],
) -> np.ndarray:
    """Return singular values of the rigidity matrix."""
    R = rigidity_matrix(nodes, members)
    return np.linalg.svd(R, compute_uv=False)


def rigidity_rank(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
) -> int:
    """Return numerical rank of the rigidity matrix."""
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    singular_values = singular_values_of_rigidity(nodes, members)
    return int(np.sum(singular_values > tol))


def mechanism_dimension(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
    rigid_tol: float = 1e-12,
) -> int:
    """Return infinitesimal mechanism count after removing rigid motions."""
    X = _validate_nodes(nodes)
    member_list = list(members)
    rank = rigidity_rank(X, member_list, tol=tol)
    nullity = 3 * X.shape[0] - rank
    rigid_modes = _rigid_body_mode_dimension(X, tol=rigid_tol)
    return max(0, nullity - rigid_modes)


def self_stress_basis(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
) -> np.ndarray:
    """Return an orthonormal basis for states of self-stress ``ker(R.T)``.

    Columns of the returned array are member force-density vectors ``q``
    satisfying ``R.T @ q == 0`` to numerical tolerance.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    member_list = list(members)
    R = rigidity_matrix(nodes, member_list)
    m = R.shape[0]
    if m == 0:
        return np.empty((0, 0), dtype=float)

    # Right singular vectors of R.T span member-force space.
    _, singular_values, Vt = np.linalg.svd(R.T, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return Vt.T[:, rank:]


def self_stress_dimension(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
) -> int:
    """Return dim ker(R.T), the number of states of self-stress."""
    X = _validate_nodes(nodes)
    member_list = list(members)
    rank = rigidity_rank(X, member_list, tol=tol)
    return max(0, len(member_list) - rank)


def analyze_rigidity(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
    rigid_tol: float = 1e-12,
) -> RigidityResult:
    """Return rank, mechanisms, and self-stress information together."""
    X = _validate_nodes(nodes)
    member_list = list(members)
    singular_values = singular_values_of_rigidity(X, member_list)
    rank = int(np.sum(singular_values > tol))
    nullity = 3 * X.shape[0] - rank
    rigid_modes = _rigid_body_mode_dimension(X, tol=rigid_tol)

    return RigidityResult(
        rank=rank,
        nullity=nullity,
        rigid_body_modes=rigid_modes,
        mechanisms=max(0, nullity - rigid_modes),
        self_stress_dimension=max(0, len(member_list) - rank),
        singular_values=singular_values,
    )


def count_mechanisms(
    nodes: np.ndarray,
    members: Iterable[Member],
    tol: float = 1e-9,
) -> int:
    """Backward-compatible alias for :func:`mechanism_dimension`."""
    return mechanism_dimension(nodes, members, tol=tol)


def remove_one_cable_scan(
    nodes: np.ndarray,
    members: List[Member],
    tol: float = 1e-9,
) -> List[Tuple[int, Member, int, int]]:
    """Remove each cable once and report mechanism count before/after."""
    mechanisms_before = mechanism_dimension(nodes, members, tol=tol)
    results: List[Tuple[int, Member, int, int]] = []

    for idx, member in enumerate(members):
        if member.kind != "cable":
            continue
        reduced = members[:idx] + members[idx + 1 :]
        mechanisms_after = mechanism_dimension(nodes, reduced, tol=tol)
        results.append((idx, member, mechanisms_before, mechanisms_after))

    return results
