"""
rigidity.py

Minimal rigidity-matrix utilities for tensegrity-like bar-joint frameworks.

This is a first-pass tool for detecting mechanisms / collapse risk by:
- building the rigidity matrix R
- computing its singular values
- counting near-zero singular values (extra floppy modes)

Notes:
- This treats members as distance constraints (bar-joint rigidity).
- It does NOT yet incorporate prestress stability or cable unilateral behavior.
- Still useful as a fast "does removing an edge introduce a mechanism?" check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import numpy as np

MemberType = Literal["bar", "cable"]


@dataclass(frozen=True)
class Member:
    i: int
    j: int
    kind: MemberType = "cable"


def rigidity_matrix(nodes: np.ndarray, members: Iterable[Member]) -> np.ndarray:
    """
    Build the standard bar-joint rigidity matrix R for a framework in 3D.

    nodes: (n, 3) array of node coordinates
    members: list of Member(i, j, kind)

    Returns:
        R: (m, 3n) rigidity matrix
    """
    X = np.asarray(nodes, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")

    n = X.shape[0]
    mem_list = list(members)
    m = len(mem_list)

    R = np.zeros((m, 3 * n), dtype=float)

    for row, mem in enumerate(mem_list):
        i, j = mem.i, mem.j
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError(f"Invalid member endpoints: {mem}")

        d = X[i] - X[j]  # direction vector
        # Fill row blocks for i and j
        R[row, 3 * i : 3 * i + 3] = d
        R[row, 3 * j : 3 * j + 3] = -d

    return R


def singular_values_of_rigidity(nodes: np.ndarray, members: Iterable[Member]) -> np.ndarray:
    """
    Compute singular values of the rigidity matrix R.
    """
    R = rigidity_matrix(nodes, members)
    s = np.linalg.svd(R, compute_uv=False)
    return s


def count_mechanisms(nodes: np.ndarray, members: Iterable[Member], tol: float = 1e-9) -> int:
    """
    Count the dimension of the nullspace of R (floppy modes) excluding rigid-body motions.

    In 3D, rigid-body motions are 6 (3 translations + 3 rotations), assuming generic positioning.

    Returns:
        mechanisms = nullity(R) - 6, lower bounded by 0.
    """
    X = np.asarray(nodes, dtype=float)
    n = X.shape[0]
    dof = 3 * n

    s = singular_values_of_rigidity(X, members)
    rank = int(np.sum(s > tol))
    nullity = dof - rank

    # Subtract rigid-body modes (6 in 3D)
    mech = max(0, nullity - 6)
    return mech


def remove_one_cable_scan(
    nodes: np.ndarray,
    members: List[Member],
    tol: float = 1e-9,
) -> List[Tuple[int, Member, int, int]]:
    """
    For each cable member, remove it and recompute mechanism count.

    Returns a list of tuples:
      (index_removed, member_removed, mechanisms_before, mechanisms_after)
    """
    mechs_before = count_mechanisms(nodes, members, tol=tol)
    results: List[Tuple[int, Member, int, int]] = []

    for idx, mem in enumerate(members):
        if mem.kind != "cable":
            continue

        reduced = members[:idx] + members[idx + 1 :]
        mechs_after = count_mechanisms(nodes, reduced, tol=tol)
        results.append((idx, mem, mechs_before, mechs_after))

    return results

