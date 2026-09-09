"""Canonical benchmark structures for tensegrity mechanics validation."""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np

from .energy import SpringMember
from .rigidity import Member, rigidity_matrix, self_stress_basis


def tetrahedron() -> Tuple[np.ndarray, List[Member], List[SpringMember]]:
    """Return a generic stress-free tetrahedral framework."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.25, 0.95, 0.0],
            [0.30, 0.35, 0.90],
        ],
        dtype=float,
    )
    edges = list(combinations(range(4), 2))
    members = [Member(i, j, kind="bar") for i, j in edges]
    springs = [
        SpringMember(
            i=i,
            j=j,
            kind="bar",
            k=1.0,
            L0=float(np.linalg.norm(nodes[i] - nodes[j])),
        )
        for i, j in edges
    ]
    return nodes, members, springs


def _prism_geometry(
    radius: float = 1.0,
    height: float = 1.5,
    twist: float = np.pi / 6.0,
) -> Tuple[np.ndarray, List[Member]]:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if height <= 0.0:
        raise ValueError("height must be positive")

    lower_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    upper_angles = lower_angles + twist

    lower = np.column_stack(
        [
            radius * np.cos(lower_angles),
            radius * np.sin(lower_angles),
            np.zeros(3),
        ]
    )
    upper = np.column_stack(
        [
            radius * np.cos(upper_angles),
            radius * np.sin(upper_angles),
            np.full(3, height),
        ]
    )
    nodes = np.vstack([lower, upper])

    struts = [(0, 4), (1, 5), (2, 3)]
    lower_cables = [(0, 1), (1, 2), (2, 0)]
    upper_cables = [(3, 4), (4, 5), (5, 3)]
    side_cables = [(0, 3), (1, 4), (2, 5)]

    members = [Member(i, j, kind="bar") for i, j in struts]
    members += [
        Member(i, j, kind="cable")
        for i, j in lower_cables + upper_cables + side_cables
    ]
    return nodes, members


def prism_force_density(
    nodes: np.ndarray,
    members: List[Member],
    prestress_scale: float = 0.20,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return an equilibrium force-density vector for the 3-strut prism.

    The vector lies in ``ker(R.T)``. Its sign is chosen so the first three
    members (struts) are in compression and the nine cables are in tension.
    It is normalized so ``max(abs(q)) == prestress_scale``.
    """
    if prestress_scale < 0.0:
        raise ValueError("prestress_scale must be non-negative")

    basis = self_stress_basis(nodes, members, tol=tol)
    if basis.shape[1] != 1:
        raise ValueError(
            "canonical prism is expected to have exactly one self-stress state"
        )

    q = basis[:, 0].copy()
    # Choose physical sign: struts negative (compression), cables positive.
    if np.mean(q[:3]) > 0.0:
        q *= -1.0

    if not (np.all(q[:3] < 0.0) and np.all(q[3:] > 0.0)):
        raise ValueError("self-stress sign pattern is not strut-/cable-compatible")

    q /= np.max(np.abs(q))
    return prestress_scale * q


def three_strut_prism(
    radius: float = 1.0,
    height: float = 1.5,
    twist: float = np.pi / 6.0,
    prestress_scale: float = 0.20,
    stiffness: float = 1.0,
) -> Tuple[np.ndarray, List[Member], List[SpringMember]]:
    """Return an equilibrated 3-strut tensegrity prism.

    Rest lengths are not assigned ad hoc. They are derived from the unique
    self-stress state of the geometry. For an axial spring,

        q_e = k_e (L_e - L0_e) / L_e,

    so ``L0_e = L_e * (1 - q_e / k_e)``. Because ``q`` lies in ``ker(R.T)``,
    the reference geometry is in force equilibrium (up to floating-point
    roundoff), making its Hessian spectrum a legitimate local-stability test.
    """
    if stiffness <= 0.0:
        raise ValueError("stiffness must be positive")
    if prestress_scale >= stiffness:
        raise ValueError("prestress_scale must be smaller than stiffness")

    nodes, members = _prism_geometry(radius=radius, height=height, twist=twist)
    q = prism_force_density(nodes, members, prestress_scale=prestress_scale)

    springs: List[SpringMember] = []
    for member, force_density in zip(members, q):
        L = float(np.linalg.norm(nodes[member.i] - nodes[member.j]))
        L0 = L * (1.0 - force_density / stiffness)
        springs.append(
            SpringMember(
                i=member.i,
                j=member.j,
                kind=member.kind,
                k=stiffness,
                L0=L0,
            )
        )

    return nodes, members, springs
