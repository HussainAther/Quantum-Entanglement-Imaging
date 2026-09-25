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

    edges = list(
        combinations(
            range(4),
            2,
        )
    )

    members = [
        Member(
            i,
            j,
            kind="bar",
        )
        for i, j in edges
    ]

    springs = [
        SpringMember(
            i=i,
            j=j,
            kind="bar",
            k=1.0,
            L0=float(
                np.linalg.norm(
                    nodes[i] - nodes[j]
                )
            ),
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
        raise ValueError(
            "radius must be positive"
        )

    if height <= 0.0:
        raise ValueError(
            "height must be positive"
        )

    lower_angles = np.array(
        [
            0.0,
            2.0 * np.pi / 3.0,
            4.0 * np.pi / 3.0,
        ]
    )

    upper_angles = (
        lower_angles + twist
    )

    lower = np.column_stack(
        [
            radius * np.cos(
                lower_angles
            ),
            radius * np.sin(
                lower_angles
            ),
            np.zeros(3),
        ]
    )

    upper = np.column_stack(
        [
            radius * np.cos(
                upper_angles
            ),
            radius * np.sin(
                upper_angles
            ),
            np.full(
                3,
                height,
            ),
        ]
    )

    nodes = np.vstack(
        [
            lower,
            upper,
        ]
    )

    struts = [
        (0, 4),
        (1, 5),
        (2, 3),
    ]

    lower_cables = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    upper_cables = [
        (3, 4),
        (4, 5),
        (5, 3),
    ]

    side_cables = [
        (0, 3),
        (1, 4),
        (2, 5),
    ]

    members = [
        Member(
            i,
            j,
            kind="bar",
        )
        for i, j in struts
    ]

    members += [
        Member(
            i,
            j,
            kind="cable",
        )
        for i, j in (
            lower_cables
            + upper_cables
            + side_cables
        )
    ]

    return nodes, members


def prism_force_density(
    nodes: np.ndarray,
    members: List[Member],
    prestress_scale: float = 0.20,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return an equilibrium force-density vector for the 3-strut prism.

    The vector lies in ker(R.T). Its sign is chosen so the first three
    members (struts) are in compression and the nine cables are in tension.
    It is normalized so max(abs(q)) == prestress_scale.
    """
    if prestress_scale < 0.0:
        raise ValueError(
            "prestress_scale must be non-negative"
        )

    basis = self_stress_basis(
        nodes,
        members,
        tol=tol,
    )

    if basis.shape[1] != 1:
        raise ValueError(
            "canonical prism is expected to have exactly "
            "one self-stress state"
        )

    q = basis[:, 0].copy()

    # Choose physical sign:
    # struts negative, cables positive.
    if np.mean(
        q[:3]
    ) > 0.0:
        q *= -1.0

    if not (
        np.all(
            q[:3] < 0.0
        )
        and np.all(
            q[3:] > 0.0
        )
    ):
        raise ValueError(
            "self-stress sign pattern is not "
            "strut-/cable-compatible"
        )

    q /= np.max(
        np.abs(q)
    )

    return (
        prestress_scale * q
    )


def three_strut_prism(
    radius: float = 1.0,
    height: float = 1.5,
    twist: float = np.pi / 6.0,
    prestress_scale: float = 0.20,
    stiffness: float = 1.0,
) -> Tuple[
    np.ndarray,
    List[Member],
    List[SpringMember],
]:
    """Return an equilibrated 3-strut tensegrity prism.

    Rest lengths are derived from the unique self-stress state of the
    geometry.

    For an axial spring,

        q_e = k_e (L_e - L0_e) / L_e,

    so

        L0_e = L_e * (1 - q_e / k_e).

    Because q lies in ker(R.T), the reference geometry is in force
    equilibrium up to floating-point roundoff.
    """
    if stiffness <= 0.0:
        raise ValueError(
            "stiffness must be positive"
        )

    if prestress_scale >= stiffness:
        raise ValueError(
            "prestress_scale must be smaller than stiffness"
        )

    nodes, members = _prism_geometry(
        radius=radius,
        height=height,
        twist=twist,
    )

    q = prism_force_density(
        nodes,
        members,
        prestress_scale=prestress_scale,
    )

    springs: List[
        SpringMember
    ] = []

    for (
        member,
        force_density,
    ) in zip(
        members,
        q,
    ):
        L = float(
            np.linalg.norm(
                nodes[member.i]
                - nodes[member.j]
            )
        )

        L0 = (
            L
            * (
                1.0
                - force_density
                / stiffness
            )
        )

        springs.append(
            SpringMember(
                i=member.i,
                j=member.j,
                kind=member.kind,
                k=stiffness,
                L0=L0,
            )
        )

    return (
        nodes,
        members,
        springs,
    )


def mirrored_t4_prism(
    radius: float = 1.0,
    half_height: float = 0.75,
    twist: float = np.pi / 6.0,
) -> Tuple[
    np.ndarray,
    List[Member],
]:
    """Return a simple mirrored double-prism T4 proxy.

    This is a computational benchmark inspired by Crawford-Young's T4
    trijunction description: two mirrored triangular-prism tensegrities
    stacked around a shared middle triangle.

    It is not yet a verbatim reconstruction of the COMSOL T4 geometry.

    Node layout
    -----------
    0,1,2 : lower triangle
    3,4,5 : middle triangle
    6,7,8 : upper triangle

    The lower prism uses +twist from lower to middle.
    The upper prism mirrors that geometry so the top triangle aligns
    rotationally with the lower triangle.
    """
    if radius <= 0.0:
        raise ValueError(
            "radius must be positive"
        )

    if half_height <= 0.0:
        raise ValueError(
            "half_height must be positive"
        )

    base_angles = np.array(
        [
            0.0,
            2.0 * np.pi / 3.0,
            4.0 * np.pi / 3.0,
        ]
    )

    middle_angles = (
        base_angles + twist
    )

    lower = np.column_stack(
        [
            radius * np.cos(
                base_angles
            ),
            radius * np.sin(
                base_angles
            ),
            np.full(
                3,
                -half_height,
            ),
        ]
    )

    middle = np.column_stack(
        [
            radius * np.cos(
                middle_angles
            ),
            radius * np.sin(
                middle_angles
            ),
            np.zeros(3),
        ]
    )

    upper = np.column_stack(
        [
            radius * np.cos(
                base_angles
            ),
            radius * np.sin(
                base_angles
            ),
            np.full(
                3,
                half_height,
            ),
        ]
    )

    nodes = np.vstack(
        [
            lower,
            middle,
            upper,
        ]
    )

    lower_struts = [
        (0, 4),
        (1, 5),
        (2, 3),
    ]

    upper_struts = [
        (6, 4),
        (7, 5),
        (8, 3),
    ]

    lower_ring = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    middle_ring = [
        (3, 4),
        (4, 5),
        (5, 3),
    ]

    upper_ring = [
        (6, 7),
        (7, 8),
        (8, 6),
    ]

    lower_side = [
        (0, 3),
        (1, 4),
        (2, 5),
    ]

    upper_side = [
        (6, 3),
        (7, 4),
        (8, 5),
    ]

    members: List[
        Member
    ] = []

    for i, j in (
        lower_struts
        + upper_struts
    ):
        members.append(
            Member(
                i,
                j,
                kind="bar",
            )
        )

    cable_edges = (
        lower_ring
        + middle_ring
        + lower_side
        + upper_ring
        + upper_side
    )

    members.extend(
        Member(
            i,
            j,
            kind="cable",
        )
        for i, j in cable_edges
    )

    return nodes, members
