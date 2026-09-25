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

def equilibrated_mirrored_t4(
    radius: float = 1.0,
    half_height: float = 0.75,
    twist: float = np.pi / 6.0,
    prestress_scale: float = 0.20,
    stiffness: float = 1.0,
) -> Tuple[
    np.ndarray,
    List[Member],
    List[SpringMember],
]:
    """Return an equilibrated mirrored T4 proxy.

    A sign-compatible state of self-stress is selected from the 2D
    self-stress space and normalized so max(abs(q)) == prestress_scale.

    Rest lengths are derived from

        q_e = k_e (L_e - L0_e) / L_e

    so that the returned geometry is in force equilibrium.
    """
    if stiffness <= 0.0:
        raise ValueError(
            "stiffness must be positive"
        )

    if prestress_scale < 0.0:
        raise ValueError(
            "prestress_scale must be non-negative"
        )

    if prestress_scale >= stiffness:
        raise ValueError(
            "prestress_scale must be smaller than stiffness"
        )

    nodes, members = mirrored_t4_prism(
        radius=radius,
        half_height=half_height,
        twist=twist,
    )

    Q = self_stress_basis(
        nodes,
        members,
        tol=1e-9,
    )

    if Q.shape[1] != 2:
        raise ValueError(
            "mirrored T4 proxy is expected to have "
            "a 2D self-stress space"
        )

    #
    # Search the self-stress plane for the
    # sign-compatible direction with the
    # largest minimum sign margin.
    #
    theta_values = np.linspace(
        0.0,
        2.0 * np.pi,
        200001,
        endpoint=False,
    )

    best_q = None
    best_margin = -np.inf

    for theta in theta_values:
        coefficients = np.array(
            [
                np.cos(theta),
                np.sin(theta),
            ]
        )

        q = Q @ coefficients

        margins = []

        for value, member in zip(
            q,
            members,
        ):
            if member.kind == "bar":
                margins.append(
                    -value
                )
            else:
                margins.append(
                    value
                )

        margin = float(
            np.min(margins)
        )

        if margin > best_margin:
            best_margin = margin
            best_q = q.copy()

    if best_q is None or best_margin <= 0.0:
        raise ValueError(
            "no strictly sign-compatible T4 self-stress found"
        )

    q = (
        best_q
        / np.max(
            np.abs(best_q)
        )
    )

    q *= prestress_scale

    springs: List[
        SpringMember
    ] = []

    for member, force_density in zip(
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

def t4_with_one_supported_t3(
    radius: float = 1.0,
    half_height: float = 0.75,
    attachment_height: float = 1.5,
) -> Tuple[
    np.ndarray,
    List[Member],
]:
    """Return a T4 proxy with one zero-twist T3 support attachment.

    The T3 interface triangle is merged with the lower triangle of the T4.
    The far triangle of the T3 acts as the post/support side.

    Node layout
    -----------
    0,1,2   : T4 lower triangle / T3 interface triangle
    3,4,5   : T4 middle triangle
    6,7,8   : T4 upper triangle
    9,10,11 : T3 support-side triangle
    """
    t4_nodes, t4_members = mirrored_t4_prism(
        radius=radius,
        half_height=half_height,
        twist=np.pi / 6.0,
    )

    #
    # Build support-side triangle directly below
    # the T4 lower/interface triangle.
    #
    interface = t4_nodes[0:3]

    support = interface.copy()
    support[:, 2] -= attachment_height

    nodes = np.vstack(
        [
            t4_nodes,
            support,
        ]
    )

    members = list(
        t4_members
    )

    #
    # Zero-twist T3 attachment topology.
    #
    # Interface triangle:
    #     0,1,2
    #
    # Support triangle:
    #     9,10,11
    #
    # Use the same 3-strut / 9-cable topology
    # as the T3 proxy.
    #
    t3_struts = [
        (9, 1),
        (10, 2),
        (11, 0),
    ]

    support_ring = [
        (9, 10),
        (10, 11),
        (11, 9),
    ]

    #
    # Interface ring already exists as the
    # T4 lower triangle, so don't duplicate it.
    #
    t3_side_cables = [
        (9, 0),
        (10, 1),
        (11, 2),
    ]

    for i, j in t3_struts:
        members.append(
            Member(
                i,
                j,
                kind="bar",
            )
        )

    for i, j in (
        support_ring
        + t3_side_cables
    ):
        members.append(
            Member(
                i,
                j,
                kind="cable",
            )
        )

    return (
        nodes,
        members,
    )

def t4_with_two_supported_t3(
    radius: float = 1.0,
    half_height: float = 0.75,
    attachment_height: float = 1.5,
) -> Tuple[
    np.ndarray,
    List[Member],
]:
    """Return a T4 proxy supported by zero-twist T3 attachments below and above.

    Node layout
    -----------
    0,1,2    : T4 lower triangle
    3,4,5    : T4 middle triangle
    6,7,8    : T4 upper triangle
    9,10,11  : lower T3 support-side triangle
    12,13,14 : upper T3 support-side triangle
    """
    t4_nodes, t4_members = mirrored_t4_prism(
        radius=radius,
        half_height=half_height,
        twist=np.pi / 6.0,
    )

    lower_interface = t4_nodes[0:3]
    upper_interface = t4_nodes[6:9]

    lower_support = lower_interface.copy()
    lower_support[:, 2] -= attachment_height

    upper_support = upper_interface.copy()
    upper_support[:, 2] += attachment_height

    nodes = np.vstack(
        [
            t4_nodes,
            lower_support,
            upper_support,
        ]
    )

    members = list(
        t4_members
    )

    #
    # Lower zero-twist T3 attachment.
    #
    lower_struts = [
        (9, 1),
        (10, 2),
        (11, 0),
    ]

    lower_support_ring = [
        (9, 10),
        (10, 11),
        (11, 9),
    ]

    lower_side_cables = [
        (9, 0),
        (10, 1),
        (11, 2),
    ]

    #
    # Upper zero-twist T3 attachment.
    #
    upper_struts = [
        (12, 7),
        (13, 8),
        (14, 6),
    ]

    upper_support_ring = [
        (12, 13),
        (13, 14),
        (14, 12),
    ]

    upper_side_cables = [
        (12, 6),
        (13, 7),
        (14, 8),
    ]

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

    for i, j in (
        lower_support_ring
        + lower_side_cables
        + upper_support_ring
        + upper_side_cables
    ):
        members.append(
            Member(
                i,
                j,
                kind="cable",
            )
        )

    return (
        nodes,
        members,
    )
