"""Analyze internal soft modes after prism cable ablation.

This experiment removes rigid-body translations and rotations before examining
the rigidity spectrum.

For a six-node 3D framework:

    total DOF = 18
    rigid-body DOF = 6
    internal DOF = 12

After one cable is removed, 11 members remain. Therefore even a full-row-rank
11 x 12 projected rigidity matrix has one exact internal mechanism.

Small nonzero singular values indicate *additional* mechanisms emerging as the
geometry approaches rank deficiency.

The script also analyzes the geometry itself using the singular values of the
centered node-coordinate matrix to quantify whether the relaxed framework is
approaching a planar or line-like configuration.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.energy import (
    nonrigid_basis,
)
from tensegrity.examples import (
    three_strut_prism,
)
from tensegrity.relaxation import (
    relax_framework,
)
from tensegrity.rigidity import (
    rigidity_matrix,
)


OUTPUT_DIR = Path(
    "outputs/prism_projected_modes"
)


def force_density_vector(
    nodes,
    springs,
):
    """Return axial spring force-density vector q."""
    values = []

    for spring in springs:
        d = nodes[spring.i] - nodes[spring.j]
        length = float(
            np.linalg.norm(d)
        )

        values.append(
            spring.k
            * (length - spring.L0)
            / length
        )

    return np.asarray(
        values,
        dtype=float,
    )


def coordinate_spectrum(
    nodes,
):
    """Return singular values of centered node coordinates.

    For Xc with shape (n, 3):

        s3 / s1 -> 0
            indicates an increasingly planar configuration.

        s2 / s1 -> 0 and s3 / s1 -> 0
            indicates an increasingly line-like configuration.
    """
    X = np.asarray(
        nodes,
        dtype=float,
    )

    centered = (
        X
        - X.mean(
            axis=0,
            keepdims=True,
        )
    )

    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
    )

    return singular_values


def save_mode(
    path,
    nodes,
    vector,
):
    """Save one Cartesian node-displacement mode."""
    mode = vector.reshape(
        (-1, 3)
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "node",
            "x",
            "y",
            "z",
            "dx",
            "dy",
            "dz",
            "magnitude",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(
            nodes.shape[0]
        ):
            writer.writerow(
                {
                    "node": i,
                    "x": nodes[i, 0],
                    "y": nodes[i, 1],
                    "z": nodes[i, 2],
                    "dx": mode[i, 0],
                    "dy": mode[i, 1],
                    "dz": mode[i, 2],
                    "magnitude": float(
                        np.linalg.norm(
                            mode[i]
                        )
                    ),
                }
            )


def analyze_removal(
    removed_idx,
    nodes,
    members,
    springs,
):
    removed = members[
        removed_idx
    ]

    reduced_members = (
        members[:removed_idx]
        + members[
            removed_idx + 1:
        ]
    )

    reduced_springs = (
        springs[:removed_idx]
        + springs[
            removed_idx + 1:
        ]
    )

    relaxation = relax_framework(
        nodes,
        reduced_springs,
        residual_tol=1e-9,
        max_iterations=5000,
        initial_step_size=0.2,
    )

    X = relaxation.nodes

    #
    # Full Cartesian rigidity matrix:
    # shape = (11, 18)
    #
    R = rigidity_matrix(
        X,
        reduced_members,
    )

    #
    # Z spans the complement of the six
    # rigid-body modes.
    #
    # For a generic six-node 3D structure:
    #
    #       Z.shape == (18, 12)
    #
    Z = nonrigid_basis(
        X,
    )

    #
    # Rigidity restricted to internal
    # deformation coordinates.
    #
    #       R_internal.shape == (11, 12)
    #
    R_internal = (
        R @ Z
    )

    #
    # IMPORTANT:
    #
    # full_matrices=True gives Vt with
    # shape (12, 12), so the exact
    # one-dimensional right nullspace is
    # explicitly present.
    #
    U, singular_values, Vt = (
        np.linalg.svd(
            R_internal,
            full_matrices=True,
        )
    )

    #
    # R_internal has 11 reported singular
    # values because min(11, 12) = 11.
    #
    # Vt[-1] is the exact internal
    # nullspace vector that has no
    # corresponding reported singular
    # value.
    #
    exact_mechanism_internal = (
        Vt[-1]
    )

    exact_mechanism_cartesian = (
        Z
        @ exact_mechanism_internal
    )

    #
    # Smallest represented singular modes.
    #
    sigma1 = (
        singular_values[-1]
    )

    sigma2 = (
        singular_values[-2]
    )

    emerging_mode1_internal = (
        Vt[-2]
    )

    emerging_mode2_internal = (
        Vt[-3]
    )

    emerging_mode1_cartesian = (
        Z
        @ emerging_mode1_internal
    )

    emerging_mode2_cartesian = (
        Z
        @ emerging_mode2_internal
    )

    #
    # Verify exact mechanism numerically.
    #
    exact_mechanism_residual = (
        np.linalg.norm(
            R
            @ exact_mechanism_cartesian
        )
    )

    emerging1_residual = (
        np.linalg.norm(
            R
            @ emerging_mode1_cartesian
        )
    )

    emerging2_residual = (
        np.linalg.norm(
            R
            @ emerging_mode2_cartesian
        )
    )

    #
    # Geometry dimensionality.
    #
    coord_sv = coordinate_spectrum(
        X,
    )

    c1, c2, c3 = coord_sv

    planarity_ratio = (
        c3 / c1
        if c1 > 0.0
        else np.nan
    )

    line_ratio = (
        c2 / c1
        if c1 > 0.0
        else np.nan
    )

    #
    # Internal force state.
    #
    q = force_density_vector(
        X,
        reduced_springs,
    )

    #
    # Left singular-vector alignment.
    #
    u1 = U[:, -1]
    u2 = U[:, -2]

    if np.dot(
        q,
        u1,
    ) < 0.0:
        u1 = -u1
        emerging_mode1_cartesian = (
            -emerging_mode1_cartesian
        )

    if np.dot(
        q,
        u2,
    ) < 0.0:
        u2 = -u2
        emerging_mode2_cartesian = (
            -emerging_mode2_cartesian
        )

    q_norm = float(
        np.linalg.norm(q)
    )

    q_u1 = float(
        np.dot(q, u1)
    )

    q_u2 = float(
        np.dot(q, u2)
    )

    cos_q_u1 = (
        q_u1 / q_norm
        if q_norm > 0.0
        else np.nan
    )

    cos_q_u2 = (
        q_u2 / q_norm
        if q_norm > 0.0
        else np.nan
    )

    #
    # Numerical ranks under several
    # thresholds. This is useful because
    # we are studying a limiting singular
    # configuration rather than claiming
    # the finite iterate has already
    # changed exact rank.
    #
    thresholds = [
        1e-6,
        1e-7,
        1e-8,
        1e-9,
    ]

    rank_by_tol = {}

    for tol in thresholds:
        rank_by_tol[tol] = int(
            np.sum(
                singular_values > tol
            )
        )

    print()
    print("=" * 76)

    print(
        f"remove cable {removed_idx} "
        f"({removed.i},{removed.j})"
    )

    print("=" * 76)

    print(
        f"relaxation converged:           "
        f"{relaxation.converged}"
    )

    print(
        f"relaxation residual:            "
        f"{relaxation.residual_final:.12e}"
    )

    print()
    print(
        "Projected rigidity spectrum"
    )

    print(
        f"R shape:                        "
        f"{R.shape}"
    )

    print(
        f"non-rigid basis Z shape:        "
        f"{Z.shape}"
    )

    print(
        f"R_internal shape:               "
        f"{R_internal.shape}"
    )

    print(
        f"sigma1:                         "
        f"{sigma1:.12e}"
    )

    print(
        f"sigma2:                         "
        f"{sigma2:.12e}"
    )

    print(
        f"exact mechanism residual:       "
        f"{exact_mechanism_residual:.12e}"
    )

    print(
        f"emerging mode 1 residual:       "
        f"{emerging1_residual:.12e}"
    )

    print(
        f"emerging mode 2 residual:       "
        f"{emerging2_residual:.12e}"
    )

    print()
    print(
        "Numerical rank / internal nullity"
    )

    for tol in thresholds:
        rank = rank_by_tol[tol]

        internal_nullity = (
            R_internal.shape[1]
            - rank
        )

        self_stress_dim = (
            R_internal.shape[0]
            - rank
        )

        print(
            f"tol={tol:.0e}: "
            f"rank={rank:2d}, "
            f"internal nullity={internal_nullity:2d}, "
            f"self-stress dim={self_stress_dim:2d}"
        )

    print()
    print(
        "Centered coordinate spectrum"
    )

    print(
        f"coord sigma1:                   "
        f"{c1:.12e}"
    )

    print(
        f"coord sigma2:                   "
        f"{c2:.12e}"
    )

    print(
        f"coord sigma3:                   "
        f"{c3:.12e}"
    )

    print(
        f"coord sigma2 / sigma1:          "
        f"{line_ratio:.12e}"
    )

    print(
        f"coord sigma3 / sigma1:          "
        f"{planarity_ratio:.12e}"
    )

    print()
    print(
        "Near-self-stress alignment"
    )

    print(
        f"q norm:                         "
        f"{q_norm:.12e}"
    )

    print(
        f"cos(q, u1):                     "
        f"{cos_q_u1:.12f}"
    )

    print(
        f"cos(q, u2):                     "
        f"{cos_q_u2:.12f}"
    )

    print()
    print(
        "Projected singular values:"
    )

    print(
        np.array2string(
            singular_values,
            precision=8,
            suppress_small=False,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Save all three important internal
    # directions:
    #
    # 1. existing exact mechanism
    # 2. smallest nonzero singular mode
    # 3. second-smallest nonzero mode
    #
    save_mode(
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            "_exact_mechanism.csv"
        ),
        X,
        exact_mechanism_cartesian,
    )

    save_mode(
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            "_emerging_mode1.csv"
        ),
        X,
        emerging_mode1_cartesian,
    )

    save_mode(
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            "_emerging_mode2.csv"
        ),
        X,
        emerging_mode2_cartesian,
    )

    #
    # Summary file.
    #
    summary_path = (
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            "_summary.csv"
        )
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "removed_index",
            "removed_i",
            "removed_j",
            "relaxation_residual",
            "sigma1",
            "sigma2",
            "exact_mechanism_residual",
            "emerging_mode1_residual",
            "emerging_mode2_residual",
            "coordinate_sigma1",
            "coordinate_sigma2",
            "coordinate_sigma3",
            "coordinate_sigma2_over_sigma1",
            "coordinate_sigma3_over_sigma1",
            "q_norm",
            "cos_q_u1",
            "cos_q_u2",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "removed_index": (
                    removed_idx
                ),
                "removed_i": (
                    removed.i
                ),
                "removed_j": (
                    removed.j
                ),
                "relaxation_residual": (
                    relaxation.residual_final
                ),
                "sigma1": sigma1,
                "sigma2": sigma2,
                "exact_mechanism_residual": (
                    exact_mechanism_residual
                ),
                "emerging_mode1_residual": (
                    emerging1_residual
                ),
                "emerging_mode2_residual": (
                    emerging2_residual
                ),
                "coordinate_sigma1": c1,
                "coordinate_sigma2": c2,
                "coordinate_sigma3": c3,
                "coordinate_sigma2_over_sigma1": (
                    line_ratio
                ),
                "coordinate_sigma3_over_sigma1": (
                    planarity_ratio
                ),
                "q_norm": q_norm,
                "cos_q_u1": cos_q_u1,
                "cos_q_u2": cos_q_u2,
            }
        )


def main():
    nodes, members, springs = (
        three_strut_prism(
            prestress_scale=0.20
        )
    )

    #
    # Representative perimeter failure.
    #
    analyze_removal(
        3,
        nodes,
        members,
        springs,
    )

    #
    # Representative side failure.
    #
    analyze_removal(
        9,
        nodes,
        members,
        springs,
    )


if __name__ == "__main__":
    main()
