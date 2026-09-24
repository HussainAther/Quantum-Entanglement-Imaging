"""Analyze soft mechanism and near-self-stress modes after cable ablation.

For the rigidity matrix R,

    R v_k = sigma_k u_k
    R.T u_k = sigma_k v_k

the right singular vector v_k is a node-velocity/deformation direction and
the left singular vector u_k is a member-force direction.

As sigma_k -> 0:
    v_k approaches an infinitesimal mechanism,
    u_k approaches a state of self-stress.

This experiment compares the relaxed spring force-density vector q against
the soft left-singular subspace and saves the corresponding node modes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.energy import energy_gradient
from tensegrity.examples import three_strut_prism
from tensegrity.relaxation import relax_framework
from tensegrity.rigidity import rigidity_matrix


OUTPUT_DIR = Path(
    "outputs/prism_soft_modes"
)


def force_density_vector(nodes, springs):
    q = []

    for spring in springs:
        d = nodes[spring.i] - nodes[spring.j]
        length = float(np.linalg.norm(d))

        q.append(
            spring.k
            * (length - spring.L0)
            / length
        )

    return np.asarray(
        q,
        dtype=float,
    )


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0.0:
        return float("nan")

    return float(
        np.dot(a, b)
        / denominator
    )


def projection_fraction(vector, basis):
    """Fraction of vector norm contained in an orthonormal basis."""

    vector = np.asarray(
        vector,
        dtype=float,
    )

    basis = np.asarray(
        basis,
        dtype=float,
    )

    norm = np.linalg.norm(vector)

    if norm == 0.0:
        return 0.0

    coefficients = (
        basis.T @ vector
    )

    projected_norm = np.linalg.norm(
        coefficients
    )

    return float(
        projected_norm / norm
    )


def save_node_mode(
    path,
    nodes,
    mode,
):
    node_mode = mode.reshape(
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
            "mode_magnitude",
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
                    "dx": node_mode[i, 0],
                    "dy": node_mode[i, 1],
                    "dz": node_mode[i, 2],
                    "mode_magnitude": (
                        np.linalg.norm(
                            node_mode[i]
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
    removed_member = members[
        removed_idx
    ]

    reduced_members = (
        members[:removed_idx]
        + members[removed_idx + 1 :]
    )

    reduced_springs = (
        springs[:removed_idx]
        + springs[removed_idx + 1 :]
    )

    result = relax_framework(
        nodes,
        reduced_springs,
        residual_tol=1e-9,
        max_iterations=5000,
        initial_step_size=0.2,
    )

    X = result.nodes

    R = rigidity_matrix(
        X,
        reduced_members,
    )

    U, singular_values, Vt = (
        np.linalg.svd(
            R,
            full_matrices=False,
        )
    )

    q = force_density_vector(
        X,
        reduced_springs,
    )

    gradient = energy_gradient(
        X,
        reduced_springs,
    ).reshape(-1)

    #
    # np.linalg.svd orders singular
    # values from largest to smallest.
    #
    u1 = U[:, -1]
    v1 = Vt[-1]

    u2 = U[:, -2]
    v2 = Vt[-2]

    sigma1 = singular_values[-1]
    sigma2 = singular_values[-2]

    #
    # Singular vectors have arbitrary
    # signs. Flip u1/v1 so u1 has a
    # positive dot product with q.
    #
    if np.dot(q, u1) < 0.0:
        u1 = -u1
        v1 = -v1

    if np.dot(q, u2) < 0.0:
        u2 = -u2
        v2 = -v2

    q_norm = np.linalg.norm(q)

    alignment1 = cosine_similarity(
        q,
        u1,
    )

    alignment2 = cosine_similarity(
        q,
        u2,
    )

    soft_basis_1 = (
        u1[:, None]
    )

    soft_basis_2 = np.column_stack(
        [
            u1,
            u2,
        ]
    )

    projection1 = projection_fraction(
        q,
        soft_basis_1,
    )

    projection2 = projection_fraction(
        q,
        soft_basis_2,
    )

    #
    # Decompose q in the complete left
    # singular-vector basis.
    #
    coefficients = (
        U.T @ q
    )

    contributions = (
        singular_values
        * coefficients
    )

    predicted_residual = (
        np.linalg.norm(
            contributions
        )
    )

    print()
    print("=" * 72)

    print(
        f"remove cable {removed_idx} "
        f"({removed_member.i},"
        f"{removed_member.j})"
    )

    print("=" * 72)

    print(
        f"sigma1:                    "
        f"{sigma1:.12e}"
    )

    print(
        f"sigma2:                    "
        f"{sigma2:.12e}"
    )

    print(
        f"q norm:                    "
        f"{q_norm:.12e}"
    )

    print(
        f"gradient norm:             "
        f"{np.linalg.norm(gradient):.12e}"
    )

    print(
        f"SVD-predicted residual:    "
        f"{predicted_residual:.12e}"
    )

    print()
    print(
        "Force-density alignment"
    )

    print(
        f"cos(q, u1):                "
        f"{alignment1:.12f}"
    )

    print(
        f"cos(q, u2):                "
        f"{alignment2:.12f}"
    )

    print(
        f"fraction of q in u1:       "
        f"{projection1:.12f}"
    )

    print(
        f"fraction of q in span(u1,u2): "
        f"{projection2:.12f}"
    )

    print()
    print(
        "Smallest-mode SVD coefficients"
    )

    print(
        f"<u1, q>:                   "
        f"{np.dot(u1, q):.12e}"
    )

    print(
        f"<u2, q>:                   "
        f"{np.dot(u2, q):.12e}"
    )

    print()
    print(
        "Soft mechanism node vectors"
    )

    mode1_nodes = v1.reshape(
        (-1, 3)
    )

    mode2_nodes = v2.reshape(
        (-1, 3)
    )

    for i in range(
        X.shape[0]
    ):
        print(
            f"node {i}: "
            f"v1="
            f"({mode1_nodes[i,0]: .6f}, "
            f"{mode1_nodes[i,1]: .6f}, "
            f"{mode1_nodes[i,2]: .6f})  "
            f"v2="
            f"({mode2_nodes[i,0]: .6f}, "
            f"{mode2_nodes[i,1]: .6f}, "
            f"{mode2_nodes[i,2]: .6f})"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_node_mode(
        OUTPUT_DIR
        / f"remove_{removed_idx:02d}_mode1.csv",
        X,
        v1,
    )

    save_node_mode(
        OUTPUT_DIR
        / f"remove_{removed_idx:02d}_mode2.csv",
        X,
        v2,
    )

    #
    # Save force-space decomposition.
    #
    decomposition_path = (
        OUTPUT_DIR
        / f"remove_{removed_idx:02d}_force_decomposition.csv"
    )

    with decomposition_path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "svd_index",
            "singular_value",
            "q_coefficient",
            "residual_contribution",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(
            len(singular_values)
        ):
            writer.writerow(
                {
                    "svd_index": i,
                    "singular_value": (
                        singular_values[i]
                    ),
                    "q_coefficient": (
                        coefficients[i]
                    ),
                    "residual_contribution": (
                        contributions[i]
                    ),
                }
            )


def main():
    nodes, members, springs = (
        three_strut_prism(
            prestress_scale=0.20
        )
    )

    #
    # Representatives of the two
    # observed symmetry classes.
    #
    analyze_removal(
        3,
        nodes,
        members,
        springs,
    )

    analyze_removal(
        9,
        nodes,
        members,
        springs,
    )


if __name__ == "__main__":
    main()
