"""Analyze the exact self-stress space of the planar remove-9 equilibrium.

The planar equilibrium obtained after removing side cable (0, 3) has

    rank(R) = 9
    number of members = 11

so

    dim ker(R.T) = 2.

This experiment reconstructs that planar equilibrium, computes an orthonormal
basis for the exact two-dimensional self-stress space, and decomposes the
physical spring force-density vector q into that basis.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from tensegrity.energy import (
    energy_gradient,
    equilibrium_residual_norm,
    total_energy,
)
from tensegrity.examples import three_strut_prism
from tensegrity.relaxation import relax_framework
from tensegrity.rigidity import (
    rigidity_matrix,
    self_stress_basis,
)


def best_fit_plane_basis(nodes):
    X = np.asarray(nodes, dtype=float)

    centroid = X.mean(axis=0)
    centered = X - centroid

    _, _, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    return (
        centroid,
        Vt[0],
        Vt[1],
        Vt[2],
    )


def project_to_plane_coordinates(
    nodes,
    centroid,
    e1,
    e2,
):
    centered = nodes - centroid

    return np.column_stack(
        [
            centered @ e1,
            centered @ e2,
        ]
    )


def reconstruct_from_plane_coordinates(
    coords,
    centroid,
    e1,
    e2,
):
    return (
        centroid
        + coords[:, 0, None] * e1[None, :]
        + coords[:, 1, None] * e2[None, :]
    )


def free_planar_coordinates(coords):
    """Remove two translations and one in-plane rotation."""
    free = []

    for i in range(coords.shape[0]):
        for j in range(2):
            if i == 0:
                continue

            if i == 1 and j == 1:
                continue

            free.append((i, j))

    return free


def spring_force_density(
    nodes,
    springs,
):
    q = []

    for spring in springs:
        d = nodes[spring.i] - nodes[spring.j]

        length = float(
            np.linalg.norm(d)
        )

        q.append(
            spring.k
            * (length - spring.L0)
            / length
        )

    return np.asarray(
        q,
        dtype=float,
    )


def main():
    nodes, members, springs = (
        three_strut_prism(
            prestress_scale=0.20
        )
    )

    removed_idx = 9

    reduced_members = (
        members[:removed_idx]
        + members[removed_idx + 1:]
    )

    reduced_springs = (
        springs[:removed_idx]
        + springs[removed_idx + 1:]
    )

    #
    # Reproduce near-planar relaxed state.
    #
    relaxation = relax_framework(
        nodes,
        reduced_springs,
        residual_tol=1e-9,
        max_iterations=5000,
        initial_step_size=0.2,
    )

    X_relaxed = relaxation.nodes

    centroid, e1, e2, normal = (
        best_fit_plane_basis(
            X_relaxed
        )
    )

    coords0 = (
        project_to_plane_coordinates(
            X_relaxed,
            centroid,
            e1,
            e2,
        )
    )

    free = free_planar_coordinates(
        coords0
    )

    fixed = coords0.copy()

    x0 = np.asarray(
        [
            coords0[i, j]
            for i, j in free
        ],
        dtype=float,
    )

    def unpack(x):
        coords = fixed.copy()

        for value, (i, j) in zip(
            x,
            free,
        ):
            coords[i, j] = value

        return coords

    def residual(x):
        coords = unpack(x)

        X = reconstruct_from_plane_coordinates(
            coords,
            centroid,
            e1,
            e2,
        )

        grad = energy_gradient(
            X,
            reduced_springs,
        )

        grad2 = np.column_stack(
            [
                grad @ e1,
                grad @ e2,
            ]
        )

        return np.asarray(
            [
                grad2[i, j]
                for i, j in free
            ]
        )

    solve = least_squares(
        residual,
        x0,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=20000,
    )

    coords = unpack(
        solve.x
    )

    X = reconstruct_from_plane_coordinates(
        coords,
        centroid,
        e1,
        e2,
    )

    #
    # Exact force state.
    #
    q = spring_force_density(
        X,
        reduced_springs,
    )

    #
    # Exact self-stress space ker(R.T).
    #
    Q = self_stress_basis(
        X,
        reduced_members,
        tol=1e-9,
    )

    R = rigidity_matrix(
        X,
        reduced_members,
    )

    coefficients = (
        Q.T @ q
    )

    q_projected = (
        Q @ coefficients
    )

    projection_error = (
        np.linalg.norm(
            q - q_projected
        )
    )

    print()
    print("=" * 76)
    print("PLANAR SELF-STRESS ANALYSIS")
    print("=" * 76)

    print(
        f"full equilibrium residual:       "
        f"{equilibrium_residual_norm(X, reduced_springs):.12e}"
    )

    print(
        f"energy:                          "
        f"{total_energy(X, reduced_springs):.12e}"
    )

    print()

    print(
        f"R shape:                         "
        f"{R.shape}"
    )

    print(
        f"self-stress basis shape:         "
        f"{Q.shape}"
    )

    print(
        f"self-stress dimension:           "
        f"{Q.shape[1]}"
    )

    print()

    print(
        f"q norm:                          "
        f"{np.linalg.norm(q):.12e}"
    )

    print(
        f"||R.T @ q||:                     "
        f"{np.linalg.norm(R.T @ q):.12e}"
    )

    print(
        f"projection error ||q-QQ.Tq||:    "
        f"{projection_error:.12e}"
    )

    print()
    print("Self-stress coordinates:")

    for i, coefficient in enumerate(
        coefficients
    ):
        print(
            f"  coefficient {i + 1}: "
            f"{coefficient:.12e}"
        )

    print()

    normalized_coefficients = (
        coefficients
        / np.linalg.norm(
            coefficients
        )
    )

    print(
        "Normalized self-stress coordinates:"
    )

    for i, coefficient in enumerate(
        normalized_coefficients
    ):
        print(
            f"  direction {i + 1}: "
            f"{coefficient:.12f}"
        )

    print()
    print("Physical q:")
    print(
        np.array2string(
            q,
            precision=8,
            suppress_small=False,
        )
    )

    print()
    print(
        "Self-stress basis vectors "
        "(columns):"
    )

    print(
        np.array2string(
            Q,
            precision=8,
            suppress_small=False,
        )
    )

    #
    # Individual equilibrium check for
    # each basis vector.
    #
    print()
    print("Basis equilibrium residuals:")

    for i in range(
        Q.shape[1]
    ):
        print(
            f"  basis {i + 1}: "
            f"{np.linalg.norm(R.T @ Q[:, i]):.12e}"
        )


if __name__ == "__main__":
    main()
