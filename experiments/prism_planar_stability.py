"""Analyze self-stress selection and second-order stability of the exact
planar equilibrium obtained after removing side cable (0, 3).

This experiment:

1. Reconstructs the exact planar equilibrium.
2. Builds a physically meaningful two-dimensional self-stress basis:
       s_loaded = q / ||q||
       s_unused = orthogonal self-stress direction
3. Computes the energy Hessian.
4. Projects out rigid-body motions.
5. Extracts the exact internal mechanisms from the projected rigidity matrix.
6. Evaluates Hessian curvature along those mechanism directions.
7. Reports the full non-rigid Hessian spectrum.

Interpretation:
A first-order mechanism can still be prestress-stabilized if the quadratic
energy curvature along that mechanism is positive.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from tensegrity.energy import (
    energy_gradient,
    equilibrium_residual_norm,
    nonrigid_basis,
    numerical_hessian,
    reduced_nonrigid_hessian,
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
    """Remove 2 translations and 1 in-plane rotation."""
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
    values = []

    for spring in springs:
        d = nodes[spring.i] - nodes[spring.j]
        L = float(np.linalg.norm(d))

        values.append(
            spring.k
            * (L - spring.L0)
            / L
        )

    return np.asarray(
        values,
        dtype=float,
    )


def reconstruct_exact_planar_equilibrium():
    """Return the exact planar remove-9 equilibrium and reduced system."""

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
    # First approach the planar state by
    # unconstrained relaxation.
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
            ],
            dtype=float,
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

    return (
        X,
        reduced_members,
        reduced_springs,
        e1,
        e2,
        normal,
    )


def physically_meaningful_self_stress_basis(
    X,
    members,
    springs,
):
    """Return loaded and unused self-stress directions."""

    q = spring_force_density(
        X,
        springs,
    )

    Q = self_stress_basis(
        X,
        members,
        tol=1e-9,
    )

    if Q.shape[1] != 2:
        raise RuntimeError(
            "Expected a two-dimensional self-stress space."
        )

    q_norm = float(
        np.linalg.norm(q)
    )

    loaded = (
        q / q_norm
    )

    #
    # Start with either exact basis vector
    # and remove the component along loaded.
    #
    candidate = Q[:, 0].copy()

    candidate -= (
        np.dot(candidate, loaded)
        * loaded
    )

    if np.linalg.norm(candidate) < 1e-10:
        candidate = Q[:, 1].copy()

        candidate -= (
            np.dot(candidate, loaded)
            * loaded
        )

    unused = (
        candidate
        / np.linalg.norm(candidate)
    )

    #
    # Deterministic sign convention.
    #
    largest = int(
        np.argmax(
            np.abs(unused)
        )
    )

    if unused[largest] < 0.0:
        unused *= -1.0

    return (
        q,
        loaded,
        unused,
        Q,
    )


def exact_internal_mechanisms(
    X,
    members,
):
    """Return exact mechanisms in Cartesian coordinates.

    R_internal has shape (11, 12). At the exact planar rank-9 state,
    its nullity is 3.
    """

    R = rigidity_matrix(
        X,
        members,
    )

    Z = nonrigid_basis(
        X,
    )

    R_internal = (
        R @ Z
    )

    U, singular_values, Vt = np.linalg.svd(
        R_internal,
        full_matrices=True,
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    nullity = (
        R_internal.shape[1]
        - rank
    )

    mechanism_internal = (
        Vt[rank:].T
    )

    mechanism_cartesian = (
        Z @ mechanism_internal
    )

    return (
        R,
        Z,
        R_internal,
        singular_values,
        mechanism_internal,
        mechanism_cartesian,
    )


def main():
    (
        X,
        members,
        springs,
        e1,
        e2,
        normal,
    ) = reconstruct_exact_planar_equilibrium()

    print()
    print("=" * 78)
    print("PLANAR EQUILIBRIUM STABILITY ANALYSIS")
    print("=" * 78)

    print(
        f"equilibrium residual:             "
        f"{equilibrium_residual_norm(X, springs):.12e}"
    )

    print(
        f"energy:                           "
        f"{total_energy(X, springs):.12e}"
    )

    #
    # Self-stress basis.
    #
    (
        q,
        s_loaded,
        s_unused,
        Q,
    ) = physically_meaningful_self_stress_basis(
        X,
        members,
        springs,
    )

    R = rigidity_matrix(
        X,
        members,
    )

    print()
    print("Physically meaningful self-stress basis")

    print(
        f"q norm:                           "
        f"{np.linalg.norm(q):.12e}"
    )

    print(
        f"||R.T @ s_loaded||:               "
        f"{np.linalg.norm(R.T @ s_loaded):.12e}"
    )

    print(
        f"||R.T @ s_unused||:               "
        f"{np.linalg.norm(R.T @ s_unused):.12e}"
    )

    print(
        f"<s_loaded, s_unused>:             "
        f"{np.dot(s_loaded, s_unused):.12e}"
    )

    print(
        f"projection of q on loaded:        "
        f"{np.dot(q, s_loaded):.12e}"
    )

    print(
        f"projection of q on unused:        "
        f"{np.dot(q, s_unused):.12e}"
    )

    print()
    print("loaded self-stress:")
    print(
        np.array2string(
            s_loaded,
            precision=8,
            suppress_small=False,
        )
    )

    print()
    print("unused orthogonal self-stress:")
    print(
        np.array2string(
            s_unused,
            precision=8,
            suppress_small=False,
        )
    )

    #
    # Exact mechanisms.
    #
    (
        R,
        Z,
        R_internal,
        rigidity_sv,
        mechanism_internal,
        mechanism_cartesian,
    ) = exact_internal_mechanisms(
        X,
        members,
    )

    print()
    print("Projected rigidity")

    print(
        f"R_internal shape:                 "
        f"{R_internal.shape}"
    )

    print(
        f"rank at tol=1e-9:                 "
        f"{np.sum(rigidity_sv > 1e-9)}"
    )

    print(
        f"exact internal mechanism count:   "
        f"{mechanism_internal.shape[1]}"
    )

    print(
        "projected rigidity singular values:"
    )

    print(
        np.array2string(
            rigidity_sv,
            precision=12,
            suppress_small=False,
        )
    )

    for i in range(
        mechanism_cartesian.shape[1]
    ):
        residual = np.linalg.norm(
            R
            @ mechanism_cartesian[:, i]
        )

        print(
            f"mechanism {i + 1} rigidity residual: "
            f"{residual:.12e}"
        )

    #
    # Full numerical Hessian.
    #
    def energy_flat(flat):
        return total_energy(
            flat.reshape((-1, 3)),
            springs,
        )

    H = numerical_hessian(
        energy_flat,
        X.reshape(-1),
        eps=2e-5,
    )

    H_nonrigid = reduced_nonrigid_hessian(
        H,
        X,
    )

    hessian_eigenvalues = np.sort(
        np.linalg.eigvalsh(
            H_nonrigid
        )
    )

    print()
    print("Non-rigid Hessian spectrum")

    print(
        np.array2string(
            hessian_eigenvalues,
            precision=12,
            suppress_small=False,
        )
    )

    eig_tol = 1e-6

    negative_modes = int(
        np.sum(
            hessian_eigenvalues < -eig_tol
        )
    )

    zero_modes = int(
        np.sum(
            np.abs(
                hessian_eigenvalues
            )
            <= eig_tol
        )
    )

    positive_modes = int(
        np.sum(
            hessian_eigenvalues > eig_tol
        )
    )

    print()

    print(
        f"negative Hessian modes:           "
        f"{negative_modes}"
    )

    print(
        f"near-zero Hessian modes:          "
        f"{zero_modes}"
    )

    print(
        f"positive Hessian modes:           "
        f"{positive_modes}"
    )

    print(
        f"minimum non-rigid eigenvalue:     "
        f"{hessian_eigenvalues[0]:.12e}"
    )

    #
    # Hessian restricted directly to the
    # exact mechanism subspace.
    #
    #
    # mechanism_internal lives in Z
    # coordinates, so we can use
    # H_nonrigid directly.
    #
    H_mech = (
        mechanism_internal.T
        @ H_nonrigid
        @ mechanism_internal
    )

    H_mech = (
        0.5
        * (
            H_mech
            + H_mech.T
        )
    )

    mechanism_curvatures = np.linalg.eigvalsh(
        H_mech
    )

    print()
    print("Hessian restricted to exact mechanism subspace")

    print(
        np.array2string(
            H_mech,
            precision=12,
            suppress_small=False,
        )
    )

    print(
        "mechanism-subspace eigenvalues:"
    )

    print(
        np.array2string(
            mechanism_curvatures,
            precision=12,
            suppress_small=False,
        )
    )

    #
    # Individual basis-vector curvature.
    #
    print()

    for i in range(
        mechanism_internal.shape[1]
    ):
        v = mechanism_internal[:, i]

        curvature = float(
            v.T
            @ H_nonrigid
            @ v
        )

        cart = mechanism_cartesian[:, i].reshape(
            (-1, 3)
        )

        in_plane_norm = float(
            np.sqrt(
                np.sum(
                    (cart @ e1) ** 2
                    + (cart @ e2) ** 2
                )
            )
        )

        out_of_plane_norm = float(
            np.linalg.norm(
                cart @ normal
            )
        )

        print(
            f"mechanism {i + 1}: "
            f"curvature={curvature:.12e}, "
            f"in-plane norm={in_plane_norm:.12e}, "
            f"out-of-plane norm={out_of_plane_norm:.12e}"
        )

    #
    # Simple classification.
    #
    min_mech = float(
        np.min(
            mechanism_curvatures
        )
    )

    print()
    print("=" * 78)

    if min_mech < -eig_tol:
        conclusion = (
            "unstable: at least one exact mechanism has "
            "negative second-order energy curvature"
        )
    elif min_mech <= eig_tol:
        conclusion = (
            "marginal at quadratic order: at least one exact mechanism "
            "has near-zero curvature"
        )
    else:
        conclusion = (
            "prestress-stabilized at quadratic order: all exact "
            "mechanisms have positive energy curvature"
        )

    print(
        f"mechanism-space conclusion: {conclusion}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
