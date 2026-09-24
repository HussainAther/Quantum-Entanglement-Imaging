"""Solve directly for a planar equilibrium after side-cable removal.

We use the relaxed remove-9 geometry as the initial guess, project it onto its
best-fit plane, then solve for an equilibrium constrained to that plane.

The in-plane coordinates are free; the out-of-plane coordinate is fixed to zero
in a local plane basis. This removes the nearly singular out-of-plane collapse
from the optimization variables and lets us ask whether an exact planar
equilibrium exists for the bilateral spring model.
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
    analyze_rigidity,
    rigidity_matrix,
)


def best_fit_plane_basis(nodes):
    """Return centroid and an orthonormal basis (e1, e2, normal)."""
    X = np.asarray(nodes, dtype=float)

    centroid = X.mean(axis=0)

    centered = X - centroid

    _, _, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    e1 = Vt[0]
    e2 = Vt[1]
    normal = Vt[2]

    return centroid, e1, e2, normal


def project_to_plane_coordinates(
    nodes,
    centroid,
    e1,
    e2,
):
    """Represent 3D nodes by 2D coordinates in the chosen plane."""
    centered = nodes - centroid

    u = centered @ e1
    v = centered @ e2

    return np.column_stack([u, v])


def reconstruct_from_plane_coordinates(
    coords,
    centroid,
    e1,
    e2,
):
    """Map 2D plane coordinates back to 3D."""
    coords = np.asarray(coords, dtype=float)

    return (
        centroid
        + coords[:, 0, None] * e1[None, :]
        + coords[:, 1, None] * e2[None, :]
    )


def remove_planar_rigid_motion(coords):
    """Gauge-fix planar rigid motion.

    Fix:
      node 0 at its initial 2D location,
      node 1's second coordinate at its initial value.

    This removes two translations and one in-plane rotation.
    """
    free = []

    for i in range(coords.shape[0]):
        for j in range(2):
            if i == 0:
                continue

            if i == 1 and j == 1:
                continue

            free.append((i, j))

    return free


def main():
    nodes, members, springs = three_strut_prism(
        prestress_scale=0.20
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
    # First get the near-planar relaxed state.
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

    coords0 = project_to_plane_coordinates(
        X_relaxed,
        centroid,
        e1,
        e2,
    )

    free = remove_planar_rigid_motion(
        coords0
    )

    fixed_coords = coords0.copy()

    x0 = np.array(
        [
            coords0[i, j]
            for i, j in free
        ],
        dtype=float,
    )

    def unpack(x):
        coords = fixed_coords.copy()

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

        grad3 = energy_gradient(
            X,
            reduced_springs,
        )

        #
        # Project forces into the plane.
        #
        gu = grad3 @ e1
        gv = grad3 @ e2

        grad2 = np.column_stack(
            [gu, gv]
        )

        #
        # Return only free-coordinate residuals.
        #
        return np.array(
            [
                grad2[i, j]
                for i, j in free
            ],
            dtype=float,
        )

    result = least_squares(
        residual,
        x0,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=20000,
        verbose=1,
    )

    coords_final = unpack(
        result.x
    )

    X_final = reconstruct_from_plane_coordinates(
        coords_final,
        centroid,
        e1,
        e2,
    )

    rigidity = analyze_rigidity(
        X_final,
        reduced_members,
    )

    R = rigidity_matrix(
        X_final,
        reduced_members,
    )

    singular_values = np.linalg.svd(
        R,
        compute_uv=False,
    )

    print()
    print("=" * 72)
    print("PLANAR EQUILIBRIUM SOLVE")
    print("=" * 72)

    print(
        f"least_squares success:      "
        f"{result.success}"
    )

    print(
        f"message:                    "
        f"{result.message}"
    )

    print(
        f"function evaluations:       "
        f"{result.nfev}"
    )

    print(
        f"in-plane residual norm:     "
        f"{np.linalg.norm(result.fun):.12e}"
    )

    print(
        f"full 3D residual norm:      "
        f"{equilibrium_residual_norm(X_final, reduced_springs):.12e}"
    )

    print(
        f"energy:                     "
        f"{total_energy(X_final, reduced_springs):.12e}"
    )

    print()
    print(
        f"rigidity rank:              "
        f"{rigidity.rank}"
    )

    print(
        f"mechanisms:                 "
        f"{rigidity.mechanisms}"
    )

    print(
        f"self-stress dimension:      "
        f"{rigidity.self_stress_dimension}"
    )

    print()
    print(
        "rigidity singular values:"
    )

    print(
        np.array2string(
            singular_values,
            precision=12,
            suppress_small=False,
        )
    )

    #
    # Confirm exact planarity.
    #
    centered = (
        X_final
        - X_final.mean(
            axis=0,
            keepdims=True,
        )
    )

    coord_sv = np.linalg.svd(
        centered,
        compute_uv=False,
    )

    print()
    print(
        "coordinate singular values:"
    )

    print(
        np.array2string(
            coord_sv,
            precision=12,
            suppress_small=False,
        )
    )

    print()
    print(
        f"smallest coordinate SV:     "
        f"{coord_sv[-1]:.12e}"
    )


if __name__ == "__main__":
    main()
