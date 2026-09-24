"""Probe candidate limiting geometries after prism cable ablation.

For each representative cable-removal class:

1. Relax the damaged prism as far as the current solver allows.
2. Compute the best-fit plane of the relaxed node cloud.
3. Project all nodes exactly onto that plane.
4. Recompute:
   - distance to the best-fit plane
   - rigidity rank
   - internal mechanism count
   - self-stress dimension
   - projected non-rigid rigidity spectrum

The goal is to test whether the nearly planar side-cable failure trajectory is
approaching an exactly planar rank-deficient configuration.

Important:
Projecting onto the nearest plane is a diagnostic construction. It is not
itself an energy relaxation and does not preserve spring lengths or force
equilibrium.
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
    analyze_rigidity,
    rigidity_matrix,
)


OUTPUT_DIR = Path(
    "outputs/prism_limiting_geometry"
)


def best_fit_plane(nodes):
    """Return centroid, plane normal, and coordinate singular values."""
    X = np.asarray(
        nodes,
        dtype=float,
    )

    centroid = X.mean(
        axis=0,
    )

    centered = (
        X - centroid
    )

    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    #
    # Right singular vector associated
    # with the smallest coordinate
    # singular value is the plane normal.
    #
    normal = Vt[-1].copy()

    normal /= np.linalg.norm(
        normal
    )

    return (
        centroid,
        normal,
        singular_values,
    )


def project_to_plane(
    nodes,
    centroid,
    normal,
):
    """Orthogonally project nodes onto a plane."""
    X = np.asarray(
        nodes,
        dtype=float,
    )

    offsets = (
        X - centroid
    )

    signed_distances = (
        offsets @ normal
    )

    projected = (
        X
        - signed_distances[:, None]
        * normal[None, :]
    )

    return (
        projected,
        signed_distances,
    )


def internal_spectrum(
    nodes,
    members,
):
    """Return rigidity spectrum after removing rigid-body motions."""
    R = rigidity_matrix(
        nodes,
        members,
    )

    Z = nonrigid_basis(
        nodes,
    )

    R_internal = (
        R @ Z
    )

    singular_values = np.linalg.svd(
        R_internal,
        compute_uv=False,
    )

    return (
        R,
        Z,
        R_internal,
        singular_values,
    )


def rank_summary(
    singular_values,
    rows,
    cols,
):
    """Return rank/nullity/self-stress under several numerical tolerances."""
    tolerances = [
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-12,
    ]

    rows_out = []

    for tol in tolerances:
        rank = int(
            np.sum(
                singular_values > tol
            )
        )

        rows_out.append(
            {
                "tol": tol,
                "rank": rank,
                "internal_nullity": (
                    cols - rank
                ),
                "self_stress_dimension": (
                    rows - rank
                ),
            }
        )

    return rows_out


def save_nodes(
    path,
    original,
    projected,
    signed_distances,
):
    with path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "node",
            "x_relaxed",
            "y_relaxed",
            "z_relaxed",
            "x_projected",
            "y_projected",
            "z_projected",
            "signed_plane_distance",
            "absolute_plane_distance",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(
            original.shape[0]
        ):
            writer.writerow(
                {
                    "node": i,
                    "x_relaxed": (
                        original[i, 0]
                    ),
                    "y_relaxed": (
                        original[i, 1]
                    ),
                    "z_relaxed": (
                        original[i, 2]
                    ),
                    "x_projected": (
                        projected[i, 0]
                    ),
                    "y_projected": (
                        projected[i, 1]
                    ),
                    "z_projected": (
                        projected[i, 2]
                    ),
                    "signed_plane_distance": (
                        signed_distances[i]
                    ),
                    "absolute_plane_distance": (
                        abs(
                            signed_distances[i]
                        )
                    ),
                }
            )


def analyze_case(
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

    #
    # First obtain the numerically relaxed
    # near-singular geometry.
    #
    relaxation = relax_framework(
        nodes,
        reduced_springs,
        residual_tol=1e-9,
        max_iterations=5000,
        initial_step_size=0.2,
    )

    X = relaxation.nodes

    #
    # Best-fit plane.
    #
    centroid, normal, coord_sv = (
        best_fit_plane(
            X
        )
    )

    X_plane, signed_distances = (
        project_to_plane(
            X,
            centroid,
            normal,
        )
    )

    max_plane_distance = float(
        np.max(
            np.abs(
                signed_distances
            )
        )
    )

    rms_plane_distance = float(
        np.sqrt(
            np.mean(
                signed_distances ** 2
            )
        )
    )

    #
    # Analyze relaxed geometry.
    #
    relaxed_rigidity = (
        analyze_rigidity(
            X,
            reduced_members,
        )
    )

    (
        _,
        _,
        relaxed_internal,
        relaxed_sv,
    ) = internal_spectrum(
        X,
        reduced_members,
    )

    #
    # Analyze exactly planar projection.
    #
    planar_rigidity = (
        analyze_rigidity(
            X_plane,
            reduced_members,
        )
    )

    (
        _,
        _,
        planar_internal,
        planar_sv,
    ) = internal_spectrum(
        X_plane,
        reduced_members,
    )

    relaxed_rank_table = (
        rank_summary(
            relaxed_sv,
            relaxed_internal.shape[0],
            relaxed_internal.shape[1],
        )
    )

    planar_rank_table = (
        rank_summary(
            planar_sv,
            planar_internal.shape[0],
            planar_internal.shape[1],
        )
    )

    projected_coord_sv = (
        np.linalg.svd(
            X_plane
            - X_plane.mean(
                axis=0,
                keepdims=True,
            ),
            compute_uv=False,
        )
    )

    print()
    print("=" * 78)

    print(
        f"remove cable {removed_idx} "
        f"({removed.i},{removed.j})"
    )

    print("=" * 78)

    print(
        f"relaxation converged:              "
        f"{relaxation.converged}"
    )

    print(
        f"relaxation residual:               "
        f"{relaxation.residual_final:.12e}"
    )

    print()
    print(
        "Best-fit plane"
    )

    print(
        "centroid:                         "
        + np.array2string(
            centroid,
            precision=8,
        )
    )

    print(
        "normal:                           "
        + np.array2string(
            normal,
            precision=8,
        )
    )

    print(
        f"maximum distance to plane:         "
        f"{max_plane_distance:.12e}"
    )

    print(
        f"RMS distance to plane:             "
        f"{rms_plane_distance:.12e}"
    )

    print()
    print(
        "Coordinate spectrum before projection"
    )

    print(
        np.array2string(
            coord_sv,
            precision=12,
            suppress_small=False,
        )
    )

    print(
        "Coordinate spectrum after projection"
    )

    print(
        np.array2string(
            projected_coord_sv,
            precision=12,
            suppress_small=False,
        )
    )

    print()
    print(
        "Relaxed geometry"
    )

    print(
        f"rigidity rank:                     "
        f"{relaxed_rigidity.rank}"
    )

    print(
        f"mechanisms:                        "
        f"{relaxed_rigidity.mechanisms}"
    )

    print(
        f"self-stress dimension:             "
        f"{relaxed_rigidity.self_stress_dimension}"
    )

    print(
        "internal singular values:"
    )

    print(
        np.array2string(
            relaxed_sv,
            precision=12,
            suppress_small=False,
        )
    )

    print()
    print(
        "Exactly planar projection"
    )

    print(
        f"rigidity rank:                     "
        f"{planar_rigidity.rank}"
    )

    print(
        f"mechanisms:                        "
        f"{planar_rigidity.mechanisms}"
    )

    print(
        f"self-stress dimension:             "
        f"{planar_rigidity.self_stress_dimension}"
    )

    print(
        "internal singular values:"
    )

    print(
        np.array2string(
            planar_sv,
            precision=12,
            suppress_small=False,
        )
    )

    print()
    print(
        "Relaxed rank table"
    )

    for row in relaxed_rank_table:
        print(
            f"tol={row['tol']:.0e}: "
            f"rank={row['rank']:2d}, "
            f"internal nullity="
            f"{row['internal_nullity']:2d}, "
            f"self-stress dim="
            f"{row['self_stress_dimension']:2d}"
        )

    print()
    print(
        "Exactly planar rank table"
    )

    for row in planar_rank_table:
        print(
            f"tol={row['tol']:.0e}: "
            f"rank={row['rank']:2d}, "
            f"internal nullity="
            f"{row['internal_nullity']:2d}, "
            f"self-stress dim="
            f"{row['self_stress_dimension']:2d}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_nodes(
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            "_plane_projection.csv"
        ),
        X,
        X_plane,
        signed_distances,
    )

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
            "max_plane_distance",
            "rms_plane_distance",
            "coord_sigma1",
            "coord_sigma2",
            "coord_sigma3",
            "relaxed_sigma_min",
            "relaxed_sigma_second",
            "planar_sigma_min",
            "planar_sigma_second",
            "relaxed_rank_1e-9",
            "planar_rank_1e-9",
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
                "max_plane_distance": (
                    max_plane_distance
                ),
                "rms_plane_distance": (
                    rms_plane_distance
                ),
                "coord_sigma1": (
                    coord_sv[0]
                ),
                "coord_sigma2": (
                    coord_sv[1]
                ),
                "coord_sigma3": (
                    coord_sv[2]
                ),
                "relaxed_sigma_min": (
                    relaxed_sv[-1]
                ),
                "relaxed_sigma_second": (
                    relaxed_sv[-2]
                ),
                "planar_sigma_min": (
                    planar_sv[-1]
                ),
                "planar_sigma_second": (
                    planar_sv[-2]
                ),
                "relaxed_rank_1e-9": (
                    int(
                        np.sum(
                            relaxed_sv
                            > 1e-9
                        )
                    )
                ),
                "planar_rank_1e-9": (
                    int(
                        np.sum(
                            planar_sv
                            > 1e-9
                        )
                    )
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
    # Representative perimeter cable.
    #
    analyze_case(
        3,
        nodes,
        members,
        springs,
    )

    #
    # Representative side cable.
    #
    analyze_case(
        9,
        nodes,
        members,
        springs,
    )


if __name__ == "__main__":
    main()
