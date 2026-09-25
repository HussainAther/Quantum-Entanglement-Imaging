"""Direct nonlinear energy sweep along the exact planar mechanism modes.

For the remove-9 planar equilibrium we already found:

    rank(R) = 9
    internal mechanisms = 3
    self-stress dimension = 2

and the Hessian restricted to the mechanism subspace had three positive
eigenvalues.

This experiment independently checks that result by perturbing the actual
node coordinates along the three mechanism-Hessian eigenvectors and comparing

    Delta E(a) = E(X + a v) - E(X)

against the quadratic prediction

    Delta E_quad(a) = 0.5 * lambda * a^2.

For a genuinely quadratically prestress-stabilized mechanism, sufficiently
small positive and negative amplitudes should both increase the energy, and
the measured Delta E should approach the quadratic prediction as a -> 0.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
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
from tensegrity.rigidity import rigidity_matrix


OUTPUT_DIR = Path(
    "outputs/prism_planar_energy_sweep"
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


def exact_planar_equilibrium():
    """Reconstruct the exact remove-9 planar equilibrium."""

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
    # Approach the planar limiting state.
    #
    relaxation = relax_framework(
        nodes,
        reduced_springs,
        residual_tol=1e-9,
        max_iterations=5000,
        initial_step_size=0.2,
    )

    X_relaxed = relaxation.nodes

    (
        centroid,
        e1,
        e2,
        normal,
    ) = best_fit_plane_basis(
        X_relaxed
    )

    coords0 = project_to_plane_coordinates(
        X_relaxed,
        centroid,
        e1,
        e2,
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

    X = reconstruct_from_plane_coordinates(
        unpack(solve.x),
        centroid,
        e1,
        e2,
    )

    return (
        X,
        reduced_members,
        reduced_springs,
        normal,
    )


def mechanism_hessian_modes(
    X,
    members,
    springs,
):
    """Return mechanism-Hessian eigenvalues and Cartesian eigenmodes."""

    R = rigidity_matrix(
        X,
        members,
    )

    #
    # Internal/non-rigid Cartesian basis.
    #
    Z = nonrigid_basis(
        X
    )

    R_internal = (
        R @ Z
    )

    #
    # full_matrices=True is necessary
    # because R_internal is 11 x 12.
    #
    _, singular_values, Vt = np.linalg.svd(
        R_internal,
        full_matrices=True,
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    #
    # Columns span the exact internal
    # mechanism space.
    #
    mechanism_internal = (
        Vt[rank:].T
    )

    if mechanism_internal.shape[1] != 3:
        raise RuntimeError(
            "Expected exactly three internal mechanisms, "
            f"found {mechanism_internal.shape[1]}"
        )

    #
    # Full energy Hessian.
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

    #
    # Hessian restricted to mechanism space.
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

    eigenvalues, eigenvectors = np.linalg.eigh(
        H_mech
    )

    #
    # Each eigenvector here is expressed
    # in the three-dimensional mechanism
    # basis. Convert it to internal
    # coordinates and then Cartesian
    # coordinates.
    #
    modes_cartesian = []

    for k in range(
        eigenvectors.shape[1]
    ):
        coeff = eigenvectors[:, k]

        v_internal = (
            mechanism_internal @ coeff
        )

        v_cartesian = (
            Z @ v_internal
        )

        #
        # Numerical safety. It should
        # already have unit norm because
        # all bases involved are
        # orthonormal.
        #
        v_cartesian /= np.linalg.norm(
            v_cartesian
        )

        modes_cartesian.append(
            v_cartesian
        )

    return (
        eigenvalues,
        np.column_stack(
            modes_cartesian
        ),
        H_mech,
        R,
    )


def write_mode_csv(
    mode_index,
    eigenvalue,
    amplitudes,
    delta_energy,
    quadratic_prediction,
):
    path = (
        OUTPUT_DIR
        / f"mode_{mode_index}_energy_sweep.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "amplitude",
            "actual_delta_energy",
            "quadratic_prediction",
            "absolute_error",
            "relative_error",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for (
            a,
            actual,
            predicted,
        ) in zip(
            amplitudes,
            delta_energy,
            quadratic_prediction,
        ):
            absolute_error = (
                actual - predicted
            )

            if predicted != 0.0:
                relative_error = (
                    absolute_error
                    / predicted
                )
            else:
                relative_error = float(
                    "nan"
                )

            writer.writerow(
                {
                    "amplitude": a,
                    "actual_delta_energy": actual,
                    "quadratic_prediction": predicted,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

    print(
        f"Saved: {path}"
    )


def plot_mode(
    mode_index,
    eigenvalue,
    amplitudes,
    delta_energy,
    quadratic_prediction,
):
    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        amplitudes,
        delta_energy,
        "o-",
        label="actual nonlinear energy",
    )

    plt.plot(
        amplitudes,
        quadratic_prediction,
        "--",
        label="quadratic Hessian prediction",
    )

    plt.xlabel(
        "Mode amplitude"
    )

    plt.ylabel(
        r"$\Delta E$"
    )

    plt.title(
        f"Planar mechanism mode {mode_index} "
        f"(lambda = {eigenvalue:.6f})"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / f"mode_{mode_index}_energy_sweep.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def symmetric_energy_check(
    amplitudes,
    delta_energy,
):
    """Measure E(+a)-E(-a) symmetry for matched amplitudes."""

    print(
        "  +/- amplitude symmetry:"
    )

    positive = (
        amplitudes[
            amplitudes > 0.0
        ]
    )

    for a in positive:
        pos_index = np.where(
            np.isclose(
                amplitudes,
                a,
                rtol=0.0,
                atol=1e-16,
            )
        )[0][0]

        neg_index = np.where(
            np.isclose(
                amplitudes,
                -a,
                rtol=0.0,
                atol=1e-16,
            )
        )[0][0]

        d_pos = delta_energy[
            pos_index
        ]

        d_neg = delta_energy[
            neg_index
        ]

        asymmetry = (
            d_pos - d_neg
        )

        scale = max(
            abs(d_pos),
            abs(d_neg),
            1e-30,
        )

        relative_asymmetry = (
            asymmetry / scale
        )

        print(
            f"    a={a:.1e}: "
            f"dE(+a)={d_pos:.12e}, "
            f"dE(-a)={d_neg:.12e}, "
            f"relative asymmetry="
            f"{relative_asymmetry:.3e}"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X,
        members,
        springs,
        plane_normal,
    ) = exact_planar_equilibrium()

    equilibrium_energy = total_energy(
        X,
        springs,
    )

    equilibrium_residual = (
        equilibrium_residual_norm(
            X,
            springs,
        )
    )

    (
        mechanism_eigenvalues,
        mechanism_modes,
        H_mech,
        R,
    ) = mechanism_hessian_modes(
        X,
        members,
        springs,
    )

    print()
    print("=" * 78)

    print(
        "PLANAR MECHANISM ENERGY SWEEP"
    )

    print("=" * 78)

    print(
        f"equilibrium residual:       "
        f"{equilibrium_residual:.12e}"
    )

    print(
        f"equilibrium energy:         "
        f"{equilibrium_energy:.12e}"
    )

    print()

    print(
        "mechanism-Hessian eigenvalues:"
    )

    print(
        np.array2string(
            mechanism_eigenvalues,
            precision=12,
            suppress_small=False,
        )
    )

    amplitudes = np.array(
        [
            -1e-2,
            -5e-3,
            -2e-3,
            -1e-3,
            -5e-4,
            -2e-4,
            -1e-4,
            0.0,
            1e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
        ],
        dtype=float,
    )

    for mode_index in range(
        mechanism_modes.shape[1]
    ):
        eigenvalue = (
            mechanism_eigenvalues[
                mode_index
            ]
        )

        v_flat = (
            mechanism_modes[
                :,
                mode_index,
            ]
        )

        v = v_flat.reshape(
            X.shape
        )

        #
        # Confirm this is an exact
        # first-order mechanism.
        #
        rigidity_residual = (
            np.linalg.norm(
                R @ v_flat
            )
        )

        in_plane_component = (
            v
            - (
                v @ plane_normal
            )[:, None]
            * plane_normal[None, :]
        )

        out_of_plane_component = (
            (v @ plane_normal)[
                :,
                None,
            ]
            * plane_normal[
                None,
                :,
            ]
        )

        in_plane_norm = float(
            np.linalg.norm(
                in_plane_component
            )
        )

        out_of_plane_norm = float(
            np.linalg.norm(
                out_of_plane_component
            )
        )

        delta_energy = []

        for amplitude in amplitudes:
            X_perturbed = (
                X
                + amplitude * v
            )

            energy = total_energy(
                X_perturbed,
                springs,
            )

            delta_energy.append(
                energy
                - equilibrium_energy
            )

        delta_energy = np.asarray(
            delta_energy,
            dtype=float,
        )

        quadratic_prediction = (
            0.5
            * eigenvalue
            * amplitudes ** 2
        )

        print()
        print("-" * 78)

        print(
            f"mode {mode_index + 1}"
        )

        print("-" * 78)

        print(
            f"lambda:                     "
            f"{eigenvalue:.12e}"
        )

        print(
            f"rigidity residual:          "
            f"{rigidity_residual:.12e}"
        )

        print(
            f"in-plane norm:              "
            f"{in_plane_norm:.12e}"
        )

        print(
            f"out-of-plane norm:          "
            f"{out_of_plane_norm:.12e}"
        )

        print()

        print(
            "  amplitude        actual dE"
            "          quadratic dE"
            "       ratio"
        )

        for (
            a,
            actual,
            predicted,
        ) in zip(
            amplitudes,
            delta_energy,
            quadratic_prediction,
        ):
            if predicted != 0.0:
                ratio = (
                    actual / predicted
                )
            else:
                ratio = float(
                    "nan"
                )

            print(
                f"  {a: .1e}   "
                f"{actual: .12e}   "
                f"{predicted: .12e}   "
                f"{ratio: .8f}"
            )

        print()

        symmetric_energy_check(
            amplitudes,
            delta_energy,
        )

        #
        # Count any actual downhill
        # perturbations, excluding zero.
        #
        nonzero = (
            amplitudes != 0.0
        )

        downhill = int(
            np.sum(
                delta_energy[
                    nonzero
                ]
                < 0.0
            )
        )

        print()

        print(
            f"  negative-energy perturbations: "
            f"{downhill}"
        )

        write_mode_csv(
            mode_index + 1,
            eigenvalue,
            amplitudes,
            delta_energy,
            quadratic_prediction,
        )

        plot_mode(
            mode_index + 1,
            eigenvalue,
            amplitudes,
            delta_energy,
            quadratic_prediction,
        )

    print()
    print("=" * 78)
    print(
        "Finished direct nonlinear mechanism-energy verification."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
