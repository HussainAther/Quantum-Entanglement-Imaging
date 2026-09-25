"""Direct nonlinear energy sweep along the two exact T4 mechanism modes.

The equilibrated mirrored T4 proxy has:

    rank = 19
    internal mechanisms = 2
    self-stress dimension = 2

and both mechanism-space Hessian eigenvalues are positive.

This experiment independently checks that result by perturbing the actual
geometry along the two mechanism-Hessian eigenvectors and comparing

    Delta E(a) = E(X + a v) - E(X)

against

    Delta E_quad(a) = 0.5 * lambda * a^2.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensegrity.energy import (
    equilibrium_residual_norm,
    nonrigid_basis,
    numerical_hessian,
    reduced_nonrigid_hessian,
    total_energy,
)
from tensegrity.examples import (
    equilibrated_mirrored_t4,
)
from tensegrity.rigidity import (
    rigidity_matrix,
)


OUTPUT_DIR = Path(
    "outputs/t4_energy_sweep"
)


def mechanism_hessian_modes(
    nodes,
    members,
    springs,
):
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

    _, singular_values, Vt = np.linalg.svd(
        R_internal,
        full_matrices=True,
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    mechanism_internal = (
        Vt[rank:].T
    )

    if mechanism_internal.shape[1] != 2:
        raise RuntimeError(
            "Expected exactly two internal mechanisms."
        )

    def energy_flat(flat):
        return total_energy(
            flat.reshape((-1, 3)),
            springs,
        )

    H = numerical_hessian(
        energy_flat,
        nodes.reshape(-1),
        eps=2e-5,
    )

    H_nonrigid = reduced_nonrigid_hessian(
        H,
        nodes,
    )

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

    modes = []

    for k in range(
        eigenvectors.shape[1]
    ):
        coefficients = (
            eigenvectors[:, k]
        )

        v_internal = (
            mechanism_internal
            @ coefficients
        )

        v_cartesian = (
            Z @ v_internal
        )

        v_cartesian /= np.linalg.norm(
            v_cartesian
        )

        modes.append(
            v_cartesian
        )

    return (
        eigenvalues,
        np.column_stack(modes),
        R,
    )


def save_csv(
    mode_index,
    amplitudes,
    actual,
    predicted,
):
    path = (
        OUTPUT_DIR
        / f"mode_{mode_index}_energy_sweep.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "amplitude",
                "actual_delta_energy",
                "quadratic_prediction",
                "ratio",
            ],
        )

        writer.writeheader()

        for (
            amplitude,
            actual_value,
            predicted_value,
        ) in zip(
            amplitudes,
            actual,
            predicted,
        ):
            ratio = (
                actual_value
                / predicted_value
                if predicted_value != 0.0
                else float("nan")
            )

            writer.writerow(
                {
                    "amplitude": amplitude,
                    "actual_delta_energy": actual_value,
                    "quadratic_prediction": predicted_value,
                    "ratio": ratio,
                }
            )

    print(
        f"Saved: {path}"
    )


def save_plot(
    mode_index,
    eigenvalue,
    amplitudes,
    actual,
    predicted,
):
    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        amplitudes,
        actual,
        "o-",
        label="actual nonlinear energy",
    )

    plt.plot(
        amplitudes,
        predicted,
        "--",
        label="quadratic prediction",
    )

    plt.xlabel(
        "Mode amplitude"
    )

    plt.ylabel(
        "Delta E"
    )

    plt.title(
        f"T4 mechanism mode {mode_index} "
        f"(lambda={eigenvalue:.6f})"
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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nodes, members, springs = (
        equilibrated_mirrored_t4(
            prestress_scale=0.20,
            stiffness=1.0,
        )
    )

    equilibrium_energy = (
        total_energy(
            nodes,
            springs,
        )
    )

    equilibrium_residual = (
        equilibrium_residual_norm(
            nodes,
            springs,
        )
    )

    (
        eigenvalues,
        modes,
        R,
    ) = mechanism_hessian_modes(
        nodes,
        members,
        springs,
    )

    print()
    print("=" * 78)
    print("T4 MECHANISM ENERGY SWEEP")
    print("=" * 78)

    print(
        f"equilibrium residual: "
        f"{equilibrium_residual:.12e}"
    )

    print(
        f"equilibrium energy:   "
        f"{equilibrium_energy:.12e}"
    )

    print()

    print(
        "mechanism-Hessian eigenvalues:"
    )

    print(
        np.array2string(
            eigenvalues,
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
        modes.shape[1]
    ):
        eigenvalue = (
            eigenvalues[
                mode_index
            ]
        )

        v_flat = (
            modes[
                :,
                mode_index,
            ]
        )

        v = v_flat.reshape(
            nodes.shape
        )

        rigidity_residual = (
            np.linalg.norm(
                R @ v_flat
            )
        )

        delta_energy = []

        for amplitude in amplitudes:
            perturbed = (
                nodes
                + amplitude * v
            )

            energy = total_energy(
                perturbed,
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
            f"lambda:             "
            f"{eigenvalue:.12e}"
        )

        print(
            f"rigidity residual:  "
            f"{rigidity_residual:.12e}"
        )

        print()

        print(
            " amplitude         actual dE"
            "          quadratic dE"
            "       ratio"
        )

        for (
            amplitude,
            actual,
            predicted,
        ) in zip(
            amplitudes,
            delta_energy,
            quadratic_prediction,
        ):
            ratio = (
                actual / predicted
                if predicted != 0.0
                else float("nan")
            )

            print(
                f"{amplitude: .1e}   "
                f"{actual: .12e}   "
                f"{predicted: .12e}   "
                f"{ratio: .8f}"
            )

        downhill = int(
            np.sum(
                delta_energy[
                    amplitudes != 0.0
                ]
                < 0.0
            )
        )

        print()

        print(
            f"negative-energy perturbations: "
            f"{downhill}"
        )

        save_csv(
            mode_index + 1,
            amplitudes,
            delta_energy,
            quadratic_prediction,
        )

        save_plot(
            mode_index + 1,
            eigenvalue,
            amplitudes,
            delta_energy,
            quadratic_prediction,
        )

    print()
    print("=" * 78)
    print(
        "Finished T4 nonlinear stability verification."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
