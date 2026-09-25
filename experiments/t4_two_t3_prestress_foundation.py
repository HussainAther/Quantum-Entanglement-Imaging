"""Joint prestress / foundation-stiffness sweep for the T4 + two-T3 composite.

The composite is assigned a uniform axial prestrain by shortening every member's
rest length relative to its geometric reference length:

    L0 = (1 - prestrain) * L_ref

This is a simplified bilateral spring model. It is not yet a full
tension-only-cable / compression-only-strut tensegrity model.

For each prestrain and foundation stiffness, the script:

1. constructs the total energy,
2. evaluates the tangent Hessian at the reference geometry,
3. measures the smallest Hessian eigenvalue,
4. solves the linear response to equal +z loads on the central T4 triangle,
5. reports mean vertical displacement and effective stiffness.

The goal is to determine how prestress shifts the foundation-stiffness response.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensegrity.energy import numerical_hessian
from tensegrity.examples import t4_with_two_supported_t3


OUTPUT_DIR = Path(
    "outputs/t4_two_t3_prestress_foundation"
)


def member_energy(
    nodes,
    members,
    reference_nodes,
    stiffness,
    prestrain,
):
    energy = 0.0

    for member in members:
        i = member.i
        j = member.j

        L = float(
            np.linalg.norm(
                nodes[i] - nodes[j]
            )
        )

        L_ref = float(
            np.linalg.norm(
                reference_nodes[i]
                - reference_nodes[j]
            )
        )

        L0 = (
            (1.0 - prestrain)
            * L_ref
        )

        energy += (
            0.5
            * stiffness
            * (L - L0) ** 2
        )

    return energy


def foundation_energy(
    nodes,
    reference_nodes,
    support_nodes,
    foundation_stiffness,
):
    displacement = (
        nodes[support_nodes]
        - reference_nodes[support_nodes]
    )

    return (
        0.5
        * foundation_stiffness
        * float(
            np.sum(
                displacement ** 2
            )
        )
    )


def total_energy(
    nodes,
    reference_nodes,
    members,
    support_nodes,
    member_stiffness,
    foundation_stiffness,
    prestrain,
):
    return (
        member_energy(
            nodes,
            members,
            reference_nodes,
            member_stiffness,
            prestrain,
        )
        + foundation_energy(
            nodes,
            reference_nodes,
            support_nodes,
            foundation_stiffness,
        )
    )


def load_vector(
    node_count,
    loaded_nodes,
    force_per_node=1.0,
):
    force = np.zeros(
        3 * node_count,
        dtype=float,
    )

    for node in loaded_nodes:
        force[
            3 * node + 2
        ] = force_per_node

    return force


def response(
    reference_nodes,
    members,
    support_nodes,
    loaded_nodes,
    member_stiffness,
    foundation_k,
    prestrain,
):
    def energy_flat(flat):
        X = flat.reshape(
            reference_nodes.shape
        )

        return total_energy(
            X,
            reference_nodes,
            members,
            support_nodes,
            member_stiffness,
            foundation_k,
            prestrain,
        )

    H = numerical_hessian(
        energy_flat,
        reference_nodes.reshape(-1),
        eps=2e-5,
    )

    H = 0.5 * (
        H + H.T
    )

    eigenvalues = np.linalg.eigvalsh(
        H
    )

    lambda_min = float(
        eigenvalues[0]
    )

    force = load_vector(
        reference_nodes.shape[0],
        loaded_nodes,
        force_per_node=1.0,
    )

    #
    # If the tangent stiffness is singular or
    # indefinite, solving H u = f may fail or
    # become physically meaningless.
    #
    if lambda_min <= 1e-10:
        return (
            lambda_min,
            float("nan"),
            float("nan"),
        )

    displacement = np.linalg.solve(
        H,
        force,
    )

    dz = np.asarray(
        [
            displacement[
                3 * node + 2
            ]
            for node in loaded_nodes
        ],
        dtype=float,
    )

    mean_dz = float(
        np.mean(dz)
    )

    total_force = float(
        len(loaded_nodes)
    )

    effective_stiffness = (
        total_force / mean_dz
        if abs(mean_dz) > 1e-15
        else np.inf
    )

    return (
        lambda_min,
        mean_dz,
        effective_stiffness,
    )


def crossing_k(
    k_values,
    stiffness_values,
    target,
):
    valid = np.isfinite(
        stiffness_values
    )

    k_values = k_values[
        valid
    ]

    stiffness_values = (
        stiffness_values[
            valid
        ]
    )

    if len(k_values) < 2:
        return float("nan")

    for i in range(
        1,
        len(k_values),
    ):
        y0 = stiffness_values[
            i - 1
        ]

        y1 = stiffness_values[
            i
        ]

        if y0 <= target <= y1:
            x0 = np.log10(
                k_values[
                    i - 1
                ]
            )

            x1 = np.log10(
                k_values[i]
            )

            if y1 == y0:
                return float(
                    k_values[i]
                )

            fraction = (
                (target - y0)
                / (y1 - y0)
            )

            x = (
                x0
                + fraction
                * (x1 - x0)
            )

            return float(
                10.0 ** x
            )

    return float("nan")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_nodes, members = (
        t4_with_two_supported_t3()
    )

    support_nodes = np.asarray(
        [
            9,
            10,
            11,
            12,
            13,
            14,
        ],
        dtype=int,
    )

    loaded_nodes = [
        3,
        4,
        5,
    ]

    member_stiffness = 1.0

    prestrain_values = np.asarray(
        [
            0.00,
            0.05,
            0.10,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    foundation_values = np.logspace(
        -3,
        2,
        61,
    )

    rows = []

    print()
    print("=" * 110)
    print(
        "T4 + TWO T3: PRESTRESS / FOUNDATION SWEEP"
    )
    print("=" * 110)

    for prestrain in prestrain_values:
        lambda_values = []
        dz_values = []
        stiffness_values = []

        for foundation_k in foundation_values:
            (
                lambda_min,
                mean_dz,
                effective_stiffness,
            ) = response(
                reference_nodes,
                members,
                support_nodes,
                loaded_nodes,
                member_stiffness,
                float(foundation_k),
                float(prestrain),
            )

            lambda_values.append(
                lambda_min
            )

            dz_values.append(
                mean_dz
            )

            stiffness_values.append(
                effective_stiffness
            )

            rows.append(
                {
                    "prestrain": prestrain,
                    "foundation_k": foundation_k,
                    "lambda_min": lambda_min,
                    "mean_dz": mean_dz,
                    "effective_stiffness": effective_stiffness,
                }
            )

        lambda_values = np.asarray(
            lambda_values
        )

        dz_values = np.asarray(
            dz_values
        )

        stiffness_values = np.asarray(
            stiffness_values
        )

        valid = np.isfinite(
            stiffness_values
        )

        if np.any(valid):
            asymptotic_stiffness = float(
                stiffness_values[
                    np.where(valid)[0][-1]
                ]
            )
        else:
            asymptotic_stiffness = float(
                "nan"
            )

        print()
        print(
            f"prestrain = {prestrain:.2f}"
        )

        print(
            f"  high-k effective stiffness: "
            f"{asymptotic_stiffness:.12e}"
        )

        for fraction in [
            0.50,
            0.90,
            0.95,
        ]:
            target = (
                fraction
                * asymptotic_stiffness
            )

            k_cross = crossing_k(
                foundation_values,
                stiffness_values,
                target,
            )

            print(
                f"  {100.0 * fraction:5.1f}% "
                f"of asymptote at k_f ~= "
                f"{k_cross:.12e}"
            )

        print(
            f"  minimum lambda_min over sweep: "
            f"{np.nanmin(lambda_values):.12e}"
        )

    #
    # Save all results.
    #
    csv_path = (
        OUTPUT_DIR
        / "prestress_foundation_sweep.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prestrain",
                "foundation_k",
                "lambda_min",
                "mean_dz",
                "effective_stiffness",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"Saved: {csv_path}"
    )

    #
    # Plot effective stiffness vs foundation.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for prestrain in prestrain_values:
        selected = [
            row
            for row in rows
            if np.isclose(
                row["prestrain"],
                prestrain,
            )
        ]

        k = np.asarray(
            [
                row[
                    "foundation_k"
                ]
                for row in selected
            ]
        )

        stiffness = np.asarray(
            [
                row[
                    "effective_stiffness"
                ]
                for row in selected
            ]
        )

        plt.semilogx(
            k,
            stiffness,
            label=(
                f"prestrain={prestrain:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Effective vertical stiffness"
    )

    plt.title(
        "Prestress shifts T4 + two-T3 foundation response"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    stiffness_plot = (
        OUTPUT_DIR
        / "effective_stiffness_vs_foundation.png"
    )

    plt.savefig(
        stiffness_plot,
        dpi=200,
    )

    plt.close()

    #
    # Plot minimum Hessian eigenvalue.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for prestrain in prestrain_values:
        selected = [
            row
            for row in rows
            if np.isclose(
                row["prestrain"],
                prestrain,
            )
        ]

        k = np.asarray(
            [
                row[
                    "foundation_k"
                ]
                for row in selected
            ]
        )

        lam = np.asarray(
            [
                row[
                    "lambda_min"
                ]
                for row in selected
            ]
        )

        plt.semilogx(
            k,
            lam,
            label=(
                f"prestrain={prestrain:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Smallest Hessian eigenvalue"
    )

    plt.title(
        "Prestress / foundation stability landscape"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    lambda_plot = (
        OUTPUT_DIR
        / "lambda_min_vs_foundation.png"
    )

    plt.savefig(
        lambda_plot,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {stiffness_plot}"
    )

    print(
        f"Saved: {lambda_plot}"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
