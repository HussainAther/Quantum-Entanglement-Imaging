"""Resolve the foundation-stiffness transition of the T4 + two-T3 composite.

This extends the coarse foundation sweep by:
1. scanning k_f densely on a log scale,
2. estimating the rigid-foundation effective-stiffness asymptote,
3. locating the foundation stiffness needed to reach 50%, 90%, and 95%
   of that asymptotic response.

The model remains the same simplified, stress-free bilateral spring proxy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tensegrity.energy import numerical_hessian
from tensegrity.examples import t4_with_two_supported_t3

from experiments.t4_two_t3_foundation import (
    force_vector_for_nodes,
    total_composite_energy,
)


OUTPUT_DIR = Path(
    "outputs/t4_two_t3_foundation_transition"
)


def response_at_foundation_k(
    reference_nodes,
    members,
    support_nodes,
    loaded_nodes,
    member_stiffness,
    foundation_k,
):
    def energy_flat(flat):
        X = flat.reshape(
            reference_nodes.shape
        )

        return total_composite_energy(
            X,
            reference_nodes,
            members,
            support_nodes,
            member_stiffness,
            foundation_k,
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

    force = force_vector_for_nodes(
        reference_nodes.shape[0],
        loaded_nodes,
        force_per_node=1.0,
    )

    displacement = np.linalg.solve(
        H,
        force,
    )

    dz = np.array(
        [
            displacement[
                3 * node + 2
            ]
            for node in loaded_nodes
        ]
    )

    mean_dz = float(
        np.mean(dz)
    )

    total_force = float(
        len(loaded_nodes)
    )

    effective_stiffness = (
        total_force / mean_dz
    )

    return (
        float(eigenvalues[0]),
        mean_dz,
        effective_stiffness,
    )


def crossing_k(
    k_values,
    stiffness_values,
    target,
):
    """Interpolate first log-k crossing of the target stiffness."""

    for i in range(
        1,
        len(k_values),
    ):
        y0 = stiffness_values[i - 1]
        y1 = stiffness_values[i]

        if y0 <= target <= y1:
            x0 = np.log10(
                k_values[i - 1]
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

    support_nodes = np.array(
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

    #
    # Resolve the transition much more finely.
    #
    foundation_values = np.logspace(
        -4,
        5,
        91,
    )

    lambda_min_values = []
    mean_dz_values = []
    stiffness_values = []

    print()
    print("=" * 94)
    print(
        "T4 + TWO T3: FOUNDATION TRANSITION SCAN"
    )
    print("=" * 94)

    for foundation_k in foundation_values:
        (
            lambda_min,
            mean_dz,
            effective_stiffness,
        ) = response_at_foundation_k(
            reference_nodes,
            members,
            support_nodes,
            loaded_nodes,
            member_stiffness,
            float(foundation_k),
        )

        lambda_min_values.append(
            lambda_min
        )

        mean_dz_values.append(
            mean_dz
        )

        stiffness_values.append(
            effective_stiffness
        )

    lambda_min_values = np.asarray(
        lambda_min_values
    )

    mean_dz_values = np.asarray(
        mean_dz_values
    )

    stiffness_values = np.asarray(
        stiffness_values
    )

    #
    # The final high-k point approximates
    # the perfectly fixed-foundation limit.
    #
    asymptotic_stiffness = float(
        stiffness_values[-1]
    )

    print(
        f"high-k asymptotic stiffness: "
        f"{asymptotic_stiffness:.12e}"
    )

    print()

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
            f"{100.0 * fraction:5.1f}% "
            f"of asymptotic stiffness: "
            f"k_f ~= {k_cross:.12e}"
        )

    #
    # CSV.
    #
    csv_path = (
        OUTPUT_DIR
        / "foundation_transition.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "foundation_k",
                "lambda_min",
                "mean_dz",
                "effective_stiffness",
                "fraction_of_asymptote",
            ]
        )

        for (
            k,
            lam,
            dz,
            stiffness,
        ) in zip(
            foundation_values,
            lambda_min_values,
            mean_dz_values,
            stiffness_values,
        ):
            writer.writerow(
                [
                    k,
                    lam,
                    dz,
                    stiffness,
                    stiffness
                    / asymptotic_stiffness,
                ]
            )

    print()
    print(
        f"Saved: {csv_path}"
    )

    #
    # Effective stiffness plot.
    #
    plt.figure(
        figsize=(7, 5)
    )

    plt.semilogx(
        foundation_values,
        stiffness_values,
        "o-",
        markersize=3,
    )

    plt.axhline(
        asymptotic_stiffness,
        linestyle="--",
        label="high-k asymptote",
    )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Effective vertical stiffness"
    )

    plt.title(
        "T4 + two T3 foundation-stiffness transition"
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
    # Smallest eigenvalue plot.
    #
    plt.figure(
        figsize=(7, 5)
    )

    plt.semilogx(
        foundation_values,
        lambda_min_values,
        "o-",
        markersize=3,
    )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Smallest Hessian eigenvalue"
    )

    plt.title(
        "Lowest stiffness mode vs foundation stiffness"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    eigen_plot = (
        OUTPUT_DIR
        / "lambda_min_vs_foundation.png"
    )

    plt.savefig(
        eigen_plot,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {stiffness_plot}"
    )

    print(
        f"Saved: {eigen_plot}"
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
