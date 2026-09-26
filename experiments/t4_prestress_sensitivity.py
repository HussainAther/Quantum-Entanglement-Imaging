"""Prestress sensitivity of the T4 + two-T3 supported composite.

This experiment fixes several representative foundation stiffnesses and sweeps
the sign-compatible prestress carried by the central T4.

For each foundation stiffness k_f and T4 prestress scale alpha, it reports:

    effective vertical stiffness
    normalized stiffness gain relative to alpha = 0
    smallest Hessian eigenvalue
    equilibrium residual

The reference geometry remains an exact equilibrium because:
- the T4 carries an internally balanced self-stress,
- the T3 attachment members are stress-free,
- the foundation springs are unstretched.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensegrity.examples import (
    t4_with_two_supported_t3,
)

from experiments.t4_prestress_with_t3_foundation import (
    build_rest_lengths,
    response,
)


OUTPUT_DIR = Path(
    "outputs/t4_prestress_sensitivity"
)


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

    foundation_values = np.asarray(
        [
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        dtype=float,
    )

    #
    # Keep alpha below stiffness=1 so all
    # derived rest lengths remain positive.
    #
    prestress_values = np.linspace(
        0.0,
        0.40,
        41,
    )

    rows = []

    print()
    print("=" * 108)
    print(
        "T4 PRESTRESS SENSITIVITY AT FIXED FOUNDATION STIFFNESS"
    )
    print("=" * 108)

    for foundation_k in foundation_values:
        stiffness_values = []
        lambda_values = []
        residual_values = []

        print()
        print(
            f"foundation k = "
            f"{foundation_k:.3e}"
        )

        for prestress_scale in prestress_values:
            rest_lengths = (
                build_rest_lengths(
                    reference_nodes,
                    members,
                    float(
                        prestress_scale
                    ),
                    stiffness=member_stiffness,
                )
            )

            (
                reference_energy,
                equilibrium_residual,
                lambda_min,
                mean_dz,
                effective_stiffness,
            ) = response(
                reference_nodes,
                members,
                rest_lengths,
                support_nodes,
                loaded_nodes,
                member_stiffness,
                float(
                    foundation_k
                ),
            )

            stiffness_values.append(
                effective_stiffness
            )

            lambda_values.append(
                lambda_min
            )

            residual_values.append(
                equilibrium_residual
            )

            rows.append(
                {
                    "foundation_k": (
                        foundation_k
                    ),
                    "prestress_scale": (
                        prestress_scale
                    ),
                    "reference_energy": (
                        reference_energy
                    ),
                    "equilibrium_residual": (
                        equilibrium_residual
                    ),
                    "lambda_min": (
                        lambda_min
                    ),
                    "mean_dz": (
                        mean_dz
                    ),
                    "effective_stiffness": (
                        effective_stiffness
                    ),
                }
            )

        stiffness_values = np.asarray(
            stiffness_values,
            dtype=float,
        )

        lambda_values = np.asarray(
            lambda_values,
            dtype=float,
        )

        residual_values = np.asarray(
            residual_values,
            dtype=float,
        )

        baseline = float(
            stiffness_values[0]
        )

        normalized_gain = (
            stiffness_values
            / baseline
        )

        #
        # Approximate initial prestress
        # sensitivity using the first step.
        #
        initial_slope = (
            stiffness_values[1]
            - stiffness_values[0]
        ) / (
            prestress_values[1]
            - prestress_values[0]
        )

        #
        # Approximate late slope near alpha=.4.
        #
        late_slope = (
            stiffness_values[-1]
            - stiffness_values[-2]
        ) / (
            prestress_values[-1]
            - prestress_values[-2]
        )

        print(
            f"  baseline k_eff:           "
            f"{baseline:.12e}"
        )

        print(
            f"  k_eff(alpha=0.40):        "
            f"{stiffness_values[-1]:.12e}"
        )

        print(
            f"  normalized gain at 0.40:  "
            f"{normalized_gain[-1]:.12f}"
        )

        print(
            f"  initial dk_eff/dalpha:    "
            f"{initial_slope:.12e}"
        )

        print(
            f"  late dk_eff/dalpha:       "
            f"{late_slope:.12e}"
        )

        print(
            f"  minimum lambda_min:       "
            f"{np.nanmin(lambda_values):.12e}"
        )

        print(
            f"  worst equilibrium residual: "
            f"{np.nanmax(residual_values):.12e}"
        )

        #
        # Add normalized gain back into
        # corresponding rows.
        #
        selected = [
            row
            for row in rows
            if np.isclose(
                row["foundation_k"],
                foundation_k,
            )
        ]

        for row, gain in zip(
            selected,
            normalized_gain,
        ):
            row[
                "normalized_stiffness_gain"
            ] = gain

    #
    # CSV.
    #
    csv_path = (
        OUTPUT_DIR
        / "t4_prestress_sensitivity.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "prestress_scale",
                "reference_energy",
                "equilibrium_residual",
                "lambda_min",
                "mean_dz",
                "effective_stiffness",
                "normalized_stiffness_gain",
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
    # Effective stiffness vs prestress.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for foundation_k in foundation_values:
        selected = [
            row
            for row in rows
            if np.isclose(
                row["foundation_k"],
                foundation_k,
            )
        ]

        alpha = np.asarray(
            [
                row[
                    "prestress_scale"
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

        plt.plot(
            alpha,
            stiffness,
            label=(
                f"k_f={foundation_k:g}"
            ),
        )

    plt.xlabel(
        "T4 prestress scale"
    )

    plt.ylabel(
        "Effective vertical stiffness"
    )

    plt.title(
        "T4 prestress sensitivity"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    stiffness_plot = (
        OUTPUT_DIR
        / "effective_stiffness_vs_prestress.png"
    )

    plt.savefig(
        stiffness_plot,
        dpi=200,
    )

    plt.close()

    #
    # Normalized stiffness gain.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for foundation_k in foundation_values:
        selected = [
            row
            for row in rows
            if np.isclose(
                row["foundation_k"],
                foundation_k,
            )
        ]

        alpha = np.asarray(
            [
                row[
                    "prestress_scale"
                ]
                for row in selected
            ]
        )

        gain = np.asarray(
            [
                row[
                    "normalized_stiffness_gain"
                ]
                for row in selected
            ]
        )

        plt.plot(
            alpha,
            gain,
            label=(
                f"k_f={foundation_k:g}"
            ),
        )

    plt.xlabel(
        "T4 prestress scale"
    )

    plt.ylabel(
        "k_eff(alpha) / k_eff(0)"
    )

    plt.title(
        "Normalized prestress stiffening"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    gain_plot = (
        OUTPUT_DIR
        / "normalized_stiffness_gain.png"
    )

    plt.savefig(
        gain_plot,
        dpi=200,
    )

    plt.close()

    #
    # Lowest Hessian eigenvalue.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for foundation_k in foundation_values:
        selected = [
            row
            for row in rows
            if np.isclose(
                row["foundation_k"],
                foundation_k,
            )
        ]

        alpha = np.asarray(
            [
                row[
                    "prestress_scale"
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

        plt.plot(
            alpha,
            lam,
            label=(
                f"k_f={foundation_k:g}"
            ),
        )

    plt.xlabel(
        "T4 prestress scale"
    )

    plt.ylabel(
        "Smallest Hessian eigenvalue"
    )

    plt.title(
        "Lowest stiffness mode vs T4 prestress"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    lambda_plot = (
        OUTPUT_DIR
        / "lambda_min_vs_prestress.png"
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
        f"Saved: {gain_plot}"
    )

    print(
        f"Saved: {lambda_plot}"
    )

    print("=" * 108)


if __name__ == "__main__":
    main()
