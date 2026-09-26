"""Relaxed prestrain / foundation-stiffness sweep for the T4 + two-T3 composite.

This corrects the earlier prestress sweep by enforcing equilibrium before
computing tangent stiffness.

For each pair:

    (prestrain, foundation stiffness)

the script:

1. assigns shortened member rest lengths,
2. includes isotropic foundation springs on the six support nodes,
3. minimizes total energy,
4. checks the equilibrium residual,
5. computes the Hessian at the relaxed equilibrium,
6. applies a small vertical load to the central T4 triangle,
7. reports effective vertical stiffness.

Important:
The current constitutive model remains bilateral and the same prestrain is
applied to all members. This is therefore a generic preloaded spring-network
benchmark, not yet a fully physical tension-only / compression-only tensegrity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tensegrity.energy import numerical_hessian
from tensegrity.examples import t4_with_two_supported_t3


OUTPUT_DIR = Path(
    "outputs/t4_two_t3_relaxed_prestress_foundation"
)


def member_rest_lengths(
    reference_nodes,
    members,
    prestrain,
):
    rest_lengths = []

    for member in members:
        i = member.i
        j = member.j

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

        rest_lengths.append(
            L0
        )

    return np.asarray(
        rest_lengths,
        dtype=float,
    )


def member_energy_and_gradient(
    nodes,
    members,
    rest_lengths,
    stiffness,
):
    energy = 0.0

    gradient = np.zeros_like(
        nodes
    )

    for member, L0 in zip(
        members,
        rest_lengths,
    ):
        i = member.i
        j = member.j

        d = (
            nodes[i]
            - nodes[j]
        )

        L = float(
            np.linalg.norm(d)
        )

        if L <= 1e-14:
            raise ValueError(
                "encountered zero-length member"
            )

        extension = (
            L - L0
        )

        energy += (
            0.5
            * stiffness
            * extension ** 2
        )

        force_density_factor = (
            stiffness
            * extension
            / L
        )

        contribution = (
            force_density_factor
            * d
        )

        gradient[i] += (
            contribution
        )

        gradient[j] -= (
            contribution
        )

    return (
        float(energy),
        gradient,
    )


def foundation_energy_and_gradient(
    nodes,
    reference_nodes,
    support_nodes,
    foundation_k,
):
    displacement = (
        nodes[support_nodes]
        - reference_nodes[support_nodes]
    )

    energy = (
        0.5
        * foundation_k
        * float(
            np.sum(
                displacement ** 2
            )
        )
    )

    gradient = np.zeros_like(
        nodes
    )

    gradient[support_nodes] = (
        foundation_k
        * displacement
    )

    return (
        energy,
        gradient,
    )


def total_energy_and_gradient(
    flat,
    shape,
    reference_nodes,
    members,
    rest_lengths,
    support_nodes,
    member_stiffness,
    foundation_k,
):
    nodes = flat.reshape(
        shape
    )

    (
        member_energy,
        member_gradient,
    ) = member_energy_and_gradient(
        nodes,
        members,
        rest_lengths,
        member_stiffness,
    )

    (
        foundation_energy,
        foundation_gradient,
    ) = foundation_energy_and_gradient(
        nodes,
        reference_nodes,
        support_nodes,
        foundation_k,
    )

    energy = (
        member_energy
        + foundation_energy
    )

    gradient = (
        member_gradient
        + foundation_gradient
    )

    return (
        float(energy),
        gradient.reshape(-1),
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


def relax_case(
    initial_nodes,
    reference_nodes,
    members,
    rest_lengths,
    support_nodes,
    member_stiffness,
    foundation_k,
):
    shape = (
        initial_nodes.shape
    )

    def objective(flat):
        energy, gradient = (
            total_energy_and_gradient(
                flat,
                shape,
                reference_nodes,
                members,
                rest_lengths,
                support_nodes,
                member_stiffness,
                foundation_k,
            )
        )

        return (
            energy,
            gradient,
        )

    result = minimize(
        objective,
        initial_nodes.reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 20000,
            "ftol": 1e-15,
            "gtol": 1e-12,
            "maxls": 100,
        },
    )

    relaxed_nodes = (
        result.x.reshape(
            shape
        )
    )

    final_energy, final_gradient = (
        total_energy_and_gradient(
            result.x,
            shape,
            reference_nodes,
            members,
            rest_lengths,
            support_nodes,
            member_stiffness,
            foundation_k,
        )
    )

    residual = float(
        np.linalg.norm(
            final_gradient
        )
    )

    max_displacement = float(
        np.max(
            np.linalg.norm(
                relaxed_nodes
                - reference_nodes,
                axis=1,
            )
        )
    )

    return (
        result,
        relaxed_nodes,
        final_energy,
        residual,
        max_displacement,
    )


def tangent_response(
    relaxed_nodes,
    reference_nodes,
    members,
    rest_lengths,
    support_nodes,
    member_stiffness,
    foundation_k,
    loaded_nodes,
):
    shape = (
        relaxed_nodes.shape
    )

    def energy_flat(flat):
        energy, _ = (
            total_energy_and_gradient(
                flat,
                shape,
                reference_nodes,
                members,
                rest_lengths,
                support_nodes,
                member_stiffness,
                foundation_k,
            )
        )

        return energy

    H = numerical_hessian(
        energy_flat,
        relaxed_nodes.reshape(-1),
        eps=2e-5,
    )

    H = (
        0.5
        * (
            H + H.T
        )
    )

    eigenvalues = (
        np.linalg.eigvalsh(
            H
        )
    )

    lambda_min = float(
        eigenvalues[0]
    )

    force = load_vector(
        relaxed_nodes.shape[0],
        loaded_nodes,
        force_per_node=1.0,
    )

    if lambda_min <= 1e-10:
        return (
            lambda_min,
            float("nan"),
            float("nan"),
        )

    displacement = (
        np.linalg.solve(
            H,
            force,
        )
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
        np.mean(
            dz
        )
    )

    total_force = float(
        len(
            loaded_nodes
        )
    )

    effective_stiffness = (
        total_force
        / mean_dz
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

    k = (
        k_values[
            valid
        ]
    )

    y = (
        stiffness_values[
            valid
        ]
    )

    if len(k) < 2:
        return float(
            "nan"
        )

    for i in range(
        1,
        len(k),
    ):
        y0 = y[
            i - 1
        ]

        y1 = y[
            i
        ]

        if y0 <= target <= y1:
            x0 = np.log10(
                k[
                    i - 1
                ]
            )

            x1 = np.log10(
                k[i]
            )

            if y1 == y0:
                return float(
                    k[i]
                )

            fraction = (
                (target - y0)
                / (y1 - y0)
            )

            return float(
                10.0 ** (
                    x0
                    + fraction
                    * (
                        x1 - x0
                    )
                )
            )

    return float(
        "nan"
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
        41,
    )

    rows = []

    print()
    print("=" * 118)
    print(
        "T4 + TWO T3: RELAXED PRESTRAIN / FOUNDATION SWEEP"
    )
    print("=" * 118)

    for prestrain in prestrain_values:
        rest_lengths = (
            member_rest_lengths(
                reference_nodes,
                members,
                float(prestrain),
            )
        )

        stiffness_values = []
        lambda_values = []

        #
        # Continuation:
        # use the relaxed geometry from the
        # previous foundation value as the
        # next initial guess.
        #
        initial_nodes = (
            reference_nodes.copy()
        )

        print()
        print(
            f"prestrain = "
            f"{prestrain:.2f}"
        )

        for foundation_k in foundation_values:
            (
                solve,
                relaxed_nodes,
                relaxed_energy,
                residual,
                max_displacement,
            ) = relax_case(
                initial_nodes,
                reference_nodes,
                members,
                rest_lengths,
                support_nodes,
                member_stiffness,
                float(
                    foundation_k
                ),
            )

            (
                lambda_min,
                mean_dz,
                effective_stiffness,
            ) = tangent_response(
                relaxed_nodes,
                reference_nodes,
                members,
                rest_lengths,
                support_nodes,
                member_stiffness,
                float(
                    foundation_k
                ),
                loaded_nodes,
            )

            stiffness_values.append(
                effective_stiffness
            )

            lambda_values.append(
                lambda_min
            )

            rows.append(
                {
                    "prestrain": (
                        prestrain
                    ),
                    "foundation_k": (
                        foundation_k
                    ),
                    "optimizer_success": (
                        solve.success
                    ),
                    "optimizer_iterations": (
                        solve.nit
                    ),
                    "equilibrium_residual": (
                        residual
                    ),
                    "relaxed_energy": (
                        relaxed_energy
                    ),
                    "max_node_displacement": (
                        max_displacement
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

            #
            # Continuation seed.
            #
            initial_nodes = (
                relaxed_nodes
            )

        stiffness_values = (
            np.asarray(
                stiffness_values,
                dtype=float,
            )
        )

        lambda_values = (
            np.asarray(
                lambda_values,
                dtype=float,
            )
        )

        valid = np.isfinite(
            stiffness_values
        )

        if np.any(
            valid
        ):
            asymptotic_stiffness = float(
                stiffness_values[
                    np.where(
                        valid
                    )[0][-1]
                ]
            )
        else:
            asymptotic_stiffness = float(
                "nan"
            )

        selected_rows = [
            row
            for row in rows
            if np.isclose(
                row["prestrain"],
                prestrain,
            )
        ]

        worst_residual = float(
            np.max(
                [
                    row[
                        "equilibrium_residual"
                    ]
                    for row
                    in selected_rows
                ]
            )
        )

        max_geometry_change = float(
            np.max(
                [
                    row[
                        "max_node_displacement"
                    ]
                    for row
                    in selected_rows
                ]
            )
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
            f"  minimum lambda_min:      "
            f"{np.nanmin(lambda_values):.12e}"
        )

        print(
            f"  worst equilibrium residual: "
            f"{worst_residual:.12e}"
        )

        print(
            f"  max geometry change:     "
            f"{max_geometry_change:.12e}"
        )

    csv_path = (
        OUTPUT_DIR
        / "relaxed_prestress_foundation_sweep.csv"
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
                "optimizer_success",
                "optimizer_iterations",
                "equilibrium_residual",
                "relaxed_energy",
                "max_node_displacement",
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
    # Effective stiffness plot.
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
                for row
                in selected
            ]
        )

        stiffness = np.asarray(
            [
                row[
                    "effective_stiffness"
                ]
                for row
                in selected
            ]
        )

        plt.semilogx(
            k,
            stiffness,
            label=(
                f"prestrain="
                f"{prestrain:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Relaxed-equilibrium effective vertical stiffness"
    )

    plt.title(
        "Relaxed prestrain / foundation response"
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
    # Equilibrium residual plot.
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
                for row
                in selected
            ]
        )

        residual = np.asarray(
            [
                row[
                    "equilibrium_residual"
                ]
                for row
                in selected
            ]
        )

        plt.loglog(
            k,
            residual,
            label=(
                f"prestrain="
                f"{prestrain:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Relaxed equilibrium residual"
    )

    plt.title(
        "Equilibrium quality after relaxation"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    residual_plot = (
        OUTPUT_DIR
        / "equilibrium_residual_vs_foundation.png"
    )

    plt.savefig(
        residual_plot,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {stiffness_plot}"
    )

    print(
        f"Saved: {residual_plot}"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
