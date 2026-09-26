"""Prestressed T4 embedded in stress-free T3 foundation attachments.

Model structure
---------------
Members 0:21
    Central mirrored T4.
    Rest lengths are derived from the validated sign-compatible T4
    self-stress.

Members 21:39
    Two zero-twist T3 attachment units.
    These are stress-free at the reference geometry.

Support nodes 9:15
    Attached to isotropic foundation springs.

Because the T4 self-stress satisfies R_T4.T @ q = 0, while the T3
attachments and foundation springs are initially unstressed, the full
reference configuration is an exact equilibrium for every prestress scale.

For each T4 prestress scale and foundation stiffness, this experiment computes:

1. equilibrium residual,
2. smallest tangent-Hessian eigenvalue,
3. mean vertical displacement of the central T4 middle triangle
   under equal +z nodal loads,
4. effective vertical stiffness,
5. foundation-stiffness crossover points relative to the high-k asymptote.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensegrity.energy import numerical_hessian
from tensegrity.examples import (
    equilibrated_mirrored_t4,
    t4_with_two_supported_t3,
)


OUTPUT_DIR = Path(
    "outputs/t4_prestress_with_t3_foundation"
)


T4_MEMBER_COUNT = 21


def build_rest_lengths(
    reference_nodes,
    members,
    prestress_scale,
    stiffness=1.0,
):
    """Return spring rest lengths for the layered composite.

    The first 21 members inherit rest lengths from the equilibrated T4
    model at the requested prestress scale.

    The remaining T3 attachment members use their geometric reference
    lengths and are therefore initially stress-free.
    """

    (
        t4_nodes,
        t4_members,
        t4_springs,
    ) = equilibrated_mirrored_t4(
        radius=1.0,
        half_height=0.75,
        twist=np.pi / 6.0,
        prestress_scale=prestress_scale,
        stiffness=stiffness,
    )

    if len(t4_members) != T4_MEMBER_COUNT:
        raise RuntimeError(
            "Unexpected T4 member count."
        )

    if len(members) < T4_MEMBER_COUNT:
        raise RuntimeError(
            "Composite has fewer members than the T4."
        )

    rest_lengths = []

    #
    # Prestressed central T4.
    #
    for spring in t4_springs:
        rest_lengths.append(
            float(
                spring.L0
            )
        )

    #
    # Stress-free T3 attachment members.
    #
    for member in members[
        T4_MEMBER_COUNT:
    ]:
        L_ref = float(
            np.linalg.norm(
                reference_nodes[member.i]
                - reference_nodes[member.j]
            )
        )

        rest_lengths.append(
            L_ref
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
                "Encountered zero-length member."
            )

        extension = (
            L - L0
        )

        energy += (
            0.5
            * stiffness
            * extension ** 2
        )

        factor = (
            stiffness
            * extension
            / L
        )

        contribution = (
            factor * d
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
        float(energy),
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

    return (
        member_energy
        + foundation_energy,
        (
            member_gradient
            + foundation_gradient
        ).reshape(-1),
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
    rest_lengths,
    support_nodes,
    loaded_nodes,
    member_stiffness,
    foundation_k,
):
    shape = (
        reference_nodes.shape
    )

    reference_flat = (
        reference_nodes.reshape(-1)
    )

    #
    # Verify equilibrium directly.
    #
    (
        reference_energy,
        reference_gradient,
    ) = total_energy_and_gradient(
        reference_flat,
        shape,
        reference_nodes,
        members,
        rest_lengths,
        support_nodes,
        member_stiffness,
        foundation_k,
    )

    equilibrium_residual = float(
        np.linalg.norm(
            reference_gradient
        )
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
        reference_flat,
        eps=2e-5,
    )

    H = (
        0.5
        * (
            H + H.T
        )
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

    if lambda_min <= 1e-10:
        return (
            reference_energy,
            equilibrium_residual,
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
        np.mean(
            dz
        )
    )

    total_force = float(
        len(loaded_nodes)
    )

    effective_stiffness = (
        total_force
        / mean_dz
        if abs(mean_dz) > 1e-15
        else np.inf
    )

    return (
        reference_energy,
        equilibrium_residual,
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

    #
    # Central T4 middle triangle.
    #
    loaded_nodes = [
        3,
        4,
        5,
    ]

    member_stiffness = 1.0

    prestress_scales = np.asarray(
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
    print("=" * 118)
    print(
        "PRESTRESSED T4 + STRESS-FREE T3 FOUNDATION SWEEP"
    )
    print("=" * 118)

    for prestress_scale in prestress_scales:
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

        stiffness_values = []
        lambda_values = []
        residual_values = []

        for foundation_k in foundation_values:
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
                    "prestress_scale": (
                        prestress_scale
                    ),
                    "foundation_k": (
                        foundation_k
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

        print()
        print(
            f"T4 prestress scale = "
            f"{prestress_scale:.2f}"
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
            f"{np.nanmax(residual_values):.12e}"
        )

    #
    # Save CSV.
    #
    csv_path = (
        OUTPUT_DIR
        / "t4_prestress_foundation_sweep.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prestress_scale",
                "foundation_k",
                "reference_energy",
                "equilibrium_residual",
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
    # Plot effective stiffness.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for prestress_scale in prestress_scales:
        selected = [
            row
            for row in rows
            if np.isclose(
                row[
                    "prestress_scale"
                ],
                prestress_scale,
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
                f"T4 prestress="
                f"{prestress_scale:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Effective vertical stiffness"
    )

    plt.title(
        "T4 prestress vs foundation stiffness"
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
    # Plot smallest Hessian eigenvalue.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for prestress_scale in prestress_scales:
        selected = [
            row
            for row in rows
            if np.isclose(
                row[
                    "prestress_scale"
                ],
                prestress_scale,
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
                f"T4 prestress="
                f"{prestress_scale:.2f}"
            ),
        )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Smallest Hessian eigenvalue"
    )

    plt.title(
        "Stability vs T4 prestress and foundation stiffness"
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

    print("=" * 118)


if __name__ == "__main__":
    main()
