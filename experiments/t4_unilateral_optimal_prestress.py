"""Unilateral optimal-prestress search for the T4 + two-T3 composite.

Cables carry tension only.
Bars carry compression only.

The central T4 receives the previously validated sign-compatible prestress.
The two zero-twist T3 attachments begin stress-free.

For each:
    foundation stiffness k_f
    T4 prestress alpha
    small applied load

we minimize the actual unilateral potential energy

    Pi(X) =
        E_unilateral(X)
        + E_foundation(X)
        - f . (X - X_ref)

and infer the effective vertical stiffness from the resulting displacement.

Several small load magnitudes are used to verify that the measured response
is in the small-load regime.

This avoids using a conventional centered Hessian at members that lie exactly
on unilateral activation boundaries.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tensegrity.energy import (
    SpringMember,
    unilateral_active_state,
    unilateral_energy_gradient,
    unilateral_total_energy,
)
from tensegrity.examples import (
    equilibrated_mirrored_t4,
    t4_with_two_supported_t3,
)


OUTPUT_DIR = Path(
    "outputs/t4_unilateral_optimal_prestress"
)

T4_MEMBER_COUNT = 21


def normalized_t4_force_density(
    reference_nodes,
):
    """Recover the unit-scale sign-compatible T4 force-density vector.

    equilibrated_mirrored_t4() performs the self-stress search. We only do
    that once here, then rescale the resulting direction for every alpha.
    """

    calibration_scale = 0.50

    (
        t4_nodes,
        t4_members,
        t4_springs,
    ) = equilibrated_mirrored_t4(
        radius=1.0,
        half_height=0.75,
        twist=np.pi / 6.0,
        prestress_scale=calibration_scale,
        stiffness=1.0,
    )

    if len(t4_members) != T4_MEMBER_COUNT:
        raise RuntimeError(
            "Unexpected T4 member count."
        )

    q_hat = []

    for member, spring in zip(
        t4_members,
        t4_springs,
    ):
        L = float(
            np.linalg.norm(
                t4_nodes[member.i]
                - t4_nodes[member.j]
            )
        )

        q = (
            spring.k
            * (
                L - spring.L0
            )
            / L
        )

        q_hat.append(
            q / calibration_scale
        )

    return np.asarray(
        q_hat,
        dtype=float,
    )


def build_unilateral_springs(
    reference_nodes,
    members,
    q_hat,
    prestress_scale,
    stiffness=1.0,
):
    """Construct layered unilateral members.

    First 21 members:
        prestressed T4 using signed self-stress q.

    Remaining members:
        stress-free T3 attachments.
    """

    springs = []

    for idx, member in enumerate(
        members
    ):
        L = float(
            np.linalg.norm(
                reference_nodes[member.i]
                - reference_nodes[member.j]
            )
        )

        if idx < T4_MEMBER_COUNT:
            q = (
                prestress_scale
                * q_hat[idx]
            )

            L0 = (
                L
                * (
                    1.0
                    - q / stiffness
                )
            )
        else:
            L0 = L

        if L0 <= 0.0:
            raise ValueError(
                "Non-positive rest length encountered."
            )

        springs.append(
            SpringMember(
                i=member.i,
                j=member.j,
                kind=member.kind,
                k=stiffness,
                L0=L0,
            )
        )

    return springs


def foundation_energy(
    nodes,
    reference_nodes,
    support_nodes,
    foundation_k,
):
    displacement = (
        nodes[support_nodes]
        - reference_nodes[support_nodes]
    )

    return (
        0.5
        * foundation_k
        * float(
            np.sum(
                displacement ** 2
            )
        )
    )


def foundation_gradient(
    nodes,
    reference_nodes,
    support_nodes,
    foundation_k,
):
    grad = np.zeros_like(
        nodes
    )

    grad[support_nodes] = (
        foundation_k
        * (
            nodes[support_nodes]
            - reference_nodes[support_nodes]
        )
    )

    return grad


def load_array(
    node_count,
    loaded_nodes,
    force_per_node,
):
    force = np.zeros(
        (
            node_count,
            3,
        ),
        dtype=float,
    )

    for node in loaded_nodes:
        force[
            node,
            2,
        ] = force_per_node

    return force


def solve_loaded_state(
    initial_nodes,
    reference_nodes,
    springs,
    support_nodes,
    loaded_nodes,
    foundation_k,
    force_per_node,
):
    shape = reference_nodes.shape

    external_force = load_array(
        reference_nodes.shape[0],
        loaded_nodes,
        force_per_node,
    )

    reference_flat = (
        reference_nodes.reshape(-1)
    )

    def objective(flat):
        X = flat.reshape(
            shape
        )

        internal_energy = (
            unilateral_total_energy(
                X,
                springs,
            )
        )

        support_energy = (
            foundation_energy(
                X,
                reference_nodes,
                support_nodes,
                foundation_k,
            )
        )

        displacement = (
            X - reference_nodes
        )

        load_potential = (
            -float(
                np.sum(
                    external_force
                    * displacement
                )
            )
        )

        energy = (
            internal_energy
            + support_energy
            + load_potential
        )

        grad = (
            unilateral_energy_gradient(
                X,
                springs,
            )
            + foundation_gradient(
                X,
                reference_nodes,
                support_nodes,
                foundation_k,
            )
            - external_force
        )

        return (
            float(energy),
            grad.reshape(-1),
        )

    result = minimize(
        objective,
        initial_nodes.reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 20000,
            "ftol": 1e-15,
            "gtol": 1e-11,
            "maxls": 100,
        },
    )

    X = result.x.reshape(
        shape
    )

    _, final_gradient = (
        objective(
            result.x
        )
    )

    residual = float(
        np.linalg.norm(
            final_gradient
        )
    )

    dz = np.asarray(
        [
            X[node, 2]
            - reference_nodes[node, 2]
            for node in loaded_nodes
        ],
        dtype=float,
    )

    mean_dz = float(
        np.mean(
            dz
        )
    )

    total_force = (
        float(
            len(
                loaded_nodes
            )
        )
        * force_per_node
    )

    effective_stiffness = (
        total_force
        / mean_dz
        if mean_dz > 1e-15
        else float("nan")
    )

    active = unilateral_active_state(
        X,
        springs,
        tol=1e-9,
    )

    max_displacement = float(
        np.max(
            np.linalg.norm(
                X - reference_nodes,
                axis=1,
            )
        )
    )

    return {
        "nodes": X,
        "success": bool(
            result.success
        ),
        "iterations": int(
            result.nit
        ),
        "residual": residual,
        "mean_dz": mean_dz,
        "effective_stiffness": (
            effective_stiffness
        ),
        "max_displacement": (
            max_displacement
        ),
        "active": active,
    }


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
    # Dense enough to locate whether the
    # bilateral ~0.30 optimum survives.
    #
    prestress_values = np.linspace(
        0.0,
        0.60,
        61,
    )

    #
    # Small-load consistency test.
    #
    force_values = np.asarray(
        [
            5e-5,
            1e-4,
            2e-4,
        ],
        dtype=float,
    )

    q_hat = (
        normalized_t4_force_density(
            reference_nodes
        )
    )

    print()
    print("=" * 116)
    print(
        "UNILATERAL T4 OPTIMAL PRESTRESS SEARCH"
    )
    print("=" * 116)

    detailed_rows = []
    summary_rows = []

    for foundation_k in foundation_values:
        alpha_stiffness = []
        alpha_linearity = []

        print()
        print(
            f"foundation k = "
            f"{foundation_k:.3e}"
        )

        for alpha in prestress_values:
            springs = (
                build_unilateral_springs(
                    reference_nodes,
                    members,
                    q_hat,
                    float(alpha),
                    stiffness=1.0,
                )
            )

            stiffness_by_force = []

            #
            # Begin each alpha from the
            # exact unloaded reference state.
            #
            initial_nodes = (
                reference_nodes.copy()
            )

            for force_per_node in force_values:
                result = (
                    solve_loaded_state(
                        initial_nodes,
                        reference_nodes,
                        springs,
                        support_nodes,
                        loaded_nodes,
                        float(
                            foundation_k
                        ),
                        float(
                            force_per_node
                        ),
                    )
                )

                stiffness_by_force.append(
                    result[
                        "effective_stiffness"
                    ]
                )

                detailed_rows.append(
                    {
                        "foundation_k": (
                            foundation_k
                        ),
                        "prestress_scale": (
                            alpha
                        ),
                        "force_per_node": (
                            force_per_node
                        ),
                        "optimizer_success": (
                            result[
                                "success"
                            ]
                        ),
                        "optimizer_iterations": (
                            result[
                                "iterations"
                            ]
                        ),
                        "equilibrium_residual": (
                            result[
                                "residual"
                            ]
                        ),
                        "mean_dz": (
                            result[
                                "mean_dz"
                            ]
                        ),
                        "effective_stiffness": (
                            result[
                                "effective_stiffness"
                            ]
                        ),
                        "max_displacement": (
                            result[
                                "max_displacement"
                            ]
                        ),
                        "active_cables": (
                            result[
                                "active"
                            ][
                                "active_cables"
                            ]
                        ),
                        "slack_cables": (
                            result[
                                "active"
                            ][
                                "slack_cables"
                            ]
                        ),
                        "active_bars": (
                            result[
                                "active"
                            ][
                                "active_bars"
                            ]
                        ),
                        "slack_bars": (
                            result[
                                "active"
                            ][
                                "slack_bars"
                            ]
                        ),
                        "threshold_members": (
                            result[
                                "active"
                            ][
                                "threshold"
                            ]
                        ),
                    }
                )

            stiffness_by_force = (
                np.asarray(
                    stiffness_by_force,
                    dtype=float,
                )
            )

            #
            # Use the middle load as our
            # representative small-load stiffness.
            #
            representative = float(
                stiffness_by_force[1]
            )

            finite = np.isfinite(
                stiffness_by_force
            )

            if np.all(
                finite
            ):
                spread = float(
                    (
                        np.max(
                            stiffness_by_force
                        )
                        - np.min(
                            stiffness_by_force
                        )
                    )
                    / np.mean(
                        stiffness_by_force
                    )
                )
            else:
                spread = float(
                    "nan"
                )

            alpha_stiffness.append(
                representative
            )

            alpha_linearity.append(
                spread
            )

        alpha_stiffness = np.asarray(
            alpha_stiffness,
            dtype=float,
        )

        alpha_linearity = np.asarray(
            alpha_linearity,
            dtype=float,
        )

        valid = np.isfinite(
            alpha_stiffness
        )

        if not np.any(
            valid
        ):
            print(
                "  no valid stiffness measurements"
            )

            continue

        valid_indices = np.where(
            valid
        )[0]

        local_index = int(
            np.argmax(
                alpha_stiffness[
                    valid
                ]
            )
        )

        best_index = int(
            valid_indices[
                local_index
            ]
        )

        optimal_alpha = float(
            prestress_values[
                best_index
            ]
        )

        maximum_stiffness = float(
            alpha_stiffness[
                best_index
            ]
        )

        baseline = float(
            alpha_stiffness[0]
        )

        normalized_gain = (
            maximum_stiffness
            / baseline
            if np.isfinite(
                baseline
            )
            and baseline != 0.0
            else float(
                "nan"
            )
        )

        linearity_spread = float(
            alpha_linearity[
                best_index
            ]
        )

        summary_rows.append(
            {
                "foundation_k": (
                    foundation_k
                ),
                "baseline_stiffness": (
                    baseline
                ),
                "optimal_prestress": (
                    optimal_alpha
                ),
                "maximum_effective_stiffness": (
                    maximum_stiffness
                ),
                "normalized_maximum_gain": (
                    normalized_gain
                ),
                "load_linearity_spread_at_optimum": (
                    linearity_spread
                ),
                "peak_at_scan_boundary": (
                    best_index == 0
                    or best_index
                    == len(
                        prestress_values
                    ) - 1
                ),
            }
        )

        print(
            f"  baseline stiffness:              "
            f"{baseline:.12e}"
        )

        print(
            f"  optimal prestress:               "
            f"{optimal_alpha:.6f}"
        )

        print(
            f"  maximum effective stiffness:     "
            f"{maximum_stiffness:.12e}"
        )

        print(
            f"  normalized maximum gain:         "
            f"{normalized_gain:.12f}"
        )

        print(
            f"  load-linearity spread at optimum:"
            f" {linearity_spread:.12e}"
        )

        print(
            f"  peak at scan boundary:           "
            f"{summary_rows[-1]['peak_at_scan_boundary']}"
        )

    #
    # Detailed CSV.
    #
    detailed_path = (
        OUTPUT_DIR
        / "unilateral_load_sweep.csv"
    )

    with detailed_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "prestress_scale",
                "force_per_node",
                "optimizer_success",
                "optimizer_iterations",
                "equilibrium_residual",
                "mean_dz",
                "effective_stiffness",
                "max_displacement",
                "active_cables",
                "slack_cables",
                "active_bars",
                "slack_bars",
                "threshold_members",
            ],
        )

        writer.writeheader()
        writer.writerows(
            detailed_rows
        )

    #
    # Optimum summary.
    #
    summary_path = (
        OUTPUT_DIR
        / "unilateral_optimal_prestress_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "baseline_stiffness",
                "optimal_prestress",
                "maximum_effective_stiffness",
                "normalized_maximum_gain",
                "load_linearity_spread_at_optimum",
                "peak_at_scan_boundary",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print(
        f"Saved: {detailed_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    #
    # Plot representative 1e-4 response.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for foundation_k in foundation_values:
        values = []

        for alpha in prestress_values:
            matches = [
                row
                for row in detailed_rows
                if np.isclose(
                    row[
                        "foundation_k"
                    ],
                    foundation_k,
                )
                and np.isclose(
                    row[
                        "prestress_scale"
                    ],
                    alpha,
                )
                and np.isclose(
                    row[
                        "force_per_node"
                    ],
                    1e-4,
                )
            ]

            if matches:
                values.append(
                    matches[0][
                        "effective_stiffness"
                    ]
                )
            else:
                values.append(
                    float("nan")
                )

        plt.plot(
            prestress_values,
            values,
            label=(
                f"k_f={foundation_k:g}"
            ),
        )

    plt.xlabel(
        "T4 prestress scale"
    )

    plt.ylabel(
        "Unilateral effective vertical stiffness"
    )

    plt.title(
        "Unilateral prestress-stiffness response"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "unilateral_stiffness_vs_prestress.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {plot_path}"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
