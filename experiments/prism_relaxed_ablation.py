"""Relaxed single-cable ablation experiment for the canonical prism.

Unlike prism_member_ablation.py, which reports instantaneous curvature at the
intact geometry, this experiment removes one cable, relaxes the damaged
framework, and only then evaluates rigidity and the non-rigid Hessian.

A stability label is emitted only when the relaxation converges to the stated
equilibrium-residual tolerance.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.energy import (
    stability_index_energy_hessian,
)
from tensegrity.examples import (
    three_strut_prism,
)
from tensegrity.relaxation import (
    relax_framework,
)
from tensegrity.rigidity import (
    analyze_rigidity,
)


def main() -> None:
    nodes, members, springs = three_strut_prism(
        prestress_scale=0.20
    )

    rows = []

    for idx, member in enumerate(members):
        if member.kind != "cable":
            continue

        reduced_members = (
            members[:idx]
            + members[idx + 1 :]
        )

        reduced_springs = (
            springs[:idx]
            + springs[idx + 1 :]
        )

        relaxation = relax_framework(
            nodes,
            reduced_springs,
            residual_tol=1e-9,
            max_iterations=5000,
            initial_step_size=0.2,
        )

        relaxed_nodes = relaxation.nodes

        rigidity = analyze_rigidity(
            relaxed_nodes,
            reduced_members,
        )

        if relaxation.converged:
            stability = stability_index_energy_hessian(
                relaxed_nodes,
                reduced_springs,
                eps=2e-5,
                eig_tol=1e-6,
            )

            lambda_min = stability.lambda_min
            classification = stability.classification

        else:
            lambda_min = float("nan")
            classification = "not_classified"

        rows.append(
            {
                "removed_index": idx,
                "removed_i": member.i,
                "removed_j": member.j,
                "converged": relaxation.converged,
                "iterations": relaxation.iterations,
                "accepted_steps": (
                    relaxation.accepted_steps
                ),
                "energy_initial": (
                    relaxation.energy_initial
                ),
                "energy_final": (
                    relaxation.energy_final
                ),
                "equilibrium_residual_initial": (
                    relaxation.residual_initial
                ),
                "equilibrium_residual_final": (
                    relaxation.residual_final
                ),
                "max_node_displacement": (
                    relaxation.max_displacement
                ),
                "rigidity_rank": rigidity.rank,
                "mechanisms_after": (
                    rigidity.mechanisms
                ),
                "self_stress_after": (
                    rigidity.self_stress_dimension
                ),
                "lambda_min_relaxed": lambda_min,
                "relaxed_classification": (
                    classification
                ),
                "relaxation_message": (
                    relaxation.message
                ),
            }
        )

        if np.isfinite(lambda_min):
            lambda_text = (
                f"{lambda_min:.6e}"
            )
        else:
            lambda_text = "not classified"

        print(
            f"remove cable {idx:2d} "
            f"({member.i},{member.j}): "
            f"converged={relaxation.converged}, "
            f"residual "
            f"{relaxation.residual_initial:.3e}"
            f"->{relaxation.residual_final:.3e}, "
            f"max displacement="
            f"{relaxation.max_displacement:.6e}, "
            f"mechanisms={rigidity.mechanisms}, "
            f"self-stress="
            f"{rigidity.self_stress_dimension}, "
            f"lambda_min={lambda_text}"
        )

    output = Path(
        "outputs/prism_relaxed_ablation.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()
