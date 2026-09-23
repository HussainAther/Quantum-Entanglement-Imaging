"""Diagnose relaxed cable-ablation states of the canonical 3-strut prism.

The goal is to distinguish a genuine finite equilibrium from an optimizer
approaching a geometrically singular configuration with a very small force
residual.
"""

from __future__ import annotations

import numpy as np

from tensegrity.energy import (
    energy_gradient,
    total_energy,
)
from tensegrity.examples import three_strut_prism
from tensegrity.relaxation import relax_framework
from tensegrity.rigidity import (
    rigidity_matrix,
    analyze_rigidity,
)


def spring_force_density(nodes, spring):
    d = nodes[spring.i] - nodes[spring.j]
    length = float(np.linalg.norm(d))

    return (
        spring.k
        * (length - spring.L0)
        / length
    )


def main() -> None:
    nodes, members, springs = three_strut_prism(
        prestress_scale=0.20
    )

    for idx, member in enumerate(members):
        if member.kind != "cable":
            continue

        reduced_members = (
            members[:idx]
            + members[idx + 1:]
        )

        reduced_springs = (
            springs[:idx]
            + springs[idx + 1:]
        )

        result = relax_framework(
            nodes,
            reduced_springs,
            residual_tol=1e-9,
            max_iterations=5000,
            initial_step_size=0.2,
        )

        X = result.nodes

        R = rigidity_matrix(
            X,
            reduced_members,
        )

        singular_values = np.linalg.svd(
            R,
            compute_uv=False,
        )

        q = np.array(
            [
                spring_force_density(X, spring)
                for spring in reduced_springs
            ],
            dtype=float,
        )

        lengths = np.array(
            [
                np.linalg.norm(
                    X[spring.i] - X[spring.j]
                )
                for spring in reduced_springs
            ]
        )

        grad = energy_gradient(
            X,
            reduced_springs,
        )

        rigidity = analyze_rigidity(
            X,
            reduced_members,
        )

        smallest_sv = singular_values[-1]

        condition_number = (
            singular_values[0]
            / smallest_sv
            if smallest_sv > 0.0
            else np.inf
        )

        print()
        print("=" * 72)
        print(
            f"removed cable {idx} "
            f"({member.i},{member.j})"
        )
        print("=" * 72)

        print(
            f"converged:               "
            f"{result.converged}"
        )

        print(
            f"iterations:               "
            f"{result.iterations}"
        )

        print(
            f"energy:                   "
            f"{total_energy(X, reduced_springs):.12e}"
        )

        print(
            f"gradient norm:            "
            f"{np.linalg.norm(grad):.12e}"
        )

        print(
            f"force-density norm:       "
            f"{np.linalg.norm(q):.12e}"
        )

        print(
            f"smallest rigidity SV:     "
            f"{smallest_sv:.12e}"
        )

        print(
            f"largest rigidity SV:      "
            f"{singular_values[0]:.12e}"
        )

        print(
            f"condition number:         "
            f"{condition_number:.12e}"
        )

        print(
            f"rigidity rank:            "
            f"{rigidity.rank}"
        )

        print(
            f"mechanisms:               "
            f"{rigidity.mechanisms}"
        )

        print(
            f"self-stress dimension:    "
            f"{rigidity.self_stress_dimension}"
        )

        print(
            f"max aligned displacement: "
            f"{result.max_displacement:.12e}"
        )

        print(
            f"minimum member length:    "
            f"{lengths.min():.12e}"
        )

        print(
            f"maximum member length:    "
            f"{lengths.max():.12e}"
        )

        print(
            "rigidity singular values:"
        )

        print(
            np.array2string(
                singular_values,
                precision=6,
                suppress_small=False,
            )
        )

        print(
            "force densities:"
        )

        print(
            np.array2string(
                q,
                precision=6,
                suppress_small=False,
            )
        )


if __name__ == "__main__":
    main()
