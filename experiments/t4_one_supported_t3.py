"""Constrained rigidity analysis of a T4 with one supported T3 attachment."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import (
    mirrored_t4_prism,
    t4_with_one_supported_t3,
)
from tensegrity.rigidity import (
    analyze_rigidity,
    rigidity_matrix,
)


def constrained_rank(
    nodes,
    members,
    fixed_nodes,
):
    R = rigidity_matrix(
        nodes,
        members,
    )

    fixed_nodes = set(
        fixed_nodes
    )

    free_columns = []

    for node in range(
        nodes.shape[0]
    ):
        if node in fixed_nodes:
            continue

        free_columns.extend(
            [
                3 * node,
                3 * node + 1,
                3 * node + 2,
            ]
        )

    R_free = R[
        :,
        free_columns,
    ]

    singular_values = np.linalg.svd(
        R_free,
        compute_uv=False,
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    free_dof = len(
        free_columns
    )

    nullity = (
        free_dof - rank
    )

    return (
        R_free,
        singular_values,
        rank,
        free_dof,
        nullity,
    )


def main():
    #
    # Baseline: isolated T4.
    #
    t4_nodes, t4_members = (
        mirrored_t4_prism()
    )

    t4_analysis = (
        analyze_rigidity(
            t4_nodes,
            t4_members,
        )
    )

    #
    # Composite.
    #
    nodes, members = (
        t4_with_one_supported_t3()
    )

    #
    # T3 support-side triangle is fixed.
    #
    fixed_nodes = [
        9,
        10,
        11,
    ]

    (
        R_free,
        singular_values,
        rank,
        free_dof,
        nullity,
    ) = constrained_rank(
        nodes,
        members,
        fixed_nodes,
    )

    print()
    print("=" * 82)
    print("T4 + ONE SUPPORTED T3 ATTACHMENT")
    print("=" * 82)

    print("Isolated T4 baseline")
    print(
        f"  nodes:                  "
        f"{t4_nodes.shape[0]}"
    )

    print(
        f"  members:                "
        f"{len(t4_members)}"
    )

    print(
        f"  internal mechanisms:    "
        f"{t4_analysis.mechanisms}"
    )

    print(
        f"  self-stress dimension:  "
        f"{t4_analysis.self_stress_dimension}"
    )

    print()

    print("Composite model")

    print(
        f"  nodes:                  "
        f"{nodes.shape[0]}"
    )

    print(
        f"  members:                "
        f"{len(members)}"
    )

    print(
        f"  fixed support nodes:     "
        f"{fixed_nodes}"
    )

    print(
        f"  constrained R shape:    "
        f"{R_free.shape}"
    )

    print(
        f"  free Cartesian DOF:     "
        f"{free_dof}"
    )

    print(
        f"  constrained rank:       "
        f"{rank}"
    )

    print(
        f"  constrained nullity:    "
        f"{nullity}"
    )

    print()

    print(
        "constrained singular values:"
    )

    print(
        np.array2string(
            singular_values,
            precision=12,
            suppress_small=False,
        )
    )

    print()

    if nullity == 0:
        conclusion = (
            "one supported T3 attachment removes all "
            "remaining infinitesimal freedom in this composite"
        )
    else:
        conclusion = (
            f"{nullity} constrained infinitesimal "
            "mechanism(s) remain"
        )

    print(
        f"conclusion: {conclusion}"
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
