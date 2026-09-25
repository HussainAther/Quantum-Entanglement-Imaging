"""Constrained rigidity analysis of a T4 supported by two T3 attachments."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import (
    t4_with_two_supported_t3,
)
from tensegrity.rigidity import (
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
    nodes, members = (
        t4_with_two_supported_t3()
    )

    #
    # Both support-side triangles fixed.
    #
    fixed_nodes = [
        9,
        10,
        11,
        12,
        13,
        14,
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
    print("T4 + TWO SUPPORTED T3 ATTACHMENTS")
    print("=" * 82)

    print(
        f"nodes:                   "
        f"{nodes.shape[0]}"
    )

    print(
        f"members:                 "
        f"{len(members)}"
    )

    print(
        f"fixed support nodes:      "
        f"{fixed_nodes}"
    )

    print(
        f"constrained R shape:     "
        f"{R_free.shape}"
    )

    print(
        f"free Cartesian DOF:      "
        f"{free_dof}"
    )

    print(
        f"constrained rank:        "
        f"{rank}"
    )

    print(
        f"constrained nullity:     "
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
            "the two supported T3 attachments remove all "
            "infinitesimal freedom in this composite"
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
