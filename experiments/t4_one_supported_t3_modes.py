"""Locate the two surviving mechanisms of the T4 + one supported T3 composite."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import (
    t4_with_one_supported_t3,
)
from tensegrity.rigidity import rigidity_matrix


def main():
    nodes, members = (
        t4_with_one_supported_t3()
    )

    fixed_nodes = {
        9,
        10,
        11,
    }

    R = rigidity_matrix(
        nodes,
        members,
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

    #
    # full_matrices=True exposes the
    # complete right nullspace.
    #
    U, singular_values, Vt = (
        np.linalg.svd(
            R_free,
            full_matrices=True,
        )
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    nullity = (
        R_free.shape[1]
        - rank
    )

    modes_free = (
        Vt[rank:].T
    )

    if nullity != 2:
        raise RuntimeError(
            f"Expected two mechanisms; found {nullity}"
        )

    #
    # Map free-coordinate modes back into
    # the full 12-node Cartesian vector.
    #
    modes_full = np.zeros(
        (
            3 * nodes.shape[0],
            nullity,
        ),
        dtype=float,
    )

    for local_col, global_col in enumerate(
        free_columns
    ):
        modes_full[
            global_col,
            :,
        ] = modes_free[
            local_col,
            :,
        ]

    print()
    print("=" * 84)
    print("T4 + ONE SUPPORTED T3: SURVIVING MECHANISMS")
    print("=" * 84)

    print(
        f"constrained rank:       "
        f"{rank}"
    )

    print(
        f"constrained nullity:    "
        f"{nullity}"
    )

    print()

    print(
        "smallest represented singular values:"
    )

    print(
        np.array2string(
            singular_values[-6:],
            precision=12,
            suppress_small=False,
        )
    )

    groups = {
        "T4 lower/interface": [
            0,
            1,
            2,
        ],
        "T4 middle": [
            3,
            4,
            5,
        ],
        "T4 upper": [
            6,
            7,
            8,
        ],
        "T3 fixed support": [
            9,
            10,
            11,
        ],
    }

    for mode_idx in range(
        nullity
    ):
        v = modes_full[
            :,
            mode_idx,
        ].reshape(
            (-1, 3)
        )

        print()
        print("-" * 84)

        print(
            f"mechanism {mode_idx + 1}"
        )

        print("-" * 84)

        full_residual = np.linalg.norm(
            R @ modes_full[:, mode_idx]
        )

        print(
            f"rigidity residual: "
            f"{full_residual:.12e}"
        )

        print()

        print(
            "node displacement vectors:"
        )

        for node in range(
            nodes.shape[0]
        ):
            magnitude = float(
                np.linalg.norm(
                    v[node]
                )
            )

            print(
                f"node {node:2d}: "
                f"("
                f"{v[node,0]: .6f}, "
                f"{v[node,1]: .6f}, "
                f"{v[node,2]: .6f}"
                f")  "
                f"|v|={magnitude:.6f}"
            )

        print()
        print(
            "group participation:"
        )

        total_norm_sq = float(
            np.sum(
                v ** 2
            )
        )

        for name, group_nodes in (
            groups.items()
        ):
            group_norm_sq = float(
                np.sum(
                    v[
                        group_nodes
                    ] ** 2
                )
            )

            fraction = (
                group_norm_sq
                / total_norm_sq
                if total_norm_sq > 0.0
                else 0.0
            )

            print(
                f"  {name:20s}: "
                f"{fraction:.6f}"
            )

    print()
    print("=" * 84)


if __name__ == "__main__":
    main()
