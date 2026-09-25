"""Supported zero-twist T3 attachment benchmark.

The zero-twist T3 proxy is internally rigid but has no state of self-stress.
This experiment treats one triangle as attached to a post/support and studies
the remaining three nodes as the interface to a larger structure.

This is a structural proxy for Crawford-Young's T3-to-post attachment, not a
verbatim reconstruction of the COMSOL geometry.
"""

from __future__ import annotations

import numpy as np

from tensegrity.rigidity import (
    rigidity_matrix,
)

from experiments.t3_attachment_scan import (
    t3_proxy_geometry,
)


def constrained_rigidity_matrix(
    nodes,
    members,
    fixed_nodes,
):
    """Remove Cartesian DOFs belonging to fixed nodes."""
    R = rigidity_matrix(
        nodes,
        members,
    )

    free_columns = []

    fixed_nodes = set(
        fixed_nodes
    )

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

    return (
        R,
        R_free,
        free_columns,
    )


def main():
    #
    # Susan's attachment version:
    # zero twist.
    #
    nodes, members = (
        t3_proxy_geometry(
            twist_degrees=0.0
        )
    )

    #
    # Lower triangle = post/support side.
    #
    fixed_nodes = [
        0,
        1,
        2,
    ]

    interface_nodes = [
        3,
        4,
        5,
    ]

    (
        R,
        R_free,
        free_columns,
    ) = constrained_rigidity_matrix(
        nodes,
        members,
        fixed_nodes,
    )

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

    print()
    print("=" * 78)
    print("SUPPORTED ZERO-TWIST T3 ATTACHMENT")
    print("=" * 78)

    print(
        f"nodes:                     "
        f"{nodes.shape[0]}"
    )

    print(
        f"members:                   "
        f"{len(members)}"
    )

    print(
        f"fixed nodes:               "
        f"{fixed_nodes}"
    )

    print(
        f"interface nodes:           "
        f"{interface_nodes}"
    )

    print()

    print(
        f"full R shape:              "
        f"{R.shape}"
    )

    print(
        f"constrained R shape:       "
        f"{R_free.shape}"
    )

    print(
        f"free Cartesian DOF:        "
        f"{free_dof}"
    )

    print(
        f"constrained rank:          "
        f"{rank}"
    )

    print(
        f"constrained nullity:       "
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

    if nullity == 0:
        conclusion = (
            "fully constrained: the post-supported "
            "attachment has no infinitesimal free mechanism"
        )
    else:
        conclusion = (
            f"{nullity} infinitesimal mechanism(s) remain "
            "after post support"
        )

    print()
    print(
        f"conclusion: {conclusion}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
