"""Initial rigidity benchmark for a mirrored T4 trijunction proxy."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import mirrored_t4_prism
from tensegrity.rigidity import (
    analyze_rigidity,
    rigidity_matrix,
    self_stress_basis,
)


def main() -> None:
    nodes, members = mirrored_t4_prism()

    result = analyze_rigidity(
        nodes,
        members,
    )

    R = rigidity_matrix(
        nodes,
        members,
    )

    self_stress = self_stress_basis(
        nodes,
        members,
    )

    singular_values = np.linalg.svd(
        R,
        compute_uv=False,
    )

    bar_count = sum(
        member.kind == "bar"
        for member in members
    )

    cable_count = sum(
        member.kind == "cable"
        for member in members
    )

    print()
    print("=" * 72)
    print("MIRRORED T4 PROXY BENCHMARK")
    print("=" * 72)

    print(
        f"nodes:                    "
        f"{nodes.shape[0]}"
    )

    print(
        f"members:                  "
        f"{len(members)}"
    )

    print(
        f"bars:                     "
        f"{bar_count}"
    )

    print(
        f"cables:                   "
        f"{cable_count}"
    )

    print()

    print(
        f"rigidity matrix shape:    "
        f"{R.shape}"
    )

    print(
        f"rigidity rank:            "
        f"{result.rank}"
    )

    print(
        f"nullity:                  "
        f"{result.nullity}"
    )

    print(
        f"rigid-body modes:         "
        f"{result.rigid_body_modes}"
    )

    print(
        f"internal mechanisms:      "
        f"{result.mechanisms}"
    )

    print(
        f"self-stress dimension:    "
        f"{result.self_stress_dimension}"
    )

    print()

    print(
        "rigidity singular values:"
    )

    print(
        np.array2string(
            singular_values,
            precision=12,
            suppress_small=False,
        )
    )

    print()

    print(
        f"self-stress basis shape:  "
        f"{self_stress.shape}"
    )

    if self_stress.shape[1] > 0:
        for k in range(
            self_stress.shape[1]
        ):
            residual = np.linalg.norm(
                R.T
                @ self_stress[:, k]
            )

            print(
                f"self-stress {k + 1} "
                f"equilibrium residual: "
                f"{residual:.12e}"
            )


if __name__ == "__main__":
    main()
