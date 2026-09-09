"""
demo_energy_stability.py

Demo: energy-Hessian stability index + remove-one-cable scan.

Run:
  python experiments/cable_removal/demo_energy_stability.py
"""

import numpy as np

from tensegrity.energy import SpringMember, stability_index_energy_hessian


def make_members(nodes: np.ndarray):
    # Toy example: all members are "cables" with k=1, and prestress via L0 < current length.
    # For bars, you'd typically set L0 > current length to induce compression,
    # or handle compression-only constraints later.
    pairs = [
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    ]
    members = []
    for (i, j) in pairs:
        L = np.linalg.norm(nodes[i] - nodes[j])
        members.append(SpringMember(i=i, j=j, kind="cable", k=1.0, L0=0.90 * L))
    return members


def main() -> None:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.9, 0.0],
            [0.5, 0.3, 0.8],
        ],
        dtype=float,
    )

    members = make_members(nodes)

    base = stability_index_energy_hessian(nodes, members, eps=1e-6)
    print(
        f"Base stability index (min eig): {base.lambda_min:.3e} "
        f"[{base.classification}]"
    )

    for idx in range(len(members)):
        reduced = members[:idx] + members[idx + 1 :]
        result = stability_index_energy_hessian(nodes, reduced, eps=1e-6)
        delta = result.lambda_min - base.lambda_min
        flag = "  <-- stability worsened" if delta < 0 else ""
        print(
            f"remove member {idx}: min eig {result.lambda_min:.3e} "
            f"[{result.classification}] (delta {delta:.3e}){flag}"
        )


if __name__ == "__main__":
    main()

