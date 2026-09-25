"""Check whether uniformly prestrained reference states are equilibria."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import t4_with_two_supported_t3


def energy_gradient_fd(
    nodes,
    members,
    reference_nodes,
    support_nodes,
    member_stiffness,
    foundation_stiffness,
    prestrain,
    eps=1e-7,
):
    def energy(X):
        value = 0.0

        for member in members:
            i = member.i
            j = member.j

            L = np.linalg.norm(
                X[i] - X[j]
            )

            L_ref = np.linalg.norm(
                reference_nodes[i]
                - reference_nodes[j]
            )

            L0 = (
                (1.0 - prestrain)
                * L_ref
            )

            value += (
                0.5
                * member_stiffness
                * (L - L0) ** 2
            )

        displacement = (
            X[support_nodes]
            - reference_nodes[support_nodes]
        )

        value += (
            0.5
            * foundation_stiffness
            * np.sum(
                displacement ** 2
            )
        )

        return float(value)

    flat = nodes.reshape(-1)
    grad = np.zeros_like(flat)

    for i in range(
        flat.size
    ):
        xp = flat.copy()
        xm = flat.copy()

        xp[i] += eps
        xm[i] -= eps

        grad[i] = (
            energy(
                xp.reshape(nodes.shape)
            )
            - energy(
                xm.reshape(nodes.shape)
            )
        ) / (
            2.0 * eps
        )

    return grad.reshape(
        nodes.shape
    )


def main():
    reference_nodes, members = (
        t4_with_two_supported_t3()
    )

    support_nodes = np.array(
        [
            9,
            10,
            11,
            12,
            13,
            14,
        ]
    )

    member_stiffness = 1.0
    foundation_stiffness = 1.0

    print()
    print("=" * 72)
    print("PRESTRAIN REFERENCE-STATE RESIDUAL")
    print("=" * 72)

    for prestrain in [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
    ]:
        grad = energy_gradient_fd(
            reference_nodes,
            members,
            reference_nodes,
            support_nodes,
            member_stiffness,
            foundation_stiffness,
            prestrain,
        )

        total_residual = float(
            np.linalg.norm(
                grad
            )
        )

        free_mask = np.ones(
            reference_nodes.shape[0],
            dtype=bool,
        )

        free_mask[
            support_nodes
        ] = False

        free_residual = float(
            np.linalg.norm(
                grad[
                    free_mask
                ]
            )
        )

        support_residual = float(
            np.linalg.norm(
                grad[
                    support_nodes
                ]
            )
        )

        print(
            f"prestrain={prestrain:.2f}  "
            f"total={total_residual:.6e}  "
            f"free={free_residual:.6e}  "
            f"support={support_residual:.6e}"
        )


if __name__ == "__main__":
    main()
