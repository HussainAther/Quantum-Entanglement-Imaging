"""Finite-foundation stiffness study for the T4 + two-T3 composite.

The two-supported T3 composite is first-order rigid when its six outer
support nodes are perfectly fixed.

Here those perfect clamps are replaced by isotropic foundation springs.
We then compute the tangent stiffness of the free structure and measure
vertical compliance of the central T4.

This is a simplified computational analogue of Crawford-Young's use of
spring-foundation boundary conditions. It is not yet a dimensional
reproduction of the COMSOL model.
"""

from __future__ import annotations

import numpy as np

from tensegrity.energy import (
    numerical_hessian,
    total_energy,
)
from tensegrity.examples import (
    t4_with_two_supported_t3,
)


def stress_free_member_energy(
    nodes,
    members,
    reference_nodes,
    stiffness=1.0,
):
    """Bilateral axial-spring energy using reference lengths."""
    energy = 0.0

    for member in members:
        i = member.i
        j = member.j

        L = float(
            np.linalg.norm(
                nodes[i] - nodes[j]
            )
        )

        L0 = float(
            np.linalg.norm(
                reference_nodes[i]
                - reference_nodes[j]
            )
        )

        energy += (
            0.5
            * stiffness
            * (L - L0) ** 2
        )

    return energy


def foundation_energy(
    nodes,
    reference_nodes,
    support_nodes,
    foundation_stiffness,
):
    """Isotropic spring foundation at selected support nodes."""
    displacement = (
        nodes[support_nodes]
        - reference_nodes[support_nodes]
    )

    return (
        0.5
        * foundation_stiffness
        * float(
            np.sum(
                displacement ** 2
            )
        )
    )


def total_composite_energy(
    nodes,
    reference_nodes,
    members,
    support_nodes,
    member_stiffness,
    foundation_stiffness,
):
    return (
        stress_free_member_energy(
            nodes,
            members,
            reference_nodes,
            stiffness=member_stiffness,
        )
        + foundation_energy(
            nodes,
            reference_nodes,
            support_nodes,
            foundation_stiffness,
        )
    )


def force_vector_for_nodes(
    node_count,
    loaded_nodes,
    force_per_node=1.0,
):
    """Apply equal +z force to selected nodes."""
    force = np.zeros(
        3 * node_count,
        dtype=float,
    )

    for node in loaded_nodes:
        force[
            3 * node + 2
        ] = force_per_node

    return force


def main():
    reference_nodes, members = (
        t4_with_two_supported_t3()
    )

    #
    # Support triangles added by the
    # two T3 attachments.
    #
    support_nodes = np.array(
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

    #
    # Central T4 middle triangle.
    #
    loaded_nodes = [
        3,
        4,
        5,
    ]

    member_stiffness = 1.0

    foundation_values = [
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]

    force = force_vector_for_nodes(
        reference_nodes.shape[0],
        loaded_nodes,
        force_per_node=1.0,
    )

    print()
    print("=" * 94)
    print(
        "T4 + TWO T3: FOUNDATION-STIFFNESS RESPONSE"
    )
    print("=" * 94)

    print(
        " foundation_k       lambda_min        "
        "mean dz / unit force       effective stiffness"
    )

    for foundation_k in foundation_values:

        def energy_flat(flat):
            X = flat.reshape(
                reference_nodes.shape
            )

            return total_composite_energy(
                X,
                reference_nodes,
                members,
                support_nodes,
                member_stiffness,
                foundation_k,
            )

        H = numerical_hessian(
            energy_flat,
            reference_nodes.reshape(-1),
            eps=2e-5,
        )

        H = (
            0.5
            * (
                H + H.T
            )
        )

        eigenvalues = np.linalg.eigvalsh(
            H
        )

        #
        # The finite foundation removes
        # global rigid-body zero modes.
        #
        lambda_min = float(
            eigenvalues[0]
        )

        #
        # Linear tangent response:
        #
        #     H u = f
        #
        displacement = np.linalg.solve(
            H,
            force,
        )

        dz = np.array(
            [
                displacement[
                    3 * node + 2
                ]
                for node in loaded_nodes
            ]
        )

        mean_dz = float(
            np.mean(dz)
        )

        total_force = float(
            len(loaded_nodes)
        )

        effective_stiffness = (
            total_force / mean_dz
            if abs(mean_dz) > 1e-15
            else np.inf
        )

        print(
            f"{foundation_k:13.3e}   "
            f"{lambda_min: .6e}   "
            f"{mean_dz: .12e}   "
            f"{effective_stiffness: .12e}"
        )

    print()
    print("=" * 94)


if __name__ == "__main__":
    main()
