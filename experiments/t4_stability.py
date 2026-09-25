"""Prestress stability analysis for the equilibrated mirrored T4 proxy."""

from __future__ import annotations

import numpy as np

from tensegrity.energy import (
    equilibrium_residual_norm,
    numerical_hessian,
    nonrigid_basis,
    reduced_nonrigid_hessian,
    total_energy,
)
from tensegrity.examples import (
    equilibrated_mirrored_t4,
)
from tensegrity.rigidity import (
    analyze_rigidity,
    rigidity_matrix,
)


def main():
    nodes, members, springs = (
        equilibrated_mirrored_t4(
            prestress_scale=0.20,
            stiffness=1.0,
        )
    )

    rigidity = analyze_rigidity(
        nodes,
        members,
    )

    residual = equilibrium_residual_norm(
        nodes,
        springs,
    )

    energy = total_energy(
        nodes,
        springs,
    )

    print()
    print("=" * 78)
    print("EQUILIBRATED MIRRORED T4 STABILITY")
    print("=" * 78)

    print(
        f"equilibrium residual:        "
        f"{residual:.12e}"
    )

    print(
        f"energy:                      "
        f"{energy:.12e}"
    )

    print()

    print(
        f"rigidity rank:               "
        f"{rigidity.rank}"
    )

    print(
        f"internal mechanisms:         "
        f"{rigidity.mechanisms}"
    )

    print(
        f"self-stress dimension:       "
        f"{rigidity.self_stress_dimension}"
    )

    #
    # Build exact mechanism basis.
    #
    R = rigidity_matrix(
        nodes,
        members,
    )

    Z = nonrigid_basis(
        nodes,
    )

    R_internal = (
        R @ Z
    )

    _, singular_values, Vt = np.linalg.svd(
        R_internal,
        full_matrices=True,
    )

    rank = int(
        np.sum(
            singular_values > 1e-9
        )
    )

    mechanism_internal = (
        Vt[rank:].T
    )

    print()

    print(
        f"R_internal shape:            "
        f"{R_internal.shape}"
    )

    print(
        f"mechanism basis shape:       "
        f"{mechanism_internal.shape}"
    )

    print(
        "projected rigidity singular values:"
    )

    print(
        np.array2string(
            singular_values,
            precision=12,
            suppress_small=False,
        )
    )

    #
    # Numerical energy Hessian.
    #
    def energy_flat(flat):
        return total_energy(
            flat.reshape((-1, 3)),
            springs,
        )

    H = numerical_hessian(
        energy_flat,
        nodes.reshape(-1),
        eps=2e-5,
    )

    H_nonrigid = reduced_nonrigid_hessian(
        H,
        nodes,
    )

    full_eigs = np.sort(
        np.linalg.eigvalsh(
            H_nonrigid
        )
    )

    print()
    print(
        "non-rigid Hessian eigenvalues:"
    )

    print(
        np.array2string(
            full_eigs,
            precision=12,
            suppress_small=False,
        )
    )

    #
    # Restrict Hessian to exact mechanisms.
    #
    H_mech = (
        mechanism_internal.T
        @ H_nonrigid
        @ mechanism_internal
    )

    H_mech = (
        0.5
        * (
            H_mech
            + H_mech.T
        )
    )

    mech_eigs = np.linalg.eigvalsh(
        H_mech
    )

    print()
    print(
        "mechanism Hessian:"
    )

    print(
        np.array2string(
            H_mech,
            precision=12,
            suppress_small=False,
        )
    )

    print(
        "mechanism-space eigenvalues:"
    )

    print(
        np.array2string(
            mech_eigs,
            precision=12,
            suppress_small=False,
        )
    )

    print()

    eig_tol = 1e-6

    if np.min(mech_eigs) < -eig_tol:
        conclusion = (
            "unstable: at least one mechanism "
            "has negative quadratic curvature"
        )
    elif np.min(mech_eigs) <= eig_tol:
        conclusion = (
            "marginal at quadratic order"
        )
    else:
        conclusion = (
            "prestress-stabilized at quadratic order"
        )

    print(
        f"conclusion: {conclusion}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
