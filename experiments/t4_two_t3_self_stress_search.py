"""Find a sign-compatible self-stress for the full T4 + two-T3 composite.

We solve a linear program for q satisfying

    R.T @ q = 0

with the tensegrity sign conditions

    bars   : q < 0
    cables : q > 0.

Rather than relying on an arbitrary SVD basis, the LP maximizes the minimum
signed member force while fixing the total signed magnitude to one.

If the optimum margin is positive, the full composite admits a strictly
sign-compatible tensegrity prestress at the reference geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from tensegrity.examples import (
    t4_with_two_supported_t3,
)
from tensegrity.rigidity import (
    analyze_rigidity,
    rigidity_matrix,
    self_stress_basis,
)


def main():
    nodes, members = (
        t4_with_two_supported_t3()
    )

    R = rigidity_matrix(
        nodes,
        members,
    )

    analysis = analyze_rigidity(
        nodes,
        members,
    )

    Q = self_stress_basis(
        nodes,
        members,
        tol=1e-9,
    )

    m = len(members)

    #
    # sign[i] * q[i] should be positive:
    #
    #     bars   -> sign=-1, q<0
    #     cables -> sign=+1, q>0
    #
    sign = np.asarray(
        [
            -1.0
            if member.kind == "bar"
            else 1.0
            for member in members
        ],
        dtype=float,
    )

    #
    # LP variables:
    #
    #     x = [q_0, ..., q_(m-1), t]
    #
    # maximize t
    #
    # scipy minimizes, so objective is -t.
    #
    c = np.zeros(
        m + 1,
        dtype=float,
    )

    c[-1] = -1.0

    #
    # Equilibrium:
    #
    #     R.T @ q = 0
    #
    # plus normalization:
    #
    #     sum(sign_i q_i) = 1.
    #
    # Once all signed values are positive,
    # this fixes the overall scale.
    #
    A_eq = np.zeros(
        (
            R.shape[1] + 1,
            m + 1,
        ),
        dtype=float,
    )

    A_eq[
        :R.shape[1],
        :m,
    ] = R.T

    A_eq[
        -1,
        :m,
    ] = sign

    b_eq = np.zeros(
        R.shape[1] + 1,
        dtype=float,
    )

    b_eq[-1] = 1.0

    #
    # Require:
    #
    #     sign_i * q_i >= t
    #
    # or:
    #
    #     -sign_i*q_i + t <= 0
    #
    A_ub = np.zeros(
        (
            m,
            m + 1,
        ),
        dtype=float,
    )

    b_ub = np.zeros(
        m,
        dtype=float,
    )

    for i in range(m):
        A_ub[
            i,
            i,
        ] = -sign[i]

        A_ub[
            i,
            -1,
        ] = 1.0

    #
    # q itself is unbounded.
    # t >= 0.
    #
    bounds = [
        (
            None,
            None,
        )
        for _ in range(m)
    ]

    bounds.append(
        (
            0.0,
            None,
        )
    )

    solve = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    print()
    print("=" * 90)
    print(
        "T4 + TWO T3: FULL-COMPOSITE SELF-STRESS SEARCH"
    )
    print("=" * 90)

    print(
        f"nodes:                    "
        f"{nodes.shape[0]}"
    )

    print(
        f"members:                  "
        f"{len(members)}"
    )

    print(
        f"rigidity rank:            "
        f"{analysis.rank}"
    )

    print(
        f"internal mechanisms:      "
        f"{analysis.mechanisms}"
    )

    print(
        f"self-stress dimension:    "
        f"{analysis.self_stress_dimension}"
    )

    print(
        f"SVD self-stress shape:    "
        f"{Q.shape}"
    )

    print()

    print(
        f"LP success:               "
        f"{solve.success}"
    )

    print(
        f"LP message:               "
        f"{solve.message}"
    )

    if not solve.success:
        print()
        print(
            "RESULT: no sign-compatible prestress "
            "was established."
        )

        return

    q = solve.x[
        :m
    ]

    margin = float(
        solve.x[-1]
    )

    signed_q = (
        sign * q
    )

    equilibrium_residual = float(
        np.linalg.norm(
            R.T @ q
        )
    )

    #
    # Normalize for easy comparison:
    #
    #     max |q| = 1
    #
    q_normalized = (
        q
        / np.max(
            np.abs(q)
        )
    )

    signed_normalized = (
        sign
        * q_normalized
    )

    print(
        f"optimal sign margin:      "
        f"{margin:.12e}"
    )

    print(
        f"equilibrium residual:     "
        f"{equilibrium_residual:.12e}"
    )

    print(
        f"minimum signed q:         "
        f"{np.min(signed_q):.12e}"
    )

    print(
        f"maximum signed q:         "
        f"{np.max(signed_q):.12e}"
    )

    print()

    print(
        "normalized self-stress:"
    )

    print(
        np.array2string(
            q_normalized,
            precision=10,
            suppress_small=False,
        )
    )

    print()
    print(
        "member-by-member:"
    )

    compatible = True

    for idx, (
        member,
        value,
        signed_value,
    ) in enumerate(
        zip(
            members,
            q_normalized,
            signed_normalized,
        )
    ):
        ok = (
            signed_value > 0.0
        )

        compatible &= ok

        required = (
            "compression"
            if member.kind == "bar"
            else "tension"
        )

        print(
            f"{idx:2d} "
            f"({member.i:2d},{member.j:2d}) "
            f"{member.kind:5s} "
            f"q={value:+.10f} "
            f"signed={signed_value:.10f} "
            f"{'OK' if ok else 'FAIL'} "
            f"[{required}]"
        )

    print()
    print("=" * 90)

    if compatible and margin > 0.0:
        print(
            "RESULT: full composite admits a strictly "
            "sign-compatible tensegrity self-stress."
        )
    else:
        print(
            "RESULT: no strictly sign-compatible "
            "prestress was found."
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
