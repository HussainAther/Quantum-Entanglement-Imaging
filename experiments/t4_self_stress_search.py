"""Search the mirrored T4 proxy's 2D self-stress space for a
strut-compression / cable-tension compatible prestress."""

from __future__ import annotations

import numpy as np

from tensegrity.examples import mirrored_t4_prism
from tensegrity.rigidity import (
    rigidity_matrix,
    self_stress_basis,
)


def sign_margin(q, members):
    """Return the minimum physical sign margin.

    Positive margin means:
        bars   -> q < 0
        cables -> q > 0
    """
    margins = []

    for value, member in zip(q, members):
        if member.kind == "bar":
            margins.append(-value)
        elif member.kind == "cable":
            margins.append(value)
        else:
            raise ValueError(
                f"unknown member kind: {member.kind}"
            )

    return float(
        np.min(margins)
    )


def main():
    nodes, members = mirrored_t4_prism()

    R = rigidity_matrix(
        nodes,
        members,
    )

    Q = self_stress_basis(
        nodes,
        members,
        tol=1e-9,
    )

    print()
    print("=" * 78)
    print("T4 SELF-STRESS SIGN-COMPATIBILITY SEARCH")
    print("=" * 78)

    print(
        f"self-stress basis shape: "
        f"{Q.shape}"
    )

    if Q.shape[1] != 2:
        raise RuntimeError(
            "Expected a 2D self-stress space."
        )

    #
    # Any unit vector in the 2D self-stress
    # coefficient plane can be written as:
    #
    #     c = [cos(theta), sin(theta)]
    #
    # and therefore
    #
    #     q = Q @ c.
    #
    # Scan densely over theta.
    #
    theta_values = np.linspace(
        0.0,
        2.0 * np.pi,
        200001,
        endpoint=False,
    )

    best_margin = -np.inf
    best_theta = None
    best_q = None

    for theta in theta_values:
        coefficients = np.array(
            [
                np.cos(theta),
                np.sin(theta),
            ]
        )

        q = Q @ coefficients

        margin = sign_margin(
            q,
            members,
        )

        if margin > best_margin:
            best_margin = margin
            best_theta = theta
            best_q = q.copy()

    assert best_q is not None
    assert best_theta is not None

    #
    # Normalize for convenient interpretation.
    #
    q_normalized = (
        best_q
        / np.max(
            np.abs(best_q)
        )
    )

    equilibrium_residual = (
        np.linalg.norm(
            R.T @ q_normalized
        )
    )

    print()
    print(
        f"best theta:                  "
        f"{best_theta:.12f}"
    )

    print(
        f"best raw sign margin:        "
        f"{best_margin:.12e}"
    )

    print(
        f"normalized equilibrium res.: "
        f"{equilibrium_residual:.12e}"
    )

    print()

    print(
        "best normalized q:"
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
    ) in enumerate(
        zip(
            members,
            q_normalized,
        )
    ):
        if member.kind == "bar":
            required = "compression (<0)"
            ok = value < 0.0
        else:
            required = "tension (>0)"
            ok = value > 0.0

        compatible &= ok

        print(
            f"{idx:2d} "
            f"({member.i},{member.j}) "
            f"{member.kind:5s} "
            f"q={value:+.10f} "
            f"{'OK' if ok else 'FAIL'} "
            f"[{required}]"
        )

    print()
    print("=" * 78)

    if compatible:
        print(
            "RESULT: physically sign-compatible "
            "self-stress exists."
        )
    else:
        print(
            "RESULT: no strictly sign-compatible "
            "self-stress found in the scanned 2D space."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
