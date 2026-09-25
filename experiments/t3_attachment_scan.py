"""Explore the T3 attachment proxy as its twist is removed.

Crawford-Young describes the ordinary T3 triangular prism as twisted, but says
that when used as a trijunction the twist must be removed.

This experiment keeps the same 3-strut / 9-cable topology and scans twist from
0 to 30 degrees to determine how rigidity, mechanisms, and self-stress change.

This is a computational proxy, not yet a verbatim COMSOL reconstruction.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.rigidity import (
    Member,
    analyze_rigidity,
    rigidity_matrix,
    self_stress_basis,
)


OUTPUT = Path(
    "outputs/t3_attachment_scan.csv"
)


def t3_proxy_geometry(
    radius: float = 1.0,
    height: float = 1.5,
    twist_degrees: float = 0.0,
):
    twist = np.deg2rad(
        twist_degrees
    )

    lower_angles = np.array(
        [
            0.0,
            2.0 * np.pi / 3.0,
            4.0 * np.pi / 3.0,
        ]
    )

    upper_angles = (
        lower_angles + twist
    )

    lower = np.column_stack(
        [
            radius * np.cos(
                lower_angles
            ),
            radius * np.sin(
                lower_angles
            ),
            np.zeros(3),
        ]
    )

    upper = np.column_stack(
        [
            radius * np.cos(
                upper_angles
            ),
            radius * np.sin(
                upper_angles
            ),
            np.full(
                3,
                height,
            ),
        ]
    )

    nodes = np.vstack(
        [
            lower,
            upper,
        ]
    )

    struts = [
        (0, 4),
        (1, 5),
        (2, 3),
    ]

    lower_ring = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    upper_ring = [
        (3, 4),
        (4, 5),
        (5, 3),
    ]

    side_cables = [
        (0, 3),
        (1, 4),
        (2, 5),
    ]

    members = [
        Member(
            i,
            j,
            kind="bar",
        )
        for i, j in struts
    ]

    members.extend(
        Member(
            i,
            j,
            kind="cable",
        )
        for i, j in (
            lower_ring
            + upper_ring
            + side_cables
        )
    )

    return (
        nodes,
        members,
    )


def best_sign_margin(
    nodes,
    members,
):
    """Find the best bar-/cable-compatible direction in self-stress space."""

    Q = self_stress_basis(
        nodes,
        members,
        tol=1e-9,
    )

    dim = Q.shape[1]

    if dim == 0:
        return (
            float("nan"),
            False,
        )

    #
    # The canonical topology will normally
    # have a one-dimensional self-stress,
    # but handle dim=2 as well.
    #
    candidates = []

    if dim == 1:
        candidates = [
            Q[:, 0],
            -Q[:, 0],
        ]

    elif dim == 2:
        for theta in np.linspace(
            0.0,
            2.0 * np.pi,
            50001,
            endpoint=False,
        ):
            candidates.append(
                Q
                @ np.array(
                    [
                        np.cos(theta),
                        np.sin(theta),
                    ]
                )
            )

    else:
        #
        # We don't try to optimize a
        # high-dimensional cone here.
        #
        return (
            float("nan"),
            False,
        )

    best = -np.inf

    for q in candidates:
        margins = []

        for value, member in zip(
            q,
            members,
        ):
            if member.kind == "bar":
                margins.append(
                    -value
                )
            else:
                margins.append(
                    value
                )

        best = max(
            best,
            float(
                np.min(margins)
            ),
        )

    return (
        best,
        best > 0.0,
    )


def main():
    twist_values = np.array(
        [
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
        ]
    )

    rows = []

    print()
    print("=" * 94)
    print("T3 ATTACHMENT TWIST SCAN")
    print("=" * 94)

    print(
        " twist   rank   mech   self-stress   sigma_min       sign margin      compatible"
    )

    for twist in twist_values:
        nodes, members = (
            t3_proxy_geometry(
                twist_degrees=float(
                    twist
                )
            )
        )

        analysis = analyze_rigidity(
            nodes,
            members,
        )

        R = rigidity_matrix(
            nodes,
            members,
        )

        singular_values = (
            np.linalg.svd(
                R,
                compute_uv=False,
            )
        )

        sigma_min = float(
            singular_values[-1]
        )

        margin, compatible = (
            best_sign_margin(
                nodes,
                members,
            )
        )

        print(
            f"{twist:6.1f} "
            f"{analysis.rank:6d} "
            f"{analysis.mechanisms:6d} "
            f"{analysis.self_stress_dimension:13d} "
            f"{sigma_min: .6e} "
            f"{margin: .6e} "
            f"{str(compatible):>11s}"
        )

        rows.append(
            {
                "twist_degrees": twist,
                "rank": analysis.rank,
                "mechanisms": (
                    analysis.mechanisms
                ),
                "self_stress_dimension": (
                    analysis.self_stress_dimension
                ),
                "sigma_min": sigma_min,
                "best_sign_margin": margin,
                "sign_compatible": compatible,
            }
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
