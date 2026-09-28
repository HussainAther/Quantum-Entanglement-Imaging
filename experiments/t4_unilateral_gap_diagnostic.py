"""Member-gap diagnostic for the unilateral T4 + two-T3 model.

This script investigates whether the many apparent active-set transitions
observed at high foundation stiffness are genuine mechanical switching events
or numerical chatter near the unilateral activation boundary.

For each selected foundation stiffness and prestress level, it:

1. builds the unilateral T4 + T3 composite,
2. solves the loaded equilibrium at force_per_node = 1e-4,
3. computes each member gap

       g = L - L0,

4. reports how many members lie near the activation threshold,
5. prints the members closest to the switching surface,
6. saves a CSV of all member-level gaps.

Interpretation
--------------
For cables:
    g > 0  -> active in tension
    g < 0  -> slack

For bars:
    g < 0  -> active in compression
    g > 0  -> slack

Members with |g| near zero lie on the unilateral switching surface.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.examples import (
    t4_with_two_supported_t3,
)

from experiments.t4_unilateral_optimal_prestress import (
    build_unilateral_springs,
    normalized_t4_force_density,
    solve_loaded_state,
)


OUTPUT_DIR = Path(
    "outputs/t4_unilateral_gap_diagnostic"
)


def member_gaps(
    nodes,
    springs,
):
    """Return current length, rest length, and gap for every member."""
    lengths = []
    rest_lengths = []
    gaps = []

    for spring in springs:
        d = (
            nodes[spring.i]
            - nodes[spring.j]
        )

        length = float(
            np.linalg.norm(d)
        )

        rest_length = float(
            spring.L0
        )

        gap = (
            length
            - rest_length
        )

        lengths.append(
            length
        )

        rest_lengths.append(
            rest_length
        )

        gaps.append(
            gap
        )

    return (
        np.asarray(
            lengths,
            dtype=float,
        ),
        np.asarray(
            rest_lengths,
            dtype=float,
        ),
        np.asarray(
            gaps,
            dtype=float,
        ),
    )


def classify_gap(
    kind,
    gap,
    tol,
):
    """Classify a member using a symmetric numerical deadband."""
    if abs(
        gap
    ) <= tol:
        return "threshold"

    if kind == "cable":
        if gap > 0.0:
            return "active"
        return "slack"

    if kind == "bar":
        if gap < 0.0:
            return "active"
        return "slack"

    raise ValueError(
        f"Unknown member kind: {kind}"
    )


def print_nearest_members(
    members,
    lengths,
    rest_lengths,
    gaps,
    count=12,
):
    """Print members closest to the unilateral switching surface."""
    order = np.argsort(
        np.abs(
            gaps
        )
    )

    print()
    print(
        f"{count} members nearest switching surface:"
    )

    print(
        "idx   endpoints   kind      "
        "length             L0                 gap"
    )

    for idx in order[
        :count
    ]:
        member = members[
            idx
        ]

        print(
            f"{idx:3d}   "
            f"({member.i:2d},{member.j:2d})   "
            f"{member.kind:5s}   "
            f"{lengths[idx]: .12e}   "
            f"{rest_lengths[idx]: .12e}   "
            f"{gaps[idx]:+.12e}"
        )


def print_deadband_counts(
    gaps,
):
    """Report how many members fall inside increasingly strict deadbands."""
    print()
    print(
        "members near switching surface:"
    )

    for tol in (
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11,
        1e-12,
    ):
        count = int(
            np.count_nonzero(
                np.abs(
                    gaps
                )
                < tol
            )
        )

        print(
            f"  |gap| < {tol:.0e}: "
            f"{count:2d}"
        )


def print_state_counts(
    members,
    gaps,
    tol,
):
    """Print mutually exclusive active/slack/threshold counts."""
    active_cables = 0
    slack_cables = 0
    active_bars = 0
    slack_bars = 0
    threshold_cables = 0
    threshold_bars = 0

    for member, gap in zip(
        members,
        gaps,
    ):
        state = classify_gap(
            member.kind,
            float(
                gap
            ),
            tol,
        )

        if (
            member.kind
            == "cable"
        ):
            if state == "active":
                active_cables += 1
            elif state == "slack":
                slack_cables += 1
            else:
                threshold_cables += 1

        elif (
            member.kind
            == "bar"
        ):
            if state == "active":
                active_bars += 1
            elif state == "slack":
                slack_bars += 1
            else:
                threshold_bars += 1

    print(
        f"tol={tol:.0e}: "
        f"activeC={active_cables:2d} "
        f"slackC={slack_cables:2d} "
        f"thresholdC={threshold_cables:2d} | "
        f"activeB={active_bars:2d} "
        f"slackB={slack_bars:2d} "
        f"thresholdB={threshold_bars:2d}"
    )


def write_case_csv(
    path,
    members,
    lengths,
    rest_lengths,
    gaps,
):
    """Save complete member-level diagnostic information."""
    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "member_index",
                "i",
                "j",
                "kind",
                "length",
                "rest_length",
                "gap",
                "abs_gap",
                "state_tol_1e-6",
                "state_tol_1e-8",
                "state_tol_1e-10",
            ],
        )

        writer.writeheader()

        for idx, (
            member,
            length,
            rest_length,
            gap,
        ) in enumerate(
            zip(
                members,
                lengths,
                rest_lengths,
                gaps,
            )
        ):
            writer.writerow(
                {
                    "member_index": idx,
                    "i": member.i,
                    "j": member.j,
                    "kind": member.kind,
                    "length": length,
                    "rest_length": rest_length,
                    "gap": gap,
                    "abs_gap": abs(
                        gap
                    ),
                    "state_tol_1e-6": classify_gap(
                        member.kind,
                        gap,
                        1e-6,
                    ),
                    "state_tol_1e-8": classify_gap(
                        member.kind,
                        gap,
                        1e-8,
                    ),
                    "state_tol_1e-10": classify_gap(
                        member.kind,
                        gap,
                        1e-10,
                    ),
                }
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_nodes, members = (
        t4_with_two_supported_t3()
    )

    support_nodes = np.asarray(
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

    loaded_nodes = [
        3,
        4,
        5,
    ]

    q_hat = (
        normalized_t4_force_density(
            reference_nodes
        )
    )

    #
    # Focus on the high-foundation cases
    # where the earlier count-based analysis
    # showed many apparent transitions.
    #
    foundation_values = [
        10.0,
        100.0,
    ]

    #
    # Concentrate around the observed
    # unilateral stiffness optima.
    #
    prestress_values = [
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.40,
        0.41,
        0.42,
    ]

    force_per_node = (
        1e-4
    )

    print()
    print("=" * 110)
    print(
        "UNILATERAL MEMBER-GAP DIAGNOSTIC"
    )
    print("=" * 110)

    summary_rows = []

    for foundation_k in foundation_values:
        print()
        print("#" * 110)
        print(
            f"FOUNDATION k = "
            f"{foundation_k:.3e}"
        )
        print("#" * 110)

        for alpha in prestress_values:
            springs = (
                build_unilateral_springs(
                    reference_nodes,
                    members,
                    q_hat,
                    prestress_scale=alpha,
                    stiffness=1.0,
                )
            )

            result = (
                solve_loaded_state(
                    reference_nodes.copy(),
                    reference_nodes,
                    springs,
                    support_nodes,
                    loaded_nodes,
                    foundation_k,
                    force_per_node,
                )
            )

            X = result[
                "nodes"
            ]

            (
                lengths,
                rest_lengths,
                gaps,
            ) = member_gaps(
                X,
                springs,
            )

            min_abs_gap = float(
                np.min(
                    np.abs(
                        gaps
                    )
                )
            )

            median_abs_gap = float(
                np.median(
                    np.abs(
                        gaps
                    )
                )
            )

            print()
            print("-" * 110)

            print(
                f"k_f={foundation_k:g}, "
                f"alpha={alpha:.2f}"
            )

            print("-" * 110)

            print(
                f"optimizer success:       "
                f"{result['success']}"
            )

            print(
                f"optimizer iterations:    "
                f"{result['iterations']}"
            )

            print(
                f"loaded residual:         "
                f"{result['residual']:.12e}"
            )

            print(
                f"mean dz:                 "
                f"{result['mean_dz']:.12e}"
            )

            print(
                f"effective stiffness:     "
                f"{result['effective_stiffness']:.12e}"
            )

            print(
                f"minimum |gap|:           "
                f"{min_abs_gap:.12e}"
            )

            print(
                f"median |gap|:            "
                f"{median_abs_gap:.12e}"
            )

            print_deadband_counts(
                gaps
            )

            print()
            print(
                "mutually exclusive state counts:"
            )

            for tol in (
                1e-4,
                1e-6,
                1e-8,
                1e-10,
                1e-12,
            ):
                print_state_counts(
                    members,
                    gaps,
                    tol,
                )

            print_nearest_members(
                members,
                lengths,
                rest_lengths,
                gaps,
                count=12,
            )

            csv_path = (
                OUTPUT_DIR
                / (
                    f"gaps_kf_{foundation_k:g}"
                    f"_alpha_{alpha:.2f}.csv"
                )
            )

            write_case_csv(
                csv_path,
                members,
                lengths,
                rest_lengths,
                gaps,
            )

            summary_rows.append(
                {
                    "foundation_k": (
                        foundation_k
                    ),
                    "prestress_scale": (
                        alpha
                    ),
                    "optimizer_success": (
                        result[
                            "success"
                        ]
                    ),
                    "optimizer_iterations": (
                        result[
                            "iterations"
                        ]
                    ),
                    "loaded_residual": (
                        result[
                            "residual"
                        ]
                    ),
                    "mean_dz": (
                        result[
                            "mean_dz"
                        ]
                    ),
                    "effective_stiffness": (
                        result[
                            "effective_stiffness"
                        ]
                    ),
                    "minimum_abs_gap": (
                        min_abs_gap
                    ),
                    "median_abs_gap": (
                        median_abs_gap
                    ),
                    "count_abs_gap_lt_1e-4": int(
                        np.count_nonzero(
                            np.abs(
                                gaps
                            )
                            < 1e-4
                        )
                    ),
                    "count_abs_gap_lt_1e-6": int(
                        np.count_nonzero(
                            np.abs(
                                gaps
                            )
                            < 1e-6
                        )
                    ),
                    "count_abs_gap_lt_1e-8": int(
                        np.count_nonzero(
                            np.abs(
                                gaps
                            )
                            < 1e-8
                        )
                    ),
                    "count_abs_gap_lt_1e-10": int(
                        np.count_nonzero(
                            np.abs(
                                gaps
                            )
                            < 1e-10
                        )
                    ),
                    "count_abs_gap_lt_1e-12": int(
                        np.count_nonzero(
                            np.abs(
                                gaps
                            )
                            < 1e-12
                        )
                    ),
                }
            )

    summary_path = (
        OUTPUT_DIR
        / "gap_diagnostic_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "prestress_scale",
                "optimizer_success",
                "optimizer_iterations",
                "loaded_residual",
                "mean_dz",
                "effective_stiffness",
                "minimum_abs_gap",
                "median_abs_gap",
                "count_abs_gap_lt_1e-4",
                "count_abs_gap_lt_1e-6",
                "count_abs_gap_lt_1e-8",
                "count_abs_gap_lt_1e-10",
                "count_abs_gap_lt_1e-12",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 110)

    print(
        f"Saved: {summary_path}"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
