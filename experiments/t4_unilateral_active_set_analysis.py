"""Analyze unilateral member activation around the optimal T4 prestress.

This script reads the detailed output from

    outputs/t4_unilateral_optimal_prestress/unilateral_load_sweep.csv

and asks whether the observed stiffness optimum near alpha ~= 0.39 coincides
with changes in:

    active cables
    slack cables
    active bars
    slack bars
    threshold members

The middle load level, force_per_node = 1e-4, is used so the analysis matches
the representative stiffness used in the optimum search.

We inspect all prestress values, with special attention to alpha in [0.30, 0.45].
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path(
    "outputs/t4_unilateral_optimal_prestress/"
    "unilateral_load_sweep.csv"
)

OUTPUT_DIR = Path(
    "outputs/t4_unilateral_active_set_analysis"
)

REPRESENTATIVE_FORCE = 1e-4


def load_rows():
    rows = []

    with INPUT_PATH.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f
        )

        for row in reader:
            rows.append(
                {
                    "foundation_k": float(
                        row[
                            "foundation_k"
                        ]
                    ),
                    "prestress_scale": float(
                        row[
                            "prestress_scale"
                        ]
                    ),
                    "force_per_node": float(
                        row[
                            "force_per_node"
                        ]
                    ),
                    "equilibrium_residual": float(
                        row[
                            "equilibrium_residual"
                        ]
                    ),
                    "mean_dz": float(
                        row[
                            "mean_dz"
                        ]
                    ),
                    "effective_stiffness": float(
                        row[
                            "effective_stiffness"
                        ]
                    ),
                    "active_cables": int(
                        row[
                            "active_cables"
                        ]
                    ),
                    "slack_cables": int(
                        row[
                            "slack_cables"
                        ]
                    ),
                    "active_bars": int(
                        row[
                            "active_bars"
                        ]
                    ),
                    "slack_bars": int(
                        row[
                            "slack_bars"
                        ]
                    ),
                    "threshold_members": int(
                        row[
                            "threshold_members"
                        ]
                    ),
                }
            )

    return rows


def state_tuple(row):
    return (
        row[
            "active_cables"
        ],
        row[
            "slack_cables"
        ],
        row[
            "active_bars"
        ],
        row[
            "slack_bars"
        ],
        row[
            "threshold_members"
        ],
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    selected = [
        row
        for row in rows
        if np.isclose(
            row[
                "force_per_node"
            ],
            REPRESENTATIVE_FORCE,
        )
    ]

    foundation_values = sorted(
        {
            row[
                "foundation_k"
            ]
            for row in selected
        }
    )

    summary_rows = []

    print()
    print("=" * 118)

    print(
        "UNILATERAL ACTIVE-SET ANALYSIS"
    )

    print("=" * 118)

    for foundation_k in foundation_values:
        cases = sorted(
            [
                row
                for row in selected
                if np.isclose(
                    row[
                        "foundation_k"
                    ],
                    foundation_k,
                )
            ],
            key=lambda row: row[
                "prestress_scale"
            ],
        )

        alpha = np.asarray(
            [
                row[
                    "prestress_scale"
                ]
                for row in cases
            ],
            dtype=float,
        )

        stiffness = np.asarray(
            [
                row[
                    "effective_stiffness"
                ]
                for row in cases
            ],
            dtype=float,
        )

        max_index = int(
            np.nanargmax(
                stiffness
            )
        )

        optimal_alpha = float(
            alpha[
                max_index
            ]
        )

        optimal_stiffness = float(
            stiffness[
                max_index
            ]
        )

        print()
        print("-" * 118)

        print(
            f"foundation k = "
            f"{foundation_k:.3e}"
        )

        print("-" * 118)

        print(
            f"optimal alpha:          "
            f"{optimal_alpha:.6f}"
        )

        print(
            f"maximum stiffness:      "
            f"{optimal_stiffness:.12e}"
        )

        #
        # Detect every active-set transition.
        #
        transitions = []

        previous_state = (
            state_tuple(
                cases[0]
            )
        )

        for index in range(
            1,
            len(
                cases
            ),
        ):
            current_state = (
                state_tuple(
                    cases[
                        index
                    ]
                )
            )

            if (
                current_state
                != previous_state
            ):
                transitions.append(
                    {
                        "alpha_before": (
                            cases[
                                index - 1
                            ][
                                "prestress_scale"
                            ]
                        ),
                        "alpha_after": (
                            cases[
                                index
                            ][
                                "prestress_scale"
                            ]
                        ),
                        "state_before": (
                            previous_state
                        ),
                        "state_after": (
                            current_state
                        ),
                    }
                )

            previous_state = (
                current_state
            )

        print(
            f"number of active-set transitions: "
            f"{len(transitions)}"
        )

        for transition in transitions:
            print()

            print(
                f"  transition "
                f"{transition['alpha_before']:.3f}"
                f" -> "
                f"{transition['alpha_after']:.3f}"
            )

            print(
                "    before "
                "(active cables, slack cables, "
                "active bars, slack bars, threshold) = "
                f"{transition['state_before']}"
            )

            print(
                "    after  "
                "(active cables, slack cables, "
                "active bars, slack bars, threshold) = "
                f"{transition['state_after']}"
            )

        #
        # Print the optimum neighborhood.
        #
        print()
        print(
            "alpha     k_eff          "
            "activeC slackC activeB slackB threshold"
        )

        for row in cases:
            a = row[
                "prestress_scale"
            ]

            if (
                0.30
                <= a
                <= 0.45
            ):
                marker = (
                    " <-- peak"
                    if np.isclose(
                        a,
                        optimal_alpha,
                    )
                    else ""
                )

                print(
                    f"{a:5.2f}   "
                    f"{row['effective_stiffness']: .9e}   "
                    f"{row['active_cables']:7d} "
                    f"{row['slack_cables']:6d} "
                    f"{row['active_bars']:7d} "
                    f"{row['slack_bars']:6d} "
                    f"{row['threshold_members']:9d}"
                    f"{marker}"
                )

        #
        # Find nearest active-set transition
        # to the stiffness optimum.
        #
        if transitions:
            distances = []

            for transition in transitions:
                midpoint = (
                    0.5
                    * (
                        transition[
                            "alpha_before"
                        ]
                        + transition[
                            "alpha_after"
                        ]
                    )
                )

                distances.append(
                    abs(
                        midpoint
                        - optimal_alpha
                    )
                )

            nearest_distance = float(
                np.min(
                    distances
                )
            )

        else:
            nearest_distance = float(
                "inf"
            )

        peak_state = (
            state_tuple(
                cases[
                    max_index
                ]
            )
        )

        #
        # Compare active state immediately
        # around the optimum.
        #
        lower_index = max(
            0,
            max_index - 1,
        )

        upper_index = min(
            len(cases) - 1,
            max_index + 1,
        )

        state_below = (
            state_tuple(
                cases[
                    lower_index
                ]
            )
        )

        state_above = (
            state_tuple(
                cases[
                    upper_index
                ]
            )
        )

        state_fixed_across_peak = (
            state_below
            == peak_state
            == state_above
        )

        print()

        print(
            f"active set fixed across peak: "
            f"{state_fixed_across_peak}"
        )

        if np.isfinite(
            nearest_distance
        ):
            print(
                f"nearest active-set transition "
                f"distance in alpha: "
                f"{nearest_distance:.6f}"
            )

        else:
            print(
                "nearest active-set transition: none"
            )

        if (
            state_fixed_across_peak
        ):
            interpretation = (
                "continuous fixed-active-set optimum"
            )
        else:
            interpretation = (
                "optimum coincides with or lies near "
                "an active-set transition"
            )

        print(
            f"local interpretation: "
            f"{interpretation}"
        )

        summary_rows.append(
            {
                "foundation_k": (
                    foundation_k
                ),
                "optimal_alpha": (
                    optimal_alpha
                ),
                "maximum_stiffness": (
                    optimal_stiffness
                ),
                "active_cables_at_peak": (
                    peak_state[0]
                ),
                "slack_cables_at_peak": (
                    peak_state[1]
                ),
                "active_bars_at_peak": (
                    peak_state[2]
                ),
                "slack_bars_at_peak": (
                    peak_state[3]
                ),
                "threshold_members_at_peak": (
                    peak_state[4]
                ),
                "active_set_fixed_across_peak": (
                    state_fixed_across_peak
                ),
                "nearest_transition_distance": (
                    nearest_distance
                ),
                "transition_count": (
                    len(
                        transitions
                    )
                ),
                "interpretation": (
                    interpretation
                ),
            }
        )

        #
        # Plot stiffness and active-member
        # counts for this foundation.
        #
        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            alpha,
            stiffness,
            "o-",
            markersize=3,
            label="effective stiffness",
        )

        plt.axvline(
            optimal_alpha,
            linestyle="--",
            label="stiffness optimum",
        )

        for transition in transitions:
            transition_alpha = (
                0.5
                * (
                    transition[
                        "alpha_before"
                    ]
                    + transition[
                        "alpha_after"
                    ]
                )
            )

            plt.axvline(
                transition_alpha,
                linestyle=":",
                alpha=0.5,
            )

        plt.xlabel(
            "T4 prestress scale"
        )

        plt.ylabel(
            "Effective vertical stiffness"
        )

        plt.title(
            "Unilateral stiffness and "
            f"active-set transitions, k_f={foundation_k:g}"
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend()
        plt.tight_layout()

        path = (
            OUTPUT_DIR
            / (
                "stiffness_active_transitions_"
                f"kf_{foundation_k:g}.png"
            )
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()

    #
    # Save summary.
    #
    summary_path = (
        OUTPUT_DIR
        / "active_set_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "optimal_alpha",
                "maximum_stiffness",
                "active_cables_at_peak",
                "slack_cables_at_peak",
                "active_bars_at_peak",
                "slack_bars_at_peak",
                "threshold_members_at_peak",
                "active_set_fixed_across_peak",
                "nearest_transition_distance",
                "transition_count",
                "interpretation",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print()
    print(
        f"Saved: {summary_path}"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
