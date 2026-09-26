"""Locate the T4 prestress that maximizes composite vertical stiffness.

The central T4 carries its validated sign-compatible self-stress.
The two zero-twist T3 attachment units remain initially stress-free.
Support nodes are connected to isotropic foundation springs.

For representative foundation stiffnesses, this script scans T4 prestress

    alpha in [0, 0.8]

and determines

    alpha*(k_f) = argmax_alpha k_eff(alpha, k_f).

The reference geometry remains an exact equilibrium at every alpha because
the T4 force state is a self-stress and all attachment/foundation elements
are initially unstressed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensegrity.examples import (
    t4_with_two_supported_t3,
)

from experiments.t4_prestress_with_t3_foundation import (
    build_rest_lengths,
    response,
)


OUTPUT_DIR = Path(
    "outputs/t4_optimal_prestress"
)


def evaluate_case(
    reference_nodes,
    members,
    support_nodes,
    loaded_nodes,
    member_stiffness,
    foundation_k,
    prestress_scale,
):
    rest_lengths = build_rest_lengths(
        reference_nodes,
        members,
        float(prestress_scale),
        stiffness=member_stiffness,
    )

    (
        reference_energy,
        equilibrium_residual,
        lambda_min,
        mean_dz,
        effective_stiffness,
    ) = response(
        reference_nodes,
        members,
        rest_lengths,
        support_nodes,
        loaded_nodes,
        member_stiffness,
        float(foundation_k),
    )

    min_rest_length = float(
        np.min(rest_lengths)
    )

    return {
        "foundation_k": float(
            foundation_k
        ),
        "prestress_scale": float(
            prestress_scale
        ),
        "reference_energy": float(
            reference_energy
        ),
        "equilibrium_residual": float(
            equilibrium_residual
        ),
        "lambda_min": float(
            lambda_min
        ),
        "mean_dz": float(
            mean_dz
        ),
        "effective_stiffness": float(
            effective_stiffness
        ),
        "min_rest_length": (
            min_rest_length
        ),
    }


def quadratic_peak(
    x,
    y,
    index,
):
    """Estimate a local maximum from three neighboring samples.

    Returns NaN if the maximum is at the scan boundary or the fitted
    parabola does not have negative curvature.
    """
    if (
        index <= 0
        or index >= len(x) - 1
    ):
        return float("nan")

    xs = np.asarray(
        x[
            index - 1:
            index + 2
        ],
        dtype=float,
    )

    ys = np.asarray(
        y[
            index - 1:
            index + 2
        ],
        dtype=float,
    )

    a, b, c = np.polyfit(
        xs,
        ys,
        deg=2,
    )

    if a >= 0.0:
        return float("nan")

    peak = (
        -b
        / (
            2.0 * a
        )
    )

    #
    # Keep refinement local.
    #
    if not (
        xs[0]
        <= peak
        <= xs[-1]
    ):
        return float("nan")

    return float(
        peak
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

    member_stiffness = 1.0

    foundation_values = np.asarray(
        [
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        dtype=float,
    )

    #
    # alpha must remain below stiffness=1.
    # 0.8 gives us a generous first scan
    # while keeping all derived rest lengths
    # strictly positive.
    #
    prestress_values = np.linspace(
        0.0,
        0.8,
        81,
    )

    scan_rows = []
    optimum_rows = []

    print()
    print("=" * 112)
    print(
        "T4 OPTIMAL PRESTRESS SEARCH"
    )
    print("=" * 112)

    for foundation_k in foundation_values:
        case_rows = []

        for alpha in prestress_values:
            row = evaluate_case(
                reference_nodes,
                members,
                support_nodes,
                loaded_nodes,
                member_stiffness,
                foundation_k,
                alpha,
            )

            scan_rows.append(
                row
            )

            case_rows.append(
                row
            )

        stiffness = np.asarray(
            [
                row[
                    "effective_stiffness"
                ]
                for row in case_rows
            ],
            dtype=float,
        )

        lambda_values = np.asarray(
            [
                row[
                    "lambda_min"
                ]
                for row in case_rows
            ],
            dtype=float,
        )

        residual_values = np.asarray(
            [
                row[
                    "equilibrium_residual"
                ]
                for row in case_rows
            ],
            dtype=float,
        )

        max_index = int(
            np.nanargmax(
                stiffness
            )
        )

        discrete_alpha = float(
            prestress_values[
                max_index
            ]
        )

        discrete_max = float(
            stiffness[
                max_index
            ]
        )

        baseline = float(
            stiffness[0]
        )

        refined_alpha = quadratic_peak(
            prestress_values,
            stiffness,
            max_index,
        )

        if np.isfinite(
            refined_alpha
        ):
            refined = evaluate_case(
                reference_nodes,
                members,
                support_nodes,
                loaded_nodes,
                member_stiffness,
                foundation_k,
                refined_alpha,
            )

            #
            # Numerical noise could make the
            # refined evaluation microscopically
            # worse than the discrete sample.
            #
            if (
                refined[
                    "effective_stiffness"
                ]
                >= discrete_max
            ):
                optimum = (
                    refined
                )
            else:
                optimum = (
                    case_rows[
                        max_index
                    ]
                )
        else:
            optimum = (
                case_rows[
                    max_index
                ]
            )

        normalized_gain = (
            optimum[
                "effective_stiffness"
            ]
            / baseline
        )

        boundary_peak = bool(
            max_index == 0
            or max_index
            == len(
                prestress_values
            ) - 1
        )

        optimum_row = {
            "foundation_k": (
                foundation_k
            ),
            "baseline_stiffness": (
                baseline
            ),
            "discrete_optimal_prestress": (
                discrete_alpha
            ),
            "discrete_max_stiffness": (
                discrete_max
            ),
            "optimal_prestress": (
                optimum[
                    "prestress_scale"
                ]
            ),
            "maximum_effective_stiffness": (
                optimum[
                    "effective_stiffness"
                ]
            ),
            "normalized_maximum_gain": (
                normalized_gain
            ),
            "lambda_min_at_optimum": (
                optimum[
                    "lambda_min"
                ]
            ),
            "equilibrium_residual_at_optimum": (
                optimum[
                    "equilibrium_residual"
                ]
            ),
            "minimum_rest_length_at_optimum": (
                optimum[
                    "min_rest_length"
                ]
            ),
            "peak_at_scan_boundary": (
                boundary_peak
            ),
        }

        optimum_rows.append(
            optimum_row
        )

        print()
        print(
            f"foundation k = "
            f"{foundation_k:.3e}"
        )

        print(
            f"  baseline stiffness:          "
            f"{baseline:.12e}"
        )

        print(
            f"  discrete peak alpha:         "
            f"{discrete_alpha:.6f}"
        )

        print(
            f"  refined optimal alpha:       "
            f"{optimum['prestress_scale']:.12e}"
        )

        print(
            f"  maximum effective stiffness: "
            f"{optimum['effective_stiffness']:.12e}"
        )

        print(
            f"  normalized maximum gain:     "
            f"{normalized_gain:.12f}"
        )

        print(
            f"  lambda_min at optimum:       "
            f"{optimum['lambda_min']:.12e}"
        )

        print(
            f"  equilibrium residual:        "
            f"{optimum['equilibrium_residual']:.12e}"
        )

        print(
            f"  minimum rest length:         "
            f"{optimum['min_rest_length']:.12e}"
        )

        print(
            f"  peak at scan boundary:       "
            f"{boundary_peak}"
        )

        print(
            f"  minimum lambda over scan:    "
            f"{np.nanmin(lambda_values):.12e}"
        )

        print(
            f"  worst equilibrium residual:  "
            f"{np.nanmax(residual_values):.12e}"
        )

    #
    # Save complete scan.
    #
    scan_path = (
        OUTPUT_DIR
        / "prestress_scan.csv"
    )

    with scan_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "prestress_scale",
                "reference_energy",
                "equilibrium_residual",
                "lambda_min",
                "mean_dz",
                "effective_stiffness",
                "min_rest_length",
            ],
        )

        writer.writeheader()
        writer.writerows(
            scan_rows
        )

    #
    # Save optimum summary.
    #
    optimum_path = (
        OUTPUT_DIR
        / "optimal_prestress_summary.csv"
    )

    with optimum_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "foundation_k",
                "baseline_stiffness",
                "discrete_optimal_prestress",
                "discrete_max_stiffness",
                "optimal_prestress",
                "maximum_effective_stiffness",
                "normalized_maximum_gain",
                "lambda_min_at_optimum",
                "equilibrium_residual_at_optimum",
                "minimum_rest_length_at_optimum",
                "peak_at_scan_boundary",
            ],
        )

        writer.writeheader()
        writer.writerows(
            optimum_rows
        )

    print()
    print(
        f"Saved: {scan_path}"
    )

    print(
        f"Saved: {optimum_path}"
    )

    #
    # Plot stiffness curves and optima.
    #
    plt.figure(
        figsize=(8, 5)
    )

    for foundation_k in foundation_values:
        selected = [
            row
            for row in scan_rows
            if np.isclose(
                row[
                    "foundation_k"
                ],
                foundation_k,
            )
        ]

        alpha = np.asarray(
            [
                row[
                    "prestress_scale"
                ]
                for row in selected
            ]
        )

        stiffness = np.asarray(
            [
                row[
                    "effective_stiffness"
                ]
                for row in selected
            ]
        )

        plt.plot(
            alpha,
            stiffness,
            label=(
                f"k_f={foundation_k:g}"
            ),
        )

        optimum = next(
            row
            for row in optimum_rows
            if np.isclose(
                row[
                    "foundation_k"
                ],
                foundation_k,
            )
        )

        plt.scatter(
            [
                optimum[
                    "optimal_prestress"
                ]
            ],
            [
                optimum[
                    "maximum_effective_stiffness"
                ]
            ],
            s=40,
        )

    plt.xlabel(
        "T4 prestress scale"
    )

    plt.ylabel(
        "Effective vertical stiffness"
    )

    plt.title(
        "Optimal T4 prestress vs foundation stiffness"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    curve_path = (
        OUTPUT_DIR
        / "effective_stiffness_optima.png"
    )

    plt.savefig(
        curve_path,
        dpi=200,
    )

    plt.close()

    #
    # Plot alpha* vs foundation stiffness.
    #
    plt.figure(
        figsize=(7, 5)
    )

    k_opt = np.asarray(
        [
            row[
                "foundation_k"
            ]
            for row in optimum_rows
        ]
    )

    alpha_opt = np.asarray(
        [
            row[
                "optimal_prestress"
            ]
            for row in optimum_rows
        ]
    )

    plt.semilogx(
        k_opt,
        alpha_opt,
        "o-",
    )

    plt.xlabel(
        "Foundation stiffness"
    )

    plt.ylabel(
        "Optimal T4 prestress scale"
    )

    plt.title(
        "Prestress optimum vs foundation stiffness"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    optimum_plot = (
        OUTPUT_DIR
        / "optimal_prestress_vs_foundation.png"
    )

    plt.savefig(
        optimum_plot,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {curve_path}"
    )

    print(
        f"Saved: {optimum_plot}"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
