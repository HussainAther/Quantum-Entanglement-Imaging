"""Plot representative prism singularity trajectories.

Shows one perimeter-cable removal and one side-cable removal so the
one-soft-mode and two-soft-mode approaches can be compared directly.

Uses only NumPy and Matplotlib.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path(
    "outputs/prism_singularity_trajectory"
)


def load_trajectory(index: int):
    path = INPUT_DIR / f"remove_{index:02d}.csv"

    iteration = []
    residual = []
    sigma_min = []
    sigma_second = []
    q_norm = []
    energy = []

    with path.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            iteration.append(
                int(row["iteration"])
            )

            residual.append(
                float(row["residual"])
            )

            sigma_min.append(
                float(row["sigma_min"])
            )

            sigma_second.append(
                float(row["sigma_second_min"])
            )

            q_norm.append(
                float(row["q_norm"])
            )

            energy.append(
                float(row["energy"])
            )

    return {
        "iteration": np.asarray(iteration),
        "residual": np.asarray(residual),
        "sigma_min": np.asarray(sigma_min),
        "sigma_second": np.asarray(sigma_second),
        "q_norm": np.asarray(q_norm),
        "energy": np.asarray(energy),
    }


def plot_singular_values(
    index: int,
    label: str,
) -> None:
    data = load_trajectory(index)

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        data["iteration"],
        data["sigma_min"],
        label=r"$\sigma_1$ (smallest)",
    )

    plt.semilogy(
        data["iteration"],
        data["sigma_second"],
        label=r"$\sigma_2$ (second-smallest)",
    )

    plt.semilogy(
        data["iteration"],
        data["residual"],
        label=r"$\|\nabla E\|$",
    )

    plt.xlabel("Relaxation iteration")
    plt.ylabel("Magnitude")

    plt.title(
        f"Prism cable removal {index}: {label}"
    )

    plt.legend()
    plt.grid(
        True,
        which="both",
        alpha=0.25,
    )

    plt.tight_layout()

    output = (
        INPUT_DIR
        / f"remove_{index:02d}_singularity.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output}")


def plot_normalized_comparison() -> None:
    """Compare representative soft modes on a common normalized scale."""

    perimeter = load_trajectory(3)
    side = load_trajectory(9)

    plt.figure(figsize=(8, 5))

    p_sigma1 = (
        perimeter["sigma_min"]
        / perimeter["sigma_min"][0]
    )

    p_sigma2 = (
        perimeter["sigma_second"]
        / perimeter["sigma_second"][0]
    )

    s_sigma1 = (
        side["sigma_min"]
        / side["sigma_min"][0]
    )

    s_sigma2 = (
        side["sigma_second"]
        / side["sigma_second"][0]
    )

    plt.semilogy(
        perimeter["iteration"],
        p_sigma1,
        label="remove 3: sigma1",
    )

    plt.semilogy(
        perimeter["iteration"],
        p_sigma2,
        label="remove 3: sigma2",
    )

    plt.semilogy(
        side["iteration"],
        s_sigma1,
        label="remove 9: sigma1",
    )

    plt.semilogy(
        side["iteration"],
        s_sigma2,
        label="remove 9: sigma2",
    )

    plt.xlabel("Relaxation iteration")
    plt.ylabel(
        "Singular value / initial singular value"
    )

    plt.title(
        "Normalized singular-mode evolution"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        alpha=0.25,
    )

    plt.tight_layout()

    output = (
        INPUT_DIR
        / "representative_mode_comparison.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output}")


def print_tail_scaling(
    index: int,
    tail_fraction: float = 0.25,
) -> None:
    """Estimate scaling relationships late in the trajectory.

    We fit log(residual) against log(sigma) over the final portion of
    the trajectory. This is exploratory rather than a formal asymptotic
    analysis.
    """
    data = load_trajectory(index)

    n = len(data["iteration"])

    start = max(
        0,
        int(
            (1.0 - tail_fraction) * n
        ),
    )

    residual = data["residual"][start:]
    sigma1 = data["sigma_min"][start:]
    sigma2 = data["sigma_second"][start:]

    mask1 = (
        np.isfinite(residual)
        & np.isfinite(sigma1)
        & (residual > 0.0)
        & (sigma1 > 0.0)
    )

    slope1, intercept1 = np.polyfit(
        np.log(sigma1[mask1]),
        np.log(residual[mask1]),
        1,
    )

    mask2 = (
        np.isfinite(residual)
        & np.isfinite(sigma2)
        & (residual > 0.0)
        & (sigma2 > 0.0)
    )

    slope2, intercept2 = np.polyfit(
        np.log(sigma2[mask2]),
        np.log(residual[mask2]),
        1,
    )

    print()
    print(
        f"Removal {index} late-trajectory scaling"
    )

    print(
        f"  residual ~ sigma1^{slope1:.4f}"
    )

    print(
        f"  residual ~ sigma2^{slope2:.4f}"
    )

    print(
        f"  final residual = "
        f"{data['residual'][-1]:.6e}"
    )

    print(
        f"  final sigma1   = "
        f"{data['sigma_min'][-1]:.6e}"
    )

    print(
        f"  final sigma2   = "
        f"{data['sigma_second'][-1]:.6e}"
    )

    print(
        f"  final q norm   = "
        f"{data['q_norm'][-1]:.6e}"
    )


def main() -> None:
    plot_singular_values(
        3,
        "perimeter-cable failure",
    )

    plot_singular_values(
        9,
        "side-cable failure",
    )

    plot_normalized_comparison()

    print_tail_scaling(3)
    print_tail_scaling(9)


if __name__ == "__main__":
    main()
