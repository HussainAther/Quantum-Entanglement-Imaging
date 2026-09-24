"""Visualize relaxed prism soft modes.

Reads the node/mode CSV files produced by prism_soft_modes.py and draws
the relaxed framework together with the infinitesimal displacement field.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path("outputs/prism_soft_modes")
OUTPUT_DIR = INPUT_DIR


def load_mode(
    removed_idx: int,
    mode_idx: int,
):
    path = (
        INPUT_DIR
        / f"remove_{removed_idx:02d}_mode{mode_idx}.csv"
    )

    rows = []

    with path.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    nodes = np.array(
        [
            [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ]
            for row in rows
        ],
        dtype=float,
    )

    mode = np.array(
        [
            [
                float(row["dx"]),
                float(row["dy"]),
                float(row["dz"]),
            ]
            for row in rows
        ],
        dtype=float,
    )

    return nodes, mode


def intact_edges():
    """Canonical prism member topology."""

    struts = [
        (0, 4),
        (1, 5),
        (2, 3),
    ]

    perimeter = [
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
    ]

    side = [
        (0, 3),
        (1, 4),
        (2, 5),
    ]

    return struts, perimeter, side


def edge_for_removed_index(index: int):
    all_members = [
        (0, 4),
        (1, 5),
        (2, 3),
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    ]

    return all_members[index]


def equal_axes(ax, nodes):
    center = nodes.mean(axis=0)

    ranges = np.ptp(
        nodes,
        axis=0,
    )

    radius = 0.6 * max(
        float(np.max(ranges)),
        1e-8,
    )

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )

    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )

    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )


def plot_mode(
    removed_idx: int,
    mode_idx: int,
    scale: float = 0.8,
):
    nodes, mode = load_mode(
        removed_idx,
        mode_idx,
    )

    struts, perimeter, side = intact_edges()

    removed_edge = edge_for_removed_index(
        removed_idx
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    #
    # Draw remaining framework.
    #
    for edge_group in (
        struts,
        perimeter,
        side,
    ):
        for i, j in edge_group:
            if (i, j) == removed_edge:
                continue

            ax.plot(
                [
                    nodes[i, 0],
                    nodes[j, 0],
                ],
                [
                    nodes[i, 1],
                    nodes[j, 1],
                ],
                [
                    nodes[i, 2],
                    nodes[j, 2],
                ],
                linewidth=1.5,
                alpha=0.7,
            )

    #
    # Removed member: dashed.
    #
    i, j = removed_edge

    ax.plot(
        [
            nodes[i, 0],
            nodes[j, 0],
        ],
        [
            nodes[i, 1],
            nodes[j, 1],
        ],
        [
            nodes[i, 2],
            nodes[j, 2],
        ],
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )

    #
    # Nodes.
    #
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        s=45,
    )

    for node_idx, point in enumerate(nodes):
        ax.text(
            point[0],
            point[1],
            point[2],
            f" {node_idx}",
        )

    #
    # Mode vectors.
    #
    ax.quiver(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        mode[:, 0],
        mode[:, 1],
        mode[:, 2],
        length=scale,
        normalize=False,
    )

    #
    # Also show displaced node positions.
    #
    displaced = (
        nodes
        + scale * mode
    )

    ax.scatter(
        displaced[:, 0],
        displaced[:, 1],
        displaced[:, 2],
        marker="x",
        s=40,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(
        f"Cable removal {removed_idx}: "
        f"soft mode {mode_idx}"
    )

    equal_axes(
        ax,
        np.vstack(
            [
                nodes,
                displaced,
            ]
        ),
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / (
            f"remove_{removed_idx:02d}"
            f"_mode{mode_idx}.png"
        )
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def main():
    #
    # One-soft-mode failure family.
    #
    plot_mode(
        3,
        1,
    )

    #
    # The second mode for removal 3
    # is intentionally included as
    # a non-soft comparison.
    #
    plot_mode(
        3,
        2,
    )

    #
    # Two-soft-mode failure family.
    #
    plot_mode(
        9,
        1,
    )

    plot_mode(
        9,
        2,
    )


if __name__ == "__main__":
    main()
