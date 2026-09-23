import numpy as np

from tensegrity.energy import SpringMember
from tensegrity.examples import three_strut_prism
from tensegrity.relaxation import (
    max_aligned_displacement,
    relax_framework,
    rigid_align_to_reference,
)


def test_rigid_alignment_removes_translation_and_rotation():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.1],
            [0.1, 0.2, 0.9],
        ],
        dtype=float,
    )

    theta = 0.6

    rotation = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta),
                0.0,
            ],
            [
                np.sin(theta),
                np.cos(theta),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    moved = (
        reference @ rotation.T
        + np.array(
            [2.0, -3.0, 1.5]
        )
    )

    aligned = rigid_align_to_reference(
        moved,
        reference,
    )

    assert np.allclose(
        aligned,
        reference,
        atol=1e-12,
    )

    assert (
        max_aligned_displacement(
            moved,
            reference,
        )
        < 1e-12
    )


def test_relaxation_leaves_equilibrated_prism_unchanged():
    nodes, _, springs = three_strut_prism(
        prestress_scale=0.20
    )

    result = relax_framework(
        nodes,
        springs,
        residual_tol=1e-10,
    )

    assert result.converged
    assert result.iterations == 0
    assert result.residual_final <= 1e-10
    assert result.max_displacement == 0.0


def test_relaxation_reduces_energy_and_residual_for_perturbed_spring():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=float,
    )

    springs = [
        SpringMember(
            i=0,
            j=1,
            kind="bar",
            k=1.0,
            L0=1.0,
        )
    ]

    result = relax_framework(
        reference,
        springs,
        residual_tol=1e-9,
        max_iterations=2000,
        initial_step_size=0.2,
    )

    assert result.converged

    assert (
        result.energy_final
        < result.energy_initial
    )

    assert (
        result.residual_final
        < result.residual_initial
    )

    final_length = np.linalg.norm(
        result.nodes[0]
        - result.nodes[1]
    )

    assert np.isclose(
        final_length,
        1.0,
        atol=1e-8,
    )
