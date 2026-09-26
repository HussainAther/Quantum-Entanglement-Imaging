import numpy as np

from tensegrity.energy import (
    SpringMember,
    unilateral_energy_gradient,
    unilateral_total_energy,
)


def test_cable_carries_tension_only():
    cable = [
        SpringMember(
            i=0,
            j=1,
            kind="cable",
            k=1.0,
            L0=1.0,
        )
    ]

    stretched = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]
    )

    compressed = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )

    assert unilateral_total_energy(
        stretched,
        cable,
    ) > 0.0

    assert unilateral_total_energy(
        compressed,
        cable,
    ) == 0.0


def test_bar_carries_compression_only():
    bar = [
        SpringMember(
            i=0,
            j=1,
            kind="bar",
            k=1.0,
            L0=1.0,
        )
    ]

    compressed = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )

    stretched = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]
    )

    assert unilateral_total_energy(
        compressed,
        bar,
    ) > 0.0

    assert unilateral_total_energy(
        stretched,
        bar,
    ) == 0.0


def test_unilateral_gradient_zero_for_slack_members():
    members = [
        SpringMember(
            i=0,
            j=1,
            kind="cable",
            k=1.0,
            L0=1.0,
        )
    ]

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )

    grad = unilateral_energy_gradient(
        nodes,
        members,
    )

    assert np.allclose(
        grad,
        0.0,
    )
