import numpy as np
import pytest

from tensegrity.viscoelastic import (
    foundation_incremental_potential_and_gradient,
    kelvin_voigt_foundation_force,
)


def test_kelvin_voigt_force_matches_kx_plus_cv():
    reference = np.zeros((2, 3))
    nodes = reference.copy()
    nodes[1, 0] = 0.20
    velocity = np.zeros_like(nodes)
    velocity[1, 0] = 0.30

    force = kelvin_voigt_foundation_force(
        nodes,
        velocity,
        reference,
        support_nodes=[1],
        stiffness=4.0,
        damping=2.0,
    )

    assert force[1, 0] == pytest.approx(-(4.0 * 0.20 + 2.0 * 0.30))
    assert np.allclose(force[0], 0.0)


def test_incremental_gradient_matches_backward_euler_force_balance():
    reference = np.zeros((1, 3))
    previous = np.zeros((1, 3))
    nodes = np.array([[0.1, 0.0, 0.0]])

    potential, gradient = foundation_incremental_potential_and_gradient(
        nodes,
        previous,
        reference,
        support_nodes=[0],
        stiffness=3.0,
        damping=2.0,
        dt=0.5,
    )

    expected_grad = 3.0 * 0.1 + 2.0 * 0.1 / 0.5
    expected_potential = 0.5 * 3.0 * 0.1**2 + 0.5 * 2.0 / 0.5 * 0.1**2
    assert gradient[0, 0] == pytest.approx(expected_grad)
    assert potential == pytest.approx(expected_potential)


def test_incremental_potential_rejects_nonpositive_dt():
    X = np.zeros((1, 3))
    with pytest.raises(ValueError):
        foundation_incremental_potential_and_gradient(
            X, X, X, [0], stiffness=1.0, damping=1.0, dt=0.0
        )
