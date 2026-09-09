import numpy as np

from tensegrity.energy import (
    SpringMember,
    nonrigid_basis,
    rigid_body_basis,
    stability_index_energy_hessian,
    total_energy,
)


def tetrahedron_nodes() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.25, 0.95, 0.0],
            [0.30, 0.35, 0.90],
        ],
        dtype=float,
    )


def tetrahedron_springs(nodes: np.ndarray):
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return [
        SpringMember(
            i=i,
            j=j,
            kind="bar",
            k=1.0,
            L0=float(np.linalg.norm(nodes[i] - nodes[j])),
        )
        for i, j in edges
    ]


def test_energy_is_invariant_under_translation_and_rotation():
    nodes = tetrahedron_nodes()
    members = tetrahedron_springs(nodes)
    strained = nodes.copy()
    strained[3] += np.array([0.08, -0.03, 0.04])
    baseline = total_energy(strained, members)

    translated = strained + np.array([2.4, -1.7, 0.9])
    assert np.isclose(total_energy(translated, members), baseline, atol=1e-12)

    theta = 0.73
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = strained @ rotation.T
    assert np.isclose(total_energy(rotated, members), baseline, atol=1e-12)


def test_generic_tetrahedron_has_six_rigid_modes():
    nodes = tetrahedron_nodes()
    rigid = rigid_body_basis(nodes)
    nonrigid = nonrigid_basis(nodes)

    assert rigid.shape == (12, 6)
    assert nonrigid.shape == (12, 6)
    assert np.allclose(rigid.T @ rigid, np.eye(6), atol=1e-12)
    assert np.allclose(nonrigid.T @ nonrigid, np.eye(6), atol=1e-12)
    assert np.allclose(rigid.T @ nonrigid, 0.0, atol=1e-12)


def test_unprestressed_tetrahedron_is_locally_stable():
    nodes = tetrahedron_nodes()
    members = tetrahedron_springs(nodes)
    result = stability_index_energy_hessian(
        nodes, members, eps=2e-5, eig_tol=1e-6
    )

    assert result.classification == "stable"
    assert result.negative_modes == 0
    assert result.zero_modes == 0
    assert result.positive_modes == 6
    assert result.lambda_min > 1e-4


def test_missing_tetrahedron_edge_is_marginal_not_numerically_unstable():
    nodes = tetrahedron_nodes()
    members = tetrahedron_springs(nodes)[:-1]
    result = stability_index_energy_hessian(
        nodes, members, eps=2e-5, eig_tol=1e-6
    )

    assert result.classification == "marginal"
    assert result.negative_modes == 0
    assert result.zero_modes >= 1
