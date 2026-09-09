from itertools import combinations

import numpy as np

from tensegrity.rigidity import (
    Member,
    analyze_rigidity,
    mechanism_dimension,
    remove_one_cable_scan,
    rigidity_matrix,
    rigidity_rank,
    self_stress_dimension,
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


def tetrahedron_members(kind="bar"):
    return [Member(i, j, kind=kind) for i, j in combinations(range(4), 2)]


def test_rigidity_matrix_shape_and_translation_null_modes():
    nodes = tetrahedron_nodes()
    members = tetrahedron_members()
    R = rigidity_matrix(nodes, members)

    assert R.shape == (6, 12)
    for axis in range(3):
        translation = np.zeros(12)
        translation[axis::3] = 1.0
        assert np.allclose(R @ translation, 0.0, atol=1e-12)


def test_tetrahedron_is_infinitesimally_rigid_without_self_stress():
    nodes = tetrahedron_nodes()
    members = tetrahedron_members()
    result = analyze_rigidity(nodes, members)

    assert rigidity_rank(nodes, members) == 6
    assert result.rank == 6
    assert result.rigid_body_modes == 6
    assert result.mechanisms == 0
    assert result.self_stress_dimension == 0
    assert mechanism_dimension(nodes, members) == 0
    assert self_stress_dimension(nodes, members) == 0


def test_removing_one_tetrahedron_edge_introduces_one_mechanism():
    nodes = tetrahedron_nodes()
    members = tetrahedron_members()
    reduced = members[:-1]

    assert mechanism_dimension(nodes, reduced) == 1
    assert self_stress_dimension(nodes, reduced) == 0


def test_complete_graph_k5_has_one_state_of_self_stress():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 1.1, 0.0],
            [0.1, 0.3, 1.0],
            [0.8, 0.6, 0.7],
        ],
        dtype=float,
    )
    members = [Member(i, j, kind="bar") for i, j in combinations(range(5), 2)]
    result = analyze_rigidity(nodes, members)

    assert result.rank == 9
    assert result.mechanisms == 0
    assert result.self_stress_dimension == 1


def test_cable_removal_scan_reports_new_mechanism():
    nodes = tetrahedron_nodes()
    members = tetrahedron_members(kind="cable")
    scan = remove_one_cable_scan(nodes, members)

    assert len(scan) == 6
    for _, _, before, after in scan:
        assert before == 0
        assert after == 1
