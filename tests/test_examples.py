import numpy as np

from tensegrity.energy import equilibrium_residual_norm, stability_index_energy_hessian
from tensegrity.examples import prism_force_density, tetrahedron, three_strut_prism
from tensegrity.rigidity import analyze_rigidity, rigidity_matrix


def test_tetrahedron_example_builds():
    nodes, members, springs = tetrahedron()
    assert nodes.shape == (4, 3)
    assert len(members) == 6
    assert len(springs) == 6


def test_three_strut_prism_builds_with_expected_topology():
    nodes, members, springs = three_strut_prism()
    assert nodes.shape == (6, 3)
    assert len(members) == 12
    assert len(springs) == 12
    assert sum(m.kind == "bar" for m in members) == 3
    assert sum(m.kind == "cable" for m in members) == 9


def test_three_strut_prism_has_one_mechanism_and_one_self_stress():
    nodes, members, _ = three_strut_prism()
    result = analyze_rigidity(nodes, members)
    assert result.rank == 11
    assert result.nullity == 7
    assert result.rigid_body_modes == 6
    assert result.mechanisms == 1
    assert result.self_stress_dimension == 1


def test_prism_force_density_is_in_equilibrium_with_physical_signs():
    nodes, members, _ = three_strut_prism()
    q = prism_force_density(nodes, members, prestress_scale=0.20)
    assert np.linalg.norm(rigidity_matrix(nodes, members).T @ q) < 1e-12
    assert np.all(q[:3] < 0.0)
    assert np.all(q[3:] > 0.0)


def test_three_strut_prism_reference_geometry_is_equilibrated():
    nodes, _, springs = three_strut_prism(prestress_scale=0.20)
    assert equilibrium_residual_norm(nodes, springs) < 1e-10


def test_prestressed_three_strut_prism_is_locally_stable():
    nodes, _, springs = three_strut_prism(prestress_scale=0.20)
    result = stability_index_energy_hessian(nodes, springs, eps=2e-5, eig_tol=1e-6)
    assert result.classification == "stable"
    assert result.negative_modes == 0
    assert result.zero_modes == 0
    assert result.lambda_min > 0.0
