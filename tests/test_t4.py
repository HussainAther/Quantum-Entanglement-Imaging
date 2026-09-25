import numpy as np

from tensegrity.examples import mirrored_t4_prism
from tensegrity.rigidity import (
    analyze_rigidity,
    rigidity_matrix,
)


from tensegrity.energy import equilibrium_residual_norm
from tensegrity.examples import equilibrated_mirrored_t4
from tensegrity.examples import t4_with_one_supported_t3

def test_mirrored_t4_proxy_has_expected_shapes():
    nodes, members = mirrored_t4_prism()

    assert nodes.shape == (9, 3)
    assert len(members) == 21

    bars = [
        member
        for member in members
        if member.kind == "bar"
    ]

    cables = [
        member
        for member in members
        if member.kind == "cable"
    ]

    assert len(bars) == 6
    assert len(cables) == 15


def test_mirrored_t4_rigidity_matrix_shape():
    nodes, members = mirrored_t4_prism()

    R = rigidity_matrix(
        nodes,
        members,
    )

    assert R.shape == (
        len(members),
        3 * len(nodes),
    )


def test_mirrored_t4_analysis_is_numerically_consistent():
    nodes, members = mirrored_t4_prism()

    result = analyze_rigidity(
        nodes,
        members,
    )

    assert result.rank <= len(members)

    assert (
        result.nullity
        == 3 * len(nodes) - result.rank
    )

    assert (
        result.self_stress_dimension
        == len(members) - result.rank
    )

    assert result.mechanisms >= 0



def test_equilibrated_mirrored_t4_is_in_force_balance():
    nodes, _, springs = equilibrated_mirrored_t4(
        prestress_scale=0.20,
        stiffness=1.0,
    )

    residual = equilibrium_residual_norm(
        nodes,
        springs,
    )

    assert residual < 1e-10



def test_t4_with_one_supported_t3_shapes():
    nodes, members = (
        t4_with_one_supported_t3()
    )

    assert nodes.shape == (
        12,
        3,
    )

    assert len(members) == 30
