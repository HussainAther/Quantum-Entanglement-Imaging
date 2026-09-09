"""Single-cable ablation diagnostics for the canonical prism.

The intact prism is an equilibrium configuration. Removing a prestressed cable
usually destroys that force balance at the *same geometry*. Therefore the
post-ablation Hessian reported here is explicitly an instantaneous curvature
diagnostic, not the stability of a relaxed post-failure equilibrium.
"""

from __future__ import annotations

import csv
from pathlib import Path

from tensegrity.energy import equilibrium_residual_norm, stability_index_energy_hessian
from tensegrity.examples import three_strut_prism
from tensegrity.rigidity import analyze_rigidity


def main() -> None:
    nodes, members, springs = three_strut_prism(prestress_scale=0.20)
    base_rigidity = analyze_rigidity(nodes, members)
    base_residual = equilibrium_residual_norm(nodes, springs)
    base_stability = stability_index_energy_hessian(
        nodes, springs, eps=2e-5, eig_tol=1e-6
    )

    rows = []
    for idx, member in enumerate(members):
        if member.kind != "cable":
            continue
        reduced_members = members[:idx] + members[idx + 1 :]
        reduced_springs = springs[:idx] + springs[idx + 1 :]
        rigidity = analyze_rigidity(nodes, reduced_members)
        residual = equilibrium_residual_norm(nodes, reduced_springs)
        curvature = stability_index_energy_hessian(
            nodes, reduced_springs, eps=2e-5, eig_tol=1e-6
        )
        rows.append(
            {
                "removed_index": idx,
                "removed_i": member.i,
                "removed_j": member.j,
                "base_mechanisms": base_rigidity.mechanisms,
                "mechanisms_after": rigidity.mechanisms,
                "base_self_stress": base_rigidity.self_stress_dimension,
                "self_stress_after": rigidity.self_stress_dimension,
                "base_equilibrium_residual": base_residual,
                "post_ablation_equilibrium_residual": residual,
                "base_lambda_min": base_stability.lambda_min,
                "instantaneous_lambda_min_after": curvature.lambda_min,
                "instantaneous_curvature_classification": curvature.classification,
            }
        )
        print(
            f"remove cable {idx:2d} ({member.i},{member.j}): "
            f"mechanisms {base_rigidity.mechanisms}->{rigidity.mechanisms}, "
            f"self-stress {base_rigidity.self_stress_dimension}->{rigidity.self_stress_dimension}, "
            f"residual={residual:.3e}, "
            f"instantaneous lambda_min={curvature.lambda_min:.6e}"
        )

    output = Path("outputs/prism_member_ablation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
