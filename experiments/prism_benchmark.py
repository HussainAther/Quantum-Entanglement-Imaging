"""Canonical equilibrated 3-strut tensegrity prism benchmark."""

from tensegrity.energy import equilibrium_residual_norm, stability_index_energy_hessian
from tensegrity.examples import three_strut_prism
from tensegrity.rigidity import analyze_rigidity


def main() -> None:
    nodes, rigidity_members, spring_members = three_strut_prism(prestress_scale=0.20)
    rigidity = analyze_rigidity(nodes, rigidity_members)
    residual = equilibrium_residual_norm(nodes, spring_members)
    stability = stability_index_energy_hessian(
        nodes, spring_members, eps=2e-5, eig_tol=1e-6
    )

    print("=== 3-Strut Tensegrity Prism ===")
    print("\nGeometry")
    print(f"Nodes:   {len(nodes)}")
    print(f"Members: {len(rigidity_members)}")
    print(f"Struts:  {sum(m.kind == 'bar' for m in rigidity_members)}")
    print(f"Cables:  {sum(m.kind == 'cable' for m in rigidity_members)}")

    print("\nRigidity")
    print(f"Rank:                {rigidity.rank}")
    print(f"Nullity:             {rigidity.nullity}")
    print(f"Rigid-body modes:    {rigidity.rigid_body_modes}")
    print(f"Mechanisms:          {rigidity.mechanisms}")
    print(f"Self-stress states:  {rigidity.self_stress_dimension}")

    print("\nEquilibrium")
    print(f"||dE/dx||:           {residual:.6e}")

    print("\nEnergy-Hessian stability")
    print(f"lambda_min:          {stability.lambda_min:.6e}")
    print(f"Negative modes:      {stability.negative_modes}")
    print(f"Zero modes:          {stability.zero_modes}")
    print(f"Positive modes:      {stability.positive_modes}")
    print(f"Classification:      {stability.classification}")


if __name__ == "__main__":
    main()
