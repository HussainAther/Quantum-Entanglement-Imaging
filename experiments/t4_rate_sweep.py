"""Rate-dependent loading of the unilateral T4 + two-T3 composite.

This experiment keeps the tensegrity constitutive law unchanged:

* cables carry tension only;
* bars/struts carry compression only.

Rate dependence is introduced only through the compliant basement/foundation
attachment, modeled as a Kelvin-Voigt spring-dashpot connection at the support
nodes.  Internal tensegrity nodes equilibrate quasi-statically at each time
step, while the basement attachment obeys a backward-Euler viscous term.

This is a deliberately small first dynamic extension.  It is intended to
answer one question cleanly: if the same final load is applied quickly or
slowly, does the supported tensegrity follow a different force-displacement
path?

All values are currently normalized/model units.  They should not be read as
physical salamander tissue parameters until experimental or literature values
are supplied.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tensegrity.energy import (
    unilateral_active_state,
    unilateral_energy_gradient,
    unilateral_total_energy,
)
from tensegrity.examples import t4_with_two_supported_t3
from tensegrity.loading import smoothstep_ramp, smoothstep_ramp_rate
from tensegrity.viscoelastic import (
    foundation_incremental_potential_and_gradient,
    kelvin_voigt_foundation_force,
)

from experiments.t4_unilateral_optimal_prestress import (
    build_unilateral_springs,
    normalized_t4_force_density,
)


OUTPUT_DIR = Path("outputs/t4_rate_sweep")


def load_array(node_count, loaded_nodes, force_per_node):
    force = np.zeros((node_count, 3), dtype=float)
    for node in loaded_nodes:
        force[node, 2] = force_per_node
    return force


def solve_increment(
    initial_nodes,
    previous_nodes,
    reference_nodes,
    springs,
    support_nodes,
    loaded_nodes,
    foundation_k,
    foundation_c,
    dt,
    force_per_node,
):
    shape = reference_nodes.shape
    external_force = load_array(
        reference_nodes.shape[0],
        loaded_nodes,
        force_per_node,
    )

    def objective(flat):
        X = flat.reshape(shape)

        structure_energy = unilateral_total_energy(X, springs)
        structure_gradient = unilateral_energy_gradient(X, springs)

        foundation_potential, foundation_gradient = (
            foundation_incremental_potential_and_gradient(
                X,
                previous_nodes,
                reference_nodes,
                support_nodes,
                stiffness=foundation_k,
                damping=foundation_c,
                dt=dt,
            )
        )

        displacement = X - reference_nodes
        external_potential = -float(np.sum(external_force * displacement))

        total = structure_energy + foundation_potential + external_potential
        gradient = structure_gradient + foundation_gradient - external_force
        return float(total), gradient.reshape(-1)

    result = minimize(
        objective,
        initial_nodes.reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 10000,
            "ftol": 1e-14,
            "gtol": 1e-10,
            "maxls": 100,
        },
    )

    X = result.x.reshape(shape)
    _, flat_gradient = objective(result.x)
    residual = float(np.linalg.norm(flat_gradient))

    velocity = (X - previous_nodes) / dt
    foundation_force = kelvin_voigt_foundation_force(
        X,
        velocity,
        reference_nodes,
        support_nodes,
        stiffness=foundation_k,
        damping=foundation_c,
    )

    dz = np.asarray(
        [X[node, 2] - reference_nodes[node, 2] for node in loaded_nodes],
        dtype=float,
    )
    mean_dz = float(np.mean(dz))
    total_force = float(len(loaded_nodes) * force_per_node)
    effective_stiffness = (
        total_force / mean_dz
        if force_per_node > 0.0 and abs(mean_dz) > 1e-15
        else float("nan")
    )

    active = unilateral_active_state(X, springs, tol=1e-8)

    return {
        "nodes": X,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "residual": residual,
        "mean_dz": mean_dz,
        "effective_stiffness": effective_stiffness,
        "support_speed_norm": float(np.linalg.norm(velocity[support_nodes])),
        "foundation_force_norm": float(np.linalg.norm(foundation_force[support_nodes])),
        "active": active,
    }


def run_case(
    reference_nodes,
    springs,
    support_nodes,
    loaded_nodes,
    ramp_time,
    final_force_per_node,
    foundation_k,
    foundation_c,
    steps_per_ramp=80,
    hold_fraction=0.5,
):
    hold_time = hold_fraction * ramp_time
    final_time = ramp_time + hold_time
    dt = ramp_time / steps_per_ramp
    step_count = int(round(final_time / dt))
    times = np.linspace(0.0, step_count * dt, step_count + 1)

    previous_nodes = reference_nodes.copy()
    initial_nodes = reference_nodes.copy()
    rows = []

    for step, time in enumerate(times):
        if step == 0:
            force_per_node = 0.0
            result = {
                "nodes": reference_nodes.copy(),
                "success": True,
                "iterations": 0,
                "residual": 0.0,
                "mean_dz": 0.0,
                "effective_stiffness": float("nan"),
                "support_speed_norm": 0.0,
                "foundation_force_norm": 0.0,
                "active": unilateral_active_state(reference_nodes, springs, tol=1e-8),
            }
        else:
            force_per_node = smoothstep_ramp(
                time,
                final_force_per_node,
                ramp_time,
            )
            result = solve_increment(
                initial_nodes,
                previous_nodes,
                reference_nodes,
                springs,
                support_nodes,
                loaded_nodes,
                foundation_k,
                foundation_c,
                dt,
                force_per_node,
            )

        rows.append(
            {
                "ramp_time": ramp_time,
                "time": time,
                "dt": dt,
                "force_per_node": force_per_node,
                "total_applied_force": len(loaded_nodes) * force_per_node,
                "force_rate_per_node": smoothstep_ramp_rate(
                    time,
                    final_force_per_node,
                    ramp_time,
                ),
                "mean_dz": result["mean_dz"],
                "effective_stiffness": result["effective_stiffness"],
                "support_speed_norm": result["support_speed_norm"],
                "foundation_force_norm": result["foundation_force_norm"],
                "optimizer_success": result["success"],
                "optimizer_iterations": result["iterations"],
                "equilibrium_residual": result["residual"],
                "active_cables": result["active"]["active_cables"],
                "slack_cables": result["active"]["slack_cables"],
                "active_bars": result["active"]["active_bars"],
                "slack_bars": result["active"]["slack_bars"],
                "threshold_members": result["active"]["threshold"],
            }
        )

        previous_nodes = result["nodes"].copy()
        initial_nodes = result["nodes"].copy()

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_nodes, members = t4_with_two_supported_t3()
    support_nodes = np.asarray([9, 10, 11, 12, 13, 14], dtype=int)
    loaded_nodes = [3, 4, 5]

    # Use the robust unilateral prestress region established by the earlier
    # experiments rather than trying to resolve 0.38 vs 0.39 further.
    prestress_scale = 0.40
    foundation_k = 1.0
    foundation_c = 0.25
    final_force_per_node = 1e-4
    ramp_times = [0.1, 1.0, 10.0]

    q_hat = normalized_t4_force_density(reference_nodes)
    springs = build_unilateral_springs(
        reference_nodes,
        members,
        q_hat,
        prestress_scale=prestress_scale,
        stiffness=1.0,
    )

    print()
    print("=" * 104)
    print("T4 RATE-DEPENDENT LOADING WITH KELVIN-VOIGT BASEMENT ATTACHMENT")
    print("=" * 104)
    print(f"prestress scale:          {prestress_scale:.3f}")
    print(f"foundation stiffness k:   {foundation_k:.3e}")
    print(f"foundation damping c:     {foundation_c:.3e}")
    print(f"final force / node:       {final_force_per_node:.3e}")
    print("units:                    normalized/model units")

    all_rows = []

    for ramp_time in ramp_times:
        rows = run_case(
            reference_nodes,
            springs,
            support_nodes,
            loaded_nodes,
            ramp_time=ramp_time,
            final_force_per_node=final_force_per_node,
            foundation_k=foundation_k,
            foundation_c=foundation_c,
        )
        all_rows.extend(rows)

        end_ramp = min(rows, key=lambda row: abs(row["time"] - ramp_time))
        final = rows[-1]
        max_residual = max(row["equilibrium_residual"] for row in rows)
        failures = sum(not row["optimizer_success"] for row in rows)

        print()
        print(f"ramp time = {ramp_time:g}")
        print(f"  displacement at end of ramp: {end_ramp['mean_dz']:.12e}")
        print(f"  final held displacement:      {final['mean_dz']:.12e}")
        print(f"  final effective stiffness:    {final['effective_stiffness']:.12e}")
        print(f"  final foundation force norm:  {final['foundation_force_norm']:.12e}")
        print(f"  maximum solve residual:       {max_residual:.12e}")
        print(f"  optimizer failures:           {failures}")

    csv_path = OUTPUT_DIR / "rate_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Force-displacement curve: the most useful precursor to stress-strain.
    plt.figure(figsize=(8, 5))
    for ramp_time in ramp_times:
        rows = [r for r in all_rows if np.isclose(r["ramp_time"], ramp_time)]
        plt.plot(
            [r["mean_dz"] for r in rows],
            [r["total_applied_force"] for r in rows],
            label=f"ramp time={ramp_time:g}",
        )
    plt.xlabel("Mean central displacement")
    plt.ylabel("Total applied force")
    plt.title("Force-displacement response at different loading rates")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    force_displacement_path = OUTPUT_DIR / "force_vs_displacement.png"
    plt.savefig(force_displacement_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for ramp_time in ramp_times:
        rows = [r for r in all_rows if np.isclose(r["ramp_time"], ramp_time)]
        normalized_time = [r["time"] / ramp_time for r in rows]
        plt.plot(
            normalized_time,
            [r["mean_dz"] for r in rows],
            label=f"ramp time={ramp_time:g}",
        )
    plt.xlabel("Time / ramp time")
    plt.ylabel("Mean central displacement")
    plt.title("Rate-dependent displacement history")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    displacement_path = OUTPUT_DIR / "displacement_vs_normalized_time.png"
    plt.savefig(displacement_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for ramp_time in ramp_times:
        rows = [r for r in all_rows if np.isclose(r["ramp_time"], ramp_time)]
        plt.plot(
            [r["time"] / ramp_time for r in rows],
            [r["foundation_force_norm"] for r in rows],
            label=f"ramp time={ramp_time:g}",
        )
    plt.xlabel("Time / ramp time")
    plt.ylabel("Basement/foundation reaction force norm")
    plt.title("Kelvin-Voigt basement reaction")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    foundation_path = OUTPUT_DIR / "foundation_reaction_vs_normalized_time.png"
    plt.savefig(foundation_path, dpi=200)
    plt.close()

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {force_displacement_path}")
    print(f"Saved: {displacement_path}")
    print(f"Saved: {foundation_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
