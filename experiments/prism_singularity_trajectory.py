"""Track the approach to singularity after prism cable ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tensegrity.energy import (
    energy_gradient,
    total_energy,
)
from tensegrity.examples import three_strut_prism
from tensegrity.relaxation import rigid_align_to_reference
from tensegrity.rigidity import rigidity_matrix


def force_density_vector(nodes, springs):
    values = []

    for spring in springs:
        d = nodes[spring.i] - nodes[spring.j]
        length = float(np.linalg.norm(d))

        values.append(
            spring.k
            * (length - spring.L0)
            / length
        )

    return np.asarray(values)


def main():
    nodes, members, springs = three_strut_prism(
        prestress_scale=0.20
    )

    output_dir = Path(
        "outputs/prism_singularity_trajectory"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for removed_idx, removed_member in enumerate(members):
        if removed_member.kind != "cable":
            continue

        reduced_members = (
            members[:removed_idx]
            + members[removed_idx + 1 :]
        )

        reduced_springs = (
            springs[:removed_idx]
            + springs[removed_idx + 1 :]
        )

        X0 = nodes.copy()
        X = nodes.copy()

        step_size = 0.2
        min_step_size = 1e-14
        backtrack_factor = 0.5
        armijo = 1e-4

        rows = []

        max_iterations = 3000

        for iteration in range(max_iterations + 1):
            grad = energy_gradient(
                X,
                reduced_springs,
            )

            residual = float(
                np.linalg.norm(grad)
            )

            energy = total_energy(
                X,
                reduced_springs,
            )

            R = rigidity_matrix(
                X,
                reduced_members,
            )

            singular_values = np.linalg.svd(
                R,
                compute_uv=False,
            )

            q = force_density_vector(
                X,
                reduced_springs,
            )

            smallest = singular_values[-1]

            second_smallest = (
                singular_values[-2]
                if len(singular_values) >= 2
                else np.nan
            )

            rows.append(
                {
                    "iteration": iteration,
                    "energy": energy,
                    "residual": residual,
                    "q_norm": float(
                        np.linalg.norm(q)
                    ),
                    "sigma_min": smallest,
                    "sigma_second_min": (
                        second_smallest
                    ),
                    "condition_number": (
                        singular_values[0]
                        / smallest
                        if smallest > 0
                        else np.inf
                    ),
                }
            )

            if iteration == max_iterations:
                break

            grad_sq = residual ** 2
            trial_step = step_size
            accepted = False

            while trial_step >= min_step_size:
                candidate = (
                    X
                    - trial_step * grad
                )

                candidate = rigid_align_to_reference(
                    candidate,
                    X0,
                )

                candidate_energy = total_energy(
                    candidate,
                    reduced_springs,
                )

                if (
                    candidate_energy
                    <= energy
                    - armijo
                    * trial_step
                    * grad_sq
                ):
                    X = candidate

                    step_size = min(
                        0.2,
                        trial_step
                        / backtrack_factor,
                    )

                    accepted = True
                    break

                trial_step *= backtrack_factor

            if not accepted:
                break

        path = output_dir / (
            f"remove_{removed_idx:02d}.csv"
        )

        with path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

        final = rows[-1]

        print(
            f"remove {removed_idx:2d} "
            f"({removed_member.i},"
            f"{removed_member.j}): "
            f"iters={final['iteration']}, "
            f"residual={final['residual']:.3e}, "
            f"sigma1={final['sigma_min']:.3e}, "
            f"sigma2="
            f"{final['sigma_second_min']:.3e}, "
            f"q={final['q_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
