"""
scan_critical_cables.py  

General-purpose tool:
- Given nodes + members, scan removal of each cable member.
- Report whether removal introduces extra mechanisms (rigidity test).
- Report whether removal worsens energetic stability (min Hessian eigenvalue).

Outputs:
- CSV file listing each cable and the before/after metrics.

Run:
  python experiments/cable_removal/scan_critical_cables.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.tensegrity.rigidity import Member, count_mechanisms
from src.tensegrity.energy import SpringMember, stability_index_energy_hessian


@dataclass(frozen=True)
class Framework:
    nodes: np.ndarray
    members: List[Member]
    spring_members: List[SpringMember]


def example_framework() -> Framework:
    """
    Replace this with your real tensegrity geometry later.

    This toy example:
    - 4 nodes in 3D
    - 6 cables
    - spring rest lengths induce mild prestress
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 0.9, 0.0],  # 2
            [0.5, 0.3, 0.8],  # 3
        ],
        dtype=float,
    )

    members = [
        Member(0, 1, "cable"),
        Member(1, 2, "cable"),
        Member(2, 0, "cable"),
        Member(0, 3, "cable"),
        Member(1, 3, "cable"),
        Member(2, 3, "cable"),
    ]

    spring_members: List[SpringMember] = []
    for m in members:
        L = float(np.linalg.norm(nodes[m.i] - nodes[m.j]))
        # cables: rest length slightly shorter (tension tendency)
        spring_members.append(SpringMember(i=m.i, j=m.j, kind=m.kind, k=1.0, L0=0.95 * L))

    return Framework(nodes=nodes, members=members, spring_members=spring_members)


def scan(fr: Framework, tol: float = 1e-9) -> List[dict]:
    nodes = fr.nodes

    mechs_before = count_mechanisms(nodes, fr.members, tol=tol)
    stab_before = stability_index_energy_hessian(nodes, fr.spring_members, eps=1e-6)

    rows: List[dict] = []
    for idx, mem in enumerate(fr.members):
        if mem.kind != "cable":
            continue

        reduced_members = fr.members[:idx] + fr.members[idx + 1 :]
        reduced_springs = fr.spring_members[:idx] + fr.spring_members[idx + 1 :]

        mechs_after = count_mechanisms(nodes, reduced_members, tol=tol)
        stab_after = stability_index_energy_hessian(nodes, reduced_springs, eps=1e-6)

        rows.append(
            {
                "removed_index": idx,
                "member_i": mem.i,
                "member_j": mem.j,
                "mechanisms_before": mechs_before,
                "mechanisms_after": mechs_after,
                "mechanisms_delta": mechs_after - mechs_before,
                "stability_min_eig_before": stab_before,
                "stability_min_eig_after": stab_after,
                "stability_delta": stab_after - stab_before,
            }
        )

    return rows


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "removed_index","member_i","member_j",
        "mechanisms_before","mechanisms_after","mechanisms_delta",
        "stability_min_eig_before","stability_min_eig_after","stability_delta"
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    fr = example_framework()
    rows = scan(fr)
    out_csv = Path("experiments/cable_removal/critical_cables_report.csv")
    write_csv(rows, out_csv)
    print(f"Wrote: {out_csv}")

    # quick console summary
    worst = sorted(rows, key=lambda r: (r["mechanisms_delta"], -r["stability_delta"]), reverse=True)
    print("\nTop candidates (worst removals first):")
    for r in worst[:5]:
        print(
            f"- remove ({r['member_i']},{r['member_j']}): "
            f"Δmech={r['mechanisms_delta']}, "
            f"Δstab={r['stability_delta']:.3e}"
        )


if __name__ == "__main__":
    main()

