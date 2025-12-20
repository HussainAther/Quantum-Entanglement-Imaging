"""
demo_scan.py

Demo: remove-one-cable scan on a small toy structure.

IMPORTANT:
This demo uses a toy framework. Replace nodes/members with your real tensegrity geometry.

Run:
  python experiments/cable_removal/demo_scan.py
"""

import numpy as np

from src.tensegrity.rigidity import Member, remove_one_cable_scan


def main() -> None:
    # Toy 3D framework:
    # 4 nodes in a tetrahedron-ish configuration
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 0.9, 0.0],  # 2
            [0.5, 0.3, 0.8],  # 3
        ],
        dtype=float,
    )

    # Members: treat some as "cables" for removal scanning
    members = [
        Member(0, 1, "cable"),
        Member(1, 2, "cable"),
        Member(2, 0, "cable"),
        Member(0, 3, "cable"),
        Member(1, 3, "cable"),
        Member(2, 3, "cable"),
    ]

    results = remove_one_cable_scan(nodes, members, tol=1e-9)

    print("Remove-one-cable scan results:")
    for idx, mem, before, after in results:
        delta = after - before
        flag = "  <-- increases mechanisms" if delta > 0 else ""
        print(f"- remove idx={idx}, cable=({mem.i},{mem.j}), mechanisms {before} -> {after}{flag}")


if __name__ == "__main__":
    main()

