"""
scan_bc_grid.py

Generate a toy "tensegrity stability chart" over (B, C) = (#bars, #cables).

This is a pipeline demo:
- generate random nodes
- randomly assign B bar members and C cable members
- define prestress via rest lengths L0
- compute energy-Hessian stability index (min eigenvalue)
- save results to CSV

Run:
  python experiments/stability_map/scan_bc_grid.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.tensegrity.energy import SpringMember, stability_index_energy_hessian


@dataclass
class TrialResult:
    B: int
    C: int
    trial: int
    stability_min_eig: float


def random_nodes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spread nodes in a 1x1x1 cube
    X = rng.random((n, 3))
    # Center around origin to reduce conditioning issues
    X = X - X.mean(axis=0, keepdims=True)
    return X


def all_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def build_random_members(
    nodes: np.ndarray,
    B: int,
    C: int,
    seed: int,
    k_bar: float = 5.0,
    k_cable: float = 1.0,
) -> List[SpringMember]:
    """
    Build a random member set with B bars and C cables.

    Prestress encoding:
    - bars: L0 slightly LONGER than current length (push/compression tendency)
    - cables: L0 slightly SHORTER than current length (pull/tension tendency)

    Note: This is a toy encoding; real tensegrities use inequality constraints.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(nodes, dtype=float)
    n = X.shape[0]

    pairs = all_pairs(n)
    rng.shuffle(pairs)

    if B + C > len(pairs):
        raise ValueError("Too many members requested for number of nodes")

    chosen = pairs[: (B + C)]
    bar_pairs = chosen[:B]
    cable_pairs = chosen[B:]

    members: List[SpringMember] = []

    # Bars
    for (i, j) in bar_pairs:
        L = float(np.linalg.norm(X[i] - X[j]))
        members.append(SpringMember(i=i, j=j, kind="bar", k=k_bar, L0=1.05 * L))

    # Cables
    for (i, j) in cable_pairs:
        L = float(np.linalg.norm(X[i] - X[j]))
        members.append(SpringMember(i=i, j=j, kind="cable", k=k_cable, L0=0.95 * L))

    return members


def run_scan(
    out_csv: Path,
    n_nodes: int = 8,
    B_range: range = range(1, 9),
    C_range: range = range(4, 25),
    trials_per_point: int = 5,
    base_seed: int = 123,
) -> List[TrialResult]:
    results: List[TrialResult] = []

    for B in B_range:
        for C in C_range:
            for t in range(trials_per_point):
                seed = base_seed + 100000 * B + 1000 * C + t
                nodes = random_nodes(n_nodes, seed=seed)

                # Skip if impossible
                max_members = n_nodes * (n_nodes - 1) // 2
                if B + C > max_members:
                    continue

                members = build_random_members(nodes, B=B, C=C, seed=seed)

                try:
                    s = stability_index_energy_hessian(nodes, members, eps=1e-6)
                except np.linalg.LinAlgError:
                    # Numerical failure; mark as NaN
                    s = float("nan")

                results.append(TrialResult(B=B, C=C, trial=t, stability_min_eig=s))

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "C", "trial", "stability_min_eig"])
        for r in results:
            w.writerow([r.B, r.C, r.trial, r.stability_min_eig])

    return results


def main() -> None:
    out_csv = Path("experiments/stability_map/results.csv")
    run_scan(out_csv=out_csv)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

