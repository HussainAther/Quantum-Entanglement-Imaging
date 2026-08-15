"""
plot_stability_map.py

Read results.csv and plot a heatmap over (B,C).
We aggregate trials by taking the median stability at each (B,C).

Run:
  python experiments/stability_map/plot_stability_map.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_results(csv_path: Path) -> Dict[Tuple[int, int], List[float]]:
    data: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            B = int(row["B"])
            C = int(row["C"])
            s = float(row["stability_min_eig"])
            if np.isfinite(s):
                data[(B, C)].append(s)
    return data


def main() -> None:
    csv_path = Path("outputs/stability_map_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Run scan_bc_grid.py first to generate results.csv")

    data = read_results(csv_path)

    Bs = sorted({k[0] for k in data.keys()})
    Cs = sorted({k[1] for k in data.keys()})

    if not Bs or not Cs:
        raise RuntimeError("No data found in results.csv")

    grid = np.full((len(Cs), len(Bs)), np.nan, dtype=float)

    for iC, C in enumerate(Cs):
        for iB, B in enumerate(Bs):
            vals = data.get((B, C), [])
            if vals:
                grid[iC, iB] = float(np.median(vals))

    plt.figure(figsize=(10, 6))
    # imshow expects row-major; we'll map rows->C and cols->B
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(label="Median min-eigenvalue (stability index)")
    plt.xticks(range(len(Bs)), Bs)
    plt.yticks(range(len(Cs)), Cs)
    plt.xlabel("B = number of bars")
    plt.ylabel("C = number of cables")
    plt.title("Toy Tensegrity Stability Map (Energy-Hessian metric)")

    out_png = Path("outputs/stability_map.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

