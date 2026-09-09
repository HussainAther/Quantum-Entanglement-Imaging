"""Sweep equilibrium prestress in the canonical 3-strut prism."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tensegrity.energy import equilibrium_residual_norm, stability_index_energy_hessian
from tensegrity.examples import three_strut_prism


def run(scales: np.ndarray, output: Path) -> None:
    rows = []
    for scale in scales:
        nodes, _, springs = three_strut_prism(prestress_scale=float(scale))
        residual = equilibrium_residual_norm(nodes, springs)
        stability = stability_index_energy_hessian(
            nodes, springs, eps=2e-5, eig_tol=1e-6
        )
        rows.append(
            {
                "prestress_scale": float(scale),
                "equilibrium_residual": residual,
                "lambda_min": stability.lambda_min,
                "negative_modes": stability.negative_modes,
                "zero_modes": stability.zero_modes,
                "positive_modes": stability.positive_modes,
                "classification": stability.classification,
            }
        )
        print(
            f"scale={scale:.3f}  residual={residual:.3e}  "
            f"lambda_min={stability.lambda_min:.6e}  "
            f"{stability.classification}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", dest="minimum", type=float, default=0.0)
    parser.add_argument("--max", dest="maximum", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prism_prestress_sweep.csv"),
    )
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be >= 2")
    if not (0.0 <= args.minimum <= args.maximum < 1.0):
        raise SystemExit("require 0 <= min <= max < 1")
    run(np.linspace(args.minimum, args.maximum, args.steps), args.output)


if __name__ == "__main__":
    main()
