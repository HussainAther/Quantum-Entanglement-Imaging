"""
scripts/run_experiment.py

Unified CLI to run common experiments.

Usage examples:
  python scripts/run_experiment.py stability-map
  python scripts/run_experiment.py stability-plot
  python scripts/run_experiment.py cable-scan
"""

from __future__ import annotations

import sys
from pathlib import Path
import subprocess


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <stability-map|stability-plot|cable-scan>")
        raise SystemExit(2)

    task = sys.argv[1].strip().lower()

    # Ensure outputs dir exists
    Path("outputs").mkdir(exist_ok=True)

    if task == "stability-map":
        run([sys.executable, "experiments/stability_map/scan_bc_grid.py"])
        print("Done. CSV should be in experiments/stability_map/ (consider moving to outputs/ next PR).")
        return

    if task == "stability-plot":
        run([sys.executable, "experiments/stability_map/plot_stability_map.py"])
        print("Done. PNG should be in experiments/stability_map/ (consider moving to outputs/ next PR).")
        return

    if task == "cable-scan":
        run([sys.executable, "experiments/cable_removal/scan_critical_cables.py"])
        print("Done. CSV should be in experiments/cable_removal/ (consider moving to outputs/ next PR).")
        return

    raise SystemExit(f"Unknown task: {task}")


if __name__ == "__main__":
    main()

