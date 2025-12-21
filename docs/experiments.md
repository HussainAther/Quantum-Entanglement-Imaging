# Experiments Guide

This repo contains small experiments to explore tensegrity stability and “collapse risk”
(e.g., removing a cable introduces mechanisms or destabilizes the system).

All generated artifacts should go into `outputs/`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
````

---

## Experiment: Critical cable scan

**Goal:** Identify “critical cables” whose removal increases mechanisms (rigidity) and/or reduces energy stability.

**Run:**

```bash
python scripts/run_experiment.py cable-scan
```

**Outputs:**

* `outputs/critical_cables_report.csv`

**Interpretation (high-level):**

* Larger `mechanisms_delta` suggests the structure becomes more floppy when that cable is removed.
* More negative `stability_delta` suggests removal worsens local energetic stability.

---

## Experiment: Stability map scan (toy)

**Goal:** Build a first “nuclide-chart-style” map over (B,C) = (#bars, #cables).
This is currently a toy generator to validate the pipeline; later we will swap in canonical tensegrity families.

**Run:**

```bash
python scripts/run_experiment.py stability-map
```

**Outputs:**

* `outputs/stability_map_results.csv`

---

## Experiment: Plot stability map

**Goal:** Visualize the stability map as a heatmap.

**Run:**

```bash
python scripts/run_experiment.py stability-plot
```

**Outputs:**

* `outputs/stability_map.png`

---

## Notes / caveats

* The current energy model treats members as springs and does not yet enforce
  cable tension-only or bar compression-only constraints.
* The rigidity matrix check is a first-pass bar-joint approximation.
* These experiments are intended to support rapid iteration on stability metrics and workflows.

