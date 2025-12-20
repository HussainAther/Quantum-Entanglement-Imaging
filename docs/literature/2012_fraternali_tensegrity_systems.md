# Fraternali & Rimoli — *Tensegrity Systems*

**Citation**  
Fraternali, F., & Rimoli, J. J.  
*Tensegrity Systems*  
(Springer / lecture notes / monograph)

---

## Why this reference matters
This work provides a rigorous mathematical framework for analyzing the stability of tensegrity structures. It formalizes the relationship between equilibrium, self-stress, mechanisms, and prestress stability using linear algebra (SVD of equilibrium matrices). This is directly relevant to determining whether a tensegrity structure collapses when a single cable is removed.

---

## Key concepts introduced

### 1. Equilibrium matrix
- Defines force balance at nodes.
- Denoted often as **A**.
- Nullspace of A → self-stress states.
- Nullspace of Aᵀ → infinitesimal mechanisms.

### 2. Self-stress vs mechanisms
- **Self-stress**: internal force patterns that satisfy equilibrium with no external loads.
- **Mechanisms**: deformation modes that cost zero energy at first order.
- A structure is potentially unstable if mechanisms exist without stabilizing prestress.

### 3. Prestress stability
- Even if mechanisms exist geometrically, sufficient prestress can stabilize the structure.
- Stability depends on the interaction between:
  - geometric stiffness
  - material stiffness
  - prestress-induced stiffness

### 4. Superstability
- A superstable tensegrity remains stable regardless of specific material properties.
- Achieved when:
  - the structure has self-stress,
  - no mechanisms exist,
  - and equilibrium is maintained under perturbations.

---

## Collapse and dimensional reduction

- Removing a single cable alters the equilibrium matrix.
- If removal increases the dimension of the nullspace of Aᵀ:
  - a new mechanism appears,
  - the structure can collapse or reduce dimensionality (e.g., 3D → 2D).
- Collapse corresponds mathematically to additional zero singular values appearing in the SVD.

---

## Connection to our project

This reference supports:
- A **general computational test** for collapse:
  1. Build equilibrium matrix A
  2. Compute SVD
  3. Remove one cable
  4. Recompute SVD
  5. Detect appearance of new mechanisms
- The idea of a **tensegrity stability chart**:
  - axes: (# bars, # cables)
  - stability metric: number

