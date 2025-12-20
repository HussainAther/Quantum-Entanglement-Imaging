"""
energy.py

Minimal energy-based stability analysis for tensegrity-like frameworks.

We model members as springs with energy:
  E = 0.5 * k * (||xi - xj|| - L0)^2

Then stability at an equilibrium configuration can be approximated by the
Hessian of E with respect to all node coordinates.

This includes prestress effects implicitly through rest lengths L0.

Notes:
- Uses numerical finite-difference Hessian (slow but simple & general).
- Good for small/medium node counts; can be optimized later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import numpy as np

MemberType = Literal["bar", "cable"]


@dataclass(frozen=True)
class SpringMember:
    i: int
    j: int
    kind: MemberType
    k: float
    L0: float


def total_energy(nodes: np.ndarray, members: Iterable[SpringMember]) -> float:
    X = np.asarray(nodes, dtype=float)
    E = 0.0
    for m in members:
        d = X[m.i] - X[m.j]
        L = float(np.linalg.norm(d))
        dl = L - m.L0
        E += 0.5 * m.k * dl * dl
    return float(E)


def numerical_hessian(
    f, x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Central-difference Hessian for scalar function f(x).
    x is 1D.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    fx = f(x)

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        for j in range(i, n):
            ej = np.zeros(n); ej[j] = 1.0

            if i == j:
                fpp = f(x + eps * ei)
                fmm = f(x - eps * ei)
                H[i, i] = (fpp - 2.0 * fx + fmm) / (eps * eps)
            else:
                fpp = f(x + eps * ei + eps * ej)
                fpm = f(x + eps * ei - eps * ej)
                fmp = f(x - eps * ei + eps * ej)
                fmm = f(x - eps * ei - eps * ej)
                Hij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                H[i, j] = Hij
                H[j, i] = Hij
    return H


def rigid_body_basis(nodes: np.ndarray) -> np.ndarray:
    """
    Build a basis for 6 rigid-body modes in 3D (translations + rotations)
    as vectors in R^(3n).

    Returns:
      B: (3n, 6) matrix whose columns span rigid-body motions.
    """
    X = np.asarray(nodes, dtype=float)
    n = X.shape[0]
    dof = 3 * n
    B = np.zeros((dof, 6), dtype=float)

    # Translations: x,y,z
    for a in range(n):
        B[3*a + 0, 0] = 1.0
        B[3*a + 1, 1] = 1.0
        B[3*a + 2, 2] = 1.0

    # Rotations about x,y,z using cross product: v = omega x r
    # omega_x = (1,0,0): v = (0, -z, y)
    # omega_y = (0,1,0): v = (z, 0, -x)
    # omega_z = (0,0,1): v = (-y, x, 0)
    for a in range(n):
        x, y, z = X[a]
        # rot x
        B[3*a + 0, 3] = 0.0
        B[3*a + 1, 3] = -z
        B[3*a + 2, 3] = y
        # rot y
        B[3*a + 0, 4] = z
        B[3*a + 1, 4] = 0.0
        B[3*a + 2, 4] = -x
        # rot z
        B[3*a + 0, 5] = -y
        B[3*a + 1, 5] = x
        B[3*a + 2, 5] = 0.0

    # Orthonormalize columns (QR)
    Q, _ = np.linalg.qr(B)
    return Q[:, :6]


def project_out_rigid_modes(H: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Project Hessian H onto subspace orthogonal to rigid-body modes.
    """
    Q = rigid_body_basis(nodes)  # (3n,6) orthonormal
    P = np.eye(H.shape[0]) - Q @ Q.T
    return P.T @ H @ P


def stability_index_energy_hessian(
    nodes: np.ndarray,
    members: List[SpringMember],
    eps: float = 1e-6,
    eig_tol: float = 1e-9,
) -> float:
    """
    Compute a simple stability index as the smallest eigenvalue of the Hessian
    projected off rigid-body motions.

    Returns:
      lambda_min (smallest eigenvalue). Positive => locally stable.
    """
    X = np.asarray(nodes, dtype=float)

    def f(flat: np.ndarray) -> float:
        Xr = flat.reshape((-1, 3))
        return total_energy(Xr, members)

    flat0 = X.reshape(-1)
    H = numerical_hessian(f, flat0, eps=eps)
    H_eff = project_out_rigid_modes(H, X)

    w = np.linalg.eigvalsh(H_eff)
    # Ignore tiny numerical junk near zero
    w_sorted = np.sort(w)
    # Return smallest eigenvalue (could be negative)
    return float(w_sorted[0])

